---
title: Elasticsearch-native hybrid retrieval (dense_vector kNN + RRF)
status: ready-for-agent
labels: [ready-for-agent]
---

# Elasticsearch-native hybrid retrieval (dense_vector kNN + RRF)

## Problem Statement

Today the funnel splits retrieval across two engines: Elasticsearch serves the
keyword pre-filter (Chinese/ik) while pgvector serves ANN, and the funnel merges
their scores with fixed coefficients (`0.6·exact + 0.35·ann + 0.05·metadata`).
Those coefficients and the hybrid short-circuit thresholds are rough constants
that SPEC.md already flags as needing calibration against real traffic. On top
of that, embeddings are stored only in Postgres (`chunk_embeddings`), so a
hybrid query touches two stores, two ranking models, and a hand-rolled score
merge that can disagree with either engine's native ordering.

The user wants embeddings integrated with Elasticsearch so that keyword and
semantic retrieval live in the same engine and are fused natively — removing the
cross-store merge and its calibration problem entirely.

## Solution

Elasticsearch becomes the single retrieval backend. The index gains an indexed
`dense_vector(1024)` cosine field; the seed script writes each document's BGE-M3
embedding into Elasticsearch alongside its content; and the funnel issues one ES
hybrid search — `match` on the ik-analyzed `content` plus `knn` on the
`dense_vector`, fused by **reciprocal rank fusion (RRF)** — for hybrid mode,
BM25-only for keyword mode, and kNN-only for semantic mode.

RRF (rank-based, not score-based) replaces the weighted-scoring merge and the
short-circuit heuristic: no coefficient calibration, no "is the keyword stage
confident enough" decision — Elasticsearch owns the ranking. Postgres keeps its
content-store role (`fetch_by_id`) and the tsvector keyword fallback (English/
code exact matching when ES errors after startup). The pgvector ANN path, the
pg_trgm fallback, and the weighted-scoring pipeline are retired from active
wiring.

## User Stories

1. As an agent, I want `search` in hybrid mode to return results fused from
   keyword and semantic signals by Elasticsearch itself, so that relevance is
   decided by one engine's native ranking rather than a hand-rolled score merge.
2. As an agent, I want a vague, intent-based Chinese query (e.g. "连接失败") to
   return the semantically nearest documents, so that I find context even when
   my wording does not match the corpus verbatim.
3. As an agent, I want a vague, intent-based English query to return the
   semantically nearest documents, so that synonym/paraphrase queries still
   retrieve the right docs.
4. As an agent, I want `keyword_search` on an exact term (error code, function
   name, precise Chinese phrase) to return precise BM25 matches with highlighted
   terms, so that exact lookups stay fast and precise without any embedding
   cost.
5. As an agent, I want `vector_search` to run kNN against Elasticsearch only, so
   that semantic-only search is exactly one embed call plus one ES call.
6. As an agent, I want hybrid results whose ordering does not depend on
   uncalibrated 0.6/0.35/0.05 coefficients, so that I can trust the ranking out
   of the box.
7. As an agent, I want the search response shape (`id`, `source`, `score`,
   `snippet`, `matched_via`, `matched_ann`) to remain unchanged, so that my
   existing tool-use patterns keep working.
8. As an agent, I want keyword-matched snippets to keep their `<em>` highlight,
   so that I can judge relevance at a glance.
9. As an agent, I want ANN-only hits to fall back to a truncated snippet, so
   that semantic matches still return usable context.
10. As an agent, I want `fetch_by_id` to keep returning full content from
    Postgres, so that progressive disclosure still works.
11. As an agent, I want a clear error when a semantic/hybrid query needs
    Elasticsearch and it is unreachable, so that I know the failure mode instead
    of getting empty results.
12. As an operator, I want the seed script to write embeddings into the ES index
    alongside content, so that the index is self-sufficient for hybrid search.
13. As an operator, I want embeddings stored only in Elasticsearch, so that there
    is no second vector store to keep in sync.
14. As an operator, I want the embedding dimension (1024) to be enforced at index
    time, so that a wrong embedder fails loudly instead of silently producing
    garbage kNN.
15. As an operator, I want keyword search to keep working through the Postgres
    tsvector path when Elasticsearch errors after startup, so that exact-term
    lookups survive an ES outage mid-flight.
16. As an operator, I want the server to keep failing fast at startup when
    Elasticsearch is unreachable, so that misconfiguration surfaces immediately
    rather than on the first call.
17. As an operator, I want the example scripts and docs updated for the new
    index mapping, so that a fresh checkout can stand up a working hybrid search.
18. As an operator, I want RRF window/rank parameters configurable via env, so
    that fusion behavior can be tuned without code changes.
19. As a maintainer, I want the funnel restructured around a single retrieval
    backend, so that there is exactly one retrieval path to reason about and
    test.
20. As a maintainer, I want the funnel unit/integration test seams preserved, so
    that tests keep running against real backends without the MCP layer.
21. As a maintainer, I want the pgvector/trigram weighted-scoring path removed
    from active wiring, so that dead backend code does not accumulate drift.
22. As an operator, I want the architecture diagram and reference docs to show
    embeddings integrated with Elasticsearch, so that the docs match the running
    system.

## Implementation Decisions

- **Retrieval architecture**: Elasticsearch becomes the sole retrieval engine.
  Postgres keeps two roles: the content store (`fetch_by_id`) and the tsvector
  keyword fallback. The pgvector ANN stage, the pg_trgm fallback, the weighted
  scoring merge, and the hybrid short-circuit heuristic are removed from the
  active funnel. The startup contract is unchanged: both Postgres and
  Elasticsearch are health-checked at startup.

- **Funnel restructure (rag-core)**: `RetrievalFunnel` now composes a single
  retrieval backend, the embedder, and the content store. A new
  `RetrievalBackend` trait replaces the `PreFilterStrategy` list + `AnnClient`
  split, and `ScoringConfig` / `ShortCircuitConfig` plus the merge/normalize
  logic are deleted. Their replacement is `RrfConfig` (window_size, rank_constant).
  Mode mapping: `keyword` → BM25-only request; `semantic` → embed query, then
  kNN-only request; `hybrid` → embed query, then a fused BM25+kNN request with
  RRF. The trait shape (decision-rich, from the design discussion):

  ```rust
  pub enum RetrievalMode { Keyword, Semantic, Hybrid }

  #[async_trait]
  pub trait RetrievalBackend: Send + Sync {
      /// Keyword: keyword = Some(query), query_vector = None.
      /// Semantic: keyword = None, query_vector = Some(embedding).
      /// Hybrid: keyword = Some(query), query_vector = Some(embedding).
      async fn search(
          &self,
          mode: RetrievalMode,
          keyword: Option<&str>,
          query_vector: Option<&[f32]>,
          limit: usize,
      ) -> RagResult<Vec<RankedHit>>;
  }
  ```

  `RankedHit` carries id, source, score, snippet, and provenance fields so the
  funnel can produce `ScoredResult`s directly (one type replaces the
  `PreFilterHit`/`AnnHit` split).

- **Response contract unchanged**: `ScoredResult` keeps its fields (`id`,
  `source`, `score`, `snippet`, `matched_via`, `matched_ann`) so the MCP tool
  response shapes are stable. Documented simplification: ES's RRF response does
  not expose which clause matched a given hit, so `matched_via` is set to the
  ES strategy kind and `matched_ann` reflects whether the request carried a kNN
  clause — provenance is request-level, not per-hit.

- **ES index mapping**: `ensure_index` adds
  `embedding: { type: "dense_vector", dims: 1024, index: true, similarity: "cosine" }`
  alongside the existing ik-analyzed `content` and keyword `source`. Mapping
  changes require index recreation, so the example index lifecycle becomes
  create-or-bump (delete + recreate), driven by the seed script. Requires ES
  >= 8.12 for the `rank: { rrf }` API (the project's `es-ik` image is 8.15.3).

- **ES client**: `index_document` accepts an embedding array and writes it into
  the document body. Search builds one of three request shapes — query-only
  (keyword), knn-only (semantic), or query + knn + `rank: { rrf }` (hybrid) —
  and response parsing keeps the existing pure-parser pattern, now also reading
  RRF scores. kNN `num_candidates` is derived from `limit` and kept small.

- **Embedding path**: the funnel embeds the query with the existing embedder
  (remote Ollama preferred, local ONNX fallback — unchanged selection) and
  passes the vector to the backend. The embedder's 1024-dim output must equal
  the `dense_vector` dims; ES enforces this at index time and the seed script
  validates it.

- **Seed script**: mirrors documents into ES with their embeddings; stops
  writing Postgres `chunk_embeddings` (removing the dual-store drift); keeps the
  1024-dim check; `--no-embed` still means keyword-only data; `--delete` cleans
  the ES docs (now including vectors) and the Postgres rows it owns.

- **Config**: `RAG_MCP_RRF_WINDOW_SIZE` (default 100) and
  `RAG_MCP_RRF_RANK_CONSTANT` (default 60) are optional env overrides. No new
  required variables.

- **Keyword fallback**: the existing fallback-wrapper pattern is retained for
  the keyword facet only — ES-primary → tsvector fallback when ES errors during
  a query. Semantic and hybrid queries surface the ES error as today's clear
  error classes, because kNN cannot fall back to tsvector.

- **Schema**: no SQL migration. `chunk_embeddings` remains in the schema (owned
  by the external ingestion process) but is no longer read by the server; the
  reference docs mark it deprecated as the ANN store.

## Testing Decisions

- **What "good" looks like**: tests assert on external behavior — returned
  ranking, result ids, response shape, and error classes for representative
  query shapes (exact Chinese term, exact English/code identifier, vague Chinese
  query, vague English query) — not on internal call counts or request details.
  The exact-term queries must still return precise matches (validating that RRF
  does not drown exact hits), and vague queries must surface semantically
  relevant docs (validating the kNN clause contributes).

- **Primary seam (one)**: `RetrievalFunnel` built by `build_funnel` — the exact
  code path `main` runs — exercised through the funnel's public API
  (`search` / `keyword_search` / `vector_search` / `fetch_by_id`) against a real
  Elasticsearch cluster that has the `dense_vector` index and real embeddings.
  These tests are env-gated (`RAG_MCP_ELASTICSEARCH_URL` plus an embedding
  backend) and skipped, never failed, when unset. MCP tool handlers are not
  exercised here; they are thin and their mapping is covered by the existing
  handler tests.

- **Modules tested**:
  - `rag-mcp` integration tests — funnel over real ES-with-vectors (the primary
    seam).
  - `rag-mcp` ES client — pure unit tests for parsing the three response shapes
    (BM25, kNN, RRF) and building the three request bodies.
  - `rag-core` — unit tests with a fake `RetrievalBackend`: mode mapping
    (keyword/semantic/hybrid), RRF parameter mapping, and keyword→tsvector
    fallback when the backend errors.
  - `rag-mcp` MCP handlers — unchanged response shape, so existing parameter-
    mapping tests stand as-is.

- **Prior art**: the existing `integration.rs` suite (per-test tokio runtime,
    uniquely-named index, unique per-process fixture tokens, poll-for-visibility
    because ES is near-real-time) and the real-ES tests in `es_prefilter.rs`
    (unique `es-`-prefixed index, env-gated, cleanup in the test body). The
    funnel unit tests in `rag-core` use fake strategies; the new tests follow
    that same fake-based pattern.

## Out of Scope

- Postgres → Elasticsearch CDC sync (separately tracked; the seed script remains
  the demo mirror).
- Query-time `source` / `language` filtering (accepted but unused, as today).
- Cross-encoder / LLM-based reranking.
- Removing `chunk_embeddings` from the schema or migrating the external
  ingestion process that writes it.
- Removing Postgres entirely (content store and keyword fallback remain).
- Filtered kNN or other ES knn query variants beyond plain indexed kNN.
- Per-hit clause provenance beyond the `matched_ann` request-level
  simplification.

## Further Notes

- ES kNN at 1024 dims costs memory and disk for the HNSW graph — negligible at
  demo-corpus scale, a real consideration at scale; noted for capacity planning,
  not handled here.
- Docs to update alongside the code: `examples/reference.md` (index mapping,
  tool semantics, funnel description), `examples/quickstart.md`, and
  `examples/architecture.svg` (ES box gains the `dense_vector` field, pgvector
  drops out of the query path, the seed arrow shows vectors written into ES).
- This spec supersedes the premise of the "pgvector AnnClient" and "Wire real
  hybrid search end-to-end" tickets (weighted scoring + short-circuit), which
  should be marked superseded on the tracker.
- The retired funnel (pgvector, trigram, weighted scoring) stays in git history;
  there is no runtime toggle to switch back.
