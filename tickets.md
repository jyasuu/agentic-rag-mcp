# Tickets: Wire real backends for agentic-rag-mcp

Implements the "Next steps" from README.md / SPEC.md: real Postgres, Elasticsearch,
pgvector, and BGE-M3 backends behind the existing `PreFilterStrategy` / `AnnClient` /
`Embedder` / `ContentStore` trait seams in `rag-core`, replacing the `NotImplemented*`
stubs in `crates/rag-mcp/src/backends.rs`. See `SPEC.md` for full design rationale.

Work the **frontier**: any ticket whose blockers are all done.

## Prefactor: Postgres + Elasticsearch connection plumbing ✅ done

**What to build:** Config/env plumbing and connection setup for Postgres (pool) and
Elasticsearch (client), wired into `AppState` in `main.rs`, with startup health checks.
No new tool-facing behavior — this is the shared substrate every Postgres- and
ES-backed ticket below needs, done first per "make the change easy, then make the
easy change."

**Blocked by:** None — can start immediately

- [x] Postgres connection pool configurable via env, constructed at startup
- [x] Elasticsearch client configurable via env, constructed at startup
- [x] Server fails fast with a clear error if either backend is unreachable at startup
- [x] No change to existing tool behavior (still returns "not wired yet" stub errors)

## Postgres tsvector PreFilterStrategy

**What to build:** `keyword_search` returns real ranked hits for English/code queries
(function names, error codes) via Postgres `tsvector`.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing

- [ ] `PreFilterStrategy` implemented against `tsvector`
- [ ] Exact English/code identifier queries return correct, ranked matches
- [ ] Wired into `RetrievalFunnel`'s prefilter list for English/code content

## pg_trgm fallback PreFilterStrategy

**What to build:** Keyword search still functions when Elasticsearch is unavailable
or unsynced, via `pg_trgm` fuzzy matching.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing

- [ ] `PreFilterStrategy` implemented against `pg_trgm`
- [ ] Produces reasonable fuzzy matches independent of ES availability

## Elasticsearch + ik_analyzer PreFilterStrategy

**What to build:** Chinese-language keyword search with proper word segmentation,
returning ES-highlight-based snippets so matched terms are visible in context.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing

- [ ] `PreFilterStrategy` implemented against Elasticsearch using `ik_analyzer`
- [ ] Chinese keyword queries return correctly segmented, precise matches
- [ ] Snippets use ES highlight to show matched terms in context

## BGE-M3 Embedder via ort

**What to build:** Local query embedding generation with no external embedding API
dependency, cost, or latency.

**Blocked by:** None — can start immediately

- [ ] `Embedder` implemented using BGE-M3 via `ort` (ONNX Runtime)
- [ ] Produces embeddings for both Chinese and English/code query text
- [ ] No external API calls made during embedding

## pgvector AnnClient

**What to build:** Real cosine/L2 ANN search against pgvector, given a precomputed
query embedding.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing

- [ ] `AnnClient` implemented against pgvector
- [ ] Returns ranked ANN hits with similarity scores for a given embedding

## Wire real vector_search end-to-end

**What to build:** `vector_search` tool returns real semantic results for vague,
intent-based queries, using the real embedder and ANN client together.

**Blocked by:** BGE-M3 Embedder via ort, pgvector AnnClient

- [ ] `vector_search` tool call returns real (non-stub) ranked semantic results
- [ ] Works for both Chinese and English/code queries

## Postgres fetch_by_id ContentStore

**What to build:** `fetch_by_id` returns real full content for a given chunk/document
id, after an agent has reviewed snippets from a prior search call.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing

- [ ] `ContentStore` implemented against Postgres
- [ ] `fetch_by_id` tool call returns full content for a valid id
- [ ] Returns a clear not-found error for an invalid id

## Wire real hybrid search end-to-end

**What to build:** The default `search` tool runs the real funnel — keyword
pre-filter, conditional ANN short-circuit, and weighted scoring — against real
backends, with no explicit `mode` required.

**Blocked by:** Postgres tsvector PreFilterStrategy, Wire real vector_search
end-to-end

(Note: Elasticsearch + ik_analyzer PreFilterStrategy is a viable alternative to the
tsvector ticket for this dependency — either confident keyword strategy unblocks
this ticket; both are recommended before shipping given the corpus is majority
Chinese.)

- [ ] `search` tool defaults to hybrid mode and returns real ranked results
- [ ] ANN stage is skipped when keyword pre-filter is already confident
- [ ] `mode: "keyword"` / `"semantic"` explicit overrides work against real backends
- [ ] Funnel wiring uses real backends end-to-end (funnel logic itself already
      covered by `rag-core` unit tests — this ticket is integration wiring only)

## Postgres to Elasticsearch CDC sync

**What to build:** Elasticsearch index stays consistent with Postgres via logical
replication (reusing `pg_x`-style CDC patterns), without dual-write inconsistency
risk.

**Blocked by:** Prefactor: Postgres + Elasticsearch connection plumbing,
Elasticsearch + ik_analyzer PreFilterStrategy

- [ ] Consumer service applies Postgres insert/update/delete changes to the ES index
- [ ] ES index reflects Postgres changes within an acceptable lag window
- [ ] No dual-write path exists elsewhere in the system
