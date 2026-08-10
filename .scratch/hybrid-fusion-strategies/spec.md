---
title: Configurable hybrid fusion — client-side RRF, score-normalized weighted mean, or server-side rank: { rrf }
status: ready-for-agent
labels: [ready-for-agent]
---

# Configurable hybrid fusion — client-side RRF, score-normalized weighted mean, or server-side rank: { rrf }

## Problem Statement

Today hybrid mode is hard-wired to one fusion method: the retrieval backend
issues two Elasticsearch requests (BM25 on the ik-analyzed `content`, kNN on
the indexed `embedding` dense_vector) and merges them **client-side with
reciprocal rank fusion** (`rrf_fuse`). That choice was forced by licensing —
Elasticsearch's native `rank: { rrf }` API is Platinum/Enterprise-only — but it
is not the only fusion option, and the operator currently cannot choose.

The two candidates differ in a fundamental way:

- **Score-based fusion** — normalize each sub-query's raw scores (min-max or
  L2), then combine via a weighted mean. Uses score *magnitude*: a much
  stronger BM25 match still outranks a middling one after normalization. Allows
  explicit per-list weighting (e.g. 0.3 keyword / 0.7 vector). Free and
  open-source. Sensitivity: min-max normalization is outlier-sensitive.
- **Pure RRF** — score = sum of `1 / (k + rank + 1)` per list. No score
  magnitude, no per-list weighting by default, robust to score-scale mismatch.
  Server-side this needs the paid ES license; the project's client-side
  `rrf_fuse` already implements the same math for free.

The user wants the fusion strategy to be a **config-selected, engine-side
supported choice**, so an operator can pick the method whose assumptions match
their corpus and their trust in BM25-vs-cosine score comparability — without
recompiling.

## Solution

Add a `HybridFusion` config, selected at startup from environment variables,
that the ES retrieval backend reads when serving hybrid mode. Three strategies:

1. **Client RRF** (default, current behavior) — two ES requests, fused by the
   existing client-side `rrf_fuse`. Runs on ES's free license.
2. **Normalized weighted mean** — two ES requests, fused client-side by a new
   pure function that min-max or L2-normalizes each list's raw scores and
   combines them with a weighted arithmetic mean. Same free-license footprint
   as option 1; adds score magnitude and per-list weights.
3. **Server-side RRF** — a single ES request carrying `query` + `knn` +
   `rank: { rrf }`, ES-native fusion. The request shape is built and sent, and
   the license requirement (Platinum/Enterprise) is documented as an
   engine-side gate rather than a silent client-side workaround.

Keyword and semantic modes are untouched — only the hybrid arm dispatches on the
fusion strategy. The MCP tool surface, the response shapes (`id`, `source`,
`score`, `snippet`, `matched_via`, `matched_ann`), and the funnel contract stay
identical.

## User Stories

1. As an operator, I want to choose the hybrid fusion method at startup via
   environment configuration, so that I can pick the method whose assumptions
   match my corpus without code changes.
2. As an operator, I want `client-rrf` to remain the default fusion method, so
   that existing deployments upgrade without changing behavior.
3. As an operator, I want client-side RRF to keep working on Elasticsearch's
   free license, so that hybrid retrieval does not force a paid license on me.
4. As an operator, I want a score-based fusion method (normalize then weighted
   mean) available on the free license, so that I can use score magnitude and
   per-list weights without paying for ES Platinum/Enterprise.
5. As an operator, I want to configure per-list fusion weights (e.g. keyword
   0.3 / vector 0.7), so that I can emphasize the sub-signal I trust more.
6. As an operator, I want to choose the score-normalization method (min-max or
   L2) for the weighted-mean strategy, so that I can avoid outlier-sensitive
   min-max behavior when my score distributions demand it.
7. As an operator, I want the server-side `rank: { rrf }` request shape to be
   built and sent when configured, so that ES-native RRF fusion is available on
   clusters whose license permits it.
8. As an operator, I want the server-side `rank: { rrf }` license requirement
   documented as an engine-side gate, so that I understand why a basic-license
   cluster rejects the request instead of getting an opaque error.
9. As an operator, I want an explicit, clearly-worded error when a configured
   fusion method is rejected by the engine (e.g. server RRF on a
   non-licensed cluster), so that the failure mode is obvious.
10. As an agent, I want `search` in hybrid mode to keep returning the same
    ranked-result shape regardless of which fusion strategy is configured, so
    that my tool-use patterns keep working.
11. As an agent, I want hybrid mode to keep surfacing both BM25-matched and
    kNN-matched hits, so that neither signal is silently dropped by the chosen
    fusion.
12. As an agent, I want keyword-clause `<em>` highlights to survive client-side
    fusion in the normalized-mean strategy exactly as they do under RRF, so
    that I can judge relevance at a glance.
13. As an agent, I want ANN-only hits to keep their truncated-content snippet
    fallback in the normalized-mean strategy, so that semantic matches still
    return usable context.
14. As an agent, I want the fused score to be positive and deterministic for
    the normalized-mean strategy, so that ranking is stable across runs.
15. As a maintainer, I want the fusion strategies implemented behind the
    existing `RetrievalBackend::search` seam, so that there is still exactly
    one retrieval path to reason about and test.
16. As a maintainer, I want the normalized-mean fusion math to be a pure
    function like `rrf_fuse`, so that it is unit-testable without a cluster.
17. As a maintainer, I want the server-side RRF request to be built by a pure
    request-builder function like the existing keyword/semantic builders, so
    that its JSON shape is unit-testable without a cluster.
18. As a maintainer, I want config parsing of the new fusion environment
    variables to fail fast with a clear message on invalid values, matching the
    existing `Config::from_env` behavior.
19. As a maintainer, I want the hybrid integration tests to exercise each
    configured strategy end-to-end against real Elasticsearch, so that the
    wiring (not just the math) is verified.
20. As a maintainer, I want the default strategy to be exercised by the exact
    same real-ES hybrid test that exists today, so that the current free-license
    path stays regression-covered.

## Implementation Decisions

- **Fusion strategy lives in rag-core config**: a new
  `HybridFusion` enum (serde snake_case) with variants `ClientRrf`,
  `NormalizedMean`, `ServerRrf`, plus a `ScoreNormalization` enum (`MinMax`,
  `L2`) and a per-list weight pair used only by `NormalizedMean`. Holds the
  arithmetic as a weighted **arithmetic** mean; harmonic/geometric means are out
  of scope (see Out of Scope). This travels alongside `RrfConfig` — the server
  RRF path reuses `RrfConfig`'s `window_size`/`rank_constant` for the `rank`
  block, so no new RRF parameters are introduced.

- **Config (rag-mcp `Config::from_env`)**: new optional env vars, all with
  sensible defaults so nothing is required:
  - `RAG_MCP_HYBRID_FUSION` = `client-rrf` (default) | `normalized-mean` |
    `server-rrf`.
  - `RAG_MCP_HYBRID_NORMALIZATION` = `min-max` (default) | `l2`, used only when
    fusion is `normalized-mean`.
  - `RAG_MCP_HYBRID_KEYWORD_WEIGHT` / `RAG_MCP_HYBRID_VECTOR_WEIGHT` — optional,
    defaulting to equal weights; when one is set the other defaults to make the
    pair sum to 1; an explicit non-1-sum pair is rejected at startup. Used only
    for `normalized-mean`.
  Invalid values fail startup with a clear message, matching the existing
  parse-failure style.

- **Backend wiring (rag-mcp)**: `EsRetrievalBackend` gains the `HybridFusion`
  config (passed via its constructor from `build_funnel`, same as `RrfConfig`
  today). Keyword and semantic modes are unchanged. Hybrid mode dispatches on
  the strategy:
  - `ClientRrf` → the existing two requests + `rrf_fuse` (byte-for-byte current
    behavior).
  - `NormalizedMean` → the same two requests + a new `score_fuse` function.
  - `ServerRrf` → a single combined request via a new pure builder.

- **New pure function `score_fuse` (rag-mcp)**: takes the two hit lists, the
  per-list weights, the normalization method, and `limit`. Normalizes each
  list's scores independently (min-max over the returned list, or L2 vector
  normalization), combines per-hit as `w_keyword * norm_keyword + w_vector *
  norm_vector` when a hit is in both lists, and scores single-list hits by their
  own normalized score times their weight. Falls back to 0 score on empty or
  zero-norm lists. Truncates to `limit`, breaks ties by id ascending (same
  deterministic convention as `rrf_fuse`). `matched_ann` stays per-hit accurate
  (a hit is an ANN match exactly when it appeared in the kNN list), and
  snippet/highlight behavior matches `rrf_fuse` (keyword highlight wins,
  truncated content fallback otherwise).

- **New pure request builder (rag-mcp `es.rs`)**: `build_hybrid_server_request`
  — `query` (BM25 match with ik analyzer) + `knn` (indexed `embedding`,
  `k`/`num_candidates` derived from `limit` as today) + `rank: { rrf:
  { window_size, rank_constant } }` from `RrfConfig`. Response parsing already
  handles RRF scores (they arrive in the same `_score` field), so no new parser
  is needed.

- **Server RRF provenance simplification**: ES RRF responses do not expose
  which clause matched a given hit, so under `server-rrf` `matched_ann` is
  request-level (`true` for hybrid), consistent with the existing documented
  simplification. The client-side strategies keep per-hit-accurate
  `matched_ann`.

- **License gate is engine-side, documented**: `server-rrf` on a basic-license
  Elasticsearch cluster returns an ES error, which the backend surfaces as a
  `RagError::Ann` with a message naming the license requirement. The reference
  docs state that `server-rrf` needs Elasticsearch Platinum/Enterprise (or a
  trial/paid cluster) while both client-side strategies are free-license.

- **Docs**: `examples/reference.md` (fusion config, the three strategies, the
  license table from the problem statement), `examples/quickstart.md`
  (env-var reference), and the SPEC.md funnel description gain the configurable
  fusion note. No schema or SQL migration; no MCP tool surface change.

## Testing Decisions

- **What "good" looks like**: tests assert on external behavior — returned
  ranking, result ids, response shape, and error classes for the same
  representative query shapes the existing suite uses (exact Chinese term,
  exact English/code identifier, vague Chinese query, vague English query) —
  not on internal call counts or request details. For a given fixture set, the
  three strategies may rank differently (that is the point of the feature); each
  strategy's test asserts its own expected ordering.

- **Primary seam (one)**: `RetrievalFunnel` built by `build_funnel` against a
  real Elasticsearch cluster with the `dense_vector` index and real embeddings,
  exactly as the existing suite does. These tests are env-gated
  (`RAG_MCP_ELASTICSEARCH_URL` plus an embedding backend) and skipped, never
  failed, when unset. `server-rrf` real-ES tests additionally require a cluster
  that accepts `rank: { rrf }` (trial/paid license); if the cluster rejects the
  request the test asserts on the surfaced error class rather than results.

- **Modules tested**:
  - `rag-mcp` integration tests — funnel over real ES: `client-rrf` keeps the
    existing hybrid test green; `normalized-mean` end-to-end (weights applied,
    both lists surface, snippet/highlight behavior); `server-rrf` end-to-end or
    clean-license-error when the cluster is basic-licensed.
  - `rag-mcp` `es_prefilter.rs` — pure unit tests for `score_fuse`: min-max and
    L2 normalization on the two lists, weighted combination, single-list hits,
    empty-list/zero-norm fallback, limit truncation, id tie-breaks,
    `matched_ann` per-hit accuracy, snippet fallbacks.
  - `rag-mcp` `es.rs` — pure unit tests for `build_hybrid_server_request`:
    carries query + knn + rank block, rank uses `RrfConfig` values, no
    highlight-regression.
  - `rag-mcp` `config.rs` — env-parsing tests: defaults, normalization choice,
    weight pairing/defaulting, and startup rejection of invalid values.
  - `rag-core` — unit tests with a fake `RetrievalBackend`: mode mapping and
    fusion-config passthrough are unchanged for keyword/semantic; hybrid
    dispatch preserves the existing behavior.

- **Prior art**: the existing `integration.rs` suite (per-test tokio runtime,
  uniquely-named index, unique per-process fixture tokens, poll-for-visibility
  because ES is near-real-time), the real-ES tests in `es_prefilter.rs`
  (unique `es-`-prefixed index, env-gated, cleanup in the test body), the pure
  unit tests on `rrf_fuse`/`map_hits` and on the keyword/semantic request
  builders, and the env-parsing tests in `config.rs`. The new tests follow
  those same patterns.

## Out of Scope

- OpenSearch `hybrid` query type or its search-pipeline normalization
  processors — the normalized-mean strategy is reimplemented client-side, so it
  works identically on any ES license and needs no OpenSearch-specific API.
- Harmonic/geometric weighted means — arithmetic mean only for v1.
- Per-query fusion selection through the MCP API — fusion is chosen at startup,
  per the operator decision.
- Fusion-strategy auto-selection / capability probing of the cluster — the
  license gate surfaces as a clear error at request time, not a startup probe.
- Hybrid short-circuit heuristics (skip ANN when keyword is confident) — the
  funnel always issues both clauses in hybrid mode, as today.
- Cross-encoder / LLM-based reranking.
- Changing the MCP tool surface, response shapes, or the keyword/semantic modes.

## Further Notes

- `server-rrf` and the client-side strategies are not drop-in equivalents: RRF
  deliberately ignores score magnitude while the weighted mean uses it. The
  reference docs should present the comparison table from the problem statement
  so an operator chooses deliberately rather than by default.
- `RrfConfig` remains the single RRF-parameter source of truth; the
  `client-rrf` and `server-rrf` strategies share its defaults (window 100, rank
  constant 60) and its env overrides.
- This spec supersedes the "hybrid is always client-side RRF" framing in
  `es_prefilter.rs`'s module docs; the doc comment should be updated to
  describe the configurable dispatch.
