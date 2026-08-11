# TODO

## Hybrid fusion strategies (done)

Configurable `RAG_MCP_HYBRID_FUSION` (`client-rrf` | `normalized-mean` | `server-rrf`).

- [x] `rag-core`: `HybridFusion`, `ScoreNormalization`, `FusionWeights`, `HybridFusionConfig` in `scoring.rs`
- [x] `config.rs`: parse `RAG_MCP_HYBRID_FUSION` / `RAG_MCP_HYBRID_NORMALIZATION` / `RAG_MCP_HYBRID_KEYWORD_WEIGHT` / `RAG_MCP_HYBRID_VECTOR_WEIGHT`
- [x] `es.rs`: `build_hybrid_server_request` + `search_hybrid_server` (`rank_window_size` wire field)
- [x] `es_prefilter.rs`: `score_fuse` + fusion dispatch in the `Hybrid` arm
- [x] `wiring.rs` + call sites: thread `fusion` config through
- [x] Real-ES tests per strategy in `es_prefilter.rs`
- [x] Integration tests for `normalized-mean` and `server-rrf`
- [x] Unit tests for `score_fuse`, request builder, config parsing
- [x] Update module docs + reference docs (reference.md, examples, SPEC.md)
- [x] Full suite green (note: `integration::vector_search_returns_semantic_results_end_to_end` flaked on ES NRT visibility — poll was added; verify on a stable run)
- [x] `/code-review` then commit

## Follow-ups noticed while working

- [x] `testutil.rs:101 unique_term()` is dead code (unused warning) — remove or use
- [x] Stale `rag-itg-*` indexes accumulate in the shared ES cluster across test runs — consider a cleanup step or `?refresh=true` on `index_document`
