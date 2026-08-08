# Examples

End-to-end reference material for running and using the `agentic-rag-mcp`
server: docs for the full config surface and MCP tool contract, plus copy-paste
shell scripts that stand up Postgres + Elasticsearch, apply the schema, seed a
bilingual corpus (Chinese + English/code), run the server, and drive it over
streamable HTTP.

Everything here assumes the same layout as `SPEC.md` and `migrations/` in
`crates/rag-mcp`:

- Postgres + pgvector holds the corpus and the ANN embeddings.
- Elasticsearch (with the `analysis-ik` plugin) backs Chinese keyword search.
- The server runs a layered funnel: keyword pre-filter -> conditional ANN ->
  weighted scoring, exposed as four MCP tools.

## Layout

| Path | What it is |
| --- | --- |
| [`quickstart.md`](quickstart.md) | The fast path: backends up, seeded, server running, first query answered. |
| [`reference.md`](reference.md) | The full docs: every env var, the schema, the MCP tool contract, funnel semantics, scoring. |
| [`env.example`](env.example) | Every supported config variable with defaults and comments. |
| [`sample-data/corpus.json`](sample-data/corpus.json) | Small bilingual fixture corpus used by the seed script. |
| [`scripts/01-start-backends.sh`](scripts/01-start-backends.sh) | Starts the Postgres(pgvector) + Elasticsearch(ik) docker containers. |
| [`scripts/02-apply-schema.sh`](scripts/02-apply-schema.sh) | Applies `migrations/*.sql` idempotently. |
| [`scripts/03-seed.sh`](scripts/03-seed.sh) | Loads the corpus into Postgres, computes + stores embeddings, mirrors docs into ES. |
| [`scripts/04-run-server.sh`](scripts/04-run-server.sh) | Builds and runs the MCP server with the example defaults. |
| [`scripts/05-mcp-call.sh`](scripts/05-mcp-call.sh) | Speaks MCP streamable HTTP to the server (init session, then one tool call). |
| [`scripts/06-sample-queries.sh`](scripts/06-sample-queries.sh) | Demos all four tools against the seeded corpus. |
| [`scripts/07-teardown.sh`](scripts/07-teardown.sh) | Stops (and optionally removes) the example containers. |

## Prerequisites

- `docker` (for `01-start-backends.sh` / `07-teardown.sh`).
- Rust 1.88+ with `cargo` (the `rmcp` crate's MSRV) for `04-run-server.sh`.
- `curl` and `jq` for the seed script and the MCP client helper.
- A Postgres client. `psql` on `PATH` is used if present; otherwise the scripts
  fall back to `docker exec <pg-container> psql`.
- An embedding backend so `vector_search` / semantic hybrid queries work:
  either a reachable Ollama serving `bge-m3` (`RAG_MCP_OLLAMA_URL`), or the
  local ONNX model directory (`RAG_MCP_EMBEDDING_MODEL_DIR`).

## 5-minute quickstart

```sh
# 1. Postgres + Elasticsearch
./examples/scripts/01-start-backends.sh

# 2. Schema (extensions, tables, indexes)
./examples/scripts/02-apply-schema.sh

# 3. Corpus + embeddings. Requires RAG_MCP_OLLAMA_URL (recommended) or
#    RAG_MCP_EMBEDDING_MODEL_DIR for the ANN stage.
export RAG_MCP_OLLAMA_URL="${RAG_MCP_OLLAMA_URL:-http://127.0.0.1:11434}"
./examples/scripts/03-seed.sh

# 4. Run the server (blocking; Ctrl-C to stop)
./examples/scripts/04-run-server.sh

# 5. In another shell, exercise the tools
export RAG_MCP_AUTH_TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"
./examples/scripts/06-sample-queries.sh
```

All scripts are idempotent: safe to re-run, and `03-seed.sh --delete` removes
exactly the fixture rows it created.

## Notes

- The `ik`-aware Elasticsearch image (`es-ik`) is not on Docker Hub — see
  [`quickstart.md`](quickstart.md) for how to build it, or point
  `RAG_MCP_ELASTICSEARCH_URL` at any ES 8.x node (Chinese word segmentation
  will degrade to fallback matching without the plugin).
- The server requires Elasticsearch to be reachable **at startup** (it
  health-checks both backends). The pg_trgm fallback only covers ES being
  down *after* startup / unsynced during queries.
- `docs` live in `reference.md`; the scripts are thin wrappers around the
  documented contract, so read the reference before modifying them.
