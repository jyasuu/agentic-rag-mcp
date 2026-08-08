# agentic-rag-mcp

Thin HTTP MCP server exposing retrieval primitives over an existing
Postgres/pgvector + Elasticsearch knowledge base. See `SPEC.md` for the full
design rationale.

## Examples

`examples/` has everything needed to stand up and drive the full stack:
`quickstart.md` for the 5-minute path, `reference.md` for the complete config /
tool / funnel reference, and idempotent scripts to start Postgres +
Elasticsearch, apply the schema, seed a bilingual corpus, run the server, and
query it over streamable HTTP.

```sh
./examples/scripts/01-start-backends.sh   # Postgres(pgvector) + ES(ik)
./examples/scripts/02-apply-schema.sh     # migrations
./examples/scripts/03-seed.sh             # corpus + BGE-M3 embeddings
./examples/scripts/04-run-server.sh       # the MCP server
./examples/scripts/06-sample-queries.sh   # drive all four tools
```

## Workspace layout

- `crates/rag-core` — transport-independent retrieval funnel
  (`RetrievalFunnel`), trait seams (`PreFilterStrategy`, `AnnClient`,
  `Embedder`, `ContentStore`), scoring, and types. No MCP/HTTP dependency —
  this is the tested seam (`cargo test -p rag-core`).
- `crates/rag-mcp` — `rmcp` + `axum` server. Wires MCP tools (`search`,
  `keyword_search`, `vector_search`, `fetch_by_id`) to `RetrievalFunnel`, adds
  bearer-token auth middleware. Backend implementations
  (Postgres/ES/pgvector/BGE-M3) are stubbed in `backends.rs` — that's the
  next implementation step, not part of this scaffold.

## Toolchain requirement

**`rag-mcp` requires Rust 1.88+** (the `rmcp` crate's MSRV). `rag-core` has
no such requirement and builds on much older toolchains.

If `rustc --version` is below 1.88, install a current toolchain via
[rustup](https://rustup.rs) before building `rag-mcp`:

```sh
rustup install stable
rustup override set stable
```

## Building

```sh
# Core logic + tests only (works on any recent-ish toolchain):
cargo test -p rag-core

# Full workspace, including the MCP server (needs Rust 1.88+):
cargo build
```

## Running the server

```sh
export RAG_MCP_AUTH_TOKEN=some-secret-token
export RAG_MCP_BIND_ADDR=127.0.0.1:8080   # optional, this is the default
cargo run -p rag-mcp
```

The MCP endpoint is mounted at `POST http://127.0.0.1:8080/mcp`, requiring
`Authorization: Bearer <RAG_MCP_AUTH_TOKEN>`.

**Note:** until `backends.rs` is wired to real Postgres/Elasticsearch/pgvector
/BGE-M3 implementations, every tool call will return an "not wired yet"
error — the server starts and the MCP protocol layer works end-to-end, but
there's no real data behind it yet.

## Next steps (see SPEC.md "Opportunity List" for the full list)

1. Implement `PreFilterStrategy` for Elasticsearch (`ik_analyzer`), Postgres
   `tsvector`, and `pg_trgm` in `backends.rs`.
2. Implement `AnnClient` against pgvector.
3. Implement `Embedder` via BGE-M3 through `ort`.
4. Implement `ContentStore` against Postgres.
5. Stand up the `pg_x`-based CDC pipeline syncing Postgres → Elasticsearch.
