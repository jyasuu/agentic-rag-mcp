# Quickstart

Get a live, searchable instance of the agentic-rag MCP server — Postgres +
pgvector + Elasticsearch up, schema applied, a bilingual corpus seeded, the
server running, and a query answered — in about five minutes.

This follows the development layout used by the project itself (see
`crates/rag-mcp/migrations` and the integration-test setup in
`crates/rag-mcp/src/integration.rs`), so what you stand up here behaves the
same as the tests.

## 0. Prerequisites

- `docker`
- `cargo` with **Rust 1.88+** (the `rmcp` crate's MSRV; `rustup install stable`)
- `curl`, `jq`
- `psql` on `PATH`, *or* just the docker containers (scripts fall back to
  `docker exec`)
- An embedding backend for semantic search (recommended: a remote Ollama serving
  `bge-m3`; alternatively the local ONNX model dir)

## 1. Start Postgres + Elasticsearch

```sh
./examples/scripts/01-start-backends.sh
```

This creates two containers (idempotently — existing containers are reused):

- `rag-pg` — `pgvector/pgvector:pg16`, user/pass/db all `rag`, published on
  port `5432`.
- `rag-es` — Elasticsearch 8.15 with the `analysis-ik` plugin preinstalled,
  published on `9200`, single-node with security disabled.

It waits until both report healthy before returning.

**The `ik` image is not on Docker Hub.** If you don't have `es-ik` locally, you
can either build it:

```sh
docker run -d --name es-plain -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS=-Xms1g -Xmx1g docker.elastic.co/elasticsearch/elasticsearch:8.15.3
# wait for green, then install the plugin on a live node:
docker exec es-plain bin/elasticsearch-plugin install \
  https://get.infini.cloud/elasticsearch/analysis-ik/8.15.3
docker restart es-plain
```

...or point `RAG_MCP_ELASTICSEARCH_URL` at any ES 8.x node and accept that
Chinese keyword search will rely on fallback matching (no `ik` segmentation).

## 2. Apply the schema

```sh
./examples/scripts/02-apply-schema.sh
```

Applies every `crates/rag-mcp/migrations/*.sql` idempotently:

- `documents` table with a stored `search_vector` (tsvector over
  `to_tsvector('simple', content)`) and a GIN index — the content store and
  the tsvector keyword fallback.
- The `vector` + `pg_trgm` extensions and `chunk_embeddings` table are legacy
  from the old pgvector path: the retrieval engine is now Elasticsearch, so
  the server never reads them. They stay in the schema so external ingestion
  keeps working.

> **Note:** `CREATE EXTENSION` requires a superuser or a role granted the
> extension's roles. The docker container from step 1 is a superuser, so the
> defaults work — but if you point the scripts at your own database, run this
> as the schema-owner/ingestion role, not the runtime user.

## 3. Seed the corpus

```sh
export RAG_MCP_OLLAMA_URL="${RAG_MCP_OLLAMA_URL:-http://127.0.0.1:11434}"
./examples/scripts/03-seed.sh
```

This loads `examples/sample-data/corpus.json` (7 bilingual documents: Chinese
support articles, English API docs, and error codes) and:

1. Inserts each document into Postgres (`source`, `language`, `content`).
   `search_vector` is generated automatically.
2. Calls Ollama's `/api/embed` once, batched, to get 1024-dim BGE-M3 embeddings
   for every document. (First call can take ~40s if the model is cold on the
   Ollama host.)
3. Mirrors each document into the Elasticsearch `documents` index with the
   `ik_max_word` analyzer mapping and the `embedding` `dense_vector(1024)`
   field, storing each embedding in its document. Elasticsearch is the sole
   retrieval engine (BM25 / kNN / hybrid RRF or weighted-mean), so the vector
   lives there — not in Postgres.

Rerun it any time; inserts are `ON CONFLICT ... DO UPDATE`. `--delete` removes
exactly the fixture rows:

```sh
./examples/scripts/03-seed.sh --delete
```

**Without Ollama**, use the local ONNX model dir:

```sh
export RAG_MCP_EMBEDDING_MODEL_DIR=/path/to/bge-m3-onnx   # model_int8.onnx + tokenizer.json
./examples/scripts/03-seed.sh
```

The shell seed script can only produce embeddings through Ollama's HTTP API.
For the local ONNX path, compute and insert embeddings with your own tooling
(or a small program using the crate's embedder), then seed with `--no-embed`.
Keyword search works without embeddings; `vector_search` needs them.

## 4. Run the server

```sh
export RAG_MCP_AUTH_TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"
./examples/scripts/04-run-server.sh
```

This sets the example defaults (database URL, ES URL, bind addr, embedder
preference) and runs `cargo run -p rag-mcp`. The server health-checks Postgres
**and** Elasticsearch at startup and exits with a clear error if either is
unreachable.

Hybrid mode merges the keyword and kNN lists per `RAG_MCP_HYBRID_FUSION`
(`client-rrf` default | `normalized-mean` | `server-rrf`); `normalized-mean`
has its own tuning (`RAG_MCP_HYBRID_NORMALIZATION`,
`RAG_MCP_HYBRID_KEYWORD_WEIGHT`, `RAG_MCP_HYBRID_VECTOR_WEIGHT`). Full env-var
reference: [`reference.md`](reference.md).

The MCP endpoint is `POST http://127.0.0.1:8080/mcp`, protected by
`Authorization: Bearer <token>`.

## 5. Talk to it

In a second shell:

```sh
export RAG_MCP_AUTH_TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"
./examples/scripts/06-sample-queries.sh
```

`05-mcp-call.sh` is the raw client — it performs the streamable-HTTP handshake
(initialize -> initialized -> tools/call) and prints the tool result:

```sh
# list the exposed tools
./examples/scripts/05-mcp-call.sh --list

# search (hybrid mode is the default)
./examples/scripts/05-mcp-call.sh search '{"query":"连接失败"}'

# exact-term lookup, keyword mode
./examples/scripts/05-mcp-call.sh search '{"query":"ERROR_10054","mode":"keyword"}'

# ANN only
./examples/scripts/05-mcp-call.sh vector_search '{"query":"苹果"}'

# full content after reviewing snippets
./examples/scripts/05-mcp-call.sh fetch_by_id '{"id":"ex-zh-conn"}'
```

## 6. Wire it into an MCP client

Claude Code / Claude Desktop MCP clients configure a remote HTTP server like
this (the bearer token rides in the request headers):

```json
{
  "mcpServers": {
    "agentic-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp",
      "headers": { "Authorization": "Bearer dev-secret" }
    }
  }
}
```

### From opencode

Register the server with `opencode mcp add` (or just run the helper script):

```sh
opencode mcp add agentic-rag \
  --url http://127.0.0.1:8080/mcp \
  --header "Authorization=Bearer dev-secret"     # KEY=VALUE, not KEY: VALUE

opencode mcp list                                  # should show "✓ agentic-rag connected"
```

Then verify it end-to-end from a fresh session:

```sh
opencode run 'try some rag'
```

The tools surface as `agentic-rag_search`, `agentic-rag_keyword_search`,
`agentic-rag_vector_search`, and `agentic-rag_fetch_by_id`.

> **Note:** an already-running opencode session loads config at startup and
> won't see the new server until you restart it. `opencode run` and
> `opencode mcp list` pick it up immediately. To remove the server later,
> delete the `mcp.agentic-rag` block from
> `~/.config/opencode/opencode.json(c)` (or your project `opencode.json`).

`./examples/scripts/08-opencode-mcp.sh` automates all of this:

```sh
./examples/scripts/08-opencode-mcp.sh          # register + show status
./examples/scripts/08-opencode-mcp.sh --list   # status only
./examples/scripts/08-opencode-mcp.sh --demo   # register + run a real search session
```

## 7. Tear down

```sh
./examples/scripts/07-teardown.sh           # stop the containers
./examples/scripts/07-teardown.sh --delete  # stop and remove them
```

## What to read next

[`reference.md`](reference.md) — every env var, the exact schema, the MCP tool
contract, funnel and scoring semantics.
