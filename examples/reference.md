# Reference

The authoritative, source-level documentation for running and extending the
`agentic-rag` MCP server: configuration, schema, the MCP tool contract, funnel
and scoring semantics. Where this doc and the code disagree, the code wins —
and this doc is wrong.

Related: [`SPEC.md`](../SPEC.md) (design rationale) and
[`quickstart.md`](quickstart.md) (getting it running).

## Architecture in one screen

```
MCP client ──POST /mcp (Bearer auth)──> rag-mcp server
                                           │  tools: search, keyword_search,
                                           │         vector_search, fetch_by_id
                                           ▼
                                  RetrievalFunnel (rag-core, transport-free)
                                           │  maps mode → one request shape
                        ┌───────────────────┼──────────────────────┬──────────┐
                        ▼                   ▼                      ▼          ▼
               Elasticsearch          Ollama / ONNX        Postgres        Postgres
               BM25 · kNN · RRF       embedder (BGE-M3)    content store   tsvector
               (sole retrieval)      semantic/hybrid      fetch_by_id     keyword
               hybrid fused           query vectors                        fallback
               (RRF / weighted mean)                                       (ES err/empty)
```

The server (`crates/rag-mcp`) is a thin `rmcp` + `axum` layer. All retrieval
logic lives in the transport-independent `RetrievalFunnel` (`crates/rag-core`).
The funnel composes a single `RetrievalBackend` — Elasticsearch owns ranking
(BM25 / kNN / RRF, or a client-side RRF / weighted-mean fusion of BM25 and
kNN) — plus the BGE-M3 embedder (remote Ollama or local ONNX)
and a Postgres content store. Postgres also hosts the tsvector keyword
fallback, used only when Elasticsearch errors or returns no keyword hits.

For the same picture as a diagram, see [`architecture.svg`](architecture.svg).

## Configuration

Sourced entirely from environment variables at startup
(`crates/rag-mcp/src/config.rs`). Missing *required* vars fail startup with a
clear error.

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `RAG_MCP_AUTH_TOKEN` | **yes** | — | Bearer token every MCP request must present. |
| `RAG_MCP_DATABASE_URL` | **yes** | — | Postgres URL, e.g. `postgres://rag:rag@127.0.0.1:5432/rag`. |
| `RAG_MCP_BIND_ADDR` | no | `127.0.0.1:8080` | Socket the HTTP server binds. |
| `RAG_MCP_ELASTICSEARCH_URL` | no | `http://127.0.0.1:9200` | ES cluster root. |
| `RAG_MCP_ES_INDEX` | no | `documents` | Index the ES retrieval engine reads (and the seed script writes). |
| `RAG_MCP_CONNECT_TIMEOUT_SECS` | no | `5` | Connect/health-check timeout for PG and ES, in seconds. |
| `RAG_MCP_RRF_WINDOW_SIZE` | no | `100` | RRF `window_size`: how many hits per list each fused request contributes. |
| `RAG_MCP_RRF_RANK_CONSTANT` | no | `60` | RRF `rank_constant` (the `k` in `1/(k + rank + 1)`). |
| `RAG_MCP_HYBRID_FUSION` | no | `client-rrf` | How hybrid mode combines the keyword and kNN lists: `client-rrf` \| `normalized-mean` \| `server-rrf` (see [Fusion (hybrid mode)](#fusion-hybrid-mode)). |
| `RAG_MCP_HYBRID_NORMALIZATION` | no | `min-max` | Score normalization for `normalized-mean`: `min-max` \| `l2` (used only when fusion is `normalized-mean`). |
| `RAG_MCP_HYBRID_KEYWORD_WEIGHT` | no | `0.5` | Weight of the keyword (BM25) list for `normalized-mean`. If set alone, the vector weight defaults to `1 - keyword`. |
| `RAG_MCP_HYBRID_VECTOR_WEIGHT` | no | `0.5` | Weight of the vector (kNN) list for `normalized-mean`. If set alone, the keyword weight defaults to `1 - vector`. An explicit pair must sum to `1` or startup fails. |
| `RAG_MCP_OLLAMA_URL` | no | — | Remote Ollama base URL. **Takes priority over** `RAG_MCP_EMBEDDING_MODEL_DIR`. |
| `RAG_MCP_OLLAMA_MODEL` | no | `bge-m3` | Model sent to Ollama `/api/embed`. Must output `embedding_length = 1024`. |
| `RAG_MCP_EMBEDDING_MODEL_DIR` | no | — | Directory with the local BGE-M3 ONNX graph + `tokenizer.json`. |
| `ORT_DYLIB_PATH` | no* | — | *Only* for the local ONNX path: path to `libonnxruntime.so` (the crate uses `load-dynamic`). |

Embedder selection (`crates/rag-mcp/src/wiring.rs`):

1. `RAG_MCP_OLLAMA_URL` set → `OllamaEmbedder` (recommended; no ONNX session to
   load or serialize).
2. else `RAG_MCP_EMBEDDING_MODEL_DIR` set → `BgeM3Embedder` (ONNX Runtime,
   in-process).
3. neither → `UnavailableEmbedder`: the server still starts and keyword search
   works, but `vector_search` / semantic-hybrid queries fail at call time with
   an actionable error.

The local ONNX model dir must contain `model_int8.onnx` (or `model.onnx`) and
`tokenizer.json`. `OllamaEmbedder` uses a 120s HTTP timeout so the cold-load
after a pull doesn't fail the first request.

### Startup contract

`AppState::connect` health-checks **both** Postgres (`SELECT 1`) and
Elasticsearch (`GET /_cluster/health`) before the server starts serving. An
unreachable backend is a startup failure, not a first-call failure.

## Schema

Owned by the external ingestion process (SPEC.md: ingestion is out of scope);
the server only reads it. Migrations are in `crates/rag-mcp/migrations`.

### `documents` (0001_documents.sql)

```sql
CREATE TABLE documents (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,        -- provenance, e.g. "wiki/errors.md"
    language      TEXT,                 -- "zh" / "en" / ...
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX documents_search_vector_idx ON documents USING GIN (search_vector);
```

`search_vector` is **generated and stored** — you never write it. The `simple`
config (not `english`) is deliberate: no stemming, so identifiers and error
codes like `validate_payload` / `ERROR_10054` match verbatim.

### `chunk_embeddings` + extensions (0002_pg_trgm_pgvector.sql)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE chunk_embeddings (
    id        TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL            -- BGE-M3 dense, 1024 dims
);
CREATE INDEX chunk_embeddings_hnsw_idx
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX documents_content_trgm_idx ON documents USING GIN (content gin_trgm_ops);
```

- **Legacy:** this is the retired pgvector path. The retrieval engine is now
  Elasticsearch (vectors live in its `embedding` `dense_vector` field), so the
  server never reads `chunk_embeddings` or the trigram index. The migration
  stays in the schema so external ingestion processes that still write it keep
  working; nothing in the query path touches it.
- `CREATE EXTENSION` needs a superuser / extension role: run migrations as the
  schema-owner/ingestion role, not the runtime user.

## MCP endpoint and protocol

- **URL:** `POST http://<RAG_MCP_BIND_ADDR>/mcp`
- **Auth:** `Authorization: Bearer <RAG_MCP_AUTH_TOKEN>` on every request.
- **Transport:** MCP **streamable HTTP**. The client must complete the
  handshake within a session:

  1. `POST` `initialize` — response returns an `Mcp-Session-Id` header.
  2. `POST` `notifications/initialized` with that session id.
  3. `POST` `tools/call` (or `tools/list`) with that session id.

  The `05-mcp-call.sh` script does exactly this; the raw exchange looks like:

```sh
# 1. initialize
curl -sS -D - http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer $RAG_MCP_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
       "params":{"protocolVersion":"2024-11-05","capabilities":{},
                 "clientInfo":{"name":"curl","version":"1"}}}'
#   → grab Mcp-Session-Id from the response headers

# 2. initialized notification
curl -sS http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer $RAG_MCP_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <session>" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. call a tool
curl -sS http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer $RAG_MCP_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <session>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"search","arguments":{"query":"连接失败"}}}'
```

The tool result arrives inside the MCP result's `content[].text` as a JSON
string (see response shapes below).

## Using it from opencode

Register the running server with opencode's CLI (writes to
`~/.config/opencode/opencode.json(c)` or a project `opencode.json`):

```sh
opencode mcp add agentic-rag \
  --url http://127.0.0.1:8080/mcp \
  --header "Authorization=Bearer dev-secret"   # KEY=VALUE form, not KEY: VALUE

opencode mcp list                              # → "● ✓ agentic-rag connected"
```

The equivalent config entry (what `opencode mcp add` writes) is:

```json
{
  "mcp": {
    "agentic-rag": {
      "type": "remote",
      "url": "http://127.0.0.1:8080/mcp",
      "headers": { "Authorization": "Bearer dev-secret" }
    }
  }
}
```

- Tools are namespaced by server name: `agentic-rag_search`,
  `agentic-rag_keyword_search`, `agentic-rag_vector_search`,
  `agentic-rag_fetch_by_id`.
- A running opencode TUI session loads config once at startup — restart it to
  see a newly added server. Fresh sessions (`opencode run 'try some rag'`,
  `opencode mcp list`) load the current config immediately.
- To remove the server, delete the `mcp.agentic-rag` block from the config
  file (there is no `opencode mcp remove`).
- `scripts/08-opencode-mcp.sh` wraps the register/verify/demo flow and fails
  with a clear hint if the server isn't running.

## Tools

Exposed via `crates/rag-mcp/src/server.rs`. All arguments pass through
`RetrievalFunnel`; the server does no re-querying or reasoning — the calling
agent decides.

### `search`

Mode dispatch → one request shape on the retrieval backend. `query`, `mode`,
`source`, `language`, `limit`.

| Param | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `query` | string | yes | — | Search text. |
| `mode` | string | no | `hybrid` | `"keyword"` \| `"semantic"` \| `"hybrid"`. |
| `source` | string | no | — | Narrow to one source (provenance path). |
| `language` | string | no | — | Narrow to one language. |
| `limit` | int | no | `10` | Max results. |

Mode mapping: `keyword` → a single BM25-only request; `semantic` → embed the
query, then a kNN-only request; `hybrid` (default) → embed the query, then
independent BM25 and kNN requests fused per `RAG_MCP_HYBRID_FUSION` (client
RRF, a normalized weighted mean, or server-side RRF — see below). Use `keyword`
for exact terms (error codes, function names); `semantic` for vague,
intent-based queries; `hybrid` for the balanced default.

### `keyword_search`

BM25-only (`keyword` mode). `query`, `limit`. Fast and precise for exact terms;
never runs ANN and never embeds the query. Postgres tsvector stands in when ES
errors or returns no hits.

### `vector_search`

kNN-only (`semantic` mode). `query`, `limit`. Requires a configured embedder.
Best for vague, intent-based queries.

### `fetch_by_id`

`id` — a chunk/document id previously returned by a search. Returns full
content (the "progressive disclosure" follow-up after reviewing snippets).

## Response shapes

Results (from `search` / `keyword_search` / `vector_search`) are arrays of:

```json
{
  "id": "ex-zh-conn",
  "source": "wiki/errors.md",
  "score": 0.72,
  "snippet": "系统发生<em>连接</em><em>失败</em>错误码 10054...",
  "matched_via": ["elasticsearch"],
  "matched_ann": false
}
```

- `snippet` is an ES highlight fragment (`<em>…</em>` marks matched terms) when
  available; otherwise a fixed 200-char truncation of the content.
- `matched_via` names the backend that produced the hit: `elasticsearch` |
  `tsvector`.
- `matched_ann` is `true` when the hit appeared in the kNN result list — i.e.
  the request carried a kNN clause (semantic mode, or a hit found by the kNN
  leg of a client-side-fused hybrid search). Under `server-rrf` (ES's native
  fused RRF), the engine doesn't expose per-hit clause provenance, so
  `matched_ann` is request-level: `true` for every hybrid hit.

`fetch_by_id` returns:

```json
{ "id": "ex-zh-conn", "source": "wiki/errors.md",
  "content": "<full content>", "metadata": {} }
```

### Errors

- Unknown id → MCP `invalid_request` with `document not found: <id>` (a data
  condition, distinguishable from backend failure).
- Backend / embedding failures → MCP `internal_error` with the underlying
  message.

## Funnel semantics (`crates/rag-core`)

`RetrievalFunnel` composes a single `RetrievalBackend`, the embedder, and the
content store. It maps the caller's mode onto exactly one request shape and
passes the backend's scores through untouched — there is no cross-engine score
merge, no calibration.

### Modes → request shapes

- `keyword`: `keyword = Some(query), query_vector = None` → BM25-only request.
- `semantic`: `keyword = None, query_vector = Some(embedding)` → kNN-only
  request (exactly one embed call).
- `hybrid`: `keyword = Some(query), query_vector = Some(embedding)` → keyword
  and kNN results combined per the configured fusion strategy (below).

### Fusion (hybrid mode)

Which method hybrid mode uses to combine the keyword and kNN lists is selected
at startup via `RAG_MCP_HYBRID_FUSION` and threaded from `config.rs` through
`wiring.rs` into `EsRetrievalBackend`. The three strategies are not drop-in
equivalents — pick by what you trust in your corpus:

| Strategy | Request shape | Fusion | License | When to choose |
| --- | --- | --- | --- | --- |
| `client-rrf` (default) | two ES requests | client-side RRF (`es_prefilter::rrf_fuse`) | free | Robust default; ignores score magnitude, immune to BM25-vs-cosine scale mismatch. |
| `normalized-mean` | two ES requests | `es_prefilter::score_fuse`: min-max or L2-normalize each list, then weighted mean | free | You want score *magnitude* (a much stronger BM25 match outranks a middling one) and per-list weights. Min-max is outlier-sensitive. |
| `server-rrf` | one ES request (`query` + `knn` + `rank: { rrf }`) | engine-native RRF | Platinum/Enterprise | You have a licensed cluster and want ES to own the fused ranking. |

Shared behavior under every strategy: the keyword clause's `<em>` highlight
wins the snippet; ANN-only hits fall back to a truncated content snippet;
`matched_ann` marks hits that came via the kNN leg (per-hit under the
client-side strategies, request-level under `server-rrf`). The client-side
fusions break ties deterministically by id (score desc, id asc);
`server-rrf` keeps Elasticsearch's own ordering.

`RrfConfig { window_size, rank_constant }` defaults to ES's own RRF values
(100 / 60), is overridable via `RAG_MCP_RRF_WINDOW_SIZE` /
`RAG_MCP_RRF_RANK_CONSTANT`, and is shared by the `client-rrf` and `server-rrf`
strategies (the rank block's `rank_window_size`/`rank_constant`).

`normalized-mean` tuning: `RAG_MCP_HYBRID_NORMALIZATION` picks the
normalization (`min-max` | `l2`), and `RAG_MCP_HYBRID_KEYWORD_WEIGHT` /
`RAG_MCP_HYBRID_VECTOR_WEIGHT` set the per-list weights (defaulting to equal
weights; a single set weight fills the other to sum to 1; an explicit pair
must sum to 1 or startup fails).

`server-rrf` on a basic-license cluster: ES rejects the request and the backend
surfaces the engine's error as a clear `RagError::Ann` ("hybrid search
failed: …") rather than silently degrading to keyword-only results.

## Backends

### tsvector (`tsvector.rs`)

Exact/identifier matching over `documents.search_vector`
(`to_tsvector('simple', ...)`) — `simple` config, no stemming. Serves as the
**keyword fallback**: Elasticsearch is primary, and only when it errors or
returns zero hits (an unavailable *or* unsynced cluster both produce that
shape) does the tsvector keyword search run.

### Elasticsearch — the sole retrieval engine (`es_prefilter.rs`, `es.rs`)

Owns all three request shapes. The index mapping (as created by
`EsClient::ensure_index` and mirrored by the seed script):

```json
{
  "settings": {
    "number_of_shards": 1, "number_of_replicas": 0,
    "analysis": { "analyzer": { "ik": { "type": "custom", "tokenizer": "ik_max_word" } } }
  },
  "mappings": {
    "properties": {
      "source":    { "type": "keyword" },
      "content":   { "type": "text", "analyzer": "ik_max_word" },
      "embedding": { "type": "dense_vector", "dims": 1024, "index": true, "similarity": "cosine" }
    }
  }
}
```

- **keyword** → `match` on `content` with the same `ik_max_word` analyzer,
  plus a one-fragment highlight (`<em>…</em>`, fragment_size 150).
- **semantic** → kNN on `embedding` (HNSW cosine). Exactly one embed call
  plus this request.
- **hybrid** → the two above (or, under `server-rrf`, one combined
  `query` + `knn` + `rank: { rrf }` request), fused per the configured
  strategy (see Funnel semantics).
- Elasticsearch is near-real-time: after indexing, a document may take ~1s to
  become searchable — the tests and the seed script poll for visibility.
- Mapping changes (e.g. adding `embedding`) require deleting the index; the
  seed script creates the index and notes this.

### Fallback (`fallback.rs`)

`FallbackRetrievalBackend` wraps the ES primary and the tsvector keyword
fallback. Only `keyword` mode can fall back (when the primary errors or returns
no hits). `semantic` and `hybrid` never fall back: kNN cannot be served by
tsvector, so the ES error surfaces as-is rather than silently degrading to
keyword-only results. Fallback hits keep `matched_via: ["tsvector"]`.

### Embedders (`embedder.rs`)

- `OllamaEmbedder` — `POST {base}/api/embed` with `{"model", "input"}`; takes
  the first returned vector; 120s timeout; 1024-dim check.
- `BgeM3Embedder` — in-process ONNX Runtime; mean-pools `last_hidden_state`
  over non-padding tokens and L2-normalizes (BGE-M3's documented pooling).
- `UnavailableEmbedder` — clear call-time error when no backend is configured.

## Tests

All integration tests are env-gated — they run against real instances when the
env is set and **skip** (never fail) otherwise:

- `RAG_MCP_DATABASE_URL` → Postgres-backed tests.
- `RAG_MCP_ELASTICSEARCH_URL` → ES-backed tests.
- `RAG_MCP_OLLAMA_URL` or `RAG_MCP_EMBEDDING_MODEL_DIR` → embedding-enabled
  end-to-end tests.

```sh
cargo test -p rag-core                                        # unit funnel tests
cargo test -p rag-mcp                                         # backend + integration
```

Notable gotchas encoded in the tests:

- Each test builds its own funnel **on its own tokio runtime** — sqlx pools and
  reqwest clients bind background tasks to the runtime that created them, so a
  shared `OnceCell` pool breaks as soon as the first test's runtime drops
  (`PoolTimedOut`, "dispatch task is gone").
- Fixture ids embed a unique per-process token so concurrent test runs never
  collide in a shared database.
- `apply_schema` runs under a Postgres advisory lock because `CREATE TABLE IF
  NOT EXISTS` is not race-free.
