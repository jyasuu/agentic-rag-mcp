#!/usr/bin/env bash
# 03-seed.sh
#
# Loads the sample corpus into Postgres + Elasticsearch and (when an Ollama
# endpoint is available) computes BGE-M3 embeddings and stores them in the ES
# `embedding` dense_vector field — Elasticsearch is the sole retrieval engine
# (BM25 / kNN / hybrid RRF), so the vector lives in the ES document, not in
# Postgres. Idempotent: documents are upserted, ES index writes replace by id.
# `--delete` removes exactly the fixture rows it manages.
#
# Embeddings: computed via RAG_MCP_OLLAMA_URL /api/embed (batch, one call).
# A shell script cannot run the local ONNX embedder, so with only
# RAG_MCP_EMBEDDING_MODEL_DIR set you must embed with your own tooling and
# pass --no-embed (or use Ollama). Keyword search works without embeddings;
# vector_search and semantic/hybrid modes need them (in ES).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

DEFAULT_CORPUS="$SCRIPT_DIR/../sample-data/corpus.json"

usage() {
  echo "usage: $0 [--no-es] [--no-embed] [--delete] [corpus.json]"
  echo "  --no-es       do not mirror documents into Elasticsearch"
  echo "  --no-embed    do not compute/store embeddings (keyword-only data)"
  echo "  --delete      remove the corpus ids from Postgres and ES"
}

DO_ES=1
DO_EMBED=1
DELETE=0
CORPUS="$DEFAULT_CORPUS"
for arg in "$@"; do
  case "$arg" in
    --no-es) DO_ES=0 ;;
    --no-embed) DO_EMBED=0 ;;
    --delete) DELETE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) CORPUS="$arg" ;;
  esac
done

if ! have jq; then
  echo "error: jq is required" >&2
  exit 1
fi
if [[ ! -f "$CORPUS" ]]; then
  echo "error: corpus not found: $CORPUS" >&2
  exit 1
fi

N="$(jq 'length' "$CORPUS")"
if [[ "$N" -eq 0 ]]; then
  echo "error: corpus is empty" >&2
  exit 1
fi
echo "corpus: $N documents from $CORPUS"

ids() { jq -r '.[].id' "$CORPUS"; }

ES_URL="${RAG_MCP_ELASTICSEARCH_URL:-http://127.0.0.1:9200}"
ES_INDEX="${RAG_MCP_ES_INDEX:-documents}"

es_delete() {
  local id="$1"
  curl -fsS -m 10 -X DELETE "$ES_URL/$ES_INDEX/_doc/$id" >/dev/null 2>&1 || true
}

es_index_doc() {
  local id="$1" source="$2" content="$3" embedding="$4"
  local body
  if [[ -n "$embedding" && "$embedding" != "null" ]]; then
    body="$(jq -nc --arg s "$source" --arg c "$content" --argjson e "$embedding" \
      '{source:$s,content:$c,embedding:$e}')"
  else
    body="$(jq -nc --arg s "$source" --arg c "$content" '{source:$s,content:$c}')"
  fi
  curl -fsS -m 10 -X PUT "$ES_URL/$ES_INDEX/_doc/$id" \
    -H 'Content-Type: application/json' -d "$body" >/dev/null
}

delete_seed() {
  wait_for_pg
  echo "deleting ${N} fixture documents from Postgres..."
  psql_run -c "$(ids | while read -r id; do
    printf "DELETE FROM documents WHERE id = '%s';" "$(sql_escape "$id")"
  done)"
  if [[ "$DO_ES" == 1 ]] && curl -fsS -m 5 "$ES_URL" >/dev/null 2>&1; then
    echo "deleting fixture documents from Elasticsearch index $ES_INDEX..."
    while read -r id; do es_delete "$id"; done <<< "$(ids)"
  fi
  echo "cleanup complete"
  exit 0
}

[[ "$DELETE" == 1 ]] && delete_seed

# ---------------------------------------------------------------------------
# 1. Postgres documents
# ---------------------------------------------------------------------------
wait_for_pg
echo "inserting documents into Postgres..."
sql_file="$(mktemp)"
trap 'rm -f "$sql_file"' EXIT
for i in $(seq 0 $((N - 1))); do
  id="$(sql_escape "$(jq -r ".[$i].id" "$CORPUS")")"
  src="$(sql_escape "$(jq -r ".[$i].source" "$CORPUS")")"
  lang="$(jq -r --argjson i "$i" '.[$i].language // empty' "$CORPUS" | sed "s/'/''/g")"
  content="$(sql_escape "$(jq -r ".[$i].content" "$CORPUS")")"
  meta="$(sql_escape "$(jq -c --argjson i "$i" '.[$i].metadata // {}' "$CORPUS")")"
  if [[ -n "$lang" ]]; then lang_sql="'$lang'"; else lang_sql="NULL"; fi
  printf "INSERT INTO documents (id, source, language, content, metadata) VALUES ('%s','%s',%s,'%s','%s'::jsonb) ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, language = EXCLUDED.language, content = EXCLUDED.content, metadata = EXCLUDED.metadata;\n" \
    "$id" "$src" "$lang_sql" "$content" "$meta" >> "$sql_file"
done
psql_run -f "$sql_file"

# ---------------------------------------------------------------------------
# 2. Embeddings (ANN stage)
# ---------------------------------------------------------------------------
if [[ "$DO_EMBED" == 1 ]]; then
  if [[ -z "${RAG_MCP_OLLAMA_URL:-}" ]]; then
    echo "note: RAG_MCP_OLLAMA_URL not set -- skipping embeddings."
    echo "      keyword search will work; vector_search needs embeddings."
    echo "      (set RAG_MCP_OLLAMA_URL, or embed with your own tooling and --no-embed)"
  else
    model="${RAG_MCP_OLLAMA_MODEL:-bge-m3}"
    base="${RAG_MCP_OLLAMA_URL%/}"
    echo "embedding $N documents via $base (model: $model) ..."
    input="$(jq -c '[.[].content]' "$CORPUS")"
    resp_file="$(mktemp)"
    trap 'rm -f "$sql_file" "$resp_file"' EXIT
    # Generous timeout: the first call after a model pull is cold (~40s).
    curl -fsS -m 120 "$base/api/embed" \
      -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg m "$model" --argjson i "$input" '{model:$m,input:$i}')" \
      > "$resp_file"

    n_emb="$(jq '.embeddings | length' "$resp_file")"
    if [[ "$n_emb" != "$N" ]]; then
      echo "error: Ollama returned $n_emb embeddings for $N inputs" >&2
      exit 1
    fi
    dim="$(jq '.embeddings[0] | length' "$resp_file")"
    if [[ "$dim" != 1024 ]]; then
      echo "error: model $model returned $dim-dim embeddings; expected 1024 (ES dense_vector(1024) field)" >&2
      exit 1
    fi

    # The vectors ride along in the ES mirror step below (stored in each
    # document's `embedding` field); keep the response file until then.
    echo "computed $N embeddings (${dim}-dim) — will store them in the ES embedding field."
  fi
fi

# ---------------------------------------------------------------------------
# 3. Elasticsearch mirror
# ---------------------------------------------------------------------------
if [[ "$DO_ES" == 1 ]]; then
  if ! curl -fsS -m 5 "$ES_URL/_cluster/health" >/dev/null 2>&1; then
    echo "note: Elasticsearch unreachable at $ES_URL -- skipping ES mirror."
  else
    echo "ensuring index $ES_INDEX (ik_max_word + embedding dense_vector(1024) mapping)..."
    curl -fsS -m 10 -X PUT "$ES_URL/$ES_INDEX" -H 'Content-Type: application/json' -d '{
      "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": { "analyzer": { "ik": { "type": "custom", "tokenizer": "ik_max_word" } } }
      },
      "mappings": { "properties": {
        "source":  { "type": "keyword" },
        "content": { "type": "text", "analyzer": "ik_max_word" },
        "embedding": { "type": "dense_vector", "dims": 1024, "index": true, "similarity": "cosine" }
      } }
    }' >/dev/null 2>&1 || echo "  (index may already exist -- continuing; mapping changes require deleting the index)"
    echo "indexing $N documents into $ES_INDEX..."
    for i in $(seq 0 $((N - 1))); do
      id="$(jq -r ".[$i].id" "$CORPUS")"
      src="$(jq -r ".[$i].source" "$CORPUS")"
      content="$(jq -r ".[$i].content" "$CORPUS")"
      emb=""
      if [[ -n "${resp_file:-}" && -f "$resp_file" ]]; then
        emb="$(jq -c --argjson i "$i" '.embeddings[$i]' "$resp_file")"
      fi
      es_index_doc "$id" "$src" "$content" "$emb"
    done
    echo "ES is near-real-time: searches may need ~1s to see fresh docs."
  fi
fi

echo
echo "seed complete. Next: ./04-run-server.sh"
echo "demo queries: ./06-sample-queries.sh (needs RAG_MCP_AUTH_TOKEN)"
