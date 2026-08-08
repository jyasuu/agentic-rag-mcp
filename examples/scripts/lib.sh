#!/usr/bin/env bash
# Shared helpers for the example scripts. Source, don't execute.
#
# Postgres access strategy: use a host `psql` pointed at RAG_MCP_DATABASE_URL
# when psql exists on PATH; otherwise fall back to `docker exec` into the
# example postgres container (the quickstart default). Elasticsearch is always
# reached over HTTP via curl.

set -euo pipefail

: "${PG_CONTAINER:=rag-pg}"
: "${PG_USER:=rag}"
: "${PG_DB:=rag}"
: "${PG_HOST:=127.0.0.1}"
: "${PG_PORT:=5432}"

have() { command -v "$1" >/dev/null 2>&1; }

# Resolve the database URL used by host psql (if psql exists on the host).
pg_url() {
  if [[ -n "${RAG_MCP_DATABASE_URL:-}" ]]; then
    printf '%s' "$RAG_MCP_DATABASE_URL"
  else
    printf 'postgres://%s@%s:%s/%s' "$PG_USER" "$PG_HOST" "$PG_PORT" "$PG_DB"
  fi
}

# Run a psql command (args like `-c "SQL"` or `-f file`).
# The docker fallback copies any `-f file` into the container first, because
# host paths don't exist inside it.
psql_run() {
  if have psql; then
    psql -X -q -v ON_ERROR_STOP=1 "$(pg_url)" "$@"
    return
  fi
  local args=("$@")
  local copied=()
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "-f" && -n "${args[$((i + 1))]:-}" && -f "${args[$((i + 1))]}" ]]; then
      local src="${args[$((i + 1))]}"
      local dst="/tmp/$(basename "$src")"
      docker cp "$src" "$PG_CONTAINER:$dst" >/dev/null
      copied+=("-f" "$dst")
      i=$((i + 1))
    else
      copied+=("${args[$i]}")
    fi
  done
  docker exec "$PG_CONTAINER" psql -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" "${copied[@]}"
}

# Block until Postgres accepts connections.
wait_for_pg() {
  echo "waiting for Postgres on $PG_HOST:$PG_PORT ..."
  for _ in $(seq 1 60); do
    if have pg_isready; then
      pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" >/dev/null 2>&1 && return 0
    else
      docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1 && return 0
    fi
    sleep 1
  done
  echo "error: Postgres not ready after 60s" >&2
  return 1
}

# Block until Elasticsearch reports a healthy cluster.
wait_for_es() {
  local url="${RAG_MCP_ELASTICSEARCH_URL:-http://127.0.0.1:9200}"
  echo "waiting for Elasticsearch at $url ..."
  for _ in $(seq 1 90); do
    if curl -fsS -m 2 "$url/_cluster/health" 2>/dev/null | grep -q '"status":"green"'; then
      return 0
    fi
    sleep 2
  done
  echo "error: Elasticsearch not green after 180s" >&2
  return 1
}

# Escape a value for single-quoted SQL literals.
sql_escape() { printf '%s' "$1" | sed "s/'/''/g"; }
