#!/usr/bin/env bash
# 01-start-backends.sh
#
# Starts the Postgres (pgvector) + Elasticsearch (analysis-ik) containers the
# quickstart and the seed script use. Idempotent: running containers are left
# alone, stopped ones are restarted. `--delete` removes and recreates them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

PG_CONTAINER="${PG_CONTAINER:-rag-pg}"
ES_CONTAINER="${ES_CONTAINER:-rag-es}"
PG_PORT="${PG_PORT:-5432}"
ES_PORT="${ES_PORT:-9200}"
PG_IMAGE="${PG_IMAGE:-pgvector/pgvector:pg16}"
ES_IMAGE="${ES_IMAGE:-es-ik:8.15.3}"
ES_MEM="${ES_MEM:-1g}"

usage() {
  echo "usage: $0 [--delete]"
  echo "  --delete  remove existing containers first (fresh state)"
}

DELETE=0
for arg in "$@"; do
  case "$arg" in
    --delete) DELETE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

if [[ "$DELETE" == 1 ]]; then
  docker rm -f "$PG_CONTAINER" "$ES_CONTAINER" >/dev/null 2>&1 || true
fi

ensure_pg() {
  if docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    echo "postgres container '$PG_CONTAINER' already running"
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    echo "restarting existing stopped container '$PG_CONTAINER'"
    docker start "$PG_CONTAINER" >/dev/null
    return
  fi
  echo "creating postgres container '$PG_CONTAINER' from $PG_IMAGE"
  docker run -d --name "$PG_CONTAINER" \
    -p "${PG_PORT}:5432" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD=rag \
    -e POSTGRES_DB="$PG_DB" \
    "$PG_IMAGE" >/dev/null
}

ensure_es() {
  if docker ps --format '{{.Names}}' | grep -qx "$ES_CONTAINER"; then
    echo "elasticsearch container '$ES_CONTAINER' already running"
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$ES_CONTAINER"; then
    echo "restarting existing stopped container '$ES_CONTAINER'"
    docker start "$ES_CONTAINER" >/dev/null
    return
  fi
  echo "creating elasticsearch container '$ES_CONTAINER' from $ES_IMAGE"
  docker run -d --name "$ES_CONTAINER" \
    -p "${ES_PORT}:9200" \
    -e discovery.type=single-node \
    -e xpack.security.enabled=false \
    -e ES_JAVA_OPTS="-Xms${ES_MEM} -Xmx${ES_MEM}" \
    "$ES_IMAGE" >/dev/null
}

ensure_pg
ensure_es
wait_for_pg
wait_for_es
echo "backends ready: postgres :$PG_PORT, elasticsearch :$ES_PORT"
