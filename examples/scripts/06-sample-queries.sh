#!/usr/bin/env bash
# 06-sample-queries.sh
#
# Exercises all four MCP tools against a seeded server. Requires the server
# (04-run-server.sh) to be running and the corpus (03-seed.sh) to be seeded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALL="$SCRIPT_DIR/05-mcp-call.sh"

export RAG_MCP_AUTH_TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"

section() { printf '\n\033[1m%s\033[0m\n' "$1"; }

section "tools/list"
"$CALL" --list

section "search  (hybrid, default) -- '连接失败'"
"$CALL" search '{"query":"连接失败"}'

section "search  (mode=keyword, exact error code) -- 'ERROR_10054'"
"$CALL" search '{"query":"ERROR_10054","mode":"keyword"}'

section "keyword_search  (English identifier) -- 'validate_payload'"
"$CALL" keyword_search '{"query":"validate_payload"}'

section "vector_search  (vague Chinese intent) -- '苹果'"
"$CALL" vector_search '{"query":"苹果"}'

section "fetch_by_id  (full content) -- 'ex-zh-conn'"
"$CALL" fetch_by_id '{"id":"ex-zh-conn"}'

section "error handling -- unknown id"
"$CALL" fetch_by_id '{"id":"does-not-exist"}'
