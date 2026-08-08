#!/usr/bin/env bash
# 04-run-server.sh
#
# Builds and runs the MCP server with the quickstart defaults. Every value is
# overridable via env (see examples/reference.md); unset example vars take
# their documented defaults here so `./04-run-server.sh` "just works".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export RAG_MCP_AUTH_TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"
export RAG_MCP_DATABASE_URL="${RAG_MCP_DATABASE_URL:-postgres://rag:rag@127.0.0.1:5432/rag}"
export RAG_MCP_BIND_ADDR="${RAG_MCP_BIND_ADDR:-127.0.0.1:8080}"
export RAG_MCP_ELASTICSEARCH_URL="${RAG_MCP_ELASTICSEARCH_URL:-http://127.0.0.1:9200}"
export RAG_MCP_ES_INDEX="${RAG_MCP_ES_INDEX:-documents}"
export RAG_MCP_CONNECT_TIMEOUT_SECS="${RAG_MCP_CONNECT_TIMEOUT_SECS:-5}"
# Leave embedding vars unset unless you set them explicitly, so the wiring in
# the server decides (Ollama > ONNX dir > clear-error stub).
export RAG_MCP_OLLAMA_URL="${RAG_MCP_OLLAMA_URL:-}"
export RAG_MCP_OLLAMA_MODEL="${RAG_MCP_OLLAMA_MODEL:-bge-m3}"
export RAG_MCP_EMBEDDING_MODEL_DIR="${RAG_MCP_EMBEDDING_MODEL_DIR:-}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "usage: $0   (all config via env; see examples/env.example)"
  exit 0
fi

echo "starting agentic-rag MCP server at http://${RAG_MCP_BIND_ADDR}/mcp"
echo "  auth token: ${RAG_MCP_AUTH_TOKEN}"
[[ -n "$RAG_MCP_OLLAMA_URL" ]] && echo "  embedder:   Ollama $RAG_MCP_OLLAMA_URL ($RAG_MCP_OLLAMA_MODEL)"
[[ -n "$RAG_MCP_EMBEDDING_MODEL_DIR" ]] && echo "  embedder:   local ONNX $RAG_MCP_EMBEDDING_MODEL_DIR"

cd "$ROOT_DIR"
exec cargo run -p rag-mcp
