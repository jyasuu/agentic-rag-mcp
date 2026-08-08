#!/usr/bin/env bash
# 08-opencode-mcp.sh
#
# Registers the agentic-rag MCP server with opencode (`opencode mcp add`),
# verifies the connection, and can run a demo session that searches the
# knowledge base from a fresh opencode session.
#
# usage:
#   ./08-opencode-mcp.sh          # register (if needed) + show connection status
#   ./08-opencode-mcp.sh --list   # show connection status only
#   ./08-opencode-mcp.sh --demo   # register + verify + `opencode run 'try some rag'`
#
# env:
#   RAG_MCP_AUTH_TOKEN  bearer token sent in the MCP request header
#                       (default: dev-secret, matches 04-run-server.sh)
#   MCP_NAME            server name in opencode config (default: agentic-rag)
#   MCP_URL             endpoint (default: http://127.0.0.1:8080/mcp)
#   OPENCODE_BIN        opencode binary (default: opencode on PATH, else
#                       ~/.opencode/bin/opencode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKEN="${RAG_MCP_AUTH_TOKEN:-dev-secret}"
MCP_NAME="${MCP_NAME:-agentic-rag}"
MCP_URL="${MCP_URL:-http://127.0.0.1:8080/mcp}"

if command -v opencode >/dev/null 2>&1; then
  OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
elif [[ -x "$HOME/.opencode/bin/opencode" ]]; then
  OPENCODE_BIN="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
else
  echo "error: opencode not found on PATH or at ~/.opencode/bin/opencode" >&2
  echo "       set OPENCODE_BIN to the opencode binary" >&2
  exit 1
fi

usage() {
  echo "usage: $0 [--list | --demo]"
  echo "  --list  show MCP connection status only"
  echo "  --demo  register + verify + run 'opencode run \"try some rag\"'"
}

MODE="add"
for arg in "$@"; do
  case "$arg" in
    --list) MODE="list" ;;
    --demo) MODE="demo" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

# --- the server must be up: any HTTP response (including 401 = auth present)
# --- means it is listening; 000 means nothing is on the port.
code="$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$MCP_URL" \
  -X POST -H 'Content-Type: application/json' -d '{}' 2>/dev/null || true)"
if [[ "$code" == "000" ]]; then
  echo "error: nothing listening at $MCP_URL (server down?)" >&2
  echo "       start it first: ./04-run-server.sh" >&2
  exit 1
fi

list_status() {
  "$OPENCODE_BIN" mcp list
}

register() {
  if "$OPENCODE_BIN" mcp list 2>/dev/null | grep -q "$MCP_NAME"; then
    echo "MCP server '$MCP_NAME' already registered -- skipping add"
  else
    echo "registering MCP server '$MCP_NAME' -> $MCP_URL"
    "$OPENCODE_BIN" mcp add "$MCP_NAME" \
      --url "$MCP_URL" \
      --header "Authorization=Bearer $TOKEN"
  fi
}

case "$MODE" in
  list)
    list_status
    ;;
  add)
    register
    echo
    echo "connection status:"
    list_status
    echo
    echo "note: a running opencode session loads config at startup -- restart it"
    echo "      to expose the tools (search, keyword_search, vector_search,"
    echo "      fetch_by_id). 'opencode run' sessions pick them up immediately."
    ;;
  demo)
    register
    list_status
    echo
    echo "running a demo opencode session that searches the knowledge base..."
    "$OPENCODE_BIN" run 'try some rag'
    ;;
esac
