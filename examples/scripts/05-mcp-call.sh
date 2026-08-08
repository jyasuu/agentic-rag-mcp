#!/usr/bin/env bash
# 05-mcp-call.sh
#
# A minimal MCP streamable-HTTP client: completes the initialize handshake
# inside a session, then makes ONE request and prints the readable result.
#
# usage:
#   ./05-mcp-call.sh --list
#   ./05-mcp-call.sh <tool> '<json-arguments>'
#
# examples:
#   ./05-mcp-call.sh search '{"query":"连接失败"}'
#   ./05-mcp-call.sh search '{"query":"ERROR_10054","mode":"keyword"}'
#   ./05-mcp-call.sh vector_search '{"query":"苹果"}'
#   ./05-mcp-call.sh fetch_by_id '{"id":"ex-zh-conn"}'
#
# env:
#   MCP_URL             endpoint (default http://127.0.0.1:8080/mcp)
#   RAG_MCP_AUTH_TOKEN  bearer token (required)
set -euo pipefail

MCP_URL="${MCP_URL:-http://127.0.0.1:8080/mcp}"
TOKEN="${RAG_MCP_AUTH_TOKEN:-}"
if [[ -z "$TOKEN" ]]; then
  echo "error: RAG_MCP_AUTH_TOKEN not set" >&2
  exit 1
fi
if ! have_jq="$(command -v jq)"; then
  echo "error: jq is required" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
HDR="$tmpdir/headers"
BODY="$tmpdir/body"

CURL=(curl -sS -m 30 -D "$HDR" -o "$BODY")
AUTH=(-H "Authorization: Bearer $TOKEN")
ACCEPT=(-H "Accept: application/json, text/event-stream")
CT=(-H "Content-Type: application/json")

# --- 1. initialize ----------------------------------------------------------
"${CURL[@]}" "${AUTH[@]}" "${ACCEPT[@]}" "${CT[@]}" \
  -X POST "$MCP_URL" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"examples-mcp-call","version":"0.1.0"}}}'

session="$(awk 'BEGIN{IGNORECASE=1} /^mcp-session-id:/{gsub("\r",""); print $2}' "$HDR" | tail -1)"
if [[ -z "$session" ]]; then
  echo "error: no Mcp-Session-Id returned in initialize response" >&2
  echo "--- response headers ---" >&2
  cat "$HDR" >&2
  echo "--- response body ---" >&2
  cat "$BODY" >&2
  exit 1
fi
SESSION=(-H "Mcp-Session-Id: $session")

# --- 2. initialized notification -------------------------------------------
"${CURL[@]}" "${AUTH[@]}" "${ACCEPT[@]}" "${CT[@]}" "${SESSION[@]}" \
  -X POST "$MCP_URL" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' || true

# --- 3. the request ---------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
  body='{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
else
  tool="${1:?a tool name is required (or use --list)}"
  args="${2-}"
  if [[ -z "$args" ]]; then
    args='{}'
  fi
  body="$(jq -nc --arg t "$tool" --argjson a "$args" \
    '{jsonrpc:"2.0",id:2,method:"tools/call",params:{name:$t,arguments:$a}}')"
fi

"${CURL[@]}" "${AUTH[@]}" "${ACCEPT[@]}" "${CT[@]}" "${SESSION[@]}" \
  -X POST "$MCP_URL" -d "$body" || {
    echo "error: request failed" >&2
    exit 1
  }

# --- 4. decode --------------------------------------------------------------
resp="$(cat "$BODY")"
# Defensive SSE handling: if the body is an event stream, take the last data
# payload (rmcp normally answers plain JSON for a single POST, but a proxy or
# an Accept negotiation can produce SSE).
if grep -q '^data:' "$BODY" 2>/dev/null; then
  resp="$(sed -n 's/^data: //p' "$BODY" | tail -1)"
fi

jq -r '
  if .error then
    "MCP error: \(.error.message)"
  elif (.result.content != null and (.result.content | length) > 0) then
    [.result.content[]?.text] | join("")
  else
    (.result // .) | tostring
  end
' <<< "$resp" 2>/dev/null || printf '%s\n' "$resp"
