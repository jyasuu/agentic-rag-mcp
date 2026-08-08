#!/usr/bin/env bash
# 02-apply-schema.sh
#
# Applies every crate migration (crates/rag-mcp/migrations/*.sql) to the
# example database, idempotently (migrations use IF NOT EXISTS). This mirrors
# the test helper apply_schema() in crates/rag-mcp/src/testutil.rs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MIGRATIONS_DIR="$ROOT_DIR/crates/rag-mcp/migrations"

if [[ ! -d "$MIGRATIONS_DIR" ]]; then
  echo "error: migrations dir not found at $MIGRATIONS_DIR" >&2
  exit 1
fi

echo "applying migrations from $MIGRATIONS_DIR"
wait_for_pg

for file in "$MIGRATIONS_DIR"/*.sql; do
  echo "applying $(basename "$file")"
  psql_run -f "$file"
done

echo "schema applied:"
psql_run -c '\dt' || true
