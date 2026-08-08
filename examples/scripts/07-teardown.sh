#!/usr/bin/env bash
# 07-teardown.sh
#
# Stops (default) or stops-and-removes (--delete) the example containers.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-rag-pg}"
ES_CONTAINER="${ES_CONTAINER:-rag-es}"

DELETE=0
for arg in "$@"; do
  case "$arg" in
    --delete) DELETE=1 ;;
    -h|--help)
      echo "usage: $0 [--delete]   (--delete also removes containers)"
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

for name in "$PG_CONTAINER" "$ES_CONTAINER"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker stop "$name" >/dev/null
    if [[ "$DELETE" == 1 ]]; then
      docker rm "$name" >/dev/null
      echo "$name: stopped and removed"
    else
      echo "$name: stopped (recreate with ./01-start-backends.sh)"
    fi
  else
    echo "$name: not present"
  fi
done
