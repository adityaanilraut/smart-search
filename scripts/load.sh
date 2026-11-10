#!/bin/sh
set -eu
port=8912
base="http://127.0.0.1:$port"
export INDEX_DIR=index/trec-covid
pid=
cleanup() {
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  pid=
}
trap cleanup EXIT
trap 'exit 130' INT
for mode in uncached warm; do
  if [ "$mode" = warm ]; then
    export REDIS_URL="${BENCH_REDIS_URL:-redis://127.0.0.1:6379}"
  else
    unset REDIS_URL
  fi
  uv run --locked uvicorn smart_search.api:app --host 127.0.0.1 --port "$port" --no-access-log &
  pid=$!
  n=0
  until curl -fs "$base/healthz" >/dev/null; do
    kill -0 "$pid"
    n=$((n + 1))
    [ "$n" -lt 60 ]
    sleep 1
  done
  uv run --locked python -m smart_search.benchmark --mode "$mode" --url "$base"
  cleanup
done
