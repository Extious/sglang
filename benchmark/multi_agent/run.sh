#!/bin/bash
# Deploy 10x Qwen3-4B-Thinking on 10 GPUs with 1 router.
# Each worker: 1 GPU, 1 model instance. Router load-balances on port 30000.
#
# Split scripts:
#   - ./run_server.sh (workers)
#   - ./run_router.sh (router)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/logs/worker_urls.txt}"

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/run_server.sh" &
SERVER_PID=$!

WAIT_WORKER_URLS_MAX_SEC="${WAIT_WORKER_URLS_MAX_SEC:-1800}"
WAIT_WORKER_URLS_SLEEP_SEC="${WAIT_WORKER_URLS_SLEEP_SEC:-2}"
elapsed=0
while [ ! -s "${WORKER_URLS_FILE}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}" 2>/dev/null || true
        echo "run_server.sh exited before workers became ready. Check ${SCRIPT_DIR}/logs/worker_*.log"
        exit 1
    fi
    if [ $elapsed -ge "${WAIT_WORKER_URLS_MAX_SEC}" ]; then
        echo "Timeout waiting for ${WORKER_URLS_FILE}"
        exit 1
    fi
    sleep "${WAIT_WORKER_URLS_SLEEP_SEC}"
    elapsed=$((elapsed + WAIT_WORKER_URLS_SLEEP_SEC))
done

"${SCRIPT_DIR}/run_router.sh"
