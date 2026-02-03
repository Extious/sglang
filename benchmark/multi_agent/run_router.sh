#!/bin/bash
# Launch the router that load-balances across workers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"

WORKER_BASE_PORT="${WORKER_BASE_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WORKER_URL_HOST="${WORKER_URL_HOST:-127.0.0.1}"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/logs/worker_urls.txt}"

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

WORKER_URLS="${WORKER_URLS:-}"
if [ -z "${WORKER_URLS}" ] && [ -f "${WORKER_URLS_FILE}" ]; then
    WORKER_URLS="$(<"${WORKER_URLS_FILE}")"
fi
if [ -z "${WORKER_URLS}" ]; then
    WORKER_URLS=""
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        WORKER_URLS="${WORKER_URLS} http://${WORKER_URL_HOST}:${port}"
    done
    WORKER_URLS="${WORKER_URLS# }"
fi

echo "Starting router on port ${ROUTER_PORT} (worker URLs: ${WORKER_URLS})..."

# Router: use launch_router from sgl-model-gateway (pip install sglang[gateway] or install from sgl-model-gateway/bindings/python)
python -m sglang_router.launch_router \
    --worker-urls ${WORKER_URLS} \
    --policy "${ROUTER_POLICY}" \
    --host "${ROUTER_HOST}" \
    --port "${ROUTER_PORT}" \
    --model-path "${MODEL_PATH}"
