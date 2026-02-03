#!/bin/bash
# Launch N sglang workers (1 per GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WORKER_BIND_HOST="${WORKER_BIND_HOST:-127.0.0.1}"
WORKER_HEALTH_HOST="${WORKER_HEALTH_HOST:-127.0.0.1}"

# Stagger worker startup to avoid NFS/disk I/O errors when many processes start at once
WORKER_STARTUP_DELAY="${WORKER_STARTUP_DELAY:-12}"
# Retry worker start if it exits quickly (e.g. NFS EIO); check after this many seconds
WORKER_START_CHECK_SEC="${WORKER_START_CHECK_SEC:-30}"
WORKER_START_RETRIES="${WORKER_START_RETRIES:-3}"
# Seconds to wait before retrying after a worker exits early (lets NFS settle)
WORKER_RETRY_DELAY_SEC="${WORKER_RETRY_DELAY_SEC:-10}"

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
# Reduce Python filesystem encoding probe; can avoid NFS EIO on init_fs_encoding
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PIDS=()
cleanup() {
    echo "Cleaning up workers..."
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Per-worker wait: longer on NFS since model load reads a lot from disk
WAIT_FOR_PORT_MAX_ATTEMPTS="${WAIT_FOR_PORT_MAX_ATTEMPTS:-90}"
WAIT_FOR_PORT_SLEEP_SEC="${WAIT_FOR_PORT_SLEEP_SEC:-2}"
wait_for_port() {
    local port=$1
    local max_attempts=${WAIT_FOR_PORT_MAX_ATTEMPTS}
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://${WORKER_HEALTH_HOST}:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep "${WAIT_FOR_PORT_SLEEP_SEC}"
        attempt=$((attempt + 1))
    done
    echo "Timeout waiting for port ${port}"
    return 1
}

echo "Starting ${NUM_WORKERS} Qwen3-4B-Thinking workers (1 per GPU)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((WORKER_BASE_PORT + i))
    export CUDA_VISIBLE_DEVICES=${i}
    : > "${LOG_DIR}/worker_${i}.log"
    pid=""
    for attempt in $(seq 1 "${WORKER_START_RETRIES}"); do
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --tp 1 \
            --port "${port}" \
            --host "${WORKER_BIND_HOST}" \
            >> "${LOG_DIR}/worker_${i}.log" 2>&1 &
        pid=$!
        echo "  Worker ${i}: GPU ${i}, port ${port}, PID ${pid} (attempt ${attempt}/${WORKER_START_RETRIES})"
        sleep "${WORKER_START_CHECK_SEC}"
        if kill -0 "${pid}" 2>/dev/null; then
            PIDS+=("${pid}")
            break
        fi
        wait "${pid}" 2>/dev/null || true
        echo "  Worker ${i} exited early (e.g. NFS I/O error)"
        if [ $attempt -lt "${WORKER_START_RETRIES}" ]; then
            echo "  Waiting ${WORKER_RETRY_DELAY_SEC}s before retry..."
            sleep "${WORKER_RETRY_DELAY_SEC}"
        fi
    done
    if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
        echo "Worker ${i} failed after ${WORKER_START_RETRIES} attempts. Check ${LOG_DIR}/worker_${i}.log"
        cleanup
        exit 1
    fi
    echo "  Waiting for worker ${i} (port ${port}) to be ready..."
    wait_for_port "${port}" || { echo "Worker ${i} (port ${port}) failed to become ready"; cleanup; exit 1; }
    echo "  Worker ${i} (port ${port}) ready"
done

WORKER_URLS=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((WORKER_BASE_PORT + i))
    WORKER_URLS="${WORKER_URLS} http://${WORKER_HEALTH_HOST}:${port}"
done
WORKER_URLS="${WORKER_URLS# }"
echo "${WORKER_URLS}" > "${LOG_DIR}/worker_urls.txt"

echo "All workers ready."
echo "Worker URLs: ${WORKER_URLS}"
echo "Saved to: ${LOG_DIR}/worker_urls.txt"
echo "Press Ctrl+C to stop workers."

wait
