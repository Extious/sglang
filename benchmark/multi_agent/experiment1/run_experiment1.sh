#!/bin/bash
# One-click script to deploy Experiment 1 infrastructure:
# 1. Submit server job via Slurm
# 2. Monitor worker deployment
# 3. Start router locally
# Note: LangChain client should be run separately after deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${REPO_ROOT}/.venv"

# Activate virtual environment
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    source "${VENV_DIR}/Scripts/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
else
    echo "WARNING: Virtual environment not found at ${VENV_DIR}" >&2
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
NUM_WORKERS="${NUM_WORKERS:-5}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-8000}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"

WORKER_URLS_FILE="${LOG_DIR}/worker_urls.txt"
ROUTER_URL_FILE="${LOG_DIR}/router_url.txt"
ROUTER_PID_FILE="${LOG_DIR}/router.pid"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -f "${ROUTER_PID_FILE}" ]; then
        ROUTER_PID=$(cat "${ROUTER_PID_FILE}")
        if kill -0 "${ROUTER_PID}" 2>/dev/null; then
            echo "Stopping router (PID: ${ROUTER_PID})..."
            kill "${ROUTER_PID}" 2>/dev/null || true
            sleep 2
            kill -9 "${ROUTER_PID}" 2>/dev/null || true
        fi
        rm -f "${ROUTER_PID_FILE}"
    fi
    echo "Cleanup done."
}
trap cleanup EXIT INT TERM

# Function to check if file exists and has content
check_file_ready() {
    local file=$1
    local max_wait=${2:-300}
    local wait_interval=${3:-5}
    local elapsed=0
    
    echo "Waiting for ${file}..."
    while [ $elapsed -lt $max_wait ]; do
        if [ -f "${file}" ] && [ -s "${file}" ]; then
            echo "  Found: ${file}"
            return 0
        fi
        sleep "${wait_interval}"
        elapsed=$((elapsed + wait_interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  Still waiting... (${elapsed}s / ${max_wait}s)"
        fi
    done
    echo "  Timeout waiting for ${file}"
    return 1
}

# Function to check if URL is accessible
check_url_ready() {
    local url=$1
    local max_wait=${2:-60}
    local wait_interval=${3:-2}
    local elapsed=0
    
    echo "Checking if ${url} is accessible..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "${url}/v1/models" > /dev/null 2>&1; then
            echo "  ${url} is ready"
            return 0
        fi
        sleep "${wait_interval}"
        elapsed=$((elapsed + wait_interval))
    done
    echo "  Timeout waiting for ${url}"
    return 1
}

echo "=========================================="
echo "Experiment 1: Deploy Workers and Router"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Workers: ${NUM_WORKERS}"
echo "Router policy: ${ROUTER_POLICY}"
echo "=========================================="
echo ""

# Step 1: Submit server job
echo "Step 1: Submitting server job..."
if [ ! -f "run_server.slurm" ]; then
    echo "ERROR: run_server.slurm not found"
    exit 1
fi

# Export variables so Slurm job can read them.
export MODEL_PATH NUM_WORKERS WORKER_BASE_PORT

SERVER_JOB=$(sbatch --parsable run_server.slurm)
echo "  Server job ID: ${SERVER_JOB}"
echo "  Monitor with: squeue -j ${SERVER_JOB}"
echo ""

# Step 2: Wait for worker URLs file
echo "Step 2: Waiting for workers to deploy..."
if ! check_file_ready "${WORKER_URLS_FILE}" 1800 10; then
    echo "ERROR: Workers failed to deploy"
    echo "Check logs: ${LOG_DIR}/run_server_*.out"
    exit 1
fi

echo "Worker URLs:"
cat "${WORKER_URLS_FILE}" | sed 's/^/  /'
echo ""

# Step 3: Start router
echo "Step 3: Starting router..."
python deploy_router.py \
    --model-path "${MODEL_PATH}" \
    --worker-urls-file "${WORKER_URLS_FILE}" \
    --router-port "${ROUTER_PORT}" \
    --router-policy "${ROUTER_POLICY}" \
    --output-dir "${LOG_DIR}" \
    > "${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!
echo "${ROUTER_PID}" > "${ROUTER_PID_FILE}"
echo "  Router PID: ${ROUTER_PID}"
echo "  Router log: ${LOG_DIR}/router.log"
echo ""

# Wait for router URL file
if ! check_file_ready "${ROUTER_URL_FILE}" 120 5; then
    echo "ERROR: Router failed to start"
    echo "Check log: ${LOG_DIR}/router.log"
    exit 1
fi

ROUTER_URL=$(cat "${ROUTER_URL_FILE}" | head -1 | tr -d '\n')
echo "Router URL: ${ROUTER_URL}"
echo ""

# Wait for router to be accessible
if ! check_url_ready "${ROUTER_URL}" 60 2; then
    echo "ERROR: Router is not accessible"
    echo "Check log: ${LOG_DIR}/router.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo "Worker URLs: ${WORKER_URLS_FILE}"
echo "Router URL: ${ROUTER_URL}"
echo "Router PID: ${ROUTER_PID}"
echo "Router log: ${LOG_DIR}/router.log"
echo "Server log: ${LOG_DIR}/run_server_${SERVER_JOB}.out"
echo ""
echo "Next steps:"
echo "  1. Run LangChain client:"
echo "     python run_langchain_app.py --router-url ${ROUTER_URL} --num-traces 20"
echo ""
echo "  2. To stop router: kill ${ROUTER_PID}"
echo "  3. Or press Ctrl+C to cleanup"
echo ""
echo "Router is running in background. Press Ctrl+C to stop."
