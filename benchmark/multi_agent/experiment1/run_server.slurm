#!/bin/bash
#SBATCH --job-name=exp1_server
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --exclude=gpu[10-13]
#SBATCH --output=/home/comp/zhanzhao/sglang/benchmark/multi_agent/experiment1/logs/run_server_%j.out
#SBATCH --error=/home/comp/zhanzhao/sglang/benchmark/multi_agent/experiment1/logs/run_server_%j.err

set -euo pipefail

# Use hardcoded script directory (BASH_SOURCE may not work correctly in SLURM)
SCRIPT_DIR="/home/comp/zhanzhao/sglang/benchmark/multi_agent/experiment1"
cd "${SCRIPT_DIR}"

REPO_ROOT="${SGLANG_REPO_ROOT:-/home/comp/zhanzhao/sglang}"
LOG_DIR="${SCRIPT_DIR}/logs"
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
    echo "  Trying to continue with system Python..." >&2
fi

# Verify critical paths exist
if [ ! -f "${SCRIPT_DIR}/deploy_server.py" ]; then
    echo "ERROR: deploy_server.py not found in ${SCRIPT_DIR}" >&2
    exit 1
fi

if [ ! -d "${REPO_ROOT}/python" ]; then
    echo "ERROR: SGLang python directory not found: ${REPO_ROOT}/python" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
NUM_WORKERS="${NUM_WORKERS:-5}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-8000}"

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "Deploying SGLang Server (Workers)"
echo "Nodes: ${SLURM_NODELIST:-unknown}"
echo "Total workers: ${NUM_WORKERS}"
echo "Model: ${MODEL_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "Output: ${LOG_DIR}"

# For multi-node deployment: distribute workers across nodes
# Each node has 2 GPUs, so we assign workers as:
# Node 0: 2 workers (GPUs 0-1)
# Node 1: 2 workers (GPUs 0-1)
# Node 2: 1 worker (GPU 0)
WORKERS_PER_NODE=2
TOTAL_NODES=${SLURM_JOB_NUM_NODES:-1}
NODE_LIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

# Calculate distribution
REMAINING_WORKERS=${NUM_WORKERS}
WORKER_ID=0
BASE_PORT=${WORKER_BASE_PORT}

# Start workers on each node
SRUN_PIDS=()
NODE_IDS=()
for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
    if [ ${REMAINING_WORKERS} -le 0 ]; then
        break
    fi
    
    NODE_WORKERS=${WORKERS_PER_NODE}
    if [ ${NODE_WORKERS} -gt ${REMAINING_WORKERS} ]; then
        NODE_WORKERS=${REMAINING_WORKERS}
    fi
    
    NODE_NAME=${NODE_LIST[$NODE_ID]}
    NODE_PORT=$((BASE_PORT + WORKER_ID))
    
    echo "Node ${NODE_ID} (${NODE_NAME}): Starting ${NODE_WORKERS} workers (ports ${NODE_PORT}-$((NODE_PORT + NODE_WORKERS - 1)))"
    
    srun -w "${NODE_NAME}" --nodes=1 --ntasks=1 \
        bash -c "
        # Activate virtual environment
        if [ -f \"${VENV_DIR}/bin/activate\" ]; then
            source \"${VENV_DIR}/bin/activate\"
        elif [ -f \"${VENV_DIR}/Scripts/activate\" ]; then
            source \"${VENV_DIR}/Scripts/activate\"
        fi
        
        export PYTHONPATH=\"${REPO_ROOT}/python:${REPO_ROOT}:\${PYTHONPATH:-}\"
        export NO_PROXY=\"localhost,127.0.0.1,0.0.0.0,::1\"
        export no_proxy=\"localhost,127.0.0.1,0.0.0.0,::1\"
        export LC_ALL=\"\${LC_ALL:-C.UTF-8}\"
        export LANG=\"\${LANG:-C.UTF-8}\"
        export PYTHONIOENCODING=\"\${PYTHONIOENCODING:-utf-8}\"
        
        cd ${SCRIPT_DIR}
        mkdir -p \"${LOG_DIR}/node_${NODE_ID}\"
        
        python deploy_server.py \
            --model-path \"${MODEL_PATH}\" \
            --num-workers ${NODE_WORKERS} \
            --worker-base-port ${NODE_PORT} \
            --output-dir \"${LOG_DIR}/node_${NODE_ID}\" \
            > \"${LOG_DIR}/node_${NODE_ID}/deploy.log\" 2>&1
    " &
    
    SRUN_PIDS+=($!)
    NODE_IDS+=(${NODE_ID})
    WORKER_ID=$((WORKER_ID + NODE_WORKERS))
    REMAINING_WORKERS=$((REMAINING_WORKERS - NODE_WORKERS))
done

# Wait for worker_urls.txt files to appear (workers ready)
echo "Waiting for workers to be ready (timeout=1800s)..."
WAIT_TIMEOUT=1800
WAIT_START=$(date +%s)
FAILED_NODES=()

for NODE_ID in "${NODE_IDS[@]}"; do
    URLS_FILE="${LOG_DIR}/node_${NODE_ID}/worker_urls.txt"
    echo "  Waiting for node ${NODE_ID} workers..."
    while true; do
        ELAPSED=$(($(date +%s) - WAIT_START))
        if [ ${ELAPSED} -ge ${WAIT_TIMEOUT} ]; then
            echo "ERROR: Timeout waiting for node ${NODE_ID} workers" >&2
            FAILED_NODES+=(${NODE_ID})
            break
        fi
        if [ -f "${URLS_FILE}" ] && [ -s "${URLS_FILE}" ]; then
            echo "  Node ${NODE_ID} workers ready."
            break
        fi
        sleep 5
    done
done

# Collect and merge worker URLs from all nodes
WORKER_URLS_FILE="${LOG_DIR}/worker_urls.txt"
: > "${WORKER_URLS_FILE}"

SUCCESS_COUNT=0
for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
    NODE_URLS_FILE="${LOG_DIR}/node_${NODE_ID}/worker_urls.txt"
    if [ -f "${NODE_URLS_FILE}" ] && [ -s "${NODE_URLS_FILE}" ]; then
        echo "Node ${NODE_ID} worker URLs:"
        cat "${NODE_URLS_FILE}" | sed 's/^/  /'
        cat "${NODE_URLS_FILE}" >> "${WORKER_URLS_FILE}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + $(wc -l < "${NODE_URLS_FILE}" | tr -d ' ')))
    else
        echo "WARNING: Node ${NODE_ID} worker_urls.txt not found or empty" >&2
        if [ -f "${LOG_DIR}/node_${NODE_ID}/deploy.log" ]; then
            echo "  Check log: ${LOG_DIR}/node_${NODE_ID}/deploy.log" >&2
        fi
    fi
done

echo ""
if [ ${SUCCESS_COUNT} -eq ${NUM_WORKERS} ]; then
    echo "Server deployment completed successfully. Worker URLs saved to ${WORKER_URLS_FILE}"
    echo "Total workers: ${SUCCESS_COUNT}"
    echo ""
    echo "Workers are running. Job will stay alive until cancelled (scancel) or timeout."
    echo "Press Ctrl+C or use 'scancel ${SLURM_JOB_ID}' to stop."

    # Keep the job running - wait indefinitely
    # This allows the worker processes to continue serving requests
    while true; do
        sleep 60
        echo "[$(date)] Workers still running..."
    done
else
    echo "WARNING: Only ${SUCCESS_COUNT}/${NUM_WORKERS} workers started" >&2
    if [ ${#FAILED_NODES[@]} -gt 0 ]; then
        echo "Failed nodes: ${FAILED_NODES[*]}" >&2
    fi
    exit 1
fi
