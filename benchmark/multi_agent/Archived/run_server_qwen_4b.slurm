#!/bin/bash
#SBATCH --job-name=exp1_server
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4080:6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/home/comp/%u/sglang/benchmark/multi_agent/logs/run_server_%j.out
#SBATCH --error=/home/comp/%u/sglang/benchmark/multi_agent/logs/run_server_%j.err

# Usage:
#   sbatch [SBATCH_OPTIONS] run_server.slurm [SCRIPT_OPTIONS]
#
# SBATCH Options (must be before script name):
#   --nodes=N             Number of nodes to allocate (REQUIRED)
# Qwen/Qwen3-4B-Instruct-2507
# Script Options:
#   --model-path PATH     Model path (default: Qwen/Qwen3-4B-Instruct-2507)
#   --num-workers N       Number of workers (default: 6)
#   --worker-base-port P  Base port for workers (default: 8000)
#   --workers-per-node N  Workers per node (default: 6)
#   --tool-call-parser P  Tool call parser (default: qwen)
#   --enable-cache-report Return cached tokens in OpenAI usage
#   --disable-cache-report Disable cached token reporting
#   --enable-hicache      Enable hierarchical KV cache (default: enabled)
#   --disable-hicache     Disable hierarchical KV cache
#   --hicache-size N      HiCache size per worker in GB (default: 8)
#   --hicache-ratio R     HiCache ratio when hicache-size<=0 (default: 2.0)
#   --enable-peer-replication  Enable peer KV cache replication for fault tolerance
#   --peer-port-base N    Base port for PeerCacheServer listeners (default: 9000)
#   --peer-buffer-size-gb N  Max DRAM per worker for peer buffer in GB (default: 8)
#   -h, --help            Show this help message
#
# Examples:
#   # 1 node, 6 workers, HiCache enabled by default
#   sbatch --nodes=1 run_server_qwen_4b.slurm
#
#   # 1 node, 1 worker, custom HiCache size
#   sbatch --nodes=1 run_server_qwen_4b.slurm --num-workers 1 --workers-per-node 1 --hicache-size 24
#
#   # Custom model
#   sbatch --nodes=2 run_server_qwen_4b.slurm --model-path meta-llama/Llama-3-8B --num-workers 4
#
# ============================================================================
# Fault Tolerance (Peer KV Cache Replication)
# ============================================================================
# To enable cross-worker fault tolerance via peer KV cache replication:
#
# 1. Start workers with --enable-peer-replication:
#
#      sbatch --nodes=1 run_server_qwen_4b.slurm --enable-peer-replication
#
#    This automatically:
#      - Enables HiCache (required)
#      - Sets --hicache-storage-backend peer for each worker
#      - Computes ring peer mapping: worker_i replicates KV to worker_(i+1)%N
#      - Starts PeerCacheServer on each worker (ports 9000+gpu_id)
#      - Writes peer_mapping.txt and router_extra_args.txt to logs/
#
# 2. Start the router with --enable-fault-tolerance:
#
#      ./run_router.sh --worker-urls-file ../logs/worker_urls.txt \
#                      --enable-fault-tolerance
#
#    This activates ring failover: when worker_i fails, the router routes
#    to worker_(i+1)%N which already has worker_i's KV cache replicas,
#    avoiding full re-prefill.
#
# Optional tuning:
#      sbatch run_server_qwen_4b.slurm --enable-peer-replication \
#             --peer-port-base 9000 --peer-buffer-size-gb 16
# ============================================================================

set -euo pipefail

# Load GCC 11.2.0 for C++20 support required by JIT kernels.
# - NVCC_PREPEND_FLAGS: injects -ccbin so nvcc uses the correct host compiler.
# - LD_LIBRARY_PATH: ensures the JIT-compiled .so can find the matching libstdc++.
module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
export CXX="${GCC_ROOT}/bin/g++"
export CC="${GCC_ROOT}/bin/gcc"
export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Default values (can be overridden by environment variables or command line args)
DEFAULT_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_NUM_WORKERS=6
DEFAULT_WORKER_BASE_PORT=8000
DEFAULT_WORKERS_PER_NODE=6
DEFAULT_TOOL_CALL_PARSER="qwen"
DEFAULT_ENABLE_CACHE_REPORT=1
DEFAULT_ENABLE_HICACHE=1
DEFAULT_HICACHE_SIZE=8
DEFAULT_HICACHE_RATIO=2.0
DEFAULT_ENABLE_PEER_REPLICATION=0
DEFAULT_PEER_PORT_BASE=9000
DEFAULT_PEER_BUFFER_SIZE_GB=8
DEFAULT_QUANTIZATION=""

# Parse command line arguments
show_help() {
    sed -n '15,40p' "$0" | sed 's/^# //'
    exit 0
}

# Initialize with environment variables or defaults
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-${DEFAULT_WORKER_BASE_PORT}}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-${DEFAULT_WORKERS_PER_NODE}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${DEFAULT_TOOL_CALL_PARSER}}"
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-${DEFAULT_ENABLE_CACHE_REPORT}}"
ENABLE_HICACHE="${ENABLE_HICACHE:-${DEFAULT_ENABLE_HICACHE}}"
HICACHE_SIZE="${HICACHE_SIZE:-${DEFAULT_HICACHE_SIZE}}"
HICACHE_RATIO="${HICACHE_RATIO:-${DEFAULT_HICACHE_RATIO}}"
ENABLE_PEER_REPLICATION="${ENABLE_PEER_REPLICATION:-${DEFAULT_ENABLE_PEER_REPLICATION}}"
PEER_PORT_BASE="${PEER_PORT_BASE:-${DEFAULT_PEER_PORT_BASE}}"
PEER_BUFFER_SIZE_GB="${PEER_BUFFER_SIZE_GB:-${DEFAULT_PEER_BUFFER_SIZE_GB}}"
QUANTIZATION="${QUANTIZATION:-${DEFAULT_QUANTIZATION}}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --worker-base-port)
            WORKER_BASE_PORT="$2"
            shift 2
            ;;
        --workers-per-node)
            WORKERS_PER_NODE="$2"
            shift 2
            ;;
        --tool-call-parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        --enable-cache-report)
            ENABLE_CACHE_REPORT=1
            shift
            ;;
        --disable-cache-report)
            ENABLE_CACHE_REPORT=0
            shift
            ;;
        --enable-hicache)
            ENABLE_HICACHE=1
            shift
            ;;
        --disable-hicache)
            ENABLE_HICACHE=0
            shift
            ;;
        --hicache-size)
            HICACHE_SIZE="$2"
            shift 2
            ;;
        --hicache-ratio)
            HICACHE_RATIO="$2"
            shift 2
            ;;
        --enable-peer-replication)
            ENABLE_PEER_REPLICATION=1
            shift
            ;;
        --peer-port-base)
            PEER_PORT_BASE="$2"
            shift 2
            ;;
        --peer-buffer-size-gb)
            PEER_BUFFER_SIZE_GB="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

if [ "${ENABLE_HICACHE}" -eq 1 ]; then
    HICACHE_FLAG="--enable-hicache"
else
    HICACHE_FLAG="--disable-hicache"
    HICACHE_SIZE=0
fi
if [ "${ENABLE_PEER_REPLICATION}" -eq 1 ]; then
    PEER_REPLICATION_FLAG="--enable-peer-replication --peer-port-base ${PEER_PORT_BASE} --peer-buffer-size-gb ${PEER_BUFFER_SIZE_GB}"
else
    PEER_REPLICATION_FLAG=""
fi
if [ "${ENABLE_CACHE_REPORT}" -eq 1 ]; then
    CACHE_REPORT_FLAG="--enable-cache-report"
else
    CACHE_REPORT_FLAG="--disable-cache-report"
fi

# Resolve repo path robustly in SLURM runtime.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
export SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

_find_repo_root() {
    local _dir="$1"
    while [ "$_dir" != "/" ]; do
        if [ -d "$_dir/benchmark/multi_agent" ]; then
            echo "$_dir"
            return 0
        fi
        _dir="$(dirname "$_dir")"
    done
    return 1
}

if [ -n "${SGLANG_REPO_ROOT:-}" ] && [ -d "${SGLANG_REPO_ROOT}/benchmark/multi_agent" ]; then
    REPO_ROOT="${SGLANG_REPO_ROOT}"
elif _found=$(_find_repo_root "${SLURM_SUBMIT_DIR:-$(pwd)}"); then
    REPO_ROOT="${_found}"
elif _found=$(_find_repo_root "$(pwd)"); then
    REPO_ROOT="${_found}"
else
    echo "ERROR: Cannot locate repo root." >&2
    echo "  Set SGLANG_REPO_ROOT or submit from repo root containing benchmark/multi_agent." >&2
    exit 1
fi

SCRIPT_DIR="${REPO_ROOT}/benchmark/multi_agent"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
VENV_DIR="${REPO_ROOT}/.venv"

# Avoid stale "ready" markers from previous runs (shared filesystem).
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}/worker_urls.txt"
rm -f "${LOG_DIR}"/node_*/worker_urls.txt
rm -f "${LOG_DIR}"/node_*/server_node.txt
rm -f "${LOG_DIR}"/node_*/deploy.log
rm -f "${LOG_DIR}"/node_*/worker_*.log

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
if [ ! -f "${SCRIPT_DIR}/src/server/deploy_server.py" ]; then
    echo "ERROR: deploy_server.py not found in ${SCRIPT_DIR}/src/server" >&2
    exit 1
fi

if [ ! -d "${REPO_ROOT}/python" ]; then
    echo "ERROR: SGLang python directory not found: ${REPO_ROOT}/python" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

echo "Deploying SGLang Server (Workers)"
echo "Nodes: ${SLURM_NODELIST:-unknown}"
echo "Total workers: ${NUM_WORKERS}"
echo "Workers per node: ${WORKERS_PER_NODE}"
echo "Model: ${MODEL_PATH}"
echo "Tool call parser: ${TOOL_CALL_PARSER}"
echo "Cache report: ${ENABLE_CACHE_REPORT}"
echo "HiCache: ${ENABLE_HICACHE}"
if [ "${ENABLE_HICACHE}" -eq 1 ]; then
    if [ "${HICACHE_SIZE}" -gt 0 ]; then
        echo "HiCache size per worker (GB): ${HICACHE_SIZE}"
    else
        echo "HiCache ratio: ${HICACHE_RATIO}"
    fi
fi
echo "Peer replication: ${ENABLE_PEER_REPLICATION}"
if [ "${ENABLE_PEER_REPLICATION}" -eq 1 ]; then
    echo "Peer port base: ${PEER_PORT_BASE}"
    echo "Peer buffer size (GB): ${PEER_BUFFER_SIZE_GB}"
fi
echo "Repo root: ${REPO_ROOT}"
echo "Output: ${LOG_DIR}"

# For multi-node deployment: distribute workers across nodes
TOTAL_NODES=${SLURM_JOB_NUM_NODES:-1}
NODE_LIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

# Calculate distribution
REMAINING_WORKERS=${NUM_WORKERS}
WORKER_ID=0
BASE_PORT=${WORKER_BASE_PORT}

# Start workers on each node
SRUN_PIDS=()
NODE_IDS=()
NODE_NAMES=()
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
    
    NODE_LOG_DIR="${LOG_DIR}/node_${NODE_ID}"
    mkdir -p "${NODE_LOG_DIR}"
    rm -f "${NODE_LOG_DIR}/worker_urls.txt" "${NODE_LOG_DIR}/server_node.txt" \
          "${NODE_LOG_DIR}/deploy.log" "${NODE_LOG_DIR}"/worker_*.log

    srun -w "${NODE_NAME}" --nodes=1 --ntasks=1 --kill-on-bad-exit=0 \
        env \
            VENV_DIR="${VENV_DIR}" \
            REPO_ROOT="${REPO_ROOT}" \
            SCRIPT_DIR="${SCRIPT_DIR}" \
            NODE_LOG_DIR="${NODE_LOG_DIR}" \
            MODEL_PATH="${MODEL_PATH}" \
            NODE_WORKERS="${NODE_WORKERS}" \
            NODE_PORT="${NODE_PORT}" \
            TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
            ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT}" \
            CACHE_REPORT_FLAG="${CACHE_REPORT_FLAG}" \
            HICACHE_FLAG="${HICACHE_FLAG}" \
            HICACHE_SIZE="${HICACHE_SIZE}" \
            HICACHE_RATIO="${HICACHE_RATIO}" \
            PEER_REPLICATION_FLAG="${PEER_REPLICATION_FLAG}" \
            QUANTIZATION="${QUANTIZATION}" \
            NODE_ID="${NODE_ID}" \
        bash -lc '
        set -euo pipefail

        module load gcc/11.2.0 2>/dev/null || true
        GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
        export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
        export CXX="${GCC_ROOT}/bin/g++"
        export CC="${GCC_ROOT}/bin/gcc"
        export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

        if [ -f "${VENV_DIR}/bin/activate" ]; then
            source "${VENV_DIR}/bin/activate"
        elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
            source "${VENV_DIR}/Scripts/activate"
        fi

        export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
        export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
        export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
        export LC_ALL="${LC_ALL:-C.UTF-8}"
        export LANG="${LANG:-C.UTF-8}"
        export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
        export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

        cd "${SCRIPT_DIR}/src/server"
        mkdir -p "${NODE_LOG_DIR}"

        PY_BIN="python"
        if [ -x "${VENV_DIR}/bin/python" ]; then
            PY_BIN="${VENV_DIR}/bin/python"
        fi

        SITE_PACKAGES_DIR="${VENV_DIR}/lib/python3.12/site-packages"
        PROBE_PATHS=(
            "${SITE_PACKAGES_DIR}"
            "${SITE_PACKAGES_DIR}/timm/models"
            "${SITE_PACKAGES_DIR}/pygments/lexers"
        )

        preflight_python() {
            "${PY_BIN}" - <<'"'"'PY'"'"' >/dev/null 2>&1
import importlib.metadata
import multiprocessing.spawn
import pygments.lexers
import subprocess
import timm.models
PY
        }

        for attempt in $(seq 1 8); do
            for p in "${PROBE_PATHS[@]}"; do
                if [ -e "${p}" ]; then
                    touch "${p}" >/dev/null 2>&1 || true
                    ls "${p}" >/dev/null 2>&1 || true
                fi
            done

            if preflight_python; then
                echo "[node ${NODE_ID}] Python import preflight OK (attempt ${attempt}/8)." >> "${NODE_LOG_DIR}/deploy.log"
                break
            fi

            echo "[node ${NODE_ID}] Python import preflight failed (attempt ${attempt}/8), retrying..." >> "${NODE_LOG_DIR}/deploy.log"
            if [ "${attempt}" -eq 8 ]; then
                echo "[node ${NODE_ID}] Python import preflight failed after retries." >> "${NODE_LOG_DIR}/deploy.log"
                exit 1
            fi
            sleep 3
        done

        for deploy_try in 1 2 3; do
            rm -f "${NODE_LOG_DIR}/worker_urls.txt" "${NODE_LOG_DIR}/server_node.txt"
            echo "[node ${NODE_ID}] Starting deploy_server.py (attempt ${deploy_try}/3)." >> "${NODE_LOG_DIR}/deploy.log"
            set +e
            QUANT_FLAG=""
            if [ -n "${QUANTIZATION}" ]; then
                QUANT_FLAG="--quantization ${QUANTIZATION}"
            fi
            "${PY_BIN}" deploy_server.py \
                --model-path "${MODEL_PATH}" \
                --num-workers "${NODE_WORKERS}" \
                --worker-base-port "${NODE_PORT}" \
                --tool-call-parser "${TOOL_CALL_PARSER}" \
                ${CACHE_REPORT_FLAG} \
                ${HICACHE_FLAG} \
                ${QUANT_FLAG} \
                --hicache-size "${HICACHE_SIZE}" \
                --hicache-ratio "${HICACHE_RATIO}" \
                ${PEER_REPLICATION_FLAG} \
                --output-dir "${NODE_LOG_DIR}" \
                >> "${NODE_LOG_DIR}/deploy.log" 2>&1
            rc=$?
            set -e

            if [ "${rc}" -eq 0 ]; then
                exit 0
            fi

            if grep -q "OSError: \\[Errno 5\\]" "${NODE_LOG_DIR}/deploy.log"; then
                echo "[node ${NODE_ID}] deploy_server.py hit transient Errno 5, retrying..." >> "${NODE_LOG_DIR}/deploy.log"
                sleep 8
                continue
            fi

            echo "[node ${NODE_ID}] deploy_server.py failed with exit ${rc} (non-retryable)." >> "${NODE_LOG_DIR}/deploy.log"
            exit "${rc}"
        done

        echo "[node ${NODE_ID}] deploy_server.py failed after retries." >> "${NODE_LOG_DIR}/deploy.log"
        exit 1
    ' &
    
    SRUN_PIDS+=($!)
    NODE_IDS+=(${NODE_ID})
    NODE_NAMES+=("${NODE_NAME}")
    WORKER_ID=$((WORKER_ID + NODE_WORKERS))
    REMAINING_WORKERS=$((REMAINING_WORKERS - NODE_WORKERS))
done

# Wait for worker_urls.txt files to appear (workers ready)
echo "Waiting for workers to be ready (timeout=1800s)..."
WAIT_TIMEOUT=1800
WAIT_START=$(date +%s)
FAILED_NODES=()

for idx in "${!NODE_IDS[@]}"; do
    NODE_ID="${NODE_IDS[$idx]}"
    NODE_NAME="${NODE_NAMES[$idx]}"
    SRUN_PID="${SRUN_PIDS[$idx]}"
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
        if ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
            set +e
            wait "${SRUN_PID}"
            SRUN_RC=$?
            set -e
            echo "ERROR: Node ${NODE_ID} (${NODE_NAME}) exited before workers became ready (exit=${SRUN_RC})." >&2
            if [ -f "${LOG_DIR}/node_${NODE_ID}/deploy.log" ]; then
                echo "----- tail: ${LOG_DIR}/node_${NODE_ID}/deploy.log -----" >&2
                tail -n 80 "${LOG_DIR}/node_${NODE_ID}/deploy.log" >&2 || true
                echo "--------------------------------------------------------" >&2
            fi
            FAILED_NODES+=(${NODE_ID})
            break
        fi
        sleep 5
    done
done

# Collect and merge worker URLs from all nodes
WORKER_URLS_FILE="${LOG_DIR}/worker_urls.txt"
: > "${WORKER_URLS_FILE}"

SUCCESS_COUNT=0
for NODE_ID in "${NODE_IDS[@]}"; do
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

    echo "Waiting on deploy processes..."
    set +e
    wait "${SRUN_PIDS[@]}"
    WAIT_RC=$?
    set -e
    if [ "${WAIT_RC}" -ne 0 ]; then
        echo "ERROR: One or more deploy processes exited unexpectedly (rc=${WAIT_RC})." >&2
        exit "${WAIT_RC}"
    fi
else
    echo "WARNING: Only ${SUCCESS_COUNT}/${NUM_WORKERS} workers started" >&2
    if [ ${#FAILED_NODES[@]} -gt 0 ]; then
        echo "Failed nodes: ${FAILED_NODES[*]}" >&2
    fi
    exit 1
fi
