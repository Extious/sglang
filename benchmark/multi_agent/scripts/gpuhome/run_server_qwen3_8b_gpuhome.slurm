#!/bin/bash
#SBATCH --job-name=exp1_server_8b
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu[10-13]
#SBATCH --output=/home/comp/%u/sglang/benchmark/multi_agent/logs/logs_qwen3_8b_gpuhome/run_server_%j.out
#SBATCH --error=/home/comp/%u/sglang/benchmark/multi_agent/logs/logs_qwen3_8b_gpuhome/run_server_%j.err

set -euo pipefail

SCRIPT_BASENAME="run_server_qwen3_8b_gpuhome.slurm"

module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
if [ -d "${GCC_ROOT}/bin" ]; then
    export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
    export CXX="${GCC_ROOT}/bin/g++"
    export CC="${GCC_ROOT}/bin/gcc"
    export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DEFAULT_MODEL_PATH="Qwen/Qwen3-8B"
DEFAULT_DP_SIZE=2
DEFAULT_PP_SIZE=2
DEFAULT_TP_SIZE=1
DEFAULT_SERVER_PORT_BASE=28000
DEFAULT_DIST_INIT_PORT_OFFSET=100
DEFAULT_TOOL_CALL_PARSER="qwen"
DEFAULT_ENABLE_CACHE_REPORT=1
DEFAULT_ENABLE_HICACHE=1
DEFAULT_HICACHE_SIZE=16
DEFAULT_HICACHE_RATIO=2.0
DEFAULT_ENABLE_PEER_REPLICATION=0
DEFAULT_PEER_PORT_BASE=29000
DEFAULT_PEER_BUFFER_SIZE_GB=16
DEFAULT_PREFETCH_POLICY=""
DEFAULT_HICACHE_EXTRA_CONFIG=""
DEFAULT_QUANTIZATION=""
DEFAULT_WAIT_TIMEOUT=1800

show_help() {
    cat <<'EOF'
Usage:
  sbatch [SBATCH_OPTIONS] run_server_qwen3_8b_gpuhome.slurm [SCRIPT_OPTIONS]

Script options:
  --model-path PATH        Model path (default: Qwen/Qwen3-8B)
  --dp-size N              Internal data parallel size (default: 2)
  --pp-size N              Pipeline parallel size per DP replica (default: 2)
  --tp-size N              Tensor parallel size per stage (default: 1)
  --server-port-base P     HTTP port used on every node (default: 28000)
  --worker-base-port P     Alias of --server-port-base
  --tool-call-parser P     Tool call parser (default: qwen)
  --enable-cache-report    Return cached tokens in OpenAI usage
  --disable-cache-report   Disable cached token reporting
  --enable-hicache         Enable hierarchical KV cache (default: enabled)
  --disable-hicache        Disable hierarchical KV cache
  --hicache-size N         HiCache size per DP replica in GB (default: 16)
  --hicache-ratio R        HiCache ratio when hicache-size<=0 (default: 2.0)
  --enable-peer-replication
                           Enable peer KV cache replication
  --peer-port-base N       Base port for peer cache listeners (default: 29000)
  --peer-buffer-size-gb N  Max DRAM per stage peer buffer in GB (default: 16)
  --prefetch-policy P      Prefetch policy: auto|best_effort|wait_complete|timeout
  --hicache-extra-config J JSON string for hicache backend extra config
  --quantization Q         Quantization method (e.g. fp8)
  --wait-timeout SECS      Timeout waiting for all nodes to become healthy
  -h, --help               Show this help

Outputs:
  benchmark/multi_agent/logs/logs_qwen3_8b_gpuhome/server_url.txt
  benchmark/multi_agent/logs/logs_qwen3_8b_gpuhome/stage_manifest.json
  benchmark/multi_agent/logs/logs_qwen3_8b_gpuhome/runtime_failover_events.jsonl

Notes:
  This script launches one internal DP+PP service across all allocated nodes.
  Only node_rank=0 is the external head endpoint used by clients.
EOF
}

wait_http_healthy() {
    local url="$1"
    local child_pid="$2"
    local timeout_s="${3:-1800}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if curl --noproxy '*' -fsS -m 3 "${url}" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${child_pid}" >/dev/null 2>&1; then
            return 1
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_s}" ]; then
            return 1
        fi
        sleep 2
    done
}

run_node_server() {
    local node_rank="${NODE_RANK:?NODE_RANK is required}"
    local node_name="${NODE_NAME:?NODE_NAME is required}"
    local node_log_dir="${NODE_LOG_DIR:?NODE_LOG_DIR is required}"
    local model_path="${MODEL_PATH:?MODEL_PATH is required}"
    local dp_size="${DP_SIZE:?DP_SIZE is required}"
    local pp_size="${PP_SIZE:?PP_SIZE is required}"
    local tp_size="${TP_SIZE:?TP_SIZE is required}"
    local nnodes="${NNODES:?NNODES is required}"
    local server_port="${SERVER_PORT:?SERVER_PORT is required}"
    local dist_init_addr="${DIST_INIT_ADDR:?DIST_INIT_ADDR is required}"
    local head_node="${HEAD_NODE:?HEAD_NODE is required}"
    local wait_timeout="${WAIT_TIMEOUT:?WAIT_TIMEOUT is required}"

    mkdir -p "${node_log_dir}"
    rm -f \
        "${node_log_dir}/server.log" \
        "${node_log_dir}/server_url.txt" \
        "${node_log_dir}/server_node.txt" \
        "${node_log_dir}/node_manifest.json" \
        "${node_log_dir}/server_main.pid" \
        "${node_log_dir}/launch_command.txt"

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
        source "${VENV_DIR}/Scripts/activate"
    fi

    export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
    export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,${node_name}"
    export no_proxy="${NO_PROXY}"
    export LC_ALL="${LC_ALL:-C.UTF-8}"
    export LANG="${LANG:-C.UTF-8}"
    export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
    export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
    export SGLANG_FAILOVER_EVENTS_FILE="${FAILOVER_EVENTS_FILE:-}"
    # The failed DP replica may own the bootstrap TCPStore. Disable NCCL
    # heartbeat monitoring so surviving replicas do not get stuck on repeated
    # Broken pipe checks against that dead store during failover recovery.
    export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
    export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"

    local local_health_url="http://127.0.0.1:${server_port}/health"
    local external_server_url="http://${node_name}:${server_port}"
    local head_server_url="http://${head_node}:${server_port}"

    local effective_hicache_extra_config="${HICACHE_EXTRA_CONFIG}"
    if [ "${ENABLE_PEER_REPLICATION}" -eq 1 ]; then
        effective_hicache_extra_config="$(
            python3 <<'PY'
import json
import os

payload = {}
raw_user_cfg = os.environ.get("HICACHE_EXTRA_CONFIG", "").strip()
if raw_user_cfg:
    payload.update(json.loads(raw_user_cfg))

payload["peer_port"] = int(os.environ["PEER_PORT_BASE"])
payload["peer_buffer_size_gb"] = float(os.environ["PEER_BUFFER_SIZE_GB"])
payload["peer_targets"] = json.loads(os.environ["PEER_TARGETS_JSON"])

print(json.dumps(payload, separators=(",", ":")))
PY
        )"
    fi

    local py_bin="python"
    if [ -x "${VENV_DIR}/bin/python" ]; then
        py_bin="${VENV_DIR}/bin/python"
    fi

    local -a launch_cmd=(
        "${py_bin}"
        -m
        sglang.launch_server
        --model-path
        "${model_path}"
        --dp-size
        "${dp_size}"
        --host
        0.0.0.0
        --port
        "${server_port}"
        --nnodes
        "${nnodes}"
        --node-rank
        "${node_rank}"
        --dist-init-addr
        "${dist_init_addr}"
        --tp
        "${tp_size}"
        --pp-size
        "${pp_size}"
        --disable-overlap-schedule
        --attention-backend
        triton
        --sampling-backend
        pytorch
        --enable-metrics
        --tool-call-parser
        "${TOOL_CALL_PARSER}"
    )

    if [ "${ENABLE_CACHE_REPORT}" -eq 1 ]; then
        launch_cmd+=(--enable-cache-report)
    fi

    if [ "${ENABLE_HICACHE}" -eq 1 ]; then
        launch_cmd+=(--enable-hierarchical-cache)
        if [ "${HICACHE_SIZE}" -gt 0 ]; then
            launch_cmd+=(--hicache-size "${HICACHE_SIZE}")
        else
            launch_cmd+=(--hicache-ratio "${HICACHE_RATIO}")
        fi
    fi

    if [ -n "${QUANTIZATION}" ]; then
        launch_cmd+=(--quantization "${QUANTIZATION}")
    fi

    if [ "${ENABLE_PEER_REPLICATION}" -eq 1 ]; then
        launch_cmd+=(--hicache-storage-backend peer)
    fi

    if [ -n "${PREFETCH_POLICY}" ]; then
        launch_cmd+=(--hicache-storage-prefetch-policy "${PREFETCH_POLICY}")
    fi

    if [ -n "${effective_hicache_extra_config}" ]; then
        launch_cmd+=(--hicache-storage-backend-extra-config "${effective_hicache_extra_config}")
    fi

    printf '%q ' "${launch_cmd[@]}" > "${node_log_dir}/launch_command.txt"
    printf '\n' >> "${node_log_dir}/launch_command.txt"

    "${launch_cmd[@]}" >> "${node_log_dir}/server.log" 2>&1 &
    local server_pid=$!
    echo "${server_pid}" > "${node_log_dir}/server_main.pid"
    export SERVER_PID="${server_pid}"

    if ! wait_http_healthy "${local_health_url}" "${server_pid}" "${wait_timeout}"; then
        if kill -0 "${server_pid}" >/dev/null 2>&1; then
            kill "${server_pid}" 2>/dev/null || true
        fi
        wait "${server_pid}" 2>/dev/null || true
        echo "[node ${node_id}] server failed to become healthy at ${external_server_url}" >&2
        exit 1
    fi

    printf '%s\n' "${external_server_url}" > "${node_log_dir}/server_url.txt"
    printf '%s\n' "${node_name}" > "${node_log_dir}/server_node.txt"

    python3 - "${node_log_dir}/node_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

node_rank = int(os.environ["NODE_RANK"])
nnodes = int(os.environ["NNODES"])
dp_size = int(os.environ["DP_SIZE"])
pp_size = int(os.environ["PP_SIZE"])
tp_size = int(os.environ["TP_SIZE"])
server_port = int(os.environ["SERVER_PORT"])
peer_port_base = int(os.environ["PEER_PORT_BASE"])
enable_peer_replication = os.environ["ENABLE_PEER_REPLICATION"] == "1"
node_name = os.environ["NODE_NAME"]
node_url = f"http://{node_name}:{server_port}"
head_server_url = f"http://{os.environ['HEAD_NODE']}:{server_port}"
server_pid = os.environ.get("SERVER_PID", "")

pp_size_per_node = max(pp_size // nnodes, 1)
nnodes_per_pp_rank = max(nnodes // pp_size, 1)
pp_rank_range = range(
    pp_size_per_node * (node_rank // nnodes_per_pp_rank),
    pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
)
nnodes_per_tp_group = nnodes_per_pp_rank
tp_size_per_node = tp_size // nnodes_per_tp_group
tp_rank_range = range(
    tp_size_per_node * (node_rank % nnodes_per_tp_group),
    tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
)

stages = []
for dp_rank in range(dp_size):
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            shard_rank = ((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank
            kill_pattern = f"sglang::scheduler_DP{dp_rank}"
            if pp_size > 1:
                kill_pattern += f"_PP{pp_rank}"
            if tp_size > 1:
                kill_pattern += f"_TP{tp_rank}"
            stage_key = f"{dp_rank}:{pp_rank}:{tp_rank}"
            stage = {
                "stage_key": stage_key,
                "dp_rank": dp_rank,
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "node_rank": node_rank,
                "node": node_name,
                "node_url": node_url,
                "head_server_url": head_server_url,
                "kill_pattern": kill_pattern,
                "peer_listen_port": peer_port_base + shard_rank,
                "peer_target_stage_key": (
                    f"{(dp_rank + 1) % dp_size}:{pp_rank}:{tp_rank}"
                    if enable_peer_replication and dp_size > 1
                    else ""
                ),
            }
            stages.append(stage)

payload = {
    "node_rank": node_rank,
    "node": node_name,
    "node_url": node_url,
    "head_server_url": head_server_url,
    "server_port": server_port,
    "server_main_pid": int(server_pid) if server_pid else None,
    "runtime_failover_events_file": os.environ.get("FAILOVER_EVENTS_FILE", ""),
    "stages": stages,
}

manifest_path = Path(sys.argv[1])
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

    wait "${server_pid}"
}

if [[ "${1:-}" == "__run_node_server" ]]; then
    export SERVER_PID=""
    run_node_server
    exit $?
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DP_SIZE="${DP_SIZE:-${DEFAULT_DP_SIZE}}"
PP_SIZE="${PP_SIZE:-${DEFAULT_PP_SIZE}}"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-${DEFAULT_SERVER_PORT_BASE}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${DEFAULT_TOOL_CALL_PARSER}}"
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-${DEFAULT_ENABLE_CACHE_REPORT}}"
ENABLE_HICACHE="${ENABLE_HICACHE:-${DEFAULT_ENABLE_HICACHE}}"
HICACHE_SIZE="${HICACHE_SIZE:-${DEFAULT_HICACHE_SIZE}}"
HICACHE_RATIO="${HICACHE_RATIO:-${DEFAULT_HICACHE_RATIO}}"
ENABLE_PEER_REPLICATION="${ENABLE_PEER_REPLICATION:-${DEFAULT_ENABLE_PEER_REPLICATION}}"
PEER_PORT_BASE="${PEER_PORT_BASE:-${DEFAULT_PEER_PORT_BASE}}"
PEER_BUFFER_SIZE_GB="${PEER_BUFFER_SIZE_GB:-${DEFAULT_PEER_BUFFER_SIZE_GB}}"
PREFETCH_POLICY="${PREFETCH_POLICY:-${DEFAULT_PREFETCH_POLICY}}"
HICACHE_EXTRA_CONFIG="${HICACHE_EXTRA_CONFIG:-${DEFAULT_HICACHE_EXTRA_CONFIG}}"
QUANTIZATION="${QUANTIZATION:-${DEFAULT_QUANTIZATION}}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-${DEFAULT_WAIT_TIMEOUT}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dp-size|--num-workers)
            DP_SIZE="$2"
            shift 2
            ;;
        --pp-size)
            PP_SIZE="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --server-port-base|--worker-base-port)
            SERVER_PORT_BASE="$2"
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
        --prefetch-policy)
            PREFETCH_POLICY="$2"
            shift 2
            ;;
        --hicache-extra-config)
            HICACHE_EXTRA_CONFIG="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

_find_repo_root() {
    local current="$1"
    while [ "${current}" != "/" ]; do
        if [ -d "${current}/benchmark/multi_agent" ]; then
            echo "${current}"
            return 0
        fi
        current="$(dirname "${current}")"
    done
    return 1
}

if [ -n "${SGLANG_REPO_ROOT:-}" ] && [ -d "${SGLANG_REPO_ROOT}/benchmark/multi_agent" ]; then
    REPO_ROOT="${SGLANG_REPO_ROOT}"
elif found_root=$(_find_repo_root "${SLURM_SUBMIT_DIR:-$(pwd)}"); then
    REPO_ROOT="${found_root}"
elif found_root=$(_find_repo_root "$(pwd)"); then
    REPO_ROOT="${found_root}"
else
    echo "ERROR: Cannot locate repo root." >&2
    exit 1
fi

MULTI_AGENT_DIR="${REPO_ROOT}/benchmark/multi_agent"
SCRIPT_PATH="${MULTI_AGENT_DIR}/slurm/${SCRIPT_BASENAME}"
LOG_DIR="${MULTI_AGENT_DIR}/logs/logs_qwen3_8b_gpuhome"
VENV_DIR="${REPO_ROOT}/.venv"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Cannot locate shared script path: ${SCRIPT_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
rm -f \
    "${LOG_DIR}/server_url.txt" \
    "${LOG_DIR}/stage_manifest.json" \
    "${LOG_DIR}/runtime_failover_events.jsonl"
for node_dir in "${LOG_DIR}"/node_*; do
    [ -d "${node_dir}" ] || continue
    rm -f \
        "${node_dir}/server.log" \
        "${node_dir}/server_url.txt" \
        "${node_dir}/server_node.txt" \
        "${node_dir}/node_manifest.json" \
        "${node_dir}/server_main.pid" \
        "${node_dir}/launch_command.txt"
done

if [ ! -d "${REPO_ROOT}/python" ]; then
    echo "ERROR: Missing ${REPO_ROOT}/python" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

TOTAL_NODES="${SLURM_JOB_NUM_NODES:-1}"
mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SERVER_PORT="${SERVER_PORT_BASE}"
DIST_INIT_PORT=$((SERVER_PORT_BASE + DEFAULT_DIST_INIT_PORT_OFFSET))
HEAD_NODE="${NODE_LIST[0]}"
HEAD_SERVER_URL="http://${HEAD_NODE}:${SERVER_PORT}"
DIST_INIT_ADDR="${HEAD_NODE}:${DIST_INIT_PORT}"
FAILOVER_EVENTS_FILE="${LOG_DIR}/runtime_failover_events.jsonl"
PEER_TARGETS_JSON="$(
    python3 - "${TOTAL_NODES}" "${DP_SIZE}" "${PP_SIZE}" "${TP_SIZE}" "${PEER_PORT_BASE}" "${NODE_LIST[@]}" <<'PY'
import json
import sys

nnodes = int(sys.argv[1])
dp_size = int(sys.argv[2])
pp_size = int(sys.argv[3])
tp_size = int(sys.argv[4])
peer_port_base = int(sys.argv[5])
node_list = sys.argv[6:]

pp_size_per_node = max(pp_size // nnodes, 1)
nnodes_per_pp_rank = max(nnodes // pp_size, 1)
nnodes_per_tp_group = nnodes_per_pp_rank
tp_size_per_node = tp_size // nnodes_per_tp_group

payload = {}
for node_rank, node_name in enumerate(node_list):
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )
    for dp_rank in range(dp_size):
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                stage_key = f"{dp_rank}:{pp_rank}:{tp_rank}"
                payload[stage_key] = f"{node_name}:{peer_port_base}"

print(json.dumps(payload, separators=(",", ":")))
PY
)"

echo "Deploying internal DP+PP service for Qwen3-8B on GPUHome"
echo "  model path:        ${MODEL_PATH}"
echo "  dp size:           ${DP_SIZE}"
echo "  pp size:           ${PP_SIZE}"
echo "  tp size:           ${TP_SIZE}"
echo "  nnodes:            ${TOTAL_NODES}"
echo "  server port:       ${SERVER_PORT}"
echo "  dist init addr:    ${DIST_INIT_ADDR}"
echo "  peer port base:    ${PEER_PORT_BASE}"
echo "  peer replication:  ${ENABLE_PEER_REPLICATION}"
echo "  head server url:   ${HEAD_SERVER_URL}"
echo "  nodes:             ${NODE_LIST[*]}"
echo "  log dir:           ${LOG_DIR}"

SRUN_PIDS=()
NODE_MANIFESTS=()
for node_rank in $(seq 0 $((TOTAL_NODES - 1))); do
    node_name="${NODE_LIST[$node_rank]}"
    node_log_dir="${LOG_DIR}/node_${node_rank}"
    mkdir -p "${node_log_dir}"

    echo "Starting node_rank=${node_rank} on ${node_name}"

    srun -w "${node_name}" --nodes=1 --ntasks=1 --kill-on-bad-exit=0 \
        env \
            SCRIPT_PATH="${SCRIPT_PATH}" \
            REPO_ROOT="${REPO_ROOT}" \
            VENV_DIR="${VENV_DIR}" \
            NODE_RANK="${node_rank}" \
            NODE_NAME="${node_name}" \
            NODE_LOG_DIR="${node_log_dir}" \
            MODEL_PATH="${MODEL_PATH}" \
            DP_SIZE="${DP_SIZE}" \
            PP_SIZE="${PP_SIZE}" \
            TP_SIZE="${TP_SIZE}" \
            NNODES="${TOTAL_NODES}" \
            SERVER_PORT="${SERVER_PORT}" \
            DIST_INIT_ADDR="${DIST_INIT_ADDR}" \
            HEAD_NODE="${HEAD_NODE}" \
            TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
            ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT}" \
            ENABLE_HICACHE="${ENABLE_HICACHE}" \
            HICACHE_SIZE="${HICACHE_SIZE}" \
            HICACHE_RATIO="${HICACHE_RATIO}" \
            ENABLE_PEER_REPLICATION="${ENABLE_PEER_REPLICATION}" \
            PEER_PORT_BASE="${PEER_PORT_BASE}" \
            PEER_BUFFER_SIZE_GB="${PEER_BUFFER_SIZE_GB}" \
            PEER_TARGETS_JSON="${PEER_TARGETS_JSON}" \
            PREFETCH_POLICY="${PREFETCH_POLICY}" \
            HICACHE_EXTRA_CONFIG="${HICACHE_EXTRA_CONFIG}" \
            QUANTIZATION="${QUANTIZATION}" \
            FAILOVER_EVENTS_FILE="${FAILOVER_EVENTS_FILE}" \
            WAIT_TIMEOUT="${WAIT_TIMEOUT}" \
        bash -lc 'bash "${SCRIPT_PATH}" __run_node_server' &

    SRUN_PIDS+=($!)
    NODE_MANIFESTS+=("${node_log_dir}/node_manifest.json")
done

echo "Waiting for nodes to become healthy (timeout=${WAIT_TIMEOUT}s)..."
WAIT_START=$(date +%s)
FAILED_NODES=()

for node_rank in $(seq 0 $((TOTAL_NODES - 1))); do
    node_name="${NODE_LIST[$node_rank]}"
    srun_pid="${SRUN_PIDS[$node_rank]}"
    node_log_dir="${LOG_DIR}/node_${node_rank}"
    server_url_file="${node_log_dir}/server_url.txt"
    manifest_file="${node_log_dir}/node_manifest.json"

    while true; do
        elapsed=$(( $(date +%s) - WAIT_START ))
        if [ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]; then
            echo "ERROR: Timeout waiting for ${node_name} to become healthy." >&2
            FAILED_NODES+=("${node_name}")
            break
        fi

        if [ -s "${server_url_file}" ] && [ -s "${manifest_file}" ]; then
            echo "  Node node_rank=${node_rank} ready: $(<"${server_url_file}")"
            break
        fi

        if ! kill -0 "${srun_pid}" >/dev/null 2>&1; then
            set +e
            wait "${srun_pid}"
            srun_rc=$?
            set -e
            echo "ERROR: node_rank=${node_rank} on ${node_name} exited early (rc=${srun_rc})." >&2
            if [ -f "${node_log_dir}/server.log" ]; then
                echo "----- tail: ${node_log_dir}/server.log -----" >&2
                tail -n 80 "${node_log_dir}/server.log" >&2 || true
                echo "--------------------------------------------" >&2
            fi
            FAILED_NODES+=("${node_name}")
            break
        fi

        sleep 5
    done
done

if [ "${#FAILED_NODES[@]}" -gt 0 ]; then
    echo "ERROR: Failed nodes: ${FAILED_NODES[*]}" >&2
    exit 1
fi

printf '%s\n' "${HEAD_SERVER_URL}" > "${LOG_DIR}/server_url.txt"

python3 - "${LOG_DIR}/stage_manifest.json" "${SLURM_JOB_ID}" "${MODEL_PATH}" "${DP_SIZE}" "${PP_SIZE}" "${TP_SIZE}" "${TOTAL_NODES}" "${SERVER_PORT}" "${DIST_INIT_ADDR}" "${PEER_PORT_BASE}" "${HEAD_SERVER_URL}" "${FAILOVER_EVENTS_FILE}" "${NODE_MANIFESTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
slurm_job_id = sys.argv[2]
model_path = sys.argv[3]
dp_size = int(sys.argv[4])
pp_size = int(sys.argv[5])
tp_size = int(sys.argv[6])
nnodes = int(sys.argv[7])
server_port = int(sys.argv[8])
dist_init_addr = sys.argv[9]
peer_port_base = int(sys.argv[10])
head_server_url = sys.argv[11]
runtime_failover_events_file = sys.argv[12]
node_manifest_paths = [Path(p) for p in sys.argv[13:]]

servers = []
stages = []
for manifest_path in node_manifest_paths:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    servers.append(payload)
    stages.extend(payload.get("stages", []))

servers.sort(key=lambda item: item["node_rank"])
stages.sort(key=lambda item: (item["dp_rank"], item["pp_rank"], item["tp_rank"]))
peer_targets = {
    stage["stage_key"]: f"{stage['node']}:{peer_port_base}"
    for stage in stages
}

combined = {
    "slurm_job_id": slurm_job_id,
    "model_path": model_path,
    "dp_size": dp_size,
    "pp_size": pp_size,
    "tp_size": tp_size,
    "nnodes": nnodes,
    "server_port": server_port,
    "dist_init_addr": dist_init_addr,
    "peer_port_base": peer_port_base,
    "server_url": head_server_url,
    "head_server_url": head_server_url,
    "runtime_failover_events_file": runtime_failover_events_file,
    "peer_targets": peer_targets,
    "servers": servers,
    "stages": stages,
}
output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
PY

echo "Head server URL:"
sed 's/^/  /' "${LOG_DIR}/server_url.txt"
echo "Stage manifest: ${LOG_DIR}/stage_manifest.json"
echo "SLURM job ${SLURM_JOB_ID} is ready and will stay alive until cancelled."

set +e
wait "${SRUN_PIDS[@]}"
wait_rc=$?
set -e

if [ "${wait_rc}" -ne 0 ]; then
    echo "ERROR: One or more node launch steps exited unexpectedly (rc=${wait_rc})." >&2
    exit "${wait_rc}"
fi
