#!/bin/bash
# SGLang deploy: DP=2, PP=2 (e.g. Qwen3-32B on 4 GPUs). Client experiments live in
# benchmark/multi_agent/applications/crewai/ (Python). Submit this job for GPU nodes only.
#
# Script Overview:
# Deploy Qwen3-32B on one node with four GPUs: DP=2 (two logical replicas) and PP=2
# (two pipeline stages per replica). Each of the four scheduler processes uses one GPU.
# Optional --enable-peer-replication maps to --enable-stage-kv-replica (cross-DP KV sync).
#
# Flow:
#   1. Parse args; locate repo root and log directory.
#   2. For each node, start run_node_server via srun (local model/venv cache,
#      sglang.launch_server with up to 3 retries, write node_manifest.json).
#   3. Wait until all nodes are healthy (server_url.txt + node_manifest.json).
#   4. Merge manifests into stage_manifest.json.
#   5. Write head URL to server_url.txt for clients and fault-injection scripts.
#   6. Block until all srun steps exit (keeps the SLURM allocation alive).
#
# Outputs under LOG_DIR (benchmark/multi_agent/logs/qwen3_32b_a100/):
#   server_url.txt               — Head HTTP URL
#   stage_manifest.json          — Global stage metadata (nodes, kill_pattern, ports)
#   runtime_failover_events.jsonl — SGLang runtime failover event stream

# ────────────────────────────────────────────────────────────
# SLURM resource request
# ────────────────────────────────────────────────────────────
#SBATCH --job-name=sglang_dp2_pp2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/comp/%u/sglang/benchmark/multi_agent/logs/qwen3_32b_a100/run_server_%j.out
#SBATCH --error=/home/comp/%u/sglang/benchmark/multi_agent/logs/qwen3_32b_a100/run_server_%j.err

set -euo pipefail

# Absolute script path (symlink-resolved); passed to srun children for __run_node_server
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# ────────────────────────────────────────────────────────────
# GCC toolchain for CUDA/NVCC
# ────────────────────────────────────────────────────────────
# Default cluster GCC may be too old; prefer GCC 11.2.0 when present
module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
if [ -d "${GCC_ROOT}/bin" ]; then
    export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
    export CXX="${GCC_ROOT}/bin/g++"
    export CC="${GCC_ROOT}/bin/gcc"
    export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ────────────────────────────────────────────────────────────
# Default parameter values
# ────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH="Qwen/Qwen3-32B"
DEFAULT_DP_SIZE=2
DEFAULT_PP_SIZE=2
DEFAULT_TP_SIZE=1
DEFAULT_SERVER_PORT_BASE=34000
DEFAULT_DIST_INIT_PORT_OFFSET=100
DEFAULT_TOOL_CALL_PARSER="qwen"
DEFAULT_ENABLE_CACHE_REPORT=1
DEFAULT_ENABLE_HICACHE=1
# Qwen3-32B PP2+DP2 on A100 needs host HiCache larger than the ~35 GB device KV pool.
DEFAULT_HICACHE_SIZE=40
MIN_HICACHE_SIZE=40
DEFAULT_HICACHE_RATIO=2.0
DEFAULT_ENABLE_PEER_REPLICATION=0
DEFAULT_PEER_PORT_BASE=35000
DEFAULT_HICACHE_EXTRA_CONFIG=""
DEFAULT_QUANTIZATION=""
DEFAULT_WAIT_TIMEOUT=1800
DEFAULT_MODEL_LOCAL_ROOT="/tmp/${USER}/sglang-model-cache"
DEFAULT_VENV_LOCAL_ROOT="/tmp/${USER}/sglang-runtime-cache"

# ────────────────────────────────────────────────────────────
# Usage
# ────────────────────────────────────────────────────────────

show_help() {
    cat <<'EOF'
Usage:
  sbatch [SBATCH_OPTIONS] deploy_sglang.slurm [SCRIPT_OPTIONS]

Script options:
  --model-path PATH        Model path (default: Qwen/Qwen3-32B)
  --dp-size N              Internal data parallel size (default: 2)
  --pp-size N              Pipeline parallel size per DP replica (default: 2)
  --tp-size N              Tensor parallel size per stage (default: 1)
  --server-port-base P     HTTP port used on every node (default: 34000)
  --tool-call-parser P     Tool call parser (default: qwen)
  --enable-cache-report    Return cached tokens in OpenAI usage
  --disable-cache-report   Disable cached token reporting
  --enable-hicache         Enable hierarchical KV cache (default: enabled)
  --disable-hicache        Disable hierarchical KV cache
  --hicache-size N         HiCache size per DP replica in GB (default: 40)
  --hicache-ratio R        HiCache ratio when hicache-size<=0 (default: 2.0)
  --enable-peer-replication
                           Enable cross-DP stage KV replica sync (--enable-stage-kv-replica)
  --peer-port-base N       Base for stage manifest listen ports and dist_init upper bound (default: 35000)
  --hicache-extra-config J JSON string for hicache storage backend extra config
  --quantization Q         Quantization method (default: none)
  --wait-timeout SECS      Timeout waiting for all nodes to become healthy
  --model-local-root DIR   Local node directory used to cache the model
  --venv-local-root DIR    Local node directory used to cache the root venv
  -h, --help               Show this help

Outputs:
  benchmark/multi_agent/logs/qwen3_32b_a100/server_url.txt
  benchmark/multi_agent/logs/qwen3_32b_a100/stage_manifest.json
  benchmark/multi_agent/logs/qwen3_32b_a100/runtime_failover_events.jsonl

Notes:
  Default SBATCH is one node and four A100 GPUs. sbatch --nodes=N can override node count.
  Only node_rank=0 exposes the head HTTP URL used by clients.
EOF
}

# ────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────

# Poll URL until HTTP 200, child exit, or timeout
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

reset_log_file() {
    mkdir -p "$(dirname "$1")" && : > "$1"
}

choose_free_port_on_node() {
    local node_name="$1"
    local start_port="$2"
    local end_port="$3"

    srun -w "${node_name}" --nodes=1 --ntasks=1 --kill-on-bad-exit=0 \
        python3 - "${start_port}" "${end_port}" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
end_port = int(sys.argv[2])

def can_bind(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sock.close()
        return False
    sock.close()
    return True

for port in range(start_port, end_port + 1):
    if can_bind(port):
        print(port)
        sys.exit(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

# ────────────────────────────────────────────────────────────
# Per-node server (invoked via srun: bash -lc '${SCRIPT_PATH} __run_node_server')
# ────────────────────────────────────────────────────────────
# Local venv/model cache, warm imports, launch_server (up to 3 retries), write
# server_url.txt and node_manifest.json, then wait on the sglang process.
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
    reset_log_file "${node_log_dir}/server.log"
    rm -f \
        "${node_log_dir}/server_url.txt" \
        "${node_log_dir}/server_node.txt" \
        "${node_log_dir}/node_manifest.json" \
        "${node_log_dir}/server_main.pid" \
        "${node_log_dir}/launch_command.txt"

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

    # Optional JSON for storage backends (file/mooncake/dynamic/...). Peer KV backup
    # across DP replicas uses --enable-stage-kv-replica, not hicache-storage-backend peer.
    local effective_hicache_extra_config="${HICACHE_EXTRA_CONFIG}"

    local source_venv_dir="${VENV_DIR}"

    resolve_base_python_root() {
        local requested_venv_dir="$1"
        local pyvenv_cfg="${requested_venv_dir}/pyvenv.cfg"
        if [ ! -f "${pyvenv_cfg}" ]; then
            return 1
        fi

        local home_bin_dir=""
        home_bin_dir="$(awk -F ' = ' '$1=="home"{print $2}' "${pyvenv_cfg}" | tail -n 1)"
        if [ -z "${home_bin_dir}" ]; then
            return 1
        fi

        dirname "${home_bin_dir}"
    }

    copy_tree_local() {
        local src_dir="$1"
        local dst_dir="$2"
        mkdir -p "${dst_dir}"
        (
            cd "${src_dir}" &&
            tar -cf - .
        ) | (
            cd "${dst_dir}" &&
            tar -xf -
        )
    }

    patch_local_pyvenv_cfg() {
        local cfg_path="$1"
        local local_home_bin_dir="$2"
        python3 - "${cfg_path}" "${local_home_bin_dir}" <<'PY'
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
local_home_bin_dir = sys.argv[2]
lines = cfg_path.read_text(encoding="utf-8").splitlines()
updated = []
replaced = False
for line in lines:
    if line.startswith("home = "):
        updated.append(f"home = {local_home_bin_dir}")
        replaced = True
    else:
        updated.append(line)
if not replaced:
    updated.append(f"home = {local_home_bin_dir}")
cfg_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
    }

    prepare_local_venv_dir() {
        local requested_venv_dir="$1"
        if [ -z "${VENV_LOCAL_ROOT:-}" ]; then
            printf '%s\n' "${requested_venv_dir}"
            return 0
        fi

        local source_base_python_root=""
        source_base_python_root="$(resolve_base_python_root "${requested_venv_dir}")" || return 1

        local runtime_key="sglang-root-venv-py312"
        local local_runtime_root="${VENV_LOCAL_ROOT}/${runtime_key}"
        local local_base_python_dir="${local_runtime_root}/python-base"
        local local_venv_dir="${local_runtime_root}/venv"
        local ready_marker="${local_runtime_root}/.runtime_cache_ready"
        local venv_lock_file="/tmp/${USER}/.${runtime_key}.lock"

        mkdir -p "${VENV_LOCAL_ROOT}" "$(dirname "${venv_lock_file}")"
        exec 8>"${venv_lock_file}"
        flock 8

        local cache_ok=0
        if [ -x "${local_venv_dir}/bin/python" ] \
            && [ -f "${local_venv_dir}/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" ] \
            && [ -x "${local_base_python_dir}/bin/python3.12" ] \
            && [ -f "${ready_marker}" ]; then
            cache_ok=1
        fi

        if [ "${cache_ok}" -ne 1 ]; then
            echo "[node ${node_rank}] Populating local runtime cache at ${local_runtime_root}." >> "${node_log_dir}/server.log"
            rm -rf "${local_runtime_root}"
            mkdir -p "${local_base_python_dir}" "${local_venv_dir}"

            echo "[node ${node_rank}] Copying base Python runtime from ${source_base_python_root}." >> "${node_log_dir}/server.log"
            copy_tree_local "${source_base_python_root}" "${local_base_python_dir}" >> "${node_log_dir}/server.log" 2>&1

            echo "[node ${node_rank}] Copying root venv from ${requested_venv_dir}." >> "${node_log_dir}/server.log"
            copy_tree_local "${requested_venv_dir}" "${local_venv_dir}" >> "${node_log_dir}/server.log" 2>&1

            patch_local_pyvenv_cfg "${local_venv_dir}/pyvenv.cfg" "${local_base_python_dir}/bin"
            rm -f \
                "${local_venv_dir}/bin/python" \
                "${local_venv_dir}/bin/python3" \
                "${local_venv_dir}/bin/python3.12"
            ln -s "${local_base_python_dir}/bin/python3.12" "${local_venv_dir}/bin/python"
            ln -s python "${local_venv_dir}/bin/python3"
            ln -s python "${local_venv_dir}/bin/python3.12"

            date -Iseconds > "${ready_marker}"
            echo "[node ${node_rank}] Local runtime cache ready: ${local_venv_dir}" >> "${node_log_dir}/server.log"
        else
            echo "[node ${node_rank}] Reusing local runtime cache at ${local_venv_dir}." >> "${node_log_dir}/server.log"
        fi

        flock -u 8
        printf '%s\n' "${local_venv_dir}"
    }

    VENV_DIR="$(prepare_local_venv_dir "${source_venv_dir}")"
    export VENV_DIR
    export RESOLVED_VENV_DIR="${VENV_DIR}"

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
        source "${VENV_DIR}/Scripts/activate"
    fi

    local py_bin="python"
    if [ -x "${VENV_DIR}/bin/python" ]; then
        py_bin="${VENV_DIR}/bin/python"
    fi

    local site_packages_dir="${VENV_DIR}/lib/python3.12/site-packages"
    local resolved_model_path="${model_path}"

    sanitize_model_cache_key() {
        local requested_model_path="$1"
        "${py_bin}" - "${requested_model_path}" <<'PY'
import os
import re
import sys

model_path = sys.argv[1].strip()
if os.path.isdir(model_path):
    cache_key = os.path.basename(os.path.normpath(model_path)) or "model"
else:
    cache_key = model_path

cache_key = re.sub(r"[^A-Za-z0-9._-]+", "__", cache_key).strip("._")
print(cache_key or "model")
PY
    }

    resolve_model_source_dir() {
        local requested_model_path="$1"
        if [ -d "${requested_model_path}" ]; then
            readlink -f "${requested_model_path}"
            return 0
        fi
        "${py_bin}" - "${requested_model_path}" <<'PY'
import sys
from huggingface_hub import snapshot_download

requested_model_path = sys.argv[1]
try:
    print(snapshot_download(repo_id=requested_model_path, local_files_only=True))
except Exception:
    sys.exit(1)
PY
    }

    download_model_to_local_dir() {
        local requested_model_path="$1"
        local target_model_dir="$2"
        "${py_bin}" - "${requested_model_path}" "${target_model_dir}" <<'PY'
import sys
from huggingface_hub import snapshot_download

requested_model_path = sys.argv[1]
target_model_dir = sys.argv[2]
print(
    snapshot_download(
        repo_id=requested_model_path,
        local_dir=target_model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
)
PY
    }

    prepare_local_model_path() {
        local requested_model_path="$1"
        if [ -z "${MODEL_LOCAL_ROOT:-}" ]; then
            printf '%s\n' "${requested_model_path}"
            return 0
        fi

        local source_model_dir=""

        local cache_key=""
        cache_key="$(sanitize_model_cache_key "${requested_model_path}")" || return 1

        local local_model_dir="${MODEL_LOCAL_ROOT}/${cache_key}"
        local ready_marker="${local_model_dir}/.model_cache_ready"
        local model_lock_file="/tmp/${USER}/.${cache_key}.model-cache.lock"

        mkdir -p "${MODEL_LOCAL_ROOT}" "$(dirname "${model_lock_file}")"
        exec 9>"${model_lock_file}"
        flock 9

        local has_weight_shards=0
        if compgen -G "${local_model_dir}/*.safetensors" >/dev/null || [ -f "${local_model_dir}/model.safetensors.index.json" ]; then
            has_weight_shards=1
        fi

        if [ ! -f "${ready_marker}" ] || [ ! -f "${local_model_dir}/config.json" ] || [ "${has_weight_shards}" -ne 1 ]; then
            rm -rf "${local_model_dir}"
            mkdir -p "${local_model_dir}"
            if source_model_dir="$(resolve_model_source_dir "${requested_model_path}")"; then
                echo "[node ${node_rank}] Populating local model cache at ${local_model_dir} from ${source_model_dir}." >> "${node_log_dir}/server.log"
                cp -aL "${source_model_dir}/." "${local_model_dir}/" >> "${node_log_dir}/server.log" 2>&1
            else
                echo "[node ${node_rank}] NFS cache miss for ${requested_model_path}; downloading to ${local_model_dir}." >> "${node_log_dir}/server.log"
                download_model_to_local_dir "${requested_model_path}" "${local_model_dir}" >> "${node_log_dir}/server.log" 2>&1
            fi
            date -Iseconds > "${ready_marker}"
            echo "[node ${node_rank}] Local model cache ready: ${local_model_dir}" >> "${node_log_dir}/server.log"
        else
            echo "[node ${node_rank}] Reusing local model cache at ${local_model_dir}." >> "${node_log_dir}/server.log"
        fi

        flock -u 9
        printf '%s\n' "${local_model_dir}"
    }

    resolved_model_path="$(prepare_local_model_path "${model_path}")"
    export RESOLVED_MODEL_PATH="${resolved_model_path}"
    echo "[node ${node_rank}] Using model path ${resolved_model_path}." >> "${node_log_dir}/server.log"

    local -a ld_library_dirs=(
        "${site_packages_dir}/torch/lib"
        "${site_packages_dir}/torchvision.libs"
        "${site_packages_dir}/triton/_C"
    )
    shopt -s nullglob
    local nvidia_lib_dir=""
    for nvidia_lib_dir in "${site_packages_dir}"/nvidia/*/lib; do
        if [ -d "${nvidia_lib_dir}" ]; then
            ld_library_dirs+=("${nvidia_lib_dir}")
        fi
    done
    shopt -u nullglob

    local extra_ld_library_path=""
    local lib_dir=""
    for lib_dir in "${ld_library_dirs[@]}"; do
        if [ ! -d "${lib_dir}" ]; then
            continue
        fi
        if [ -z "${extra_ld_library_path}" ]; then
            extra_ld_library_path="${lib_dir}"
        else
            extra_ld_library_path="${extra_ld_library_path}:${lib_dir}"
        fi
    done

    export PYTHONPATH="${site_packages_dir}:${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
    if [ -n "${extra_ld_library_path}" ]; then
        export LD_LIBRARY_PATH="${extra_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    local -a probe_paths=(
        "${site_packages_dir}"
        "${site_packages_dir}/torch"
        "${site_packages_dir}/torch/lib"
        "${site_packages_dir}/torchvision"
        "${site_packages_dir}/torchvision.libs"
        "${site_packages_dir}/timm/models"
        "${site_packages_dir}/pygments/lexers"
        "${site_packages_dir}/nvidia"
        "${site_packages_dir}/triton/_C"
    )
    local -a shared_lib_patterns=(
        "${site_packages_dir}/torch/lib/*.so*"
        "${site_packages_dir}/nvidia/*/lib/*.so*"
        "${site_packages_dir}/torchvision/*.so*"
        "${site_packages_dir}/torchvision.libs/*.so*"
        "${site_packages_dir}/triton/_C/*.so*"
    )

    echo "[node ${node_rank}] Warming Python import paths before launch." >> "${node_log_dir}/server.log"
    for path in "${probe_paths[@]}"; do
        if [ -e "${path}" ]; then
            touch "${path}" >/dev/null 2>&1 || true
            ls "${path}" >/dev/null 2>&1 || true
        fi
    done
    echo "[node ${node_rank}] Python import path warming finished." >> "${node_log_dir}/server.log"
    echo "[node ${node_rank}] Warming shared libraries before launch." >> "${node_log_dir}/server.log"
    shopt -s nullglob
    local lib_pattern=""
    local lib_path=""
    for lib_pattern in "${shared_lib_patterns[@]}"; do
        for lib_path in ${lib_pattern}; do
            dd if="${lib_path}" of=/dev/null bs=8M status=none || true
        done
    done
    shopt -u nullglob
    echo "[node ${node_rank}] Shared library warming finished." >> "${node_log_dir}/server.log"

    warm_errno5_paths() {
        local log_file="$1"
        local start_line="$2"
        python3 - "${log_file}" "${start_line}" <<'PY'
import re
import sys
from pathlib import Path

log_file = Path(sys.argv[1])
start_line = int(sys.argv[2])
pattern = re.compile(r"OSError: \[Errno 5\] Input/output error: '([^']+)'")

if not log_file.exists():
    sys.exit(0)

lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
seen = set()
for line in lines[max(start_line - 1, 0):]:
    match = pattern.search(line)
    if not match:
        continue
    raw_path = match.group(1)
    for candidate in (Path(raw_path), Path(raw_path).parent):
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        print(candidate_str)
PY
    }

    local -a launch_cmd=(
        "${py_bin}"
        -m
        sglang.launch_server
        --model-path
        "${resolved_model_path}"
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
        --disable-hicache-numa-detect
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
        launch_cmd+=(--enable-stage-kv-replica)
    fi

    if [ -n "${effective_hicache_extra_config}" ]; then
        launch_cmd+=(--hicache-storage-backend-extra-config "${effective_hicache_extra_config}")
    fi

    printf '%q ' "${launch_cmd[@]}" > "${node_log_dir}/launch_command.txt"
    printf '\n' >> "${node_log_dir}/launch_command.txt"

    local server_pid=""
    local launch_try=""
    local launch_attempt_start_line=""
    local launch_succeeded=0
    for launch_try in 1 2 3; do
        launch_attempt_start_line=$(( $(wc -l < "${node_log_dir}/server.log" 2>/dev/null || echo 0) + 1 ))
        echo "[node ${node_rank}] Starting launch_server attempt ${launch_try}/3." >> "${node_log_dir}/server.log"

        "${launch_cmd[@]}" >> "${node_log_dir}/server.log" 2>&1 &
        server_pid=$!
        echo "${server_pid}" > "${node_log_dir}/server_main.pid"
        export SERVER_PID="${server_pid}"

        if wait_http_healthy "${local_health_url}" "${server_pid}" "${wait_timeout}"; then
            launch_succeeded=1
            break
        fi

        if kill -0 "${server_pid}" >/dev/null 2>&1; then
            kill "${server_pid}" 2>/dev/null || true
        fi
        wait "${server_pid}" 2>/dev/null || true

        if sed -n "${launch_attempt_start_line},\$p" "${node_log_dir}/server.log" | grep -q "OSError: \\[Errno 5\\]"; then
            while IFS= read -r err_path; do
                [ -z "${err_path}" ] && continue
                echo "[node ${node_rank}] warming Errno 5 path: ${err_path}" >> "${node_log_dir}/server.log"
                if [ -e "${err_path}" ]; then
                    touch "${err_path}" >/dev/null 2>&1 || true
                    ls "${err_path}" >/dev/null 2>&1 || true
                fi
            done < <(warm_errno5_paths "${node_log_dir}/server.log" "${launch_attempt_start_line}")
            echo "[node ${node_rank}] launch_server hit transient Errno 5 on attempt ${launch_try}/3, retrying..." >> "${node_log_dir}/server.log"
            sleep 8
            continue
        fi

        echo "[node ${node_rank}] server failed to become healthy at ${external_server_url}" >&2
        exit 1
    done

    if [ "${launch_succeeded}" -ne 1 ]; then
        echo "[node ${node_rank}] server failed to become healthy at ${external_server_url}" >&2
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
    "venv_dir": os.environ.get("VENV_DIR", ""),
    "resolved_venv_dir": os.environ.get("RESOLVED_VENV_DIR", ""),
    "model_path": os.environ.get("MODEL_PATH", ""),
    "resolved_model_path": os.environ.get("RESOLVED_MODEL_PATH", ""),
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

# ────────────────────────────────────────────────────────────
# Child entry: srun invokes '${SCRIPT_PATH} __run_node_server'
# ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "__run_node_server" ]]; then
    export SERVER_PID=""
    run_node_server
    exit $?
fi

# ────────────────────────────────────────────────────────────
# Main: apply defaults and coordinate head node
# ────────────────────────────────────────────────────────────

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
HICACHE_EXTRA_CONFIG="${HICACHE_EXTRA_CONFIG:-${DEFAULT_HICACHE_EXTRA_CONFIG}}"
QUANTIZATION="${QUANTIZATION:-${DEFAULT_QUANTIZATION}}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-${DEFAULT_WAIT_TIMEOUT}}"
MODEL_LOCAL_ROOT="${MODEL_LOCAL_ROOT:-${DEFAULT_MODEL_LOCAL_ROOT}}"
VENV_LOCAL_ROOT="${VENV_LOCAL_ROOT:-${DEFAULT_VENV_LOCAL_ROOT}}"

if [ "${ENABLE_HICACHE}" -eq 1 ] && [ "${HICACHE_SIZE}" -gt 0 ] && [ "${HICACHE_SIZE}" -lt "${MIN_HICACHE_SIZE}" ]; then
    echo "WARNING: hicache-size=${HICACHE_SIZE} GB is too small for Qwen3-32B DP=2 PP=2 on A100; bumping to ${MIN_HICACHE_SIZE} GB." >&2
    HICACHE_SIZE="${MIN_HICACHE_SIZE}"
fi

# ────────────────────────────────────────────────────────────
# Command line parameter parsing
# ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)            MODEL_PATH="$2";            shift 2 ;;
        --dp-size)               DP_SIZE="$2";               shift 2 ;;
        --pp-size)               PP_SIZE="$2";               shift 2 ;;
        --tp-size)               TP_SIZE="$2";               shift 2 ;;
        --server-port-base)      SERVER_PORT_BASE="$2";      shift 2 ;;
        --tool-call-parser)      TOOL_CALL_PARSER="$2";      shift 2 ;;
        --enable-cache-report)   ENABLE_CACHE_REPORT=1;      shift   ;;
        --disable-cache-report)  ENABLE_CACHE_REPORT=0;      shift   ;;
        --enable-hicache)        ENABLE_HICACHE=1;           shift   ;;
        --disable-hicache)       ENABLE_HICACHE=0;           shift   ;;
        --hicache-size)          HICACHE_SIZE="$2";          shift 2 ;;
        --hicache-ratio)         HICACHE_RATIO="$2";         shift 2 ;;
        --enable-peer-replication) ENABLE_PEER_REPLICATION=1; shift   ;;
        --peer-port-base)        PEER_PORT_BASE="$2";        shift 2 ;;
        --hicache-extra-config)  HICACHE_EXTRA_CONFIG="$2";  shift 2 ;;
        --quantization)          QUANTIZATION="$2";          shift 2 ;;
        --wait-timeout)          WAIT_TIMEOUT="$2";          shift 2 ;;
        --model-local-root)      MODEL_LOCAL_ROOT="$2";      shift 2 ;;
        --venv-local-root)       VENV_LOCAL_ROOT="$2";       shift 2 ;;
        -h|--help)               show_help; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ────────────────────────────────────────────────────────────
# Repository root discovery
# ────────────────────────────────────────────────────────────

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
LOG_DIR="${MULTI_AGENT_DIR}/logs/qwen3_32b_a100"
VENV_DIR="${REPO_ROOT}/.venv"

# ────────────────────────────────────────────────────────────
# Clear stale outputs from previous runs
# ────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}"
rm -f \
    "${LOG_DIR}/server_url.txt" \
    "${LOG_DIR}/stage_manifest.json" \
    "${LOG_DIR}/runtime_failover_events.jsonl"
for node_dir in "${LOG_DIR}"/node_*; do
    [ -d "${node_dir}" ] || continue
    reset_log_file "${node_dir}/server.log"
    rm -f \
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

# ────────────────────────────────────────────────────────────
# Node list and networking
# ────────────────────────────────────────────────────────────

TOTAL_NODES="${SLURM_JOB_NUM_NODES:-1}"
mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SERVER_PORT="${SERVER_PORT_BASE}"
HEAD_NODE="${NODE_LIST[0]}"
HEAD_SERVER_URL="http://${HEAD_NODE}:${SERVER_PORT}"
DIST_INIT_PORT_START=$((SERVER_PORT_BASE + DEFAULT_DIST_INIT_PORT_OFFSET))
DIST_INIT_PORT_END=$((PEER_PORT_BASE - 1))
DIST_INIT_PORT="$(choose_free_port_on_node "${HEAD_NODE}" "${DIST_INIT_PORT_START}" "${DIST_INIT_PORT_END}")"
DIST_INIT_ADDR="${HEAD_NODE}:${DIST_INIT_PORT}"
FAILOVER_EVENTS_FILE="${LOG_DIR}/runtime_failover_events.jsonl"

# ────────────────────────────────────────────────────────────
# Deployment summary
# ────────────────────────────────────────────────────────────

echo "Deploying Qwen3-32B: DP+PP on ${TOTAL_NODES} node(s), peer replication=${ENABLE_PEER_REPLICATION}"
echo "  model path:        ${MODEL_PATH}"
echo "  dp size:           ${DP_SIZE}"
echo "  pp size:           ${PP_SIZE}"
echo "  tp size:           ${TP_SIZE}"
echo "  nnodes:            ${TOTAL_NODES}"
echo "  server port:       ${SERVER_PORT}"
echo "  dist init addr:    ${DIST_INIT_ADDR}"
echo "  peer port base:    ${PEER_PORT_BASE}"
echo "  peer replication:  ${ENABLE_PEER_REPLICATION}"
echo "  quantization:      ${QUANTIZATION:-none}"
echo "  model local root:  ${MODEL_LOCAL_ROOT}"
echo "  venv local root:   ${VENV_LOCAL_ROOT}"
echo "  head server url:   ${HEAD_SERVER_URL}"
echo "  nodes:             ${NODE_LIST[*]}"
echo "  log dir:           ${LOG_DIR}"

# ────────────────────────────────────────────────────────────
# Launch servers on all nodes (async srun)
# ────────────────────────────────────────────────────────────

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
            HICACHE_EXTRA_CONFIG="${HICACHE_EXTRA_CONFIG}" \
            QUANTIZATION="${QUANTIZATION}" \
            MODEL_LOCAL_ROOT="${MODEL_LOCAL_ROOT}" \
            VENV_LOCAL_ROOT="${VENV_LOCAL_ROOT}" \
            FAILOVER_EVENTS_FILE="${FAILOVER_EVENTS_FILE}" \
            WAIT_TIMEOUT="${WAIT_TIMEOUT}" \
        bash -lc '"${SCRIPT_PATH}" __run_node_server' &

    SRUN_PIDS+=($!)
    NODE_MANIFESTS+=("${node_log_dir}/node_manifest.json")
done

# ────────────────────────────────────────────────────────────
# Wait until all nodes are healthy
# ────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────
# Write head server_url.txt and merged stage_manifest.json
# ────────────────────────────────────────────────────────────

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
    stage["stage_key"]: f"{stage['node']}:{stage.get('peer_listen_port', peer_port_base)}"
    for stage in stages
}

combined = {
    "slurm_job_id": slurm_job_id,
    "resolved_venv_dir": servers[0].get("resolved_venv_dir", "") if servers else "",
    "model_path": model_path,
    "resolved_model_path": servers[0].get("resolved_model_path", "") if servers else "",
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

# ────────────────────────────────────────────────────────────
# Ready: keep job alive until cancelled
# ────────────────────────────────────────────────────────────

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
