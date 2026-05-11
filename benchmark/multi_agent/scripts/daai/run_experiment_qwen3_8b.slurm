#!/bin/bash
#SBATCH --job-name=crewai_exp_8b_daai_dp2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=hkbugpudgx01,hkbugpusrv08
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
    local start_dir="$1"
    local current_dir
    current_dir="$(cd "${start_dir}" 2>/dev/null && pwd)" || return 1

    for depth in 1 2 3 4 5 6 7 8; do
        if [ -f "${current_dir}/benchmark/multi_agent/src/main.py" ]; then
            printf '%s\n' "${current_dir}"
            return 0
        fi
        if [ "${current_dir}" = "/" ]; then
            break
        fi
        current_dir="$(dirname "${current_dir}")"
    done

    return 1
}

check_assigned_gpus_idle() {
    local max_used_mib="${GPU_IDLE_MAX_USED_MIB:-2048}"
    local devices_raw="${CUDA_VISIBLE_DEVICES:-}"
    local smi_output
    local token
    local match
    local idx
    local uuid
    local used_mib
    local total_mib
    local bad_gpu=0

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARN: nvidia-smi is unavailable; skipping GPU idle check."
        return 0
    fi

    if [ -z "${devices_raw}" ]; then
        echo "WARN: CUDA_VISIBLE_DEVICES is not set; skipping GPU idle check."
        return 0
    fi

    smi_output="$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits)" || {
        echo "ERROR: Failed to query GPU memory via nvidia-smi."
        return 1
    }

    echo "Checking assigned GPUs for pre-existing memory usage (threshold: ${max_used_mib} MiB)..."
    echo "CUDA_VISIBLE_DEVICES=${devices_raw}"

    IFS=',' read -r -a gpu_tokens <<< "${devices_raw}"
    for token in "${gpu_tokens[@]}"; do
        token="${token// /}"
        [ -n "${token}" ] || continue

        match="$(
            printf '%s\n' "${smi_output}" | awk -F',' -v token="${token}" '
                function trim(s) {
                    gsub(/^[ \t]+|[ \t]+$/, "", s)
                    return s
                }
                {
                    idx = trim($1)
                    uuid = trim($2)
                    used = trim($3)
                    total = trim($4)
                    if (idx == token || uuid == token) {
                        print idx "," uuid "," used "," total
                        exit
                    }
                }
            '
        )"

        if [ -z "${match}" ]; then
            echo "ERROR: Could not resolve allocated GPU '${token}' from nvidia-smi output."
            return 1
        fi

        IFS=',' read -r idx uuid used_mib total_mib <<< "${match}"
        echo "  GPU ${idx} (${uuid}): used=${used_mib} MiB / total=${total_mib} MiB"

        if [ "${used_mib}" -gt "${max_used_mib}" ]; then
            bad_gpu=1
        fi
    done

    if [ "${bad_gpu}" -ne 0 ]; then
        echo "ERROR: At least one allocated GPU already has more than ${max_used_mib} MiB in use."
        echo "ERROR: Refusing to start on a dirty GPU. Override the threshold with GPU_IDLE_MAX_USED_MIB if needed."
        echo "Active compute processes reported by nvidia-smi:"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true
        return 1
    fi
}

# sglang root path
REPO_ROOT=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$(find_repo_root "${SLURM_SUBMIT_DIR}" || true)"
fi
if [ -z "${REPO_ROOT}" ] && [ -n "${SGLANG_REPO_ROOT:-}" ]; then
    REPO_ROOT="$(find_repo_root "${SGLANG_REPO_ROOT}" || true)"
fi
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}" || true)"
fi
if [ -z "${REPO_ROOT}" ]; then
    echo "ERROR: Cannot locate sglang repo root from SLURM_SUBMIT_DIR, SGLANG_REPO_ROOT, or script path." >&2
    exit 1
fi
# src core path
SRC_ROOT="${REPO_ROOT}/benchmark/multi_agent/src"
# venv
VENV_DIR="${SGLANG_VENV:-${REPO_ROOT}/.venv}"
# log directory
LOG_DIR="${REPO_ROOT}/benchmark/multi_agent/logs"
mkdir -p "${LOG_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
export MULTI_AGENT_MAIN_LOG="${LOG_DIR}/main_${JOB_ID}.log"
export MULTI_AGENT_SERVER_LOG="${LOG_DIR}/server_${JOB_ID}.log"
export MULTI_AGENT_BACKUP_SERVER_LOG="${LOG_DIR}/backup_server_${JOB_ID}.log"
export MULTI_AGENT_INJECTOR_LOG="${LOG_DIR}/injector_${JOB_ID}.log"
export MULTI_AGENT_CLIENT_LOG="${LOG_DIR}/client_${JOB_ID}.log"
exec >> "${MULTI_AGENT_MAIN_LOG}" 2>&1

# GCC setup (for CUDA)
module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
if [ -d "${GCC_ROOT}/bin" ]; then
    export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
    export CXX="${GCC_ROOT}/bin/g++"
    export CC="${GCC_ROOT}/bin/gcc"
    export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Python path
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH:-}"
# no proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"
# UTF-8 encoding
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="utf-8"
export PYTHONUNBUFFERED=1

# debug logging or not
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"

# sglang root path
export SGLANG_REPO_ROOT="${REPO_ROOT}"

# Fail fast if Slurm handed us a GPU that is already heavily occupied.
check_assigned_gpus_idle

# execute main function
cd "${SRC_ROOT}"
exec "${VENV_DIR}/bin/python" -m main "$@" --log-dir "${LOG_DIR}"
