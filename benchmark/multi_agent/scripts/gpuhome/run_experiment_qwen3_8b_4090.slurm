#!/bin/bash
#SBATCH --job-name=remote_backup_test
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --exclude=gpu[10-13]
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
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

# debug logging or not
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"
export PYTHONUNBUFFERED=1

# sglang root path
export SGLANG_REPO_ROOT="${REPO_ROOT}"

# execute main function
# Override hicache_size for RTX 4090 VRAM; other settings come from config/*.json
cd "${SRC_ROOT}"
exec "${VENV_DIR}/bin/python" -m main \
    --log-dir "${LOG_DIR}" \
    "$@"
