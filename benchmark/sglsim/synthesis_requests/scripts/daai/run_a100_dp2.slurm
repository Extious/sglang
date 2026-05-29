#!/bin/bash
#SBATCH --job-name=synth_gsp_dp2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
    local start_dir="$1"
    local current_dir
    current_dir="$(cd "${start_dir}" 2>/dev/null && pwd)" || return 1
    for _ in 1 2 3 4 5 6 7 8; do
        if [ -f "${current_dir}/benchmark/sglsim/synthesis_requests/run.py" ]; then
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
    echo "ERROR: Cannot locate sglang repo root." >&2
    exit 1
fi

VENV_DIR="${SGLANG_VENV:-${REPO_ROOT}/.venv}"
CONFIG_DIR="${REPO_ROOT}/benchmark/sglsim/synthesis_requests/config"
LOG_DIR="${REPO_ROOT}/benchmark/sglsim/synthesis_requests/logs"
mkdir -p "${LOG_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
MAIN_LOG="${LOG_DIR}/slurm_${JOB_ID}.log"
exec >> "${MAIN_LOG}" 2>&1

module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
if [ -d "${GCC_ROOT}/bin" ]; then
    export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
    export CXX="${GCC_ROOT}/bin/g++"
    export CC="${GCC_ROOT}/bin/gcc"
    export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="utf-8"
export PYTHONUNBUFFERED=1
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"

if [ -n "${HF_HOME:-}" ]; then
    export HF_HOME
elif [ -d "/tmp/sglang-model-cache" ]; then
    export HF_HOME="/tmp/sglang-model-cache"
fi

echo "=== synthesis_requests A100 dp=2 ==="
echo "repo=${REPO_ROOT}"
echo "node=$(hostname)"
echo "job_id=${JOB_ID}"
nvidia-smi -L || true

cd "${REPO_ROOT}"
"${VENV_DIR}/bin/python" -m pip install -q transformers numpy tqdm 2>/dev/null || true
"${VENV_DIR}/bin/python" -m benchmark.sglsim.synthesis_requests.run \
    --config-dir "${CONFIG_DIR}" \
    --log-level info

echo "=== done; results under benchmark/sglsim/synthesis_requests/results/ ==="
