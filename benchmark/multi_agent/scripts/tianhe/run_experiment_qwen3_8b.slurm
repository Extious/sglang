#!/bin/bash
#SBATCH --job-name=crewai_exp_8b_tianhe_dp2
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# Slurm copies this script to /tmp; do not use SCRIPT_DIR to find the repo. Runtime logs are
# written by the script itself into benchmark/multi_agent/logs.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# sglang root: on Slurm, sbatch uses a temp copy; prefer submit dir. Else local runs use SCRIPT_DIR.
if [[ -n "${SGLANG_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${SGLANG_REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../../../.." && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
if [[ ! -d "${REPO_ROOT}/benchmark/multi_agent/src" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} is not the sglang tree; set SGLANG_REPO_ROOT or run sbatch from scripts/tianhe" >&2
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

# CUDA toolkit root (required for SGLang JIT kernels during cuda-graph capture, e.g. fused RoPE)
# Tianhe modulepath uses GPU_compiler names like CUDA/12.2 (see: module avail)
if [[ -z "${CUDA_HOME:-}" ]]; then
  _cuda_candidates="${TIANHE_CUDA_MODULE:-CUDA/12.3 CUDA/12.2 CUDA/12.1 CUDA/12.0 CUDA/11.8}"
  for _cuda_mod in ${_cuda_candidates}; do
    module load "${_cuda_mod}" 2>/dev/null && break
  done || true
fi
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  _nvcc="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "${_nvcc}")/.." && pwd)"
fi
if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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

# Qwen3-8B native context = 40960; baseline-200k/remote_backup-200k apply YaRN and cap at 200000.
# Run examples:
#   sbatch run_experiment_qwen3_8b.slurm --config baseline-200k
#   sbatch run_experiment_qwen3_8b.slurm --config remote_backup-200k
# Optional flags (--log-dir, --output-dir, ...) are read from src/config/env.yml and/or the command line.
cd "${SRC_ROOT}"
exec "${VENV_DIR}/bin/python" -m main \
  --log-dir "${LOG_DIR}" \
  "$@"
