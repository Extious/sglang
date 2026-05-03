#!/bin/bash
#SBATCH --job-name=sglang_server_8b
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu[10-13]
#SBATCH --output=/home/comp/%u/sglang/benchmark/multi_agent/logs/logs_qwen3_8b/run_server_%j.out
#SBATCH --error=/home/comp/%u/sglang/benchmark/multi_agent/logs/logs_qwen3_8b/run_server_%j.err

set -euo pipefail

REPO_ROOT="/home/comp/zhanzhao/sglang"

# GCC setup (for CUDA)
module load gcc/11.2.0 2>/dev/null || true
GCC_ROOT="/usr/local/gcc/gcc-11.2.0"
if [ -d "${GCC_ROOT}/bin" ]; then
    export NVCC_PREPEND_FLAGS="-ccbin ${GCC_ROOT}/bin"
    export CXX="${GCC_ROOT}/bin/g++"
    export CC="${GCC_ROOT}/bin/gcc"
    export LD_LIBRARY_PATH="${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# src core path
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"

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

# deploy script path
DEPLOY_PY="${REPO_ROOT}/benchmark/multi_agent/src/server/deploy_server.py"

# venv
VENV_DIR="${REPO_ROOT}/.venv"

# log directory
LOG_DIR="${REPO_ROOT}/benchmark/multi_agent/logs/logs_qwen3_8b"
mkdir -p "${LOG_DIR}"

# execute deploy script
cd "${REPO_ROOT}"
exec "${VENV_DIR}/bin/python" "${DEPLOY_PY}" --log-dir "${LOG_DIR}" "$@"
