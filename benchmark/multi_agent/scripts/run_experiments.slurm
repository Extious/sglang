#!/bin/bash
# CPU-only SLURM job: run baseline + gpu_backup CrewAI experiments via Python.
# GPU deploy is separate: deploy_sglang_dp2_pp2.slurm (DP=2, PP=2).
#
# Usage:
#   sbatch run_experiments.slurm [extra args passed to experiment_pipeline.py]
#
# Example:
#   sbatch run_experiments.slurm --group all --job-limit 6

#SBATCH --job-name=crewai_ab_cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/comp/%u/sglang/benchmark/multi_agent/logs/qwen3_32b_a100/run_experiments_%j.out
#SBATCH --error=/home/comp/%u/sglang/benchmark/multi_agent/logs/qwen3_32b_a100/run_experiments_%j.err

set -euo pipefail

# Hardcoded absolute path for this deployment
REPO_ROOT="/home/comp/25482351/sglang"
if [ ! -d "${REPO_ROOT}/python" ]; then
    echo "ERROR: Repo root not found at ${REPO_ROOT}" >&2
    exit 1
fi
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${REPO_ROOT}/benchmark/multi_agent/logs/qwen3_32b_a100"

exec python "${REPO_ROOT}/benchmark/multi_agent/applications/crewai/experiment_pipeline.py" "$@"
