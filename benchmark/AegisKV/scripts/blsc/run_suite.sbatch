#!/bin/bash
# Usage (from the login node):
#   sbatch -t 01:00:00 --export=ALL,BRANCH_DIR=$HOME/aegis/src/aegiskv,CONFIG=configs/a100_main.json,\
#SUITE=nofail:rebuild:aegiskv,WARMUP=4,OUT=$HOME/aegis/runs/main/aegiskv scripts/blsc/run_suite.sbatch
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -o /home/bingxing2/home/scx8333/aegis/logs/%x.%j.log
set -uo pipefail
source "$BRANCH_DIR/benchmark/AegisKV/scripts/blsc/env.sh"
cd "$BRANCH_DIR/benchmark/AegisKV"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
hostname
# Fail fast if the forward-compat driver cannot initialize CUDA on this node.
python -c "import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print('cuda ok', torch.version.cuda)" || { echo "CUDA_INIT_FAILED on $(hostname)"; exit 2; }
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
# Baseline suites reuse the failure time fixed by the AegisKV suite's No-Failure run.
FAIL_AT=()
[ -n "${FAIL_AT_FILE:-}" ] && FAIL_AT=(--fail-at "$(cat "$FAIL_AT_FILE")")
# Entries <policy>@<worker>/<pct> need the No-Failure makespan of the AegisKV suite.
[ -n "${MAKESPAN_FILE:-}" ] && FAIL_AT+=(--makespan "$(cat "$MAKESPAN_FILE")")
# sbatch --export splits on commas, so SUITE uses colons.
PYTHONPATH="$AEGIS_MASTER_SITE" python -m harness.run --config "$CONFIG" --suite "${SUITE//:/,}" --warmup "${WARMUP:-0}" --out-dir "$OUT" "${FAIL_AT[@]}"
echo "SUITE_EXIT=$?"
