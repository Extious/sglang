#!/bin/bash
# Usage (from the login node):
#   sbatch -t 01:00:00 --export=ALL,BRANCH_DIR=$HOME/aegis/src/aegiskv,CONFIG=configs/a100_main.json,\
#SUITE=nofail,rebuild,aegiskv,WARMUP=4,OUT=$HOME/aegis/runs/main/aegiskv scripts/blsc/run_suite.sbatch
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -o /home/bingxing2/home/scx8333/aegis/logs/%x.%j.log
set -uo pipefail
source "$BRANCH_DIR/benchmark/AegisKV/scripts/blsc/env.sh"
cd "$BRANCH_DIR/benchmark/AegisKV"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -m harness.run --config "$CONFIG" --suite "$SUITE" --warmup "${WARMUP:-0}" --out-dir "$OUT"
echo "SUITE_EXIT=$?"
