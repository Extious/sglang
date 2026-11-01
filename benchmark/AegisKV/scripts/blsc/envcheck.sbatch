#!/bin/bash
# One-GPU check of the CUDA 13 compat runtime, kernels and the AegisKV endpoint.
#   sbatch --export=ALL,BRANCH_DIR=$HOME/aegis/src/aegiskv,MODEL=... scripts/blsc/envcheck.sbatch
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 00:15:00
#SBATCH -o /home/bingxing2/home/scx8333/aegis/logs/%x.%j.log
set -uo pipefail
source "$BRANCH_DIR/benchmark/AegisKV/scripts/blsc/env.sh"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -m sglang.launch_server --model-path "$MODEL" --port 30999 --host 127.0.0.1 \
  --enable-aegiskv --page-size 1 --trust-remote-code --context-length 8192 \
  --cuda-graph-backend-prefill disabled --mem-fraction-static 0.85 &
PID=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:30999/health_generate >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:30999/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
echo
curl -s http://127.0.0.1:30999/aegiskv/control -H 'Content-Type: application/json' -d '{"op": "poll", "args": {}}'
echo
kill $PID; wait $PID
echo ENVCHECK_DONE
