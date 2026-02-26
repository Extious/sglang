#!/bin/bash
#SBATCH --job-name=qwen_image_t2i
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --exclude=gpu[10-13]
#SBATCH --output=/home/comp/zhanzhao/sglang/benchmark/multi_agent/logs/run_t2i_server_%j.out
#SBATCH --error=/home/comp/zhanzhao/sglang/benchmark/multi_agent/logs/run_t2i_server_%j.err

# Usage:
#   sbatch run_t2i_server.slurm [OPTIONS]
#
# Options:
#   --model-path PATH     Model path (default: Qwen/Qwen-Image-2512)
#   --port PORT           Server port (default: 30010)
#   --mode MODE           "serve" for persistent server, "generate" for one-off (default: serve)
#   --prompt TEXT          Prompt for generate mode (default: "A beautiful sunset over the mountains")
#   --num-inference-steps N  Number of denoising steps (default: 50)
#   --guidance-scale F    Guidance scale (default: 4.0)
#   -h, --help            Show this help message
#
# Examples:
#   # Launch persistent server
#   sbatch run_t2i_server.slurm
#
#   # One-off generation
#   sbatch run_t2i_server.slurm --mode generate --prompt "A cat sitting on a table"
#
#   # Custom model path (local weights)
#   sbatch run_t2i_server.slurm --model-path /path/to/local/Qwen-Image-2512

set -euo pipefail

# Default values
DEFAULT_MODEL_PATH="Qwen/Qwen-Image-2512"
DEFAULT_PORT=30010
DEFAULT_MODE="serve"
DEFAULT_PROMPT="A beautiful sunset over the mountains"
DEFAULT_NUM_INFERENCE_STEPS=50
DEFAULT_GUIDANCE_SCALE=4.0

# Parse command line arguments
show_help() {
    sed -n '16,35p' "$0" | sed 's/^# //'
    exit 0
}

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
PORT="${PORT:-${DEFAULT_PORT}}"
MODE="${MODE:-${DEFAULT_MODE}}"
PROMPT="${PROMPT:-${DEFAULT_PROMPT}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${DEFAULT_NUM_INFERENCE_STEPS}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-${DEFAULT_GUIDANCE_SCALE}}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --num-inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Paths
SCRIPT_DIR="/home/comp/zhanzhao/sglang/benchmark/multi_agent"
REPO_ROOT="${SGLANG_REPO_ROOT:-/home/comp/zhanzhao/sglang}"
LOG_DIR="${SCRIPT_DIR}/logs"
VENV_DIR="${REPO_ROOT}/.venv"
OUTPUT_DIR="${SCRIPT_DIR}/results/t2i_outputs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Activate virtual environment
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    source "${VENV_DIR}/Scripts/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
else
    echo "WARNING: Virtual environment not found at ${VENV_DIR}" >&2
    echo "  Trying to continue with system Python..." >&2
fi

# Verify SGLang is available
if [ ! -d "${REPO_ROOT}/python" ]; then
    echo "ERROR: SGLang python directory not found: ${REPO_ROOT}/python" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "============================================"
echo "Deploying Qwen-Image Text-to-Image Server"
echo "============================================"
echo "Node:               ${SLURM_NODELIST:-unknown}"
echo "GPU:                1x A100 (80GB)"
echo "Model:              ${MODEL_PATH}"
echo "Mode:               ${MODE}"
echo "CPU offload:        disabled (A100 80GB has sufficient VRAM)"
echo "Repo root:          ${REPO_ROOT}"
echo "Output:             ${OUTPUT_DIR}"
echo "============================================"

# NOTE: Qwen-Image-2512 shares the same architecture as Qwen-Image (20B MMDiT).
# Total model size ~40GB in bf16 (DIT ~20GB + text encoder ~15GB + VAE ~1GB).
# A100 80GB has sufficient VRAM to load the entire model without CPU offload.

# Server args for single A100 (no CPU offload needed)
SERVER_ARGS=(
    --model-path "${MODEL_PATH}"
    --num-gpus 1
    --tp-size 1
    --text-encoder-cpu-offload false
    --dit-cpu-offload false
    --vae-cpu-offload false
)

if [ "${MODE}" = "serve" ]; then
    echo "Starting persistent HTTP server on port ${PORT}..."
    sglang serve "${SERVER_ARGS[@]}" --port "${PORT}"
elif [ "${MODE}" = "generate" ]; then
    SAMPLING_ARGS=(
        --prompt "${PROMPT}"
        --num-inference-steps "${NUM_INFERENCE_STEPS}"
        --guidance-scale "${GUIDANCE_SCALE}"
        --save-output
        --output-path "${OUTPUT_DIR}"
    )
    echo "Running one-off generation..."
    echo "Prompt: ${PROMPT}"
    echo "Steps:  ${NUM_INFERENCE_STEPS}"
    sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
    echo "Generation complete. Output saved to ${OUTPUT_DIR}"
else
    echo "ERROR: Unknown mode '${MODE}'. Use 'serve' or 'generate'." >&2
    exit 1
fi
