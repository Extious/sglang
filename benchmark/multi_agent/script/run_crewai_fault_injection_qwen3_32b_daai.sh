#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export STAGE_MANIFEST="${STAGE_MANIFEST:-${MULTI_AGENT_DIR}/logs/qwen3_32b_a100/stage_manifest.json}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MULTI_AGENT_DIR}/logs/qwen3_32b_a100/crewai_fault_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_crewai_fault_injection_gpuhome.sh" "$@"
