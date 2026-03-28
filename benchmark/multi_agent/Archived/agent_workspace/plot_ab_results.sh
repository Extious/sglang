#!/bin/bash
# Generate publication-quality AB comparison figures from completed experiments.
#
# Usage:
#   ./plot_ab_results.sh [--output-dir DIR]
#
# By default reads from the standard output location and auto-detects which
# experiment groups were completed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PLOT_SCRIPT="${MULTI_AGENT_DIR}/src/application/crewai/plot_ab_comparison.py"
OUTPUT_BASE="${MULTI_AGENT_DIR}/logs_qwen3_8b_gpuhome/agent_workspace/crewai_ab_qwen3_8b_gpuhome"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./plot_ab_results.sh [--output-dir DIR]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "${PLOT_SCRIPT}" ]; then
    echo "ERROR: plot script not found: ${PLOT_SCRIPT}" >&2
    exit 1
fi

BASELINE="${OUTPUT_BASE}/flush_fault__no_backup__stream"

declare -A TREATMENTS=(
    ["best_effort"]="${OUTPUT_BASE}/flush_fault__with_backup__stream__best_effort"
    ["wait_complete"]="${OUTPUT_BASE}/flush_fault__with_backup__stream__wait_complete"
    ["timeout"]="${OUTPUT_BASE}/flush_fault__with_backup__stream__timeout"
)
declare -A LABELS=(
    ["best_effort"]="Peer Backup (best-effort)"
    ["wait_complete"]="Peer Backup (wait-complete)"
    ["timeout"]="Peer Backup (timeout)"
)

if [ ! -f "${BASELINE}/trace_log.json" ]; then
    echo "ERROR: baseline experiment not found: ${BASELINE}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"

generated=0
for key in best_effort wait_complete timeout; do
    tdir="${TREATMENTS[$key]}"
    if [ -f "${tdir}/trace_log.json" ] && [ -f "${tdir}/retry_metrics.json" ]; then
        for ext in pdf png; do
            outfile="${OUTPUT_BASE}/ab_comparison__no_backup_vs_${key}.${ext}"
            echo "Generating: ${outfile}"
            python "${PLOT_SCRIPT}" \
                --baseline "${BASELINE}" \
                --treatment "${tdir}" \
                --baseline-label "No Backup" \
                --treatment-label "${LABELS[$key]}" \
                --output "${outfile}"
            generated=$((generated + 1))
        done
    else
        echo "Skipping ${key}: results not found in ${tdir}"
    fi
done

if [ "${generated}" -eq 0 ]; then
    echo "No treatment experiments found. Nothing to plot."
    exit 1
fi

echo ""
echo "Done. Generated ${generated} figures in: ${OUTPUT_BASE}/"
