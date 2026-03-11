#!/bin/bash
# Simplified fault injection: Just run heavy_swarm and let router's retry handle it
# We'll document when the fault "should" occur for analysis purposes
#
# This version doesn't actually inject faults - it just runs the workload
# and you can manually observe/inject faults, or analyze natural failures.
#
# For actual fault injection without sudo/SSH/SLURM, you would need to:
# 1. Modify the worker code to accept shutdown signals
# 2. Use a separate control script on the worker node
# 3. Or use network-level tools (requires permissions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TASK_LIMIT="${TASK_LIMIT:-80}"
TASK_PROCESSES="${TASK_PROCESSES:-6}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_BASE_URL="${ROUTER_BASE_URL:-http://${ROUTER_HOST}:${ROUTER_PORT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HEAVY_SWARM_SCRIPT="${HEAVY_SWARM_SCRIPT:-${HOME}/swarms/heavy_swarm.py}"
RESULTS_PATH="${RESULTS_PATH:-}"
ENABLE_TIMING_REPORTS="${ENABLE_TIMING_REPORTS:-1}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/../logs/heavy_swarm_$(date +%Y%m%d_%H%M%S).log}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log "=========================================="
log "Heavy Swarm Test"
log "=========================================="
log "Task limit:           $TASK_LIMIT"
log "Task processes:       $TASK_PROCESSES"
log "Router URL:           $ROUTER_BASE_URL"
log "Log file:             $LOG_FILE"
log "=========================================="

# Set up environment
export HEAVY_SWARM_TASK_LIMIT="$TASK_LIMIT"
export HEAVY_SWARM_TASK_PROCESSES="$TASK_PROCESSES"
export HEAVY_SWARM_ENABLE_TIMING_REPORTS="$ENABLE_TIMING_REPORTS"
export LLM_BASE_URL="${ROUTER_BASE_URL}/v1"

if [ -n "$RESULTS_PATH" ]; then
    export HEAVY_SWARM_RESULTS_PATH="$RESULTS_PATH"
fi

log "Starting heavy_swarm.py..."
START_TIME=$(date +%s)

"$PYTHON_BIN" "$HEAVY_SWARM_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "=========================================="
log "Test Completed"
log "=========================================="
log "Exit code:            $EXIT_CODE"
log "Total duration:       ${DURATION}s"
log "=========================================="

exit "$EXIT_CODE"
