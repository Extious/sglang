#!/bin/bash
# Launcher for heavy_swarm_fault_test.py — runs all tasks in one shot and
# injects worker failures based on real-time task completion count.
#
# Prerequisites:
#   - Must be run within an active SLURM allocation (same job as workers)
#   - Or workers must be running in a separate SLURM job (use --slurm-job-id)
#
# Usage Examples:
#   # Default: 200 tasks, 18 processes, fault every 30 tasks on 6 workers
#   ./run_heavy_swarm_with_fault_injection.sh
#
#   # Quick smoke test: 50 tasks, 1 fault after 25 tasks
#   ./run_heavy_swarm_with_fault_injection.sh --quick
#
#   # Custom batch size and recovery time
#   ./run_heavy_swarm_with_fault_injection.sh \
#     --inject-every 50 --recover-after 60

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration (all overridable via env vars or CLI flags)
# ============================================================================

# Task
TASK_LIMIT="${TASK_LIMIT:-200}"
TASK_PROCESSES="${TASK_PROCESSES:-18}"

# Fault injection
INJECT_EVERY_N="${INJECT_EVERY_N:-30}"
INJECT_DELAY="${INJECT_DELAY:-10}"
RECOVER_AFTER="${RECOVER_AFTER:-30}"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/../logs/worker_urls.txt}"
SLURM_JOB_ID_ARG="${SLURM_JOB_ID_ARG:-}"

# Router
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-30000}"

# Python / script paths
PYTHON_BIN="${PYTHON_BIN:-${HOME}/swarms/swarms/.venv/bin/python}"
HEAVY_SWARM_SCRIPT="${HEAVY_SWARM_SCRIPT:-${HOME}/swarms/heavy_swarm.py}"
FAULT_TEST_SCRIPT="${FAULT_TEST_SCRIPT:-${SCRIPT_DIR}/heavy_swarm_fault_test.py}"

# Output
RESULTS_PATH="${RESULTS_PATH:-}"
ENABLE_TIMING_REPORTS="${ENABLE_TIMING_REPORTS:-1}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/../logs/fault_injection_$(date +%Y%m%d_%H%M%S).log}"

# ============================================================================
# Usage
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [options]

Runs heavy_swarm tasks in one shot and injects worker failures based on
real-time task completion count.  After every N completed tasks, the next
worker is suspended; it auto-recovers after a configurable delay while
remaining tasks keep running.

Options:
  --task-limit N              Total number of tasks (default: 200)
  --task-processes N          Parallel processes (default: 18)
  --quick                     Quick mode: 50 tasks, 1 fault after 25 tasks
  --inject-every N            Inject a fault every N completed tasks (default: 30)
  --inject-delay N            Seconds to wait after threshold before injecting (default: 10)
  --recover-after N           Recover suspended worker after N seconds (default: 30)
  --worker-urls-file PATH     File with worker URLs, one per line
  --slurm-job-id ID           Target SLURM job ID (optional, auto-detected)
  --router-host HOST          Router host (default: 127.0.0.1)
  --router-port PORT          Router port (default: 30000)
  --python-bin PATH           Python executable
  --results-path PATH         Output results JSON path
  --log-file PATH             Log file path
  -h, --help                  Show this help

Examples:
  $0
  $0 --quick
  $0 --inject-every 50 --recover-after 60
  $0 --worker-urls-file /path/to/urls.txt --slurm-job-id 12345
EOF
}

# ============================================================================
# Parse CLI
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-limit)          TASK_LIMIT="$2";        shift 2 ;;
        --task-processes)      TASK_PROCESSES="$2";    shift 2 ;;
        --quick)               TASK_LIMIT=50; INJECT_EVERY_N=25; INJECT_DELAY=10; RECOVER_AFTER=30; shift ;;
        --inject-every)        INJECT_EVERY_N="$2";    shift 2 ;;
        --inject-delay)        INJECT_DELAY="$2";      shift 2 ;;
        --recover-after)       RECOVER_AFTER="$2";     shift 2 ;;
        --worker-urls-file)    WORKER_URLS_FILE="$2";  shift 2 ;;
        --slurm-job-id)        SLURM_JOB_ID_ARG="$2";  shift 2 ;;
        --router-host)         ROUTER_HOST="$2";       shift 2 ;;
        --router-port)         ROUTER_PORT="$2";       shift 2 ;;
        --python-bin)          PYTHON_BIN="$2";        shift 2 ;;
        --results-path)        RESULTS_PATH="$2";      shift 2 ;;
        --log-file)            LOG_FILE="$2";          shift 2 ;;
        -h|--help)             usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

ROUTER_BASE_URL="http://${ROUTER_HOST}:${ROUTER_PORT}"

if [ ! -f "$WORKER_URLS_FILE" ]; then
    echo "ERROR: Worker URLs file not found: $WORKER_URLS_FILE" >&2
    exit 1
fi

if [ ! -f "$HEAVY_SWARM_SCRIPT" ]; then
    echo "ERROR: heavy_swarm.py not found: $HEAVY_SWARM_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$FAULT_TEST_SCRIPT" ]; then
    echo "ERROR: heavy_swarm_fault_test.py not found: $FAULT_TEST_SCRIPT" >&2
    exit 1
fi

if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
    echo "ERROR: Python binary not found: $PYTHON_BIN" >&2
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    echo "ERROR: srun command not found. This script requires SLURM." >&2
    exit 1
fi

# Quick router health check
if ! curl --noproxy '*' -fsS -m 2 "${ROUTER_BASE_URL}/health" >/dev/null 2>&1; then
    echo "ERROR: Router is not accessible at ${ROUTER_BASE_URL}" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# ============================================================================
# Export environment variables for the Python script
# ============================================================================

export HEAVY_SWARM_TASK_LIMIT="$TASK_LIMIT"
export HEAVY_SWARM_TASK_PROCESSES="$TASK_PROCESSES"
export HEAVY_SWARM_ENABLE_TIMING_REPORTS="$ENABLE_TIMING_REPORTS"
export LLM_BASE_URL="${ROUTER_BASE_URL}/v1"
export HEAVY_SWARM_SCRIPT

# Suppress verbose agent output — show only progress on terminal
export HEAVY_SWARM_VERBOSE="${HEAVY_SWARM_VERBOSE:-false}"
export HEAVY_SWARM_AGENT_PRINTS_ON="${HEAVY_SWARM_AGENT_PRINTS_ON:-false}"
export HEAVY_SWARM_SHOW_DASHBOARD="${HEAVY_SWARM_SHOW_DASHBOARD:-false}"
export HEAVY_SWARM_STREAMING="${HEAVY_SWARM_STREAMING:-false}"
export TASK_STAGGER_DELAY="${TASK_STAGGER_DELAY:-0}"
export SUPPRESS_AGENT_OUTPUT="${SUPPRESS_AGENT_OUTPUT:-1}"

export FAULT_WORKER_URLS_FILE="$WORKER_URLS_FILE"
export FAULT_INJECT_EVERY_N="$INJECT_EVERY_N"
export FAULT_INJECT_DELAY="$INJECT_DELAY"
export FAULT_RECOVER_AFTER="$RECOVER_AFTER"

if [ -n "$SLURM_JOB_ID_ARG" ]; then
    export FAULT_SLURM_JOB_ID="$SLURM_JOB_ID_ARG"
fi

# Kill-mode fault injection support
if [ -n "${FAULT_MODE:-}" ]; then
    export FAULT_MODE
fi
if [ -n "${FAULT_ROUTER_URL:-}" ]; then
    export FAULT_ROUTER_URL
fi
if [ -n "${WORKER_RESTART_CMD:-}" ]; then
    export WORKER_RESTART_CMD
fi

if [ -n "$RESULTS_PATH" ]; then
    export HEAVY_SWARM_RESULTS_PATH="$RESULTS_PATH"
fi

# ============================================================================
# Log configuration and launch
# ============================================================================

{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fault Injection Test Configuration"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task limit:           $TASK_LIMIT"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task processes:       $TASK_PROCESSES"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inject every:         ${INJECT_EVERY_N} tasks"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inject delay:         ${INJECT_DELAY}s"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Recover after:        ${RECOVER_AFTER}s"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stagger delay:        ${TASK_STAGGER_DELAY}s"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker URLs file:     $WORKER_URLS_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Router URL:           $ROUTER_BASE_URL"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python binary:        $PYTHON_BIN"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file:             $LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
} | tee -a "$LOG_FILE" >&2

START_TIME=$(date +%s)

# stdout (agent output) → log file only
# stderr (progress / fault-injection messages) → terminal + log file
set +e
"$PYTHON_BIN" "$FAULT_TEST_SCRIPT" >> "$LOG_FILE" 2> >(tee -a "$LOG_FILE" >&2)
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test Completed"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code:            $EXIT_CODE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total duration:       ${DURATION}s ($(date -u -d @${DURATION} +%H:%M:%S 2>/dev/null || echo "${DURATION}s"))"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file:             $LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
} | tee -a "$LOG_FILE" >&2

exit "$EXIT_CODE"
