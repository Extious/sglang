#!/bin/bash
# Shell-based fault injection wrapper for heavy_swarm.py with SLURM support
#
# This script runs heavy_swarm.py and injects worker failures by suspending
# worker processes via SLURM srun on compute nodes.
#
# Prerequisites:
#   - Must be run within an active SLURM allocation (same job as workers)
#   - Or workers must be running in a separate SLURM job (use --slurm-job-id)
#
# Usage Examples:
#   # Phase 1: Warmup — run first 30 tasks to populate KV cache
#   ./run_heavy_swarm_with_fault_injection.sh \
#     --warmup --warmup-tasks 30
#
#   # Phase 2: Test — start from task 30, fault at 3 min
#   ./run_heavy_swarm_with_fault_injection.sh \
#     --task-start-offset 30 \
#     --failed-worker-port 8003 \
#     --worker-node hkbugpusrv10
#
#   # Basic — 80 tasks, fault at 3 min (no warmup)
#   ./run_heavy_swarm_with_fault_injection.sh \
#     --failed-worker-port 8003 \
#     --worker-node hkbugpusrv10
#
#   # Target specific SLURM job
#   ./run_heavy_swarm_with_fault_injection.sh \
#     --failed-worker-port 8003 \
#     --worker-node hkbugpusrv10 \
#     --slurm-job-id 12345

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration Parameters
# ============================================================================

# Task configuration
TASK_LIMIT="${TASK_LIMIT:-80}"
TASK_PROCESSES="${TASK_PROCESSES:-6}"

# Cache warmup configuration
WARMUP_MODE="${WARMUP_MODE:-0}"                       # 1 = run warmup only (no fault injection)
WARMUP_TASKS="${WARMUP_TASKS:-30}"                    # Number of tasks for warmup
TASK_START_OFFSET="${TASK_START_OFFSET:-0}"            # Skip first N tasks (use after warmup)

# Fault injection timing (in seconds)
INJECT_AFTER_SECONDS="${INJECT_AFTER_SECONDS:-180}"   # Default: 3 minutes
RECOVER_AFTER_SECONDS="${RECOVER_AFTER_SECONDS:-30}"  # Default: 30 seconds
FAILED_WORKER_PORT="${FAILED_WORKER_PORT:-8003}"      # Default: port 8003
WORKER_NODE="${WORKER_NODE:-}"                        # Worker node name
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-}"       # Optional: target SLURM job ID

# Router configuration
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_BASE_URL="${ROUTER_BASE_URL:-http://${ROUTER_HOST}:${ROUTER_PORT}}"

# Python configuration
PYTHON_BIN="${PYTHON_BIN:-python3}"
HEAVY_SWARM_SCRIPT="${HEAVY_SWARM_SCRIPT:-${SCRIPT_DIR}/../src/application/swarms/heavy_swarm.py}"

# Output configuration
RESULTS_PATH="${RESULTS_PATH:-}"
ENABLE_TIMING_REPORTS="${ENABLE_TIMING_REPORTS:-1}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/../logs/fault_injection_$(date +%Y%m%d_%H%M%S).log}"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >> "$LOG_FILE"
    echo "$msg" >&2
}

error() {
    log "ERROR: $*"
}

# Execute command on worker node via SLURM srun
exec_on_worker_node() {
    local node="$1"
    local cmd="$2"

    local srun_opts="-w ${node} --nodes=1 --ntasks=1 --overlap"

    # If targeting specific job, add job ID
    if [ -n "${SLURM_JOB_ID_TARGET:-}" ]; then
        srun_opts="--jobid=${SLURM_JOB_ID_TARGET} ${srun_opts}"
    fi

    # Disable proxy for srun commands
    env NO_PROXY='*' no_proxy='*' srun ${srun_opts} bash -c "$cmd" 2>/dev/null
}

# Check if router is accessible
check_router() {
    local url="$1"
    if curl --noproxy '*' -fsS -m 2 "${url}/health" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Find worker process by port on worker node
find_worker_pid() {
    local node="$1"
    local port="$2"

    local cmd="ss -ltnp 'sport = :${port}' 2>/dev/null | awk -F'pid=' '/pid=/{split(\$2,a,\",\"); print a[1]}' | head -1"

    local pid
    pid=$(exec_on_worker_node "$node" "$cmd" 2>/dev/null || true)

    if [ -n "$pid" ] && [ "$pid" != "0" ]; then
        echo "$pid"
        return 0
    fi
    return 1
}

# Check worker is alive via HTTP /health (does not need srun)
check_worker_health() {
    local node="$1"
    local port="$2"
    curl --noproxy '*' -fsS -m 3 "http://${node}:${port}/health" >/dev/null 2>&1
}

# Stop worker process
stop_worker() {
    local node="$1"
    local port="$2"
    local pid

    log ">>> INJECTING FAULT: Stopping worker on ${node}:${port}"

    if pid=$(find_worker_pid "$node" "$port"); then
        log "Found worker PID: $pid on ${node}"

        local stop_cmd="kill -STOP $pid 2>/dev/null && echo 'ok' || echo 'failed'"
        local result
        result=$(exec_on_worker_node "$node" "$stop_cmd")

        if [ "$result" = "ok" ]; then
            log ">>> FAULT INJECTED: Worker process suspended (PID: $pid on ${node})"
            echo "$pid"
            return 0
        else
            error "Failed to suspend worker process (PID: $pid on ${node})"
            return 1
        fi
    else
        error "No worker found on ${node}:${port}"
        return 1
    fi
}

# Resume worker process
resume_worker() {
    local node="$1"
    local pid="$2"

    log ">>> RECOVERING FAULT: Resuming worker (PID: ${pid} on ${node})"

    # Directly try to resume without checking if process exists
    # (STOPPED processes may not show up in ps checks)
    local resume_cmd="kill -CONT $pid 2>/dev/null && echo 'ok' || echo 'failed'"
    local result
    result=$(exec_on_worker_node "$node" "$resume_cmd")

    if [ "$result" = "ok" ]; then
        log ">>> FAULT RECOVERED: Worker process resumed (PID: $pid on ${node})"
        return 0
    else
        error "Failed to resume worker process (PID: $pid on ${node})"
        error "Process may have been killed or terminated"
        return 1
    fi
}

# Fault injection scheduler - runs in background
fault_injection_scheduler() {
    local worker_node="$1"
    local worker_port="$2"
    local inject_after="$3"
    local recover_after="$4"
    local heavy_swarm_pid="$5"

    log "Fault injection scheduler started"
    log "Will inject fault after ${inject_after}s, recover after ${recover_after}s"
    log "Target worker: ${worker_node}:${worker_port}"

    # Wait for injection time
    log "Waiting ${inject_after}s before injecting fault..."
    local elapsed=0
    while [ "$elapsed" -lt "$inject_after" ]; do
        if ! kill -0 "$heavy_swarm_pid" 2>/dev/null; then
            log "heavy_swarm.py completed before fault injection time"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 60)) -eq 0 ]; then
            log "Elapsed: ${elapsed}s / ${inject_after}s (waiting for injection time)"
        fi
    done

    # Inject fault
    if ! kill -0 "$heavy_swarm_pid" 2>/dev/null; then
        log "heavy_swarm.py completed before fault injection"
        return 0
    fi

    local worker_pid
    if worker_pid=$(stop_worker "$worker_node" "$worker_port"); then
        log "FAULT INJECTED at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        error "Failed to inject fault, aborting recovery"
        return 1
    fi

    # Wait for recovery time
    log "Waiting ${recover_after}s before recovering fault..."
    elapsed=0
    while [ "$elapsed" -lt "$recover_after" ]; do
        if ! kill -0 "$heavy_swarm_pid" 2>/dev/null; then
            log "heavy_swarm.py completed during fault period"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 15)) -eq 0 ] && [ "$elapsed" -lt "$recover_after" ]; then
            log "Fault period: ${elapsed}s / ${recover_after}s"
        fi
    done

    # Recover fault
    if resume_worker "$worker_node" "$worker_pid"; then
        log "FAULT RECOVERED at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        error "Failed to recover fault"
        return 1
    fi

    log "Fault injection cycle completed"
}

# ============================================================================
# Main Script
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [options]

This script runs heavy_swarm.py and injects worker failures by suspending
worker processes via SLURM srun.

It supports a two-phase workflow:
  Phase 1 (warmup):  Run with --warmup to execute the first N tasks for cache
                     pre-population. No fault injection occurs in this phase.
  Phase 2 (test):    Run without --warmup. Tasks start from offset N (set via
                     --task-start-offset). Fault injection fires after the
                     configured delay.

Options:
  --task-limit N              Number of tasks to run (default: 80)
  --task-processes N          Number of parallel processes (default: 6)
  --warmup                    Run in warmup mode (cache pre-population only, no fault injection)
  --warmup-tasks N            Number of warmup tasks (default: 30)
  --task-start-offset N       Skip first N tasks; start from task N (default: 0)
  --inject-after N            Inject fault after N seconds (default: 180 = 3 min)
  --recover-after N           Recover after N seconds (default: 30)
  --failed-worker-port PORT   Worker port to fail (default: 8003)
  --worker-node NODE          Worker node name (REQUIRED for fault injection; optional for warmup)
  --slurm-job-id ID           Target SLURM job ID (optional)
  --router-host HOST          Router host (default: 127.0.0.1)
  --router-port PORT          Router port (default: 30000)
  --python-bin PATH           Python executable (default: python3)
  --results-path PATH         Output results JSON path
  --log-file PATH             Log file path
  -h, --help                  Show this help

Examples:
  # Phase 1: Warmup — run first 30 tasks to populate KV cache
  $0 --warmup --warmup-tasks 30 --router-host 10.0.0.1

  # Phase 2: Test — start from task 30, inject fault at 3 min
  $0 --task-start-offset 30 \\
     --worker-node hkbugpusrv10 --failed-worker-port 8003

  # Custom warmup count, then test with custom timing
  $0 --warmup --warmup-tasks 50
  $0 --task-start-offset 50 --task-limit 80 \\
     --worker-node hkbugpusrv10 --failed-worker-port 8003 \\
     --inject-after 60 --recover-after 30

  # Target specific SLURM job
  $0 --worker-node hkbugpusrv10 --failed-worker-port 8003 \\
     --slurm-job-id 12345
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-limit)
            TASK_LIMIT="$2"
            shift 2
            ;;
        --task-processes)
            TASK_PROCESSES="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_MODE=1
            shift
            ;;
        --warmup-tasks)
            WARMUP_TASKS="$2"
            shift 2
            ;;
        --task-start-offset)
            TASK_START_OFFSET="$2"
            shift 2
            ;;
        --inject-after)
            INJECT_AFTER_SECONDS="$2"
            shift 2
            ;;
        --recover-after)
            RECOVER_AFTER_SECONDS="$2"
            shift 2
            ;;
        --failed-worker-port)
            FAILED_WORKER_PORT="$2"
            shift 2
            ;;
        --worker-node)
            WORKER_NODE="$2"
            shift 2
            ;;
        --slurm-job-id)
            SLURM_JOB_ID_TARGET="$2"
            shift 2
            ;;
        --failed-worker-url)
            # Extract host and port from URL
            WORKER_NODE=$(echo "$2" | sed -E 's|https?://([^:]+):.*|\1|')
            FAILED_WORKER_PORT=$(echo "$2" | grep -oP ':\K[0-9]+$' || echo "8003")
            shift 2
            ;;
        --router-host)
            ROUTER_HOST="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --results-path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

# Reconstruct ROUTER_BASE_URL after all arguments are parsed
ROUTER_BASE_URL="http://${ROUTER_HOST}:${ROUTER_PORT}"

if [ "$WARMUP_MODE" -eq 0 ] && [ -z "$WORKER_NODE" ]; then
    error "Worker node name is required. Use --worker-node or --failed-worker-url"
    error "Or use --warmup for cache warmup mode (no fault injection)"
    usage >&2
    exit 1
fi

if [ ! -f "$HEAVY_SWARM_SCRIPT" ]; then
    error "heavy_swarm.py not found: $HEAVY_SWARM_SCRIPT"
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    error "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    error "srun command not found. This script requires SLURM."
    exit 1
fi

# Auto-detect SLURM job ID if not provided (skip in warmup mode)
if [ "$WARMUP_MODE" -eq 0 ]; then
    if [ -z "${SLURM_JOB_ID_TARGET:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        detected_job_id=$(squeue -u "$(whoami)" -h -w "${WORKER_NODE}" -o "%.10i" 2>/dev/null \
            | head -1 | tr -d ' ')
        if [ -n "$detected_job_id" ]; then
            SLURM_JOB_ID_TARGET="$detected_job_id"
            log "Auto-detected SLURM job ID: ${SLURM_JOB_ID_TARGET} (node: ${WORKER_NODE})"
        else
            log "WARNING: Could not auto-detect SLURM job ID for node ${WORKER_NODE}"
        fi
    fi
fi

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log "=========================================="
if [ "$WARMUP_MODE" -eq 1 ]; then
    log "Cache Warmup Configuration"
else
    log "Fault Injection Test Configuration"
fi
log "=========================================="
log "Mode:                 $([ "$WARMUP_MODE" -eq 1 ] && echo 'WARMUP (cache pre-population)' || echo 'TEST (with fault injection)')"
log "Task limit:           $TASK_LIMIT"
log "Task processes:       $TASK_PROCESSES"
log "Task start offset:    $TASK_START_OFFSET"
if [ "$WARMUP_MODE" -eq 1 ]; then
    log "Warmup tasks:         $WARMUP_TASKS"
else
    log "Inject after:         ${INJECT_AFTER_SECONDS}s ($(date -u -d @${INJECT_AFTER_SECONDS} +%H:%M:%S 2>/dev/null || echo "${INJECT_AFTER_SECONDS}s"))"
    log "Recover after:        ${RECOVER_AFTER_SECONDS}s"
    log "Failed worker:        ${WORKER_NODE}:${FAILED_WORKER_PORT}"
    log "SLURM job ID:         ${SLURM_JOB_ID_TARGET:-<current allocation>}"
fi
log "Router URL:           $ROUTER_BASE_URL"
log "Python binary:        $PYTHON_BIN"
log "Heavy swarm script:   $HEAVY_SWARM_SCRIPT"
log "Results path:         ${RESULTS_PATH:-<not set>}"
log "Log file:             $LOG_FILE"
log "=========================================="

# Check router connectivity
log "Checking router connectivity..."
if ! check_router "$ROUTER_BASE_URL"; then
    error "Router is not accessible at ${ROUTER_BASE_URL}"
    error "Please ensure the router is running"
    exit 1
fi
log "Router is accessible"

# Check if target worker exists (skip in warmup mode)
if [ "$WARMUP_MODE" -eq 0 ]; then
    log "Checking if worker on ${WORKER_NODE}:${FAILED_WORKER_PORT} exists..."
    if find_worker_pid "$WORKER_NODE" "$FAILED_WORKER_PORT" >/dev/null 2>&1; then
        log "Worker found via srun PID lookup on ${WORKER_NODE}:${FAILED_WORKER_PORT}"
    elif check_worker_health "$WORKER_NODE" "$FAILED_WORKER_PORT"; then
        log "Worker found via HTTP health check on ${WORKER_NODE}:${FAILED_WORKER_PORT}"
        log "NOTE: srun PID lookup failed but worker HTTP endpoint is reachable"
    else
        error "No worker found on ${WORKER_NODE}:${FAILED_WORKER_PORT}"
        error "Please verify:"
        error "  1. Worker is running on ${WORKER_NODE}"
        error "  2. Port ${FAILED_WORKER_PORT} is correct"
        error "  3. SLURM allocation includes ${WORKER_NODE}"
        error "  4. If outside SLURM allocation, use --slurm-job-id <JOB_ID>"
        error "  Hint: run 'squeue -u $(whoami)' to find your SLURM job ID"
        exit 1
    fi
fi

# ============================================================================
# Run heavy_swarm.py with fault injection
# ============================================================================

# Set up common environment for heavy_swarm.py
export HEAVY_SWARM_TASK_PROCESSES="$TASK_PROCESSES"
export HEAVY_SWARM_ENABLE_TIMING_REPORTS="$ENABLE_TIMING_REPORTS"
export LLM_BASE_URL="${ROUTER_BASE_URL}/v1"

if [ -n "$RESULTS_PATH" ]; then
    export HEAVY_SWARM_RESULTS_PATH="$RESULTS_PATH"
fi

START_TIME=$(date +%s)

# Track state for cleanup
_SUSPENDED_WORKER_PID=""
_SUSPENDED_WORKER_NODE=""
_HEAVY_SWARM_PID=""
_SCHEDULER_PID=""

cleanup() {
    local sig="${1:-EXIT}"
    if [ -n "$_SUSPENDED_WORKER_PID" ] && [ -n "$_SUSPENDED_WORKER_NODE" ]; then
        log "Cleanup: resuming suspended worker (PID: $_SUSPENDED_WORKER_PID on $_SUSPENDED_WORKER_NODE)"
        resume_worker "$_SUSPENDED_WORKER_NODE" "$_SUSPENDED_WORKER_PID" || true
        _SUSPENDED_WORKER_PID=""
    fi
    if [ -n "$_SCHEDULER_PID" ]; then
        kill "$_SCHEDULER_PID" 2>/dev/null || true
        wait "$_SCHEDULER_PID" 2>/dev/null || true
        _SCHEDULER_PID=""
    fi
    if [ "$sig" != "EXIT" ] && [ -n "$_HEAVY_SWARM_PID" ]; then
        kill "$_HEAVY_SWARM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Override stop_worker to track suspended state for cleanup
_original_stop_worker=$(declare -f stop_worker)
stop_worker() {
    local node="$1"
    local port="$2"
    local pid
    local result

    log ">>> INJECTING FAULT: Stopping worker on ${node}:${port}"

    if pid=$(find_worker_pid "$node" "$port"); then
        log "Found worker PID: $pid on ${node}"

        local stop_cmd="kill -STOP $pid 2>/dev/null && echo 'ok' || echo 'failed'"
        result=$(exec_on_worker_node "$node" "$stop_cmd")

        if [ "$result" = "ok" ]; then
            log ">>> FAULT INJECTED: Worker process suspended (PID: $pid on ${node})"
            _SUSPENDED_WORKER_PID="$pid"
            _SUSPENDED_WORKER_NODE="$node"
            echo "$pid"
            return 0
        else
            error "Failed to suspend worker process (PID: $pid on ${node})"
            return 1
        fi
    else
        error "No worker found on ${node}:${port}"
        return 1
    fi
}

# Override resume_worker to clear suspended state
_original_resume_worker=$(declare -f resume_worker)
resume_worker() {
    local node="$1"
    local pid="$2"

    log ">>> RECOVERING FAULT: Resuming worker (PID: ${pid} on ${node})"

    local resume_cmd="kill -CONT $pid 2>/dev/null && echo 'ok' || echo 'failed'"
    local result
    result=$(exec_on_worker_node "$node" "$resume_cmd")

    if [ "$result" = "ok" ]; then
        log ">>> FAULT RECOVERED: Worker process resumed (PID: $pid on ${node})"
        _SUSPENDED_WORKER_PID=""
        _SUSPENDED_WORKER_NODE=""
        return 0
    else
        error "Failed to resume worker process (PID: $pid on ${node})"
        error "Process may have been killed or terminated"
        _SUSPENDED_WORKER_PID=""
        _SUSPENDED_WORKER_NODE=""
        return 1
    fi
}

# ============================================================================
# Warmup Mode: run first N tasks for cache pre-population, no fault injection
# ============================================================================
if [ "$WARMUP_MODE" -eq 1 ]; then
    log ">>> WARMUP MODE: Running ${WARMUP_TASKS} tasks for cache pre-population"
    export HEAVY_SWARM_TASK_LIMIT="$WARMUP_TASKS"
    export HEAVY_SWARM_TASK_START_OFFSET="0"

    "$PYTHON_BIN" "$HEAVY_SWARM_SCRIPT" >> "$LOG_FILE" 2>&1 &
    _HEAVY_SWARM_PID=$!
    log "heavy_swarm.py (warmup) started with PID: $_HEAVY_SWARM_PID"

    set +e
    wait "$_HEAVY_SWARM_PID"
    HEAVY_SWARM_EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    log "=========================================="
    log "Warmup Completed"
    log "=========================================="
    log "Exit code:            $HEAVY_SWARM_EXIT_CODE"
    log "Warmup tasks:         $WARMUP_TASKS"
    log "Total duration:       ${DURATION}s ($(date -u -d @${DURATION} +%H:%M:%S 2>/dev/null || echo "${DURATION}s"))"
    log "Log file:             $LOG_FILE"
    log ""
    log "Next step — run the actual test with fault injection:"
    log "  $0 --task-start-offset ${WARMUP_TASKS} \\"
    log "     --worker-node <NODE> --failed-worker-port <PORT>"
    log "=========================================="

    exit "$HEAVY_SWARM_EXIT_CODE"
fi

# ============================================================================
# Test Mode: run tasks (from offset) with fault injection
# ============================================================================
export HEAVY_SWARM_TASK_LIMIT="$TASK_LIMIT"
export HEAVY_SWARM_TASK_START_OFFSET="$TASK_START_OFFSET"

log "Starting heavy_swarm.py (task offset: ${TASK_START_OFFSET}, limit: ${TASK_LIMIT})..."

# Run heavy_swarm.py in background
"$PYTHON_BIN" "$HEAVY_SWARM_SCRIPT" >> "$LOG_FILE" 2>&1 &
_HEAVY_SWARM_PID=$!

log "heavy_swarm.py started with PID: $_HEAVY_SWARM_PID"

# Start fault injection scheduler in background
fault_injection_scheduler \
    "$WORKER_NODE" \
    "$FAILED_WORKER_PORT" \
    "$INJECT_AFTER_SECONDS" \
    "$RECOVER_AFTER_SECONDS" \
    "$_HEAVY_SWARM_PID" &
_SCHEDULER_PID=$!

log "Fault injection scheduler started with PID: $_SCHEDULER_PID"

# Wait for heavy_swarm.py to complete (disable set -e for wait)
set +e
wait "$_HEAVY_SWARM_PID"
HEAVY_SWARM_EXIT_CODE=$?
set -e

# Wait for scheduler to finish
wait "$_SCHEDULER_PID" 2>/dev/null || true
_SCHEDULER_PID=""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "=========================================="
log "Test Completed"
log "=========================================="
log "Exit code:            $HEAVY_SWARM_EXIT_CODE"
log "Task start offset:    $TASK_START_OFFSET"
log "Total duration:       ${DURATION}s ($(date -u -d @${DURATION} +%H:%M:%S 2>/dev/null || echo "${DURATION}s"))"
log "Log file:             $LOG_FILE"
log "=========================================="

exit "$HEAVY_SWARM_EXIT_CODE"
