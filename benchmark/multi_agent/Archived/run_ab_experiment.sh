#!/bin/bash
# A/B Experiment: peer-replication vs no peer-replication under fault injection.
#
# Runs two sequential experiments on the same SLURM cluster:
#   A) WITH peer-replication  + fault-tolerance routing
#   B) WITHOUT peer-replication (baseline)
#
# Each experiment:
#   1. Submits a SLURM job to start workers
#   2. Waits for workers to be ready
#   3. Starts the Python router
#   4. Runs fault injection test
#   5. Saves timing reports to a separate directory
#   6. Tears down router and workers
#
# Usage:
#   ./run_ab_experiment.sh [options]
#
# Options:
#   --task-limit N           Total tasks per experiment (default: 200)
#   --task-processes N       Parallel processes (default: 18)
#   --inject-every N         Fault every N tasks (default: 30)
#   --recover-after N        Recovery delay in seconds (default: 30)
#   --num-workers N          Number of workers (default: 6)
#   --model-path PATH        Model path (default: Qwen/Qwen3-4B-Instruct-2507)
#   --hicache-size N         HiCache size per worker in GB (default: 24)
#   --output-dir DIR         Base output directory for results
#   --only-with              Run only the WITH peer-replication experiment
#   --only-without           Run only the WITHOUT peer-replication experiment
#   --quick                  Quick mode: 50 tasks, fault after 25
#   -h, --help               Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MULTI_AGENT_DIR}/logs"
SLURM_DIR="${MULTI_AGENT_DIR}/slurm"

# ============================================================================
# Defaults
# ============================================================================
TASK_LIMIT=200
TASK_PROCESSES=18
INJECT_EVERY_N=30
INJECT_DELAY=10
RECOVER_AFTER=40
NUM_WORKERS=6
WORKERS_PER_NODE=6
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
HICACHE_SIZE=24
QUANTIZATION=""
ROUTER_PORT=30000
ROUTER_HOST="127.0.0.1"
OUTPUT_BASE_DIR="${HOME}/swarms/agent_workspace/timing_reports"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

RUN_WITH=1
RUN_WITHOUT=1

# ============================================================================
# Parse CLI
# ============================================================================
usage() {
    sed -n '/^# Usage:/,/^# *-h/p' "$0" | sed 's/^# //'
    echo "  -h, --help               Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-limit)        TASK_LIMIT="$2";        shift 2 ;;
        --task-processes)    TASK_PROCESSES="$2";     shift 2 ;;
        --inject-every)      INJECT_EVERY_N="$2";    shift 2 ;;
        --recover-after)     RECOVER_AFTER="$2";     shift 2 ;;
        --num-workers)       NUM_WORKERS="$2";       shift 2 ;;
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --hicache-size)      HICACHE_SIZE="$2";      shift 2 ;;
        --quantization)      QUANTIZATION="$2";      shift 2 ;;
        --output-dir)        OUTPUT_BASE_DIR="$2";   shift 2 ;;
        --only-with)         RUN_WITH=1; RUN_WITHOUT=0; shift ;;
        --only-without)      RUN_WITH=0; RUN_WITHOUT=1; shift ;;
        --quick)             TASK_LIMIT=50; INJECT_EVERY_N=25; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_LOG="${LOG_DIR}/ab_experiment_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# ============================================================================
# Helpers
# ============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${EXPERIMENT_LOG}"
}

die() {
    log "FATAL: $*"
    exit 1
}

wait_for_file() {
    local filepath="$1"
    local timeout_s="${2:-600}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if [ -f "${filepath}" ] && [ -s "${filepath}" ]; then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_s}" ]; then
            return 1
        fi
        sleep 5
    done
}

wait_for_router() {
    local host="$1"
    local port="$2"
    local timeout_s="${3:-120}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if curl --noproxy '*' -fsS -m 2 "http://${host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_s}" ]; then
            return 1
        fi
        sleep 2
    done
}

kill_router() {
    local port="$1"
    local pids
    pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
        | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' \
        | sort -u)
    for pid in ${pids}; do
        [ -z "${pid}" ] && continue
        log "  Killing router process pid=${pid} on port ${port}"
        kill "${pid}" 2>/dev/null || true
    done
    sleep 2
}

cancel_slurm_job() {
    local job_id="$1"
    if [ -n "${job_id}" ] && squeue -j "${job_id}" -h >/dev/null 2>&1; then
        log "  Cancelling SLURM job ${job_id}"
        scancel "${job_id}" 2>/dev/null || true
        sleep 5
    fi
}

kill_worker_ports() {
    local base_port="${1:-8000}"
    local num_workers="${2:-6}"
    local peer_port_base="${3:-9000}"
    for i in $(seq 0 $((num_workers - 1))); do
        local wport=$((base_port + i))
        local pport=$((peer_port_base + i))
        for port in "${wport}" "${pport}"; do
            local pids
            pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
                | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' \
                | sort -u)
            for pid in ${pids}; do
                [ -z "${pid}" ] && continue
                log "  Killing leftover process pid=${pid} on port ${port}"
                kill "${pid}" 2>/dev/null || true
            done
        done
    done
    sleep 2
}

# ============================================================================
# Run one experiment
# ============================================================================
run_experiment() {
    local label="$1"         # "with_peer_replication" or "without_peer_replication"
    local peer_flag="$2"     # "--enable-peer-replication" or ""
    local ft_flag="$3"       # "--enable-fault-tolerance" or ""
    local result_dir="$4"    # output directory for timing reports

    log "============================================================"
    log "EXPERIMENT: ${label}"
    log "============================================================"
    log "  peer-replication: $([ -n "${peer_flag}" ] && echo 'YES' || echo 'NO')"
    log "  fault-tolerance:  $([ -n "${ft_flag}" ] && echo 'YES' || echo 'NO')"
    log "  tasks: ${TASK_LIMIT}, processes: ${TASK_PROCESSES}"
    log "  inject every: ${INJECT_EVERY_N}, recover after: ${RECOVER_AFTER}s"
    log "  results: ${result_dir}"
    log ""

    mkdir -p "${result_dir}"

    # -- Step 1: Submit SLURM job --
    log "[${label}] Step 1: Submitting SLURM job for workers..."
    kill_worker_ports 8000 "${NUM_WORKERS}" 9000

    local sbatch_args=(
        --nodes=1
        "${SLURM_DIR}/run_server_qwen_4b.slurm"
        --num-workers "${NUM_WORKERS}"
        --workers-per-node "${WORKERS_PER_NODE}"
        --model-path "${MODEL_PATH}"
        --enable-hicache
        --hicache-size "${HICACHE_SIZE}"
    )
    if [ -n "${peer_flag}" ]; then
        sbatch_args+=( --enable-peer-replication )
    fi
    if [ -n "${QUANTIZATION}" ]; then
        sbatch_args+=( --quantization "${QUANTIZATION}" )
    fi

    local sbatch_output
    sbatch_output=$(sbatch "${sbatch_args[@]}" 2>&1)
    local job_id
    job_id=$(echo "${sbatch_output}" | grep -oP '\d+$' || true)

    if [ -z "${job_id}" ]; then
        log "ERROR: Failed to submit SLURM job: ${sbatch_output}"
        return 1
    fi
    log "  SLURM job submitted: ${job_id}"

    # -- Step 2: Wait for SLURM job to start running --
    log "[${label}] Step 2a: Waiting for SLURM job ${job_id} to start running (timeout=300s)..."

    local wait_start
    wait_start=$(date +%s)
    while true; do
        local job_state
        job_state=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        if [ "${job_state}" = "RUNNING" ]; then
            log "  SLURM job ${job_id} is RUNNING"
            break
        fi
        if [ "${job_state}" = "FAILED" ] || [ "${job_state}" = "CANCELLED" ] || \
           [ "${job_state}" = "TIMEOUT" ] || [ "${job_state}" = "UNKNOWN" ]; then
            if ! squeue -j "${job_id}" -h >/dev/null 2>&1; then
                log "ERROR: SLURM job ${job_id} is no longer in queue (state=${job_state})"
                return 1
            fi
        fi
        if [ $(( $(date +%s) - wait_start )) -ge 300 ]; then
            log "ERROR: SLURM job ${job_id} did not start within 300s (state=${job_state})"
            cancel_slurm_job "${job_id}"
            return 1
        fi
        log "  Job ${job_id} state: ${job_state}, waiting..."
        sleep 10
    done

    # Clean stale worker_urls.txt so we don't pick up results from a previous run
    local worker_urls_file="${LOG_DIR}/worker_urls.txt"
    rm -f "${worker_urls_file}"

    # -- Step 2b: Wait for workers to be ready --
    log "[${label}] Step 2b: Waiting for worker_urls.txt to appear (timeout=600s)..."

    if ! wait_for_file "${worker_urls_file}" 600; then
        log "ERROR: Workers did not become ready within 600s"
        cancel_slurm_job "${job_id}"
        return 1
    fi

    local num_urls
    num_urls=$(wc -l < "${worker_urls_file}" | tr -d ' ')
    log "  Workers ready: ${num_urls} URLs in ${worker_urls_file}"

    # -- Step 3: Start router --
    log "[${label}] Step 3: Starting router..."

    kill_router "${ROUTER_PORT}"

    local router_args=(
        --worker-urls-file "${worker_urls_file}"
        --port "${ROUTER_PORT}"
    )
    if [ -n "${ft_flag}" ]; then
        router_args+=( --enable-fault-tolerance )
    fi

    # Launch router in a subshell with sglang venv activated.
    # This keeps the parent shell's environment clean for the swarms venv.
    (
        if [ -f "${VENV_DIR}/bin/activate" ]; then
            source "${VENV_DIR}/bin/activate"
        fi
        export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
        exec "${SCRIPT_DIR}/run_router.sh" "${router_args[@]}"
    ) >> "${LOG_DIR}/router_${label}_${TIMESTAMP}.log" 2>&1 &
    local router_launcher_pid=$!

    if ! wait_for_router "${ROUTER_HOST}" "${ROUTER_PORT}" 120; then
        log "ERROR: Router did not become healthy within 120s"
        kill "${router_launcher_pid}" 2>/dev/null || true
        cancel_slurm_job "${job_id}"
        return 1
    fi
    log "  Router healthy at http://${ROUTER_HOST}:${ROUTER_PORT}"

    # -- Step 4: Run fault injection test --
    log "[${label}] Step 4: Running fault injection test..."

    local fault_log="${LOG_DIR}/fault_injection_${label}_${TIMESTAMP}.log"

    local exp_start_ts
    exp_start_ts=$(date +%s)

    set +e
    TASK_LIMIT="${TASK_LIMIT}" \
    TASK_PROCESSES="${TASK_PROCESSES}" \
    INJECT_EVERY_N="${INJECT_EVERY_N}" \
    INJECT_DELAY="${INJECT_DELAY}" \
    RECOVER_AFTER="${RECOVER_AFTER}" \
    WORKER_URLS_FILE="${worker_urls_file}" \
    FAULT_SLURM_JOB_ID="${job_id}" \
    HEAVY_SWARM_MODEL_NAME="openai/${MODEL_PATH}" \
    HEAVY_SWARM_TIMING_OUTPUT_DIR="${result_dir}" \
    LOG_FILE="${fault_log}" \
        "${SCRIPT_DIR}/run_heavy_swarm_with_fault_injection.sh" \
            --task-limit "${TASK_LIMIT}" \
            --task-processes "${TASK_PROCESSES}" \
            --inject-every "${INJECT_EVERY_N}" \
            --recover-after "${RECOVER_AFTER}" \
            --worker-urls-file "${worker_urls_file}" \
            --slurm-job-id "${job_id}" \
            --log-file "${fault_log}"
    local test_rc=$?
    set -e

    local exp_end_ts
    exp_end_ts=$(date +%s)
    local duration=$(( exp_end_ts - exp_start_ts ))

    if [ "${test_rc}" -ne 0 ]; then
        log "WARNING: Fault injection test exited with code ${test_rc}"
    fi
    log "  Test completed in ${duration}s (exit=${test_rc})"
    log "  Timing reports saved to: ${result_dir}"

    # -- Step 5: Cleanup --
    log "[${label}] Step 5: Cleaning up..."

    kill_router "${ROUTER_PORT}"
    kill "${router_launcher_pid}" 2>/dev/null || true
    wait "${router_launcher_pid}" 2>/dev/null || true
    cancel_slurm_job "${job_id}"
    kill_worker_ports 8000 "${NUM_WORKERS}" 9000

    log "[${label}] Experiment complete."
    log ""
    return "${test_rc}"
}

# ============================================================================
# Main
# ============================================================================

log "============================================================"
log "A/B Fault Tolerance Experiment"
log "============================================================"
log "  Timestamp:       ${TIMESTAMP}"
log "  Model:           ${MODEL_PATH}"
log "  Workers:         ${NUM_WORKERS}"
log "  HiCache size:    ${HICACHE_SIZE} GB"
log "  Quantization:    ${QUANTIZATION:-none}"
log "  Tasks:           ${TASK_LIMIT}"
log "  Processes:       ${TASK_PROCESSES}"
log "  Inject every:    ${INJECT_EVERY_N} tasks"
log "  Recover after:   ${RECOVER_AFTER}s"
log "  Output base:     ${OUTPUT_BASE_DIR}"
log "  Run WITH:        $([ ${RUN_WITH} -eq 1 ] && echo 'YES' || echo 'NO')"
log "  Run WITHOUT:     $([ ${RUN_WITHOUT} -eq 1 ] && echo 'YES' || echo 'NO')"
log "  Experiment log:  ${EXPERIMENT_LOG}"
log "============================================================"
log ""

OVERALL_RC=0

# -- Experiment A: WITH peer-replication --
if [ "${RUN_WITH}" -eq 1 ]; then
    run_experiment \
        "with_peer_replication" \
        "--enable-peer-replication" \
        "--enable-fault-tolerance" \
        "${OUTPUT_BASE_DIR}/with_full_backup" \
        || OVERALL_RC=1

    log "Cooldown 15s before next experiment..."
    sleep 15
fi

# -- Experiment B: WITHOUT peer-replication --
if [ "${RUN_WITHOUT}" -eq 1 ]; then
    run_experiment \
        "without_peer_replication" \
        "" \
        "" \
        "${OUTPUT_BASE_DIR}/without_backup" \
        || OVERALL_RC=1
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "============================================================"
log "A/B Experiment Complete"
log "============================================================"
log "  Full log: ${EXPERIMENT_LOG}"
if [ "${RUN_WITH}" -eq 1 ]; then
    log "  WITH results:    ${OUTPUT_BASE_DIR}/with_full_backup/"
fi
if [ "${RUN_WITHOUT}" -eq 1 ]; then
    log "  WITHOUT results: ${OUTPUT_BASE_DIR}/without_backup/"
fi
log "  Overall exit code: ${OVERALL_RC}"
log "============================================================"

exit "${OVERALL_RC}"
