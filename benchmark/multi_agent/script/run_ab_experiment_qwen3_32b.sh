#!/bin/bash
# 2-Group Fault Tolerance Experiment: Qwen3-32B on H20 GPUs (TP=2, 4 GPUs).
#
# Runs two sequential experiments:
#   1. flush_fault__no_backup   — Cache flush without backup
#   2. flush_fault__with_backup — Cache flush with backup (shows peer benefit)
#
# Fault injection: after 45 tasks, remove worker from router, abort all
# requests, flush KV cache, then re-add to router after a delay.
# Each fault targets a different worker.
# Processes start with a 5s stagger to spread tasks across agent stages.
#
# Port allocation (avoids conflict with 4B experiment):
#   Workers:     18000-18001
#   Peer cache:  19000-19001
#   Router:      40000
#
# Usage:
#   ./run_ab_experiment_qwen3_32b.sh [options]
#
# Options:
#   --task-limit N           Total tasks per experiment (default: 100)
#   --task-processes N       Parallel processes (default: 25)
#   --inject-every N         Fault every N tasks (default: 50)
#   --num-workers N          Number of workers (default: 2)
#   --model-path PATH        Model path (default: Qwen/Qwen3-32B)
#   --hicache-size N         HiCache size per worker in GB (default: 60)
#   --tp-size N              Tensor parallelism per worker (default: 2)
#   --output-dir DIR         Base output directory for results
#   -h, --help               Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MULTI_AGENT_DIR}/logs_qwen3_32b"
SLURM_DIR="${MULTI_AGENT_DIR}/slurm"

# ============================================================================
# Defaults
# ============================================================================
TASK_LIMIT=80
TASK_PROCESSES=12
INJECT_EVERY_N=45
INJECT_DELAY=10
TASK_STAGGER_DELAY=5
RECOVER_AFTER=60
NUM_WORKERS=2
WORKERS_PER_NODE=2
TP_SIZE=2
MODEL_PATH="Qwen/Qwen3-32B"
HICACHE_SIZE=60
QUANTIZATION=""
ROUTER_PORT=40000
PROMETHEUS_PORT=39000
ROUTER_HOST="127.0.0.1"
WORKER_BASE_PORT=18000
PEER_PORT_BASE=19000
OUTPUT_BASE_DIR="${MULTI_AGENT_DIR}/logs_qwen3_32b/agent_workspace/timing_reports_qwen3_32b"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

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
        --num-workers)       NUM_WORKERS="$2";       shift 2 ;;
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --hicache-size)      HICACHE_SIZE="$2";      shift 2 ;;
        --tp-size)           TP_SIZE="$2";           shift 2 ;;
        --quantization)      QUANTIZATION="$2";      shift 2 ;;
        --output-dir)        OUTPUT_BASE_DIR="$2";   shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_LOG="${LOG_DIR}/ab_experiment_32b_${TIMESTAMP}.log"
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
    local launcher_pid="${4:-}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if curl --noproxy '*' -fsS -m 2 "http://${host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ -n "${launcher_pid}" ] && ! kill -0 "${launcher_pid}" 2>/dev/null; then
            log "  Router launcher process (pid=${launcher_pid}) died"
            return 1
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
        log "  Killing router process pid=${pid} on port ${port} (SIGTERM)"
        kill "${pid}" 2>/dev/null || true
    done
    sleep 2
    pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
        | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' \
        | sort -u)
    for pid in ${pids}; do
        [ -z "${pid}" ] && continue
        log "  Force-killing router process pid=${pid} on port ${port} (SIGKILL)"
        kill -9 "${pid}" 2>/dev/null || true
    done
    sleep 1
}

wait_for_port_free() {
    local port="$1"
    local timeout_s="${2:-30}"
    local start_ts
    start_ts=$(date +%s)
    while ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; do
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_s}" ]; then
            log "  WARNING: port ${port} still in use after ${timeout_s}s"
            return 1
        fi
        sleep 1
    done
    return 0
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
    local base_port="${1:-18000}"
    local num_workers="${2:-2}"
    local peer_port_base="${3:-19000}"
    local tp="${TP_SIZE:-1}"
    for i in $(seq 0 $((num_workers - 1))); do
        local wport=$((base_port + i))
        local pids
        pids=$(ss -ltnp "sport = :${wport}" 2>/dev/null \
            | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' \
            | sort -u)
        for pid in ${pids}; do
            [ -z "${pid}" ] && continue
            log "  Killing leftover process pid=${pid} on port ${wport}"
            kill "${pid}" 2>/dev/null || true
        done
        for r in $(seq 0 $((tp - 1))); do
            local pport=$((peer_port_base + i * tp + r))
            pids=$(ss -ltnp "sport = :${pport}" 2>/dev/null \
                | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' \
                | sort -u)
            for pid in ${pids}; do
                [ -z "${pid}" ] && continue
                log "  Killing leftover process pid=${pid} on port ${pport}"
                kill "${pid}" 2>/dev/null || true
            done
        done
    done
    sleep 2
}

# Build the command template that the fault test script will use to restart a
# killed worker.  Placeholders {PORT} and {GPU_ID} are substituted at runtime.
build_worker_restart_cmd() {
    local venv_activate=""
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        venv_activate="source ${VENV_DIR}/bin/activate && "
    fi
    local quant_flag=""
    if [ -n "${QUANTIZATION}" ]; then
        quant_flag="--quantization ${QUANTIZATION} "
    fi
    echo "${venv_activate}PYTHONPATH=${REPO_ROOT}/python:${REPO_ROOT}:\${PYTHONPATH:-} exec python -m sglang.launch_server --model-path ${MODEL_PATH} --tp ${TP_SIZE} --host 0.0.0.0 --port {PORT} --enable-metrics --attention-backend triton --sampling-backend pytorch --tool-call-parser qwen --enable-cache-report --enable-hierarchical-cache --hicache-size ${HICACHE_SIZE} ${quant_flag}2>&1"
}

# ============================================================================
# Run one experiment
# ============================================================================
run_experiment() {
    local label="$1"         # e.g. "no_fault__no_backup"
    local peer_flag="$2"     # "--enable-peer-replication" or ""
    local ft_flag="$3"       # "--enable-fault-tolerance" or ""
    local fault_mode="$4"    # "none" or "kill"
    local result_dir="$5"    # output directory for timing reports
    local streaming="$6"     # "true" or "false"

    log "============================================================"
    log "EXPERIMENT: ${label}"
    log "============================================================"
    log "  peer-replication: $([ -n "${peer_flag}" ] && echo 'YES' || echo 'NO')"
    log "  fault-tolerance:  $([ -n "${ft_flag}" ] && echo 'YES' || echo 'NO')"
    log "  fault-mode:       ${fault_mode}"
    log "  tasks: ${TASK_LIMIT}, processes: ${TASK_PROCESSES}"
    log "  inject every: ${INJECT_EVERY_N}"
    log "  worker ports: ${WORKER_BASE_PORT}-$((WORKER_BASE_PORT + NUM_WORKERS - 1))"
    log "  peer ports: ${PEER_PORT_BASE}-$((PEER_PORT_BASE + NUM_WORKERS * TP_SIZE - 1))"
    log "  router port: ${ROUTER_PORT}"
    log "  results: ${result_dir}"
    log ""

    mkdir -p "${result_dir}"

    # -- Step 1: Submit SLURM job --
    log "[${label}] Step 1: Submitting SLURM job for workers..."
    kill_worker_ports "${WORKER_BASE_PORT}" "${NUM_WORKERS}" "${PEER_PORT_BASE}"

    local sbatch_args=(
        --nodes=1
        "${SLURM_DIR}/run_server_qwen3_32b.slurm"
        --num-workers "${NUM_WORKERS}"
        --workers-per-node "${WORKERS_PER_NODE}"
        --tp-size "${TP_SIZE}"
        --model-path "${MODEL_PATH}"
        --worker-base-port "${WORKER_BASE_PORT}"
        --peer-port-base "${PEER_PORT_BASE}"
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
    log "[${label}] Step 2a: Waiting for SLURM job ${job_id} to start running (timeout=86400s)..."

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
        if [ $(( $(date +%s) - wait_start )) -ge 86400 ]; then
            log "ERROR: SLURM job ${job_id} did not start within 86400s (state=${job_state})"
            cancel_slurm_job "${job_id}"
            return 1
        fi
        log "  Job ${job_id} state: ${job_state}, waiting..."
        sleep 10
    done

    local worker_urls_file="${LOG_DIR}/worker_urls.txt"
    rm -f "${worker_urls_file}"

    # -- Step 2b: Wait for workers to be ready (also check SLURM job health) --
    log "[${label}] Step 2b: Waiting for worker_urls.txt to appear (timeout=1200s)..."

    local wait2b_start
    wait2b_start=$(date +%s)
    local workers_ready=0
    while true; do
        if [ -f "${worker_urls_file}" ] && [ -s "${worker_urls_file}" ]; then
            workers_ready=1
            break
        fi
        if [ $(( $(date +%s) - wait2b_start )) -ge 1200 ]; then
            log "ERROR: Workers did not become ready within 1200s"
            cancel_slurm_job "${job_id}"
            return 1
        fi
        local jstate
        jstate=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        if [ "${jstate}" = "FAILED" ] || [ "${jstate}" = "COMPLETED" ] || [ "${jstate}" = "CANCELLED" ] || [ -z "${jstate}" ] || [ "${jstate}" = "UNKNOWN" ]; then
            local exit_info
            exit_info=$(sacct -j "${job_id}" --format=State,ExitCode -n 2>/dev/null | head -1 || echo "N/A")
            log "ERROR: SLURM job ${job_id} is no longer running (state=${jstate}, info=${exit_info})"
            log "  Check logs: ${LOG_DIR}/run_server_${job_id}.err"
            return 1
        fi
        sleep 5
    done

    local num_urls
    num_urls=$(wc -l < "${worker_urls_file}" | tr -d ' ')
    log "  Workers ready: ${num_urls} URLs in ${worker_urls_file}"

    # -- Step 3: Start router (with retry) --
    log "[${label}] Step 3: Starting router..."

    local router_args=(
        --worker-urls-file "${worker_urls_file}"
        --port "${ROUTER_PORT}"
        --prometheus-port "${PROMETHEUS_PORT}"
        --chunk-timeout-secs 60
        --prefill-timeout-secs 120
        --request-timeout-secs 600
        --no-auto-port
    )
    if [ -n "${ft_flag}" ]; then
        router_args+=( --enable-fault-tolerance )
    fi

    local router_launcher_pid=""
    local router_up=0
    for router_attempt in 1 2 3; do
        log "  Router start attempt ${router_attempt}/3..."

        kill_router "${ROUTER_PORT}"
        kill_router "${PROMETHEUS_PORT}"
        if ! wait_for_port_free "${ROUTER_PORT}" 30; then
            log "  WARNING: port ${ROUTER_PORT} not free, retrying..."
            continue
        fi

        (
            if [ -f "${VENV_DIR}/bin/activate" ]; then
                source "${VENV_DIR}/bin/activate"
            fi
            export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
            export ROUTER_URL_FILE="${LOG_DIR}/router_url.txt"
            export ROUTER_NODE_FILE="${LOG_DIR}/router_node.txt"
            exec "${SCRIPT_DIR}/run_router.sh" "${router_args[@]}"
        ) >> "${LOG_DIR}/router_${label}_${TIMESTAMP}.log" 2>&1 &
        router_launcher_pid=$!

        if wait_for_router "${ROUTER_HOST}" "${ROUTER_PORT}" 120 "${router_launcher_pid}"; then
            router_up=1
            break
        fi
        log "  Router attempt ${router_attempt} failed"
        kill "${router_launcher_pid}" 2>/dev/null || true
        wait "${router_launcher_pid}" 2>/dev/null || true
        sleep 5
    done

    if [ "${router_up}" -ne 1 ]; then
        log "ERROR: Router failed to start after 3 attempts"
        kill_router "${ROUTER_PORT}"
        [ -n "${router_launcher_pid}" ] && kill "${router_launcher_pid}" 2>/dev/null || true
        cancel_slurm_job "${job_id}"
        return 1
    fi
    log "  Router healthy at http://${ROUTER_HOST}:${ROUTER_PORT}"

    # -- Step 4: Run test --
    log "[${label}] Step 4: Running test (fault_mode=${fault_mode})..."

    local fault_log="${LOG_DIR}/fault_injection_${label}_${TIMESTAMP}.log"

    local exp_start_ts
    exp_start_ts=$(date +%s)

    local fault_env_vars=(
        TASK_LIMIT="${TASK_LIMIT}"
        TASK_PROCESSES="${TASK_PROCESSES}"
        WORKER_URLS_FILE="${worker_urls_file}"
        FAULT_SLURM_JOB_ID="${job_id}"
        ROUTER_PORT="${ROUTER_PORT}"
        HEAVY_SWARM_MODEL_NAME="openai/${MODEL_PATH}"
        HEAVY_SWARM_TIMING_OUTPUT_DIR="${result_dir}"
        LOG_FILE="${fault_log}"
        SUPPRESS_AGENT_OUTPUT="1"
        HEAVY_SWARM_VERBOSE="false"
        HEAVY_SWARM_AGENT_PRINTS_ON="false"
        HEAVY_SWARM_SHOW_DASHBOARD="false"
        HEAVY_SWARM_STREAMING="${streaming}"
        TASK_STAGGER_DELAY="${TASK_STAGGER_DELAY}"
    )

    if [ "${fault_mode}" = "kill" ]; then
        local restart_cmd
        restart_cmd=$(build_worker_restart_cmd)
        fault_env_vars+=(
            INJECT_EVERY_N="${INJECT_EVERY_N}"
            INJECT_DELAY="${INJECT_DELAY}"
            FAULT_MODE="kill"
            FAULT_ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}"
            WORKER_RESTART_CMD="${restart_cmd}"
        )
    elif [ "${fault_mode}" = "flush" ]; then
        fault_env_vars+=(
            INJECT_EVERY_N="${INJECT_EVERY_N}"
            INJECT_DELAY="${INJECT_DELAY}"
            RECOVER_AFTER="${RECOVER_AFTER}"
            FAULT_MODE="flush"
            FAULT_ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}"
        )
    else
        fault_env_vars+=(
            INJECT_EVERY_N="999999"
            INJECT_DELAY="0"
            RECOVER_AFTER="0"
        )
    fi

    local inject_every_arg="999999"
    if [ "${fault_mode}" = "kill" ] || [ "${fault_mode}" = "flush" ]; then
        inject_every_arg="${INJECT_EVERY_N}"
    fi

    set +e
    env "${fault_env_vars[@]}" \
        "${SCRIPT_DIR}/run_heavy_swarm_with_fault_injection.sh" \
            --task-limit "${TASK_LIMIT}" \
            --task-processes "${TASK_PROCESSES}" \
            --inject-every "${inject_every_arg}" \
            --worker-urls-file "${worker_urls_file}" \
            --slurm-job-id "${job_id}" \
            --log-file "${fault_log}"
    local test_rc=$?
    set -e

    local exp_end_ts
    exp_end_ts=$(date +%s)
    local duration=$(( exp_end_ts - exp_start_ts ))

    if [ "${test_rc}" -ne 0 ]; then
        log "WARNING: Test exited with code ${test_rc}"
    fi
    log "  Test completed in ${duration}s (exit=${test_rc})"
    log "  Timing reports saved to: ${result_dir}"

    # -- Step 4b: Dump per-request retry metrics from router --
    log "[${label}] Step 4b: Fetching per-request retry metrics..."
    local stats_json
    stats_json=$(curl -s "http://${ROUTER_HOST}:${ROUTER_PORT}/stats" 2>/dev/null || echo "{}")
    if echo "${stats_json}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ns = d.get('normal_request_stats', {})
rs = d.get('retry_request_stats', {})
print('=== Per-Request Metrics ===')
if ns:
    print(f'  Normal requests:    {ns[\"normal_requests\"]}')
    print(f'  Avg latency:        {ns[\"avg_latency_ms\"]:.0f} ms')
    print(f'  P50 latency:        {ns[\"p50_latency_ms\"]:.0f} ms')
    print(f'  P99 latency:        {ns[\"p99_latency_ms\"]:.0f} ms')
else:
    print('  Normal requests:    (no data)')
if rs:
    print(f'  Retried requests:   {rs[\"retried_requests\"]}')
    print(f'  Avg tokens before fault: {rs[\"avg_tokens_before_fault\"]}')
    print(f'  Avg tokens total:   {rs[\"avg_tokens_total\"]}')
    print(f'  Avg total latency:  {rs[\"avg_total_ms\"]:.0f} ms')
    print(f'  Avg retry latency:  {rs[\"avg_retry_ms\"]:.0f} ms  (from fault detection to completion)')
    print('  --- Individual retried requests ---')
    for i, r in enumerate(rs.get('retry_records', []), 1):
        print(f'    [{i}] user={r.get(\"user\",\"?\")}, mode={r.get(\"mode\",\"?\")}, '
              f'tokens_before={r[\"tokens_before_fault\"]}, '
              f'tokens_after={r[\"tokens_total\"]-r[\"tokens_before_fault\"]}, '
              f'total={r[\"total_ms\"]:.0f}ms, retry={r[\"retry_ms\"]:.0f}ms, '
              f'{r[\"from_worker\"]} -> {r[\"to_worker\"]}')
else:
    print('  Retried requests:   0')
print('=== End Per-Request Metrics ===')
" 2>/dev/null; then
        log "  Retry metrics saved to: ${result_dir}/retry_metrics.json"
        echo "${stats_json}" > "${result_dir}/retry_metrics.json"
    else
        log "  WARNING: Could not fetch retry metrics from router"
    fi

    # -- Step 5: Cleanup --
    log "[${label}] Step 5: Cleaning up..."

    kill_router "${ROUTER_PORT}"
    kill_router "${PROMETHEUS_PORT}"
    kill "${router_launcher_pid}" 2>/dev/null || true
    wait "${router_launcher_pid}" 2>/dev/null || true
    cancel_slurm_job "${job_id}"
    kill_worker_ports "${WORKER_BASE_PORT}" "${NUM_WORKERS}" "${PEER_PORT_BASE}"

    log "[${label}] Experiment complete."
    log ""
    return "${test_rc}"
}

# ============================================================================
# Main
# ============================================================================

log "============================================================"
log "4-Group Fault Tolerance Experiment — Qwen3-32B on H20 (FP8)"
log "============================================================"
log "  Timestamp:       ${TIMESTAMP}"
log "  Model:           ${MODEL_PATH}"
log "  Workers:         ${NUM_WORKERS}"
log "  Worker ports:    ${WORKER_BASE_PORT}-$((WORKER_BASE_PORT + NUM_WORKERS - 1))"
log "  Peer ports:      ${PEER_PORT_BASE}-$((PEER_PORT_BASE + NUM_WORKERS - 1))"
log "  Router port:     ${ROUTER_PORT}"
log "  HiCache size:    ${HICACHE_SIZE} GB"
log "  Quantization:    ${QUANTIZATION:-none}"
log "  Tasks:           ${TASK_LIMIT}"
log "  Processes:       ${TASK_PROCESSES}"
log "  Inject every:    ${INJECT_EVERY_N} tasks (flush mode)"
log "  Stagger delay:   ${TASK_STAGGER_DELAY}s between processes"
log "  TP size:         ${TP_SIZE}"
log "  Output base:     ${OUTPUT_BASE_DIR}"
log "  Experiment log:  ${EXPERIMENT_LOG}"
log "============================================================"
log ""

OVERALL_RC=0

# -- Group 1: Flush fault, no backup (cache wiped, no peer recovery) --
# fault-tolerance enabled so the router retries failed requests
run_experiment \
    "flush_fault__no_backup" \
    "" \
    "--enable-fault-tolerance" \
    "flush" \
    "${OUTPUT_BASE_DIR}/flush_fault__no_backup" \
    "false" \
    || OVERALL_RC=1

log "Cooldown 15s before next experiment..."
sleep 15

# -- Group 2: Flush fault, with backup (cache wiped, peer recovery) --
run_experiment \
    "flush_fault__with_backup" \
    "--enable-peer-replication" \
    "--enable-fault-tolerance" \
    "flush" \
    "${OUTPUT_BASE_DIR}/flush_fault__with_backup" \
    "true" \
    || OVERALL_RC=1

# ============================================================================
# Summary
# ============================================================================
log ""
log "============================================================"
log "2-Group Experiment Complete — Qwen3-32B"
log "============================================================"
log "  Full log: ${EXPERIMENT_LOG}"
log "  Group 1: ${OUTPUT_BASE_DIR}/flush_fault__no_backup/"
log "  Group 2: ${OUTPUT_BASE_DIR}/flush_fault__with_backup/"
log "  Overall exit code: ${OVERALL_RC}"
log "============================================================"

exit "${OVERALL_RC}"
