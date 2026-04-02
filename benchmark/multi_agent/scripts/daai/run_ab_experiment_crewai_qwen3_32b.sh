#!/bin/bash
# CrewAI PP2+DP2 fault-tolerance experiment on A100.
#
# Script Overview:
# Run CrewAI multi-agent A/B experiment on A100 GPU cluster, verifying Qwen3-32B model's fault-tolerance capability under PP2+DP2 (pipeline parallel x2 + data parallel x2) deployment.
#
# Four control groups:
#   1. no_backup              —— No backup, fail immediately on fault
#   2. with_backup_wait_complete —— With backup, wait for prefetch completion before serving
#   3. with_backup_best_effort   —— With backup, do best effort prefetch (not guaranteed to complete)
#   4. with_backup_timeout       —— With backup, give up prefetch on timeout

# Exit on error or pipe failure
set -euo pipefail

# ────────────────────────────────────────────────────────────
# Directory and Path Configuration
# ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${MULTI_AGENT_DIR}/logs/qwen3_32b_a100"
RESULTS_DIR="${MULTI_AGENT_DIR}/results/crewai_ab_qwen3_32b_a100"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
ROOT_VENV_DIR="${REPO_ROOT}/.venv"
CREWAI_VENV_DIR="${MULTI_AGENT_DIR}/src/application/crewai/.venv"
CREWAI_FAULT_SCRIPT="${SCRIPT_DIR}/run_crewai_fault_injection_qwen3_32b.sh"
CREWAI_APP_DIR="${MULTI_AGENT_DIR}/src/application/crewai"
OUTPUT_BASE_DIR="${RESULTS_DIR}"

# ────────────────────────────────────────────────────────────
# Runtime State Variables
# ────────────────────────────────────────────────────────────

# Current active SLURM Job ID (for signal handling)
_ACTIVE_SLURM_JOB=""

# ────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────

# Group related
GROUP_TO_RUN="all"
VALID_GROUPS=(
    "no_backup"
    "with_backup_wait_complete"
    "with_backup_best_effort"
    "with_backup_timeout"
)

# Server related
MODEL_PATH="Qwen/Qwen3-32B"
DP_SIZE=2
PP_SIZE=2
TP_SIZE=1
NNODES=2
SERVER_PORT_BASE=34000
PEER_PORT_BASE=35000
HICACHE_SIZE=40
MIN_HICACHE_SIZE=40
QUANTIZATION=""
MODEL_LOCAL_ROOT="${MODEL_LOCAL_ROOT:-/tmp/${USER}/sglang-model-cache}"

# Fault injection related
INJECT_AFTER_JOB="1"
INJECT_AFTER_TASK=""
INJECT_DELAY=10
FAULT_DP_RANK="0"
FAULT_PP_RANK="0"
FAULT_TP_RANK="0"

# CrewAI related
JOB_LIMIT=6
APP_WORKERS=2
JOBS_CSV="${MULTI_AGENT_DIR}/src/application/crewai/topics.csv"
DEFAULT_YEAR="2025"
CREWAI_SHORT_MAX_TOKENS="${CREWAI_SHORT_MAX_TOKENS:-128}"
CREWAI_LONG_MAX_TOKENS="${CREWAI_LONG_MAX_TOKENS:-384}"
CREWAI_FORCE_FULL_BUDGET="${CREWAI_FORCE_FULL_BUDGET:-0}"
CREWAI_IGNORE_EOS="${CREWAI_IGNORE_EOS:-0}"
CREWAI_MIN_TOKENS="${CREWAI_MIN_TOKENS:-0}"


# Usage
usage() {
    cat <<'EOF'
Usage:
  ./run_ab_experiment_crewai_qwen3_32b.sh [options]

Options:
  --job-limit N              Total jobs per experiment (default: 6)
  --app-workers N            CrewAI worker processes (default: 2)
  --inject-after-job N[,M]   Inject after job completion count(s)
  --inject-after-task N[,M]  Inject after task completion count(s)
  --inject-delay SECS        Delay after trigger before fault injection
  --dp-size N                Replica count / node count (default: 2)
  --pp-size N                Pipeline parallel size per replica (default: 2)
  --tp-size N                Tensor parallel size per stage (default: 1)
  --fault-dp-rank N          Replica rank to kill (default: 0)
  --fault-pp-rank N          Pipeline stage rank to kill (default: 0)
  --fault-tp-rank N          Tensor rank to kill (default: 0)
  --model-path PATH          Model path (default: Qwen/Qwen3-32B)
  --model-local-root DIR     Local model cache root on the target node
  --hicache-size N           HiCache size per replica in GB (default: 40)
  --quantization Q           Quantization method (default: none)
  --jobs-csv PATH            Job CSV file for CrewAI
  --default-year YEAR        Default year for missing CSV year
  --output-dir DIR           Base output directory for results
  --group NAME[,NAME...]     all | no_backup | with_backup_wait_complete |
                             with_backup_best_effort | with_backup_timeout
  -h, --help                 Show this help
EOF
}

# ────────────────────────────────────────────────────────────
# Command line parameter parsing
# ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-limit) JOB_LIMIT="$2"; shift 2 ;;
        --app-workers) APP_WORKERS="$2"; shift 2 ;;
        --inject-after-job) INJECT_AFTER_JOB="$2"; INJECT_AFTER_TASK=""; shift 2 ;;
        --inject-after-task) INJECT_AFTER_TASK="$2"; INJECT_AFTER_JOB=""; shift 2 ;;
        --inject-delay) INJECT_DELAY="$2"; shift 2 ;;
        --dp-size) DP_SIZE="$2"; shift 2 ;;
        --pp-size) PP_SIZE="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --fault-dp-rank) FAULT_DP_RANK="$2"; shift 2 ;;
        --fault-pp-rank) FAULT_PP_RANK="$2"; shift 2 ;;
        --fault-tp-rank) FAULT_TP_RANK="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-local-root) MODEL_LOCAL_ROOT="$2"; shift 2 ;;
        --hicache-size) HICACHE_SIZE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --jobs-csv) JOBS_CSV="$2"; shift 2 ;;
        --default-year) DEFAULT_YEAR="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE_DIR="$2"; shift 2 ;;
        --group)
            GROUP_TO_RUN="$2"
            shift 2
            while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                GROUP_TO_RUN="${GROUP_TO_RUN},$1"
                shift
            done
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# ────────────────────────────────────────────────────────────
# HiCache size safety check
# ────────────────────────────────────────────────────────────

if [ "${HICACHE_SIZE}" -gt 0 ] && [ "${HICACHE_SIZE}" -lt "${MIN_HICACHE_SIZE}" ]; then
    echo "WARNING: hicache-size=${HICACHE_SIZE} GB is too small for Qwen3-32B DP=2 PP=2 on A100; bumping to ${MIN_HICACHE_SIZE} GB." >&2
    HICACHE_SIZE="${MIN_HICACHE_SIZE}"
fi

# ────────────────────────────────────────────────────────────
# Signal handling
# ────────────────────────────────────────────────────────────

on_signal() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Signal caught - cleaning up..."
    if [ -n "${_ACTIVE_SLURM_JOB}" ] && squeue -j "${_ACTIVE_SLURM_JOB}" -h >/dev/null 2>&1; then
        scancel "${_ACTIVE_SLURM_JOB}" 2>/dev/null || true
    fi
    kill -- -$$ 2>/dev/null || true
    exit 130
}

trap 'on_signal' INT TERM

# ────────────────────────────────────────────────────────────
# Group check
# ────────────────────────────────────────────────────────────

group_is_selected() {
    local target="$1"
    local selected
    for selected in "${SELECTED_GROUPS[@]}"; do
        if [[ "${selected}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

# ────────────────────────────────────────────────────────────
# Parse --group parameter
# ────────────────────────────────────────────────────────────

SELECTED_GROUPS=()
if [[ "${GROUP_TO_RUN}" == "all" ]]; then
    SELECTED_GROUPS=("${VALID_GROUPS[@]}")
else
    IFS=',' read -r -a raw_groups <<< "${GROUP_TO_RUN}"
    for raw_group in "${raw_groups[@]}"; do
        group_name="$(echo "${raw_group}" | xargs)"
        [[ -z "${group_name}" ]] && continue
        valid=0
        for candidate in "${VALID_GROUPS[@]}"; do
            if [[ "${group_name}" == "${candidate}" ]]; then
                valid=1
                break
            fi
        done
        if [[ "${valid}" -ne 1 ]]; then
            echo "Invalid --group entry: ${group_name}" >&2
            usage >&2
            exit 1
        fi
        if ! group_is_selected "${group_name}"; then
            SELECTED_GROUPS+=("${group_name}")
        fi
    done
    if [[ "${#SELECTED_GROUPS[@]}" -eq 0 ]]; then
        echo "No valid groups selected via --group: ${GROUP_TO_RUN}" >&2
        exit 1
    fi
fi

# ────────────────────────────────────────────────────────────
# Initialize log file
# ────────────────────────────────────────────────────────────

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE_DIR}"
EXPERIMENT_LOG="${LOG_DIR}/ab_experiment_crewai_32b_a100_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${EXPERIMENT_LOG}"
}

# ────────────────────────────────────────────────────────────
# Cancel SLURM job
# ────────────────────────────────────────────────────────────

cancel_slurm_job() {
    local job_id="$1"
    if [ -n "${job_id}" ] && squeue -j "${job_id}" -h >/dev/null 2>&1; then
        log "  Cancelling SLURM job ${job_id}"
        scancel "${job_id}" 2>/dev/null || true
        sleep 5
    fi
}

# ────────────────────────────────────────────────────────────
# Wait for server artifacts
# ────────────────────────────────────────────────────────────

wait_for_server_artifacts() {
    local job_id="$1"
    local server_url_file="$2"
    local stage_manifest_file="$3"
    local timeout_s="${4:-1200}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if [ -s "${server_url_file}" ] && [ -s "${stage_manifest_file}" ]; then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_s}" ]; then
            log "ERROR: Server artifacts did not become ready within ${timeout_s}s"
            cancel_slurm_job "${job_id}"
            return 1
        fi
        local jstate
        jstate=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        if [ "${jstate}" = "FAILED" ] || [ "${jstate}" = "COMPLETED" ] || [ "${jstate}" = "CANCELLED" ] || [ -z "${jstate}" ] || [ "${jstate}" = "UNKNOWN" ]; then
            log "ERROR: SLURM job ${job_id} is no longer running (state=${jstate})"
            return 1
        fi
        sleep 5
    done
}

# ────────────────────────────────────────────────────────────
# Run a Python script inside a venv (for plotting)
# ────────────────────────────────────────────────────────────

run_in_venv() {
    local venv_dir="$1"; shift
    local script="$1"; shift
    [ -f "${script}" ] && [ -f "${venv_dir}/bin/activate" ] || return 1
    set +e
    ( source "${venv_dir}/bin/activate"; exec python "${script}" "$@" ) >> "${EXPERIMENT_LOG}" 2>&1
    local rc=$?; set -e; return "${rc}"
}

# ────────────────────────────────────────────────────────────
# Run one experiment group
# Parameters:
#   $1 = label              —— group name
#   $2 = peer_enabled       —— whether to enable Peer replication (1=yes, 0=no)
#   $3 = result_dir         —— result directory for this group
#   $4 = prefetch_policy    —— wait_complete / best_effort / timeout
#   $5 = hicache_extra_config —— example: '{"prefetch_timeout_base":5,...}'
# ────────────────────────────────────────────────────────────

run_experiment() {
    local label="$1"
    local peer_enabled="$2"
    local result_dir="$3"
    local prefetch_policy="${4:-}"
    local hicache_extra_config="${5:-}"
    local figures_dir="${result_dir}/figures"
    mkdir -p "${result_dir}" "${figures_dir}"
    log "============================================================"
    log "EXPERIMENT: ${label}"
    log "============================================================"
    log "  peer-replication:  $([ "${peer_enabled}" = "1" ] && echo 'YES' || echo 'NO')"
    log "  prefetch-policy:   ${prefetch_policy:-auto}"
    log "  results:           ${result_dir}"
    [ -n "${hicache_extra_config}" ] && log "  hicache-extra:     ${hicache_extra_config}"

    local server_url_file="${LOG_DIR}/server_url.txt"
    local stage_manifest_file="${LOG_DIR}/stage_manifest.json"
    rm -f "${server_url_file}" "${stage_manifest_file}"

    # Step 1：Submit server job using sbatch 

    log "[${label}] Step 1: Submitting SLURM job..."
    local sbatch_args=(
        --nodes="${NNODES}"
        "${SCRIPT_DIR}/run_server_qwen3_32b.slurm"
        --dp-size "${DP_SIZE}"
        --pp-size "${PP_SIZE}"
        --tp-size "${TP_SIZE}"
        --model-path "${MODEL_PATH}"
        --model-local-root "${MODEL_LOCAL_ROOT}"
        --venv-local-root ""
        --server-port-base "${SERVER_PORT_BASE}"
        --peer-port-base "${PEER_PORT_BASE}"
        --enable-hicache
        --hicache-size "${HICACHE_SIZE}"
    )
    if [ "${peer_enabled}" = "1" ]; then
        sbatch_args+=(--enable-peer-replication)
    fi
    if [ -n "${prefetch_policy}" ]; then
        sbatch_args+=(--prefetch-policy "${prefetch_policy}")
    fi
    if [ -n "${hicache_extra_config}" ]; then
        sbatch_args+=(--hicache-extra-config "${hicache_extra_config}")
    fi
    if [ -n "${QUANTIZATION}" ]; then
        sbatch_args+=(--quantization "${QUANTIZATION}")
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
    _ACTIVE_SLURM_JOB="${job_id}"

    # Step 2：Wait for SLURM job to enter RUNNING state

    log "[${label}] Step 2: Waiting for SLURM job ${job_id} to start..."
    local wait_start
    wait_start=$(date +%s)
    while true; do
        local job_state
        job_state=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        if [ "${job_state}" = "RUNNING" ]; then
            break
        fi
        if [ $(( $(date +%s) - wait_start )) -ge 86400 ]; then
            log "ERROR: SLURM job ${job_id} did not start within 86400s"
            cancel_slurm_job "${job_id}"
            return 1
        fi
        sleep 10
    done

    # Step 3：Wait for head server URL and stage manifest file to be ready

    log "[${label}] Step 3: Waiting for head server URL and stage manifest..."
    if ! wait_for_server_artifacts "${job_id}" "${server_url_file}" "${stage_manifest_file}" 1200; then
        cancel_slurm_job "${job_id}"
        return 1
    fi

    local head_server_url
    head_server_url="$(<"${server_url_file}")"
    cp "${server_url_file}" "${result_dir}/server_url.txt"
    cp "${stage_manifest_file}" "${result_dir}/stage_manifest.json"
    printf '%s\n' "${job_id}" > "${result_dir}/slurm_job_id.txt"

    # Step 4：Run CrewAI workload and inject fault

    log "[${label}] Step 4: Running CrewAI fault injection against ${head_server_url}..."
    local run_trace_file="${result_dir}/trace_log.json"

    local -a inject_trigger_args=()
    if [ -n "${INJECT_AFTER_TASK}" ]; then
        inject_trigger_args+=(--inject-after-task "${INJECT_AFTER_TASK}")
    else
        inject_trigger_args+=(--inject-after-job "${INJECT_AFTER_JOB}")
    fi

    set +e
    CREWAI_ENABLE_STREAM=0 \
    CREWAI_SHORT_MAX_TOKENS="${CREWAI_SHORT_MAX_TOKENS}" \
    CREWAI_LONG_MAX_TOKENS="${CREWAI_LONG_MAX_TOKENS}" \
    CREWAI_FORCE_FULL_BUDGET="${CREWAI_FORCE_FULL_BUDGET}" \
    CREWAI_IGNORE_EOS="${CREWAI_IGNORE_EOS}" \
    CREWAI_MIN_TOKENS="${CREWAI_MIN_TOKENS}" \
    "${CREWAI_FAULT_SCRIPT}" \
        --server-url "${head_server_url}" \
        --stage-manifest "${stage_manifest_file}" \
        --slurm-job-id "${job_id}" \
        --model-path "${MODEL_PATH}" \
        --jobs-csv "${JOBS_CSV}" \
        --job-limit "${JOB_LIMIT}" \
        --workers "${APP_WORKERS}" \
        --default-year "${DEFAULT_YEAR}" \
        "${inject_trigger_args[@]}" \
        --inject-delay "${INJECT_DELAY}" \
        --fault-dp-rank "${FAULT_DP_RANK}" \
        --fault-pp-rank "${FAULT_PP_RANK}" \
        --fault-tp-rank "${FAULT_TP_RANK}" \
        --output-dir "${result_dir}" \
        --trace-file "${run_trace_file}"
    local test_rc=$?
    set -e
    if [ "${test_rc}" -ne 0 ]; then
        log "WARNING: CrewAI run exited with code ${test_rc}"
    fi

    # Step 5: Post-processing: generate figures

    if [ -f "${run_trace_file}" ]; then
        run_in_venv "${CREWAI_VENV_DIR}" "${CREWAI_APP_DIR}/plot_trace_log_profile.py" \
            --input "${run_trace_file}" --output "${figures_dir}/trace_log_profile.png" || true
        if [ -f "${result_dir}/internal_failover_metrics.json" ]; then
            run_in_venv "${CREWAI_VENV_DIR}" "${CREWAI_APP_DIR}/plot_peer_cache_hits.py" \
                --failover-metrics "${result_dir}/internal_failover_metrics.json" \
                --output "${figures_dir}/peer_cache_hit_profile.png" || true
        fi
    fi

    # Step 6：Clean up SLURM job

    log "[${label}] Step 6: Cleaning up..."
    cancel_slurm_job "${job_id}"
    _ACTIVE_SLURM_JOB=""

    log "[${label}] Experiment complete."
    log ""
    return "${test_rc}"
}

# ════════════════════════════════════════════════════════════
# Print global configuration summary
# ════════════════════════════════════════════════════════════

log "============================================================"
log "CrewAI PP2+DP2 Fault Tolerance Experiment - Qwen3-32B A100"
log "============================================================"
log "  Timestamp:          ${TIMESTAMP}"
log "  Model:              ${MODEL_PATH}"
log "  Job limit:          ${JOB_LIMIT}"
log "  App workers:        ${APP_WORKERS}"
log "  CrewAI tokens:      short=${CREWAI_SHORT_MAX_TOKENS}, long=${CREWAI_LONG_MAX_TOKENS}"
log "  Force full budget:  ${CREWAI_FORCE_FULL_BUDGET}"
log "  Ignore EOS:         ${CREWAI_IGNORE_EOS}"
log "  Min tokens:         ${CREWAI_MIN_TOKENS}"
log "  Topology:           DP=${DP_SIZE}, PP=${PP_SIZE}, TP=${TP_SIZE}"
log "  Nodes:              ${NNODES}"
log "  Fault target:       dp=${FAULT_DP_RANK}, pp=${FAULT_PP_RANK}, tp=${FAULT_TP_RANK}"
log "  Model local root:   ${MODEL_LOCAL_ROOT}"
log "  HiCache size:       ${HICACHE_SIZE} GB"
log "  Quantization:       ${QUANTIZATION:-none}"
log "  Jobs CSV:           ${JOBS_CSV}"
if [ -n "${INJECT_AFTER_TASK}" ]; then
    log "  Inject after task:  ${INJECT_AFTER_TASK}"
else
    log "  Inject after job:   ${INJECT_AFTER_JOB}"
fi
log "  Inject delay:       ${INJECT_DELAY}s"
log "  Output base:        ${OUTPUT_BASE_DIR}"
log "  Groups to run:      ${SELECTED_GROUPS[*]}"
log "  Experiment log:     ${EXPERIMENT_LOG}"
log "============================================================"
log ""

# Global exit code: if any group fails, set to 1
OVERALL_RC=0

# ════════════════════════════════════════════════════════════
# Main loop: run each experiment group
# ════════════════════════════════════════════════════════════

for idx in "${!SELECTED_GROUPS[@]}"; do
    group="${SELECTED_GROUPS[$idx]}"
    if [[ "${idx}" -gt 0 ]]; then
        log "Cooldown 15s before next experiment..."
        sleep 15
    fi
    case "${group}" in
        no_backup)
            run_experiment \
                "no_backup" \
                "0" \
                "${OUTPUT_BASE_DIR}/no_backup" \
                || OVERALL_RC=1
            ;;
        with_backup_wait_complete)
            run_experiment \
                "with_backup_wait_complete" \
                "1" \
                "${OUTPUT_BASE_DIR}/with_backup_wait_complete" \
                "wait_complete" \
                || OVERALL_RC=1
            ;;
        with_backup_best_effort)
            run_experiment \
                "with_backup_best_effort" \
                "1" \
                "${OUTPUT_BASE_DIR}/with_backup_best_effort" \
                "best_effort" \
                || OVERALL_RC=1
            ;;
        with_backup_timeout)
            run_experiment \
                "with_backup_timeout" \
                "1" \
                "${OUTPUT_BASE_DIR}/with_backup_timeout" \
                "timeout" \
                '{"prefetch_timeout_base":5,"prefetch_timeout_per_ki_token":0}' \
                || OVERALL_RC=1
            ;;
    esac
done

# ════════════════════════════════════════════════════════════
# Plot A/B comparison figures
# ════════════════════════════════════════════════════════════

log ""
log "Generating cross-experiment comparison figures..."

BASELINE_DIR="${OUTPUT_BASE_DIR}/no_backup"
declare -A TREATMENT_DIRS=(
    ["wait_complete"]="${OUTPUT_BASE_DIR}/with_backup_wait_complete"
    ["best_effort"]="${OUTPUT_BASE_DIR}/with_backup_best_effort"
    ["timeout"]="${OUTPUT_BASE_DIR}/with_backup_timeout"
)
declare -A TREATMENT_LABELS=(
    ["wait_complete"]="With Backup (wait-complete)"
    ["best_effort"]="With Backup (best-effort)"
    ["timeout"]="With Backup (timeout)"
)

AB_PLOT="${CREWAI_APP_DIR}/plot_ab_comparison.py"
for key in wait_complete best_effort timeout; do
    treatment_dir="${TREATMENT_DIRS[$key]}"
    if [ -d "${BASELINE_DIR}" ] && [ -d "${treatment_dir}" ]; then
        for ext in pdf png; do
            run_in_venv "${ROOT_VENV_DIR}" "${AB_PLOT}" \
                --baseline "${BASELINE_DIR}" --treatment "${treatment_dir}" \
                --baseline-label "No Backup" --treatment-label "${TREATMENT_LABELS[$key]}" \
                --output "${OUTPUT_BASE_DIR}/ab_comparison__no_backup_vs_${key}.${ext}" || true
        done
    fi
done

# ════════════════════════════════════════════════════════════
# Print summary information
# ════════════════════════════════════════════════════════════

log ""
log "============================================================"
log "Experiment Complete - CrewAI Qwen3-32B A100"
log "============================================================"
log "  Full log: ${EXPERIMENT_LOG}"
for group in "${SELECTED_GROUPS[@]}"; do
    log "  ${group}: ${OUTPUT_BASE_DIR}/${group}"
done
log "  Overall exit code: ${OVERALL_RC}"
log "============================================================"

exit "${OVERALL_RC}"
