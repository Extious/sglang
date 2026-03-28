#!/bin/bash
# CrewAI PP2+DP2 fault-tolerance experiment on GPUHome.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MULTI_AGENT_DIR}/logs_qwen3_8b_gpuhome"
RESULTS_DIR="${MULTI_AGENT_DIR}/results/crewai_ab_qwen3_8b_gpuhome"
SLURM_DIR="${MULTI_AGENT_DIR}/slurm"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
ROOT_VENV_DIR="${REPO_ROOT}/.venv"
CREWAI_VENV_DIR="${MULTI_AGENT_DIR}/src/application/crewai/.venv"
CREWAI_FAULT_SCRIPT="${SCRIPT_DIR}/run_crewai_fault_injection_gpuhome.sh"
CREWAI_PLOT_SCRIPT="${MULTI_AGENT_DIR}/src/application/crewai/plot_trace_log_profile.py"
CREWAI_PEER_CACHE_PLOT_SCRIPT="${MULTI_AGENT_DIR}/src/application/crewai/plot_peer_cache_hits.py"
CREWAI_AB_COMPARISON_PLOT_SCRIPT="${MULTI_AGENT_DIR}/src/application/crewai/plot_ab_comparison.py"

_ACTIVE_SLURM_JOB=""

JOB_LIMIT=6
APP_WORKERS=2
INJECT_AFTER_JOB="1"
INJECT_AFTER_TASK=""
INJECT_DELAY=10
RECOVER_AFTER=""
MODEL_PATH="Qwen/Qwen3-8B"
HICACHE_SIZE=16
QUANTIZATION=""
JOBS_CSV="${MULTI_AGENT_DIR}/src/application/crewai/topics.csv"
DEFAULT_YEAR="2025"
SERVER_PORT_BASE=28000
PEER_PORT_BASE=29000
DP_SIZE=2
PP_SIZE=2
TP_SIZE=1
NNODES=2
FAULT_DP_RANK="0"
FAULT_PP_RANK="0"
FAULT_TP_RANK="0"
OUTPUT_BASE_DIR="${RESULTS_DIR}"
GROUP_TO_RUN="all"
VALID_GROUPS=(
    "no_backup"
    "with_backup_wait_complete"
    "with_backup_best_effort"
    "with_backup_timeout"
)

usage() {
    cat <<'EOF'
Usage:
  ./run_ab_experiment_crewai_qwen3_8b_gpuhome.sh [options]

Options:
  --job-limit N              Total jobs per experiment (default: 6)
  --app-workers N            CrewAI worker processes (default: 2)
  --inject-after-job N[,M]   Inject after job completion count(s)
  --inject-after-task N[,M]  Inject after task completion count(s)
  --inject-delay SECS        Delay after trigger before fault injection
  --recover-after SECS       Deprecated compatibility option, ignored
  --dp-size N                Replica count / node count (default: 2)
  --pp-size N                Pipeline parallel size per replica (default: 2)
  --tp-size N                Tensor parallel size per stage (default: 1)
  --fault-dp-rank N          Replica rank to kill (default: 0)
  --fault-pp-rank N          Pipeline stage rank to kill (default: 0)
  --fault-tp-rank N          Tensor rank to kill (default: 0)
  --model-path PATH          Model path (default: Qwen/Qwen3-8B)
  --hicache-size N           HiCache size per replica in GB (default: 16)
  --quantization Q           Quantization method (default: none)
  --jobs-csv PATH            Job CSV file for CrewAI
  --default-year YEAR        Default year for missing CSV year
  --output-dir DIR           Base output directory for results
  --group NAME[,NAME...]     all | no_backup | with_backup_wait_complete |
                             with_backup_best_effort | with_backup_timeout
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-limit|--topic-limit) JOB_LIMIT="$2"; shift 2 ;;
        --app-workers) APP_WORKERS="$2"; shift 2 ;;
        --inject-after-job|--inject-after-topic) INJECT_AFTER_JOB="$2"; INJECT_AFTER_TASK=""; shift 2 ;;
        --inject-after-task) INJECT_AFTER_TASK="$2"; INJECT_AFTER_JOB=""; shift 2 ;;
        --inject-delay) INJECT_DELAY="$2"; shift 2 ;;
        --recover-after) RECOVER_AFTER="$2"; shift 2 ;;
        --dp-size) DP_SIZE="$2"; shift 2 ;;
        --pp-size) PP_SIZE="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --fault-dp-rank) FAULT_DP_RANK="$2"; shift 2 ;;
        --fault-pp-rank) FAULT_PP_RANK="$2"; shift 2 ;;
        --fault-tp-rank) FAULT_TP_RANK="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --hicache-size) HICACHE_SIZE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --jobs-csv|--topics-csv) JOBS_CSV="$2"; shift 2 ;;
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

on_signal() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Signal caught — cleaning up..."
    if [ -n "${_ACTIVE_SLURM_JOB}" ] && squeue -j "${_ACTIVE_SLURM_JOB}" -h >/dev/null 2>&1; then
        scancel "${_ACTIVE_SLURM_JOB}" 2>/dev/null || true
    fi
    kill -- -$$ 2>/dev/null || true
    exit 130
}

trap 'on_signal' INT TERM

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE_DIR}"
EXPERIMENT_LOG="${LOG_DIR}/ab_experiment_crewai_8b_gpuhome_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${EXPERIMENT_LOG}"
}

cancel_slurm_job() {
    local job_id="$1"
    if [ -n "${job_id}" ] && squeue -j "${job_id}" -h >/dev/null 2>&1; then
        log "  Cancelling SLURM job ${job_id}"
        scancel "${job_id}" 2>/dev/null || true
        sleep 5
    fi
}

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

write_group_summary() {
    local trace_json="$1"
    local summary_json="$2"
    python3 - "${trace_json}" "${summary_json}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
records = json.loads(trace_path.read_text(encoding="utf-8"))

wall_times = [item.get("summary", {}).get("wall_time_s") for item in records if item.get("status") == "success"]
wall_times = [float(x) for x in wall_times if x not in (None, "")]
payload = {
    "jobs_total": len(records),
    "jobs_success": sum(1 for item in records if item.get("status") == "success"),
    "jobs_failed": sum(1 for item in records if item.get("status") != "success"),
    "avg_wall_time_s": round(statistics.mean(wall_times), 3) if wall_times else None,
    "p50_wall_time_s": round(statistics.median(wall_times), 3) if wall_times else None,
    "max_wall_time_s": round(max(wall_times), 3) if wall_times else None,
}
summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

plot_group_profile() {
    local trace_json="$1"
    local output_png="$2"
    if [ ! -f "${trace_json}" ] || [ ! -f "${CREWAI_PLOT_SCRIPT}" ] || [ ! -f "${CREWAI_VENV_DIR}/bin/activate" ]; then
        return 1
    fi
    set +e
    (
        source "${CREWAI_VENV_DIR}/bin/activate"
        exec python "${CREWAI_PLOT_SCRIPT}" --input "${trace_json}" --output "${output_png}"
    ) >> "${EXPERIMENT_LOG}" 2>&1
    local plot_rc=$?
    set -e
    return "${plot_rc}"
}

plot_peer_cache_profile() {
    local failover_metrics_json="$1"
    local output_png="$2"
    if [ ! -f "${failover_metrics_json}" ] || [ ! -f "${CREWAI_PEER_CACHE_PLOT_SCRIPT}" ] || [ ! -f "${CREWAI_VENV_DIR}/bin/activate" ]; then
        return 1
    fi
    set +e
    (
        source "${CREWAI_VENV_DIR}/bin/activate"
        exec python "${CREWAI_PEER_CACHE_PLOT_SCRIPT}" \
            --failover-metrics "${failover_metrics_json}" \
            --output "${output_png}"
    ) >> "${EXPERIMENT_LOG}" 2>&1
    local plot_rc=$?
    set -e
    return "${plot_rc}"
}

plot_ab_comparison() {
    local baseline_dir="$1"
    local treatment_dir="$2"
    local output_path="$3"
    local baseline_label="${4:-No Backup}"
    local treatment_label="${5:-Peer Backup}"

    if [ ! -f "${baseline_dir}/trace_log.json" ] || [ ! -f "${baseline_dir}/internal_failover_metrics.json" ]; then
        return 1
    fi
    if [ ! -f "${treatment_dir}/trace_log.json" ] || [ ! -f "${treatment_dir}/internal_failover_metrics.json" ]; then
        return 1
    fi
    if [ ! -f "${CREWAI_AB_COMPARISON_PLOT_SCRIPT}" ] || [ ! -f "${ROOT_VENV_DIR}/bin/activate" ]; then
        return 1
    fi

    set +e
    (
        source "${ROOT_VENV_DIR}/bin/activate"
        exec python "${CREWAI_AB_COMPARISON_PLOT_SCRIPT}" \
            --baseline "${baseline_dir}" \
            --treatment "${treatment_dir}" \
            --baseline-label "${baseline_label}" \
            --treatment-label "${treatment_label}" \
            --output "${output_path}"
    ) >> "${EXPERIMENT_LOG}" 2>&1
    local plot_rc=$?
    set -e
    return "${plot_rc}"
}

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
    log "  peer-replication: $([ "${peer_enabled}" = "1" ] && echo 'YES' || echo 'NO')"
    log "  topology:         DP=${DP_SIZE}, PP=${PP_SIZE}, TP=${TP_SIZE}"
    log "  nodes:            ${NNODES}"
    log "  fault target:     dp=${FAULT_DP_RANK}, pp=${FAULT_PP_RANK}, tp=${FAULT_TP_RANK}"
    log "  job-limit:        ${JOB_LIMIT}"
    log "  app-workers:      ${APP_WORKERS}"
    if [ -n "${INJECT_AFTER_TASK}" ]; then
        log "  inject-after-task: ${INJECT_AFTER_TASK}"
    else
        log "  inject-after-job:  ${INJECT_AFTER_JOB}"
    fi
    log "  inject-delay:     ${INJECT_DELAY}s"
    log "  server port base: ${SERVER_PORT_BASE}"
    log "  peer port base:   ${PEER_PORT_BASE}"
    log "  results:          ${result_dir}"
    log "  prefetch-policy:  ${prefetch_policy:-auto}"
    if [ -n "${hicache_extra_config}" ]; then
        log "  hicache-extra:    ${hicache_extra_config}"
    fi

    local server_url_file="${LOG_DIR}/server_url.txt"
    local stage_manifest_file="${LOG_DIR}/stage_manifest.json"
    rm -f "${server_url_file}" "${stage_manifest_file}"

    log "[${label}] Step 1: Submitting SLURM job..."
    local sbatch_args=(
        --nodes="${NNODES}"
        "${SLURM_DIR}/run_server_qwen3_8b_gpuhome.slurm"
        --dp-size "${DP_SIZE}"
        --pp-size "${PP_SIZE}"
        --tp-size "${TP_SIZE}"
        --model-path "${MODEL_PATH}"
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
    log "[${label}] Step 4: Running CrewAI fault injection against ${head_server_url}..."
    local run_trace_file="${result_dir}/trace_log.json"
    set +e
    if [ -n "${INJECT_AFTER_TASK}" ]; then
        CREWAI_ENABLE_STREAM=0 "${CREWAI_FAULT_SCRIPT}" \
            --server-url "${head_server_url}" \
            --stage-manifest "${stage_manifest_file}" \
            --slurm-job-id "${job_id}" \
            --model-path "${MODEL_PATH}" \
            --jobs-csv "${JOBS_CSV}" \
            --job-limit "${JOB_LIMIT}" \
            --workers "${APP_WORKERS}" \
            --default-year "${DEFAULT_YEAR}" \
            --inject-after-task "${INJECT_AFTER_TASK}" \
            --inject-delay "${INJECT_DELAY}" \
            --fault-dp-rank "${FAULT_DP_RANK}" \
            --fault-pp-rank "${FAULT_PP_RANK}" \
            --fault-tp-rank "${FAULT_TP_RANK}" \
            --output-dir "${result_dir}" \
            --trace-file "${run_trace_file}"
    else
        CREWAI_ENABLE_STREAM=0 "${CREWAI_FAULT_SCRIPT}" \
            --server-url "${head_server_url}" \
            --stage-manifest "${stage_manifest_file}" \
            --slurm-job-id "${job_id}" \
            --model-path "${MODEL_PATH}" \
            --jobs-csv "${JOBS_CSV}" \
            --job-limit "${JOB_LIMIT}" \
            --workers "${APP_WORKERS}" \
            --default-year "${DEFAULT_YEAR}" \
            --inject-after-job "${INJECT_AFTER_JOB}" \
            --inject-delay "${INJECT_DELAY}" \
            --fault-dp-rank "${FAULT_DP_RANK}" \
            --fault-pp-rank "${FAULT_PP_RANK}" \
            --fault-tp-rank "${FAULT_TP_RANK}" \
            --output-dir "${result_dir}" \
            --trace-file "${run_trace_file}"
    fi
    local test_rc=$?
    set -e
    if [ "${test_rc}" -ne 0 ]; then
        log "WARNING: CrewAI run exited with code ${test_rc}"
    fi

    if [ -f "${run_trace_file}" ]; then
        write_group_summary "${run_trace_file}" "${result_dir}/summary.json"
        plot_group_profile "${run_trace_file}" "${figures_dir}/trace_log_profile.png" || true
        if [ -f "${result_dir}/internal_failover_metrics.json" ]; then
            plot_peer_cache_profile \
                "${result_dir}/internal_failover_metrics.json" \
                "${figures_dir}/peer_cache_hit_profile.png" \
                || true
        fi
    fi

    log "[${label}] Step 5: Cleaning up..."
    cancel_slurm_job "${job_id}"
    _ACTIVE_SLURM_JOB=""

    log "[${label}] Experiment complete."
    log ""
    return "${test_rc}"
}

log "============================================================"
log "CrewAI PP2+DP2 Fault Tolerance Experiment — Qwen3-8B GPUHome"
log "============================================================"
log "  Timestamp:          ${TIMESTAMP}"
log "  Model:              ${MODEL_PATH}"
log "  Job limit:          ${JOB_LIMIT}"
log "  App workers:        ${APP_WORKERS}"
log "  Topology:           DP=${DP_SIZE}, PP=${PP_SIZE}, TP=${TP_SIZE}"
log "  Nodes:              ${NNODES}"
log "  Fault target:       dp=${FAULT_DP_RANK}, pp=${FAULT_PP_RANK}, tp=${FAULT_TP_RANK}"
log "  HiCache size:       ${HICACHE_SIZE} GB"
log "  Quantization:       ${QUANTIZATION:-none}"
log "  Jobs CSV:           ${JOBS_CSV}"
if [ -n "${INJECT_AFTER_TASK}" ]; then
    log "  Inject after task:  ${INJECT_AFTER_TASK}"
else
    log "  Inject after job:   ${INJECT_AFTER_JOB}"
fi
log "  Inject delay:       ${INJECT_DELAY}s"
if [ -n "${RECOVER_AFTER}" ]; then
    log "  Recover after:      ${RECOVER_AFTER}s (deprecated, ignored)"
fi
log "  Output base:        ${OUTPUT_BASE_DIR}"
log "  Groups to run:      ${SELECTED_GROUPS[*]}"
log "  Experiment log:     ${EXPERIMENT_LOG}"
log "============================================================"
log ""

OVERALL_RC=0

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

for key in wait_complete best_effort timeout; do
    treatment_dir="${TREATMENT_DIRS[$key]}"
    if [ -d "${BASELINE_DIR}" ] && [ -d "${treatment_dir}" ]; then
        plot_ab_comparison \
            "${BASELINE_DIR}" \
            "${treatment_dir}" \
            "${OUTPUT_BASE_DIR}/ab_comparison__no_backup_vs_${key}.pdf" \
            "No Backup" \
            "${TREATMENT_LABELS[$key]}" \
            || true
        plot_ab_comparison \
            "${BASELINE_DIR}" \
            "${treatment_dir}" \
            "${OUTPUT_BASE_DIR}/ab_comparison__no_backup_vs_${key}.png" \
            "No Backup" \
            "${TREATMENT_LABELS[$key]}" \
            || true
    fi
done

log ""
log "============================================================"
log "Experiment Complete — CrewAI Qwen3-8B GPUHome"
log "============================================================"
log "  Full log: ${EXPERIMENT_LOG}"
for group in "${SELECTED_GROUPS[@]}"; do
    log "  ${group}: ${OUTPUT_BASE_DIR}/${group}"
done
log "  Overall exit code: ${OVERALL_RC}"
log "============================================================"

exit "${OVERALL_RC}"
