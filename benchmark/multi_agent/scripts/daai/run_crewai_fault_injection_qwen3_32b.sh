#!/bin/bash
# CrewAI fault injection (Qwen3-32B / A100 DAAI).
#
# Script Overview:
# Run CrewAI against a deployed SGLang server while scheduling fault injection via
# srun. Consumes stage_manifest from the SLURM server script; optional filters
# restrict which task completion events count toward injection thresholds.

# Exit on error or pipe failure
set -euo pipefail

# ────────────────────────────────────────────────────────────
# Directory and Path Configuration
# ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
CREWAI_APP_DIR="${MULTI_AGENT_DIR}/src/application/crewai"
CREWAI_VENV_DIR="${CREWAI_APP_DIR}/.venv"
CREWAI_SCRIPT="${CREWAI_APP_DIR}/crewai_collaboration.py"
FAILOVER_METRICS_SCRIPT="${CREWAI_APP_DIR}/build_internal_failover_metrics.py"

# ────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────

# Server / manifest
SERVER_URL="${SERVER_URL:-}"
STAGE_MANIFEST="${STAGE_MANIFEST:-${MULTI_AGENT_DIR}/logs/qwen3_32b_a100/stage_manifest.json}"
RUNTIME_FAILOVER_EVENTS_FILE="${RUNTIME_FAILOVER_EVENTS_FILE:-}"
SLURM_JOB_ID_VALUE="${SLURM_JOB_ID_VALUE:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
JOB_LIMIT="${JOB_LIMIT:-10}"
APP_WORKERS="${APP_WORKERS:-2}"
DEFAULT_YEAR="${DEFAULT_YEAR:-2025}"
JOBS_CSV="${JOBS_CSV:-${CREWAI_APP_DIR}/topics.csv}"
# Injection triggers (comma-separated); INJECT_AFTER_JOB vs INJECT_AFTER_TASK are mutually exclusive
INJECT_AFTER_JOB="${INJECT_AFTER_JOB:-}"
INJECT_AFTER_TASK="${INJECT_AFTER_TASK:-}"
INJECT_DELAY="${INJECT_DELAY:-10}"
INJECT_MATCH_TASK_LABEL="${INJECT_MATCH_TASK_LABEL:-}"
INJECT_MATCH_AGENT_ROLE="${INJECT_MATCH_AGENT_ROLE:-}"
INJECT_MATCH_WORKER_ID="${INJECT_MATCH_WORKER_ID:-}"
FAULT_DP_RANK="${FAULT_DP_RANK:-}"
# PP/TP filter: numeric = that rank only; all/* = include all ranks (each trigger still kills one stage, rotated)
FAULT_PP_RANK="${FAULT_PP_RANK:-0}"
FAULT_TP_RANK="${FAULT_TP_RANK:-0}"
UNHEALTHY_TIMEOUT="${UNHEALTHY_TIMEOUT:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${MULTI_AGENT_DIR}/logs/qwen3_32b_a100/crewai_fault_$(date +%Y%m%d_%H%M%S)}"
# Trace JSON path (empty = ${OUTPUT_DIR}/trace_log.json)
TRACE_FILE=""

# ────────────────────────────────────────────────────────────
# Usage
# ────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
  ./run_crewai_fault_injection_qwen3_32b.sh [options]

Options:
  --server-url URL            Head server base URL
  --stage-manifest PATH       Stage manifest produced by the SLURM deploy script
  --slurm-job-id ID           Active SLURM job id for srun-based fault injection
  --model-path PATH           Model path passed to the CrewAI app
  --jobs-csv PATH             Job CSV file
  --job-limit N               Job limit
  --workers N                 Number of CrewAI worker processes
  --default-year YEAR         Default year for jobs without year
  --inject-after-job N[,M]    Inject after job completion count(s)
  --inject-after-task N[,M]   Inject after task completion count(s)
  --inject-delay SECS         Delay after trigger before fault injection
  --inject-match-task-label S Only count matching task_label events for injection
  --inject-match-agent-role S Only count matching agent_role events for injection
  --inject-match-worker-id N  Only count matching worker_id events for injection
  --fault-dp-rank N           Optional fixed replica rank to kill
  --fault-pp-rank N           Filter pipeline stage (default: 0; use all/* to allow any)
  --fault-tp-rank N           Filter tensor rank (default: 0; use all/* to allow any)
  --unhealthy-timeout SECS    Wait time for failed worker health to drop
  --output-dir DIR            Output directory
  --trace-file PATH           Trace JSON output path
  -h, --help                  Show this help
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# ────────────────────────────────────────────────────────────
# Command line parameter parsing
# ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url) SERVER_URL="$2"; shift 2 ;;
        --stage-manifest) STAGE_MANIFEST="$2"; shift 2 ;;
        --slurm-job-id) SLURM_JOB_ID_VALUE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --jobs-csv) JOBS_CSV="$2"; shift 2 ;;
        --job-limit) JOB_LIMIT="$2"; shift 2 ;;
        --workers) APP_WORKERS="$2"; shift 2 ;;
        --default-year) DEFAULT_YEAR="$2"; shift 2 ;;
        --inject-after-job) INJECT_AFTER_JOB="$2"; shift 2 ;;
        --inject-after-task) INJECT_AFTER_TASK="$2"; shift 2 ;;
        --inject-delay) INJECT_DELAY="$2"; shift 2 ;;
        --inject-match-task-label) INJECT_MATCH_TASK_LABEL="$2"; shift 2 ;;
        --inject-match-agent-role) INJECT_MATCH_AGENT_ROLE="$2"; shift 2 ;;
        --inject-match-worker-id) INJECT_MATCH_WORKER_ID="$2"; shift 2 ;;
        --fault-dp-rank) FAULT_DP_RANK="$2"; shift 2 ;;
        --fault-pp-rank) FAULT_PP_RANK="$2"; shift 2 ;;
        --fault-tp-rank) FAULT_TP_RANK="$2"; shift 2 ;;
        --unhealthy-timeout) UNHEALTHY_TIMEOUT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --trace-file) TRACE_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ────────────────────────────────────────────────────────────
# Preconditions
# ────────────────────────────────────────────────────────────

# Mutually exclusive trigger modes
if [[ -n "${INJECT_AFTER_JOB}" && -n "${INJECT_AFTER_TASK}" ]]; then
    die "Use only one of --inject-after-job or --inject-after-task"
fi

# Default: inject after first job completion when neither mode is set
if [[ -z "${INJECT_AFTER_JOB}" && -z "${INJECT_AFTER_TASK}" ]]; then
    INJECT_AFTER_JOB="1"
fi

if [[ ! -f "${CREWAI_SCRIPT}" ]]; then
    die "CrewAI script not found: ${CREWAI_SCRIPT}"
fi

if [[ ! -f "${STAGE_MANIFEST}" ]]; then
    die "Stage manifest not found: ${STAGE_MANIFEST}"
fi

if [[ ! -f "${JOBS_CSV}" ]]; then
    die "Jobs CSV file not found: ${JOBS_CSV}"
fi

if [[ ! -f "${CREWAI_VENV_DIR}/bin/activate" ]]; then
    die "CrewAI virtual environment not found: ${CREWAI_VENV_DIR}"
fi

# ────────────────────────────────────────────────────────────
# Output directory and log paths
# ────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/run.log"
APP_STDOUT_LOG="${OUTPUT_DIR}/crewai_stdout.log"
JOB_SUMMARY_CSV="${OUTPUT_DIR}/job_summary.csv"
TASK_SUMMARY_CSV="${OUTPUT_DIR}/task_summary.csv"
EVENTS_FILE="${OUTPUT_DIR}/events.jsonl"

if [[ -z "${TRACE_FILE}" ]]; then
    TRACE_FILE="${OUTPUT_DIR}/trace_log.json"
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${RUN_LOG}"
}

# ────────────────────────────────────────────────────────────
# Load missing fields from stage_manifest in one pass
# ────────────────────────────────────────────────────────────

{
    IFS=$'\t' read -r _m_slurm_job_id _m_server_url _m_failover_file < <(
        python3 - "${STAGE_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("\t".join([
    str(m.get("slurm_job_id", "")),
    str(m.get("server_url", "")).rstrip("/"),
    str(m.get("runtime_failover_events_file", "")),
]))
PY
    )
    [[ -z "${SLURM_JOB_ID_VALUE}" ]] && SLURM_JOB_ID_VALUE="${_m_slurm_job_id}"
    [[ -z "${SERVER_URL}" ]]          && SERVER_URL="${_m_server_url}"
    [[ -z "${RUNTIME_FAILOVER_EVENTS_FILE}" ]] && RUNTIME_FAILOVER_EVENTS_FILE="${_m_failover_file}"
}

[[ -z "${SLURM_JOB_ID_VALUE}" ]] && die "Missing --slurm-job-id and stage manifest does not contain slurm_job_id"
[[ -z "${SERVER_URL}" ]]          && die "No head server URL found in stage manifest: ${STAGE_MANIFEST}"

# ────────────────────────────────────────────────────────────
# Copy server logs to output directory
# ────────────────────────────────────────────────────────────

snapshot_server_logs() {
    [[ -n "${RUNTIME_FAILOVER_EVENTS_FILE}" ]] || return 0
    local log_dir
    log_dir="$(dirname "${RUNTIME_FAILOVER_EVENTS_FILE}")"
    [[ -d "${log_dir}" ]] || return 0

    local snapshot_dir="${OUTPUT_DIR}/server_logs"
    mkdir -p "${snapshot_dir}"
    shopt -s nullglob
    for path in "${log_dir}"/node_*/server.log; do
        local node_name
        node_name="$(basename "$(dirname "${path}")")"
        mkdir -p "${snapshot_dir}/${node_name}"
        cp "${path}" "${snapshot_dir}/${node_name}/server.log"
    done
    shopt -u nullglob
}

# ────────────────────────────────────────────────────────────
# Append event record to events.jsonl
# ────────────────────────────────────────────────────────────

append_events_file_record() {
    local event_type="$1"; shift
    python3 - "${EVENTS_FILE}" "${event_type}" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

events_path = Path(sys.argv[1])
record = {"event": sys.argv[2], "timestamp": datetime.now(timezone.utc).isoformat()}
for pair in sys.argv[3:]:
    k, v = pair.split("=", 1)
    record[k] = int(v) if v.isdigit() else v
events_path.parent.mkdir(parents=True, exist_ok=True)
with events_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

# ────────────────────────────────────────────────────────────
# Fault target stage selection
# ────────────────────────────────────────────────────────────

mapfile -t STAGE_TARGETS < <(
    python3 - "${STAGE_MANIFEST}" "${FAULT_DP_RANK}" "${FAULT_PP_RANK}" "${FAULT_TP_RANK}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
fault_dp = sys.argv[2].strip()
fault_pp = sys.argv[3].strip()
fault_tp = sys.argv[4].strip()

for stage in payload.get("stages", []):
    if fault_dp and str(stage.get("dp_rank")) != fault_dp:
        continue
    if fault_pp and fault_pp not in {"all", "*"} and str(stage.get("pp_rank")) != fault_pp:
        continue
    if fault_tp and fault_tp not in {"all", "*"} and str(stage.get("tp_rank")) != fault_tp:
        continue
    fields = [
        str(stage.get("dp_rank", "")),
        str(stage.get("pp_rank", "")),
        str(stage.get("tp_rank", "")),
        str(stage.get("node", "")),
        str(stage.get("node_url", "")),
        str(stage.get("kill_pattern", "")),
    ]
    print("\t".join(fields))
PY
)

if [[ "${#STAGE_TARGETS[@]}" -eq 0 ]]; then
    die "No stage targets matched the requested (dp, pp, tp) filter"
fi

# ────────────────────────────────────────────────────────────
# Injection trigger mode (job/task)
# ────────────────────────────────────────────────────────────

if [[ -n "${INJECT_AFTER_TASK}" ]]; then
    INJECT_KIND="task"
    INJECT_THRESHOLDS_RAW="${INJECT_AFTER_TASK}"
    export CREWAI_PRINT_COMPLETED_ONLY=0
else
    INJECT_KIND="job"
    INJECT_THRESHOLDS_RAW="${INJECT_AFTER_JOB}"
    export CREWAI_PRINT_COMPLETED_ONLY=1
fi

IFS=',' read -r -a INJECT_THRESHOLDS <<< "${INJECT_THRESHOLDS_RAW}"
if [[ "${#INJECT_THRESHOLDS[@]}" -eq 0 ]]; then
    die "No injection threshold parsed"
fi

for threshold in "${INJECT_THRESHOLDS[@]}"; do
    [[ "${threshold}" =~ ^[0-9]+$ ]] || die "Invalid threshold: ${threshold}"
    (( threshold > 0 )) || die "Threshold must be greater than zero"
done

# ────────────────────────────────────────────────────────────
# Runtime state
# ────────────────────────────────────────────────────────────

FAULT_BG_PIDS=()
JOB_DONE_COUNT=0
TASK_DONE_COUNT=0
NEXT_INJECT_IDX=0
APP_PID=""
_FI_CLEANUP_HANDLED=0

# ────────────────────────────────────────────────────────────
# Signal handling
# ────────────────────────────────────────────────────────────

_fi_cleanup() {
    if [ "${_FI_CLEANUP_HANDLED}" -eq 1 ]; then
        exit 130
    fi
    _FI_CLEANUP_HANDLED=1
    trap - INT TERM
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fault-injection script interrupted — killing child processes..."
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill -TERM "${APP_PID}" 2>/dev/null || true
        sleep 1
        kill -9 "${APP_PID}" 2>/dev/null || true
    fi
    if [ "${#FAULT_BG_PIDS[@]}" -gt 0 ]; then
        for pid in "${FAULT_BG_PIDS[@]}"; do
            [ -z "${pid}" ] && continue
            kill -9 "${pid}" 2>/dev/null || true
        done
    fi
    jobs -p 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    exit 130
}

trap '_fi_cleanup' INT TERM

# ────────────────────────────────────────────────────────────
# Health checks and runtime failover reaction
# ────────────────────────────────────────────────────────────

# Poll /health until failure or timeout; returns 0 if unhealthy, 1 if still healthy at timeout
wait_worker_unhealthy() {
    local worker_url="$1"
    local timeout_s="${2:-20}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if ! curl --noproxy '*' -fsS -m 3 "${worker_url}/health" >/dev/null 2>&1; then
            return 0
        fi
        if (( $(date +%s) - start_ts >= timeout_s )); then
            return 1
        fi
        sleep 2
    done
}

# Poll runtime_failover_events_file for failover events after not_before_epoch for failed_dp_rank
wait_for_runtime_reaction() {
    local failed_dp_rank="$1"
    local not_before_epoch="$2"
    local timeout_s="${3:-20}"

    if [[ -z "${RUNTIME_FAILOVER_EVENTS_FILE}" ]]; then
        return 1
    fi

    python3 - "${RUNTIME_FAILOVER_EVENTS_FILE}" "${failed_dp_rank}" "${not_before_epoch}" "${timeout_s}" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
failed_dp_rank = sys.argv[2]
not_before_epoch = float(sys.argv[3])
timeout_s = float(sys.argv[4])
deadline = time.time() + timeout_s
interesting_events = {
    "failover_dispatched",
    "failover_aborted",
    "resume_accepted",
    "resume_rejected",
}

def parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

while time.time() < deadline:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") not in interesting_events:
                continue
            if str(record.get("failed_owner_dp_rank", "")) != failed_dp_rank:
                continue
            if parse_timestamp(str(record.get("timestamp", ""))) < not_before_epoch:
                continue
            print(json.dumps(record, ensure_ascii=True))
            sys.exit(0)
    time.sleep(1.0)

sys.exit(1)
PY
}

# ────────────────────────────────────────────────────────────
# Fault injection core
# ────────────────────────────────────────────────────────────

# srun --overlap on target node with pkill -9; tries --gres=none, --gpus=0, then no GPU flag
kill_stage_via_slurm() {
    local slurm_job_id="$1"
    local node_name="$2"
    local kill_pattern="$3"
    local remote_script
    remote_script=$(cat <<'EOF'
set -euo pipefail
matches=$(pgrep -f "${KILL_PATTERN}" || true)
if [ -z "${matches}" ]; then
    echo "No process matched ${KILL_PATTERN}" >&2
    exit 3
fi
pgrep -af "${KILL_PATTERN}" || true
pkill -9 -f "${KILL_PATTERN}"
EOF
)

    local -a srun_base=(
        srun
        --jobid "${slurm_job_id}"
        -w "${node_name}"
        --nodes=1
        --ntasks=1
        --overlap
        --kill-on-bad-exit=0
    )
    local -a gpu_release_opts=(
        "--gres=none"
        "--gpus=0"
        ""
    )
    local gpu_opt=""
    local rc=1

    for gpu_opt in "${gpu_release_opts[@]}"; do
        local -a cmd=("${srun_base[@]}")
        if [ -n "${gpu_opt}" ]; then
            cmd+=("${gpu_opt}")
        fi
        cmd+=(
            env
            "KILL_PATTERN=${kill_pattern}"
            bash
            -lc
            "${remote_script}"
        )

        set +e
        env NO_PROXY="*" no_proxy="*" "${cmd[@]}" >> "${RUN_LOG}" 2>&1
        rc=$?
        set -e

        if [ "${rc}" -eq 0 ]; then
            return 0
        fi

        if [ -n "${gpu_opt}" ]; then
            log "Kill step retry: srun ${gpu_opt} failed with rc=${rc}, falling back..."
        fi
    done

    return "${rc}"
}

# Wait ${INJECT_DELAY}s, then kill one stage via srun and verify the fault took effect.
delayed_inject_fault() {
    local event_id="$1"
    local trigger_kind="$2"
    local trigger_count="$3"
    local stage_spec="$4"

    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${stage_spec}"

    log "Fault ${event_id}: waiting ${INJECT_DELAY}s after ${trigger_kind}=${trigger_count}"
    sleep "${INJECT_DELAY}"

    log "Fault ${event_id}: killing dp=${dp_rank} pp=${pp_rank} tp=${tp_rank} on ${node_name}"
    append_events_file_record "fault_injected" \
        "event_id=${event_id}" "trigger_kind=${trigger_kind}" "trigger_count=${trigger_count}" \
        "dp_rank=${dp_rank}" "pp_rank=${pp_rank}" "tp_rank=${tp_rank}" \
        "node=${node_name}" "node_url=${node_url}" "kill_pattern=${kill_pattern}" \
        "slurm_job_id=${SLURM_JOB_ID_VALUE}"

    local fault_start_epoch
    fault_start_epoch=$(date +%s)

    if ! kill_stage_via_slurm "${SLURM_JOB_ID_VALUE}" "${node_name}" "${kill_pattern}"; then
        local rc=$?
        log "Fault ${event_id}: failed to kill ${kill_pattern} (rc=${rc})"
        append_events_file_record "fault_injection_failed" \
            "event_id=${event_id}" "dp_rank=${dp_rank}" "node=${node_name}" \
            "kill_pattern=${kill_pattern}" "status=kill_failed"
        return "${rc}"
    fi
    log "Fault ${event_id}: kill command completed for ${kill_pattern}"

    local runtime_event=""
    if runtime_event=$(wait_for_runtime_reaction "${dp_rank}" "${fault_start_epoch}" "${UNHEALTHY_TIMEOUT}" 2>/dev/null); then
        log "Fault ${event_id}: runtime failover reaction observed"
        append_events_file_record "runtime_failover_reaction" \
            "event_id=${event_id}" "dp_rank=${dp_rank}" "node_url=${node_url}" "status=runtime_reaction"
        return 0
    fi

    if wait_worker_unhealthy "${node_url}" "${UNHEALTHY_TIMEOUT}"; then
        log "Fault ${event_id}: health became unhealthy at ${node_url}"
        append_events_file_record "node_became_unhealthy" \
            "event_id=${event_id}" "dp_rank=${dp_rank}" "node_url=${node_url}" "status=unhealthy"
        return 0
    fi

    log "Fault ${event_id}: no failure reaction observed within ${UNHEALTHY_TIMEOUT}s"
    append_events_file_record "runtime_reaction_timeout" \
        "event_id=${event_id}" "dp_rank=${dp_rank}" "node_url=${node_url}" "status=timeout"
    return 1
}

# ────────────────────────────────────────────────────────────
# Injection scheduling and task event filters
# ────────────────────────────────────────────────────────────

schedule_injection_if_needed() {
    local current_count="$1"
    local task_label="${2:-}"
    local agent_role="${3:-}"
    local worker_id="${4:-}"
    local job_id="${5:-}"
    local job_name="${6:-}"
    if (( NEXT_INJECT_IDX >= ${#INJECT_THRESHOLDS[@]} )); then
        return
    fi

    local threshold="${INJECT_THRESHOLDS[$NEXT_INJECT_IDX]}"
    if (( current_count < threshold )); then
        return
    fi

    local event_id=$((NEXT_INJECT_IDX + 1))
    local stage_index=$((NEXT_INJECT_IDX % ${#STAGE_TARGETS[@]}))
    local stage_spec="${STAGE_TARGETS[$stage_index]}"

    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${stage_spec}"

    log "Fault ${event_id}: trigger matched ${INJECT_KIND}=${current_count}, target=dp${dp_rank}/pp${pp_rank}/tp${tp_rank} on ${node_name}"
    append_events_file_record "fault_scheduled" \
        "event_id=${event_id}" "trigger_kind=${INJECT_KIND}" "trigger_count=${current_count}" \
        "dp_rank=${dp_rank}" "pp_rank=${pp_rank}" "tp_rank=${tp_rank}" \
        "node=${node_name}" "node_url=${node_url}" "inject_delay=${INJECT_DELAY}" \
        "task_label=${task_label}" "agent_role=${agent_role}" "worker_id=${worker_id}" \
        "job_id=${job_id}" "job=${job_name}"

    delayed_inject_fault "${event_id}" "${INJECT_KIND}" "${current_count}" "${stage_spec}" &
    FAULT_BG_PIDS+=($!)
    NEXT_INJECT_IDX=$((NEXT_INJECT_IDX + 1))
}

task_event_matches_filters() {
    local task_label="${1:-}"
    local agent_role="${2:-}"
    local worker_id="${3:-}"

    if [[ -n "${INJECT_MATCH_TASK_LABEL}" && "${task_label}" != "${INJECT_MATCH_TASK_LABEL}" ]]; then
        return 1
    fi
    if [[ -n "${INJECT_MATCH_AGENT_ROLE}" && "${agent_role}" != "${INJECT_MATCH_AGENT_ROLE}" ]]; then
        return 1
    fi
    if [[ -n "${INJECT_MATCH_WORKER_ID}" && "${worker_id}" != "${INJECT_MATCH_WORKER_ID}" ]]; then
        return 1
    fi
    return 0
}

# ────────────────────────────────────────────────────────────
# Result aggregation
# ────────────────────────────────────────────────────────────

write_job_summary() {
    if [[ ! -f "${TRACE_FILE}" ]]; then
        return
    fi

    python3 - "${TRACE_FILE}" "${JOB_SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

records = json.loads(trace_path.read_text(encoding="utf-8"))
with csv_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "job_id",
        "job",
        "year",
        "worker_id",
        "status",
        "job_completion_time_s",
        "llm_calls",
        "prefill_cached_tokens",
        "total_tokens",
        "error",
    ])
    for item in records:
        summary = item.get("summary", {})
        writer.writerow([
            item.get("topic_id", ""),
            item.get("topic", ""),
            item.get("year", ""),
            item.get("worker_id", ""),
            item.get("status", ""),
            summary.get("wall_time_s", ""),
            summary.get("llm_calls", ""),
            summary.get("prefill_cached_tokens", ""),
            summary.get("total_tokens", ""),
            item.get("error", "") or "",
        ])
PY
}

write_task_summary() {
    if [[ ! -f "${TRACE_FILE}" ]]; then
        return
    fi

    python3 - "${TRACE_FILE}" "${TASK_SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

AGENT_TO_TASK = {
    "Planning Coordinator": "Phase 1 / Planning",
    "Data Collector": "Phase 2 / Data Collection",
    "Deep Analyst": "Phase 2 / Deep Analysis (critical)",
    "Trend Scout": "Phase 2 / Trend Scan",
    "Risk Assessor": "Phase 2 / Risk Assessment",
    "Report Synthesizer": "Phase 3 / Synthesis",
}

trace_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
records = json.loads(trace_path.read_text(encoding="utf-8"))

with csv_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "job_id",
        "job",
        "year",
        "worker_id",
        "job_status",
        "agent",
        "task_label",
        "task_completion_time_s",
        "llm_calls",
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prefill_cache_hit_rate",
        "error",
    ])
    for item in records:
        for timing in item.get("agent_timings", []):
            agent = timing.get("agent", "")
            writer.writerow([
                item.get("topic_id", ""),
                item.get("topic", ""),
                item.get("year", ""),
                item.get("worker_id", ""),
                item.get("status", ""),
                agent,
                AGENT_TO_TASK.get(agent, agent),
                timing.get("duration_s", ""),
                timing.get("llm_calls", ""),
                timing.get("prompt_tokens", ""),
                timing.get("cached_prompt_tokens", ""),
                timing.get("completion_tokens", ""),
                timing.get("total_tokens", ""),
                timing.get("prefill_cache_hit_rate", ""),
                timing.get("error", "") or "",
            ])
PY
}

# ────────────────────────────────────────────────────────────
# Event handling (one Python process per batch, not per field)
# ────────────────────────────────────────────────────────────

process_new_events() {
    local start_line="$1"
    local end_line="$2"
    while IFS=$'\t' read -r ev_type job_id job_name worker_id task_label agent_role; do
        case "${ev_type}" in
            job_completed)
                JOB_DONE_COUNT=$((JOB_DONE_COUNT + 1))
                log "Observed job completion ${JOB_DONE_COUNT}: job_id=${job_id} job=${job_name}"
                [[ "${INJECT_KIND}" == "job" ]] && \
                    schedule_injection_if_needed "${JOB_DONE_COUNT}" "" "" "${worker_id}" "${job_id}" "${job_name}"
                ;;
            job_failed)
                log "Observed job failure: job_id=${job_id} job=${job_name}"
                ;;
            task_completed)
                if ! task_event_matches_filters "${task_label}" "${agent_role}" "${worker_id}"; then
                    log "Observed task completion (ignored by filter): ${task_label} -> ${agent_role}"
                    continue
                fi
                TASK_DONE_COUNT=$((TASK_DONE_COUNT + 1))
                log "Observed task completion ${TASK_DONE_COUNT}: ${task_label} -> ${agent_role}"
                [[ "${INJECT_KIND}" == "task" ]] && \
                    schedule_injection_if_needed "${TASK_DONE_COUNT}" "${task_label}" "${agent_role}" "${worker_id}" "${job_id}" "${job_name}"
                ;;
            task_failed)
                log "Observed task failure: ${task_label}"
                ;;
        esac
    done < <(sed -n "${start_line},${end_line}p" "${EVENTS_FILE}" | python3 -c '
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    print("\t".join(str(e.get(k, "") or "") for k in
        ("event", "job_id", "job", "worker_id", "task_label", "agent_role")))
')
}

# ────────────────────────────────────────────────────────────
# Main execution
# ────────────────────────────────────────────────────────────

rm -f "${RUN_LOG}" "${APP_STDOUT_LOG}" "${EVENTS_FILE}"

log "Starting CrewAI fault injection run"
log "Server URL: ${SERVER_URL}"
log "Stage manifest: ${STAGE_MANIFEST}"
log "SLURM job id: ${SLURM_JOB_ID_VALUE}"
log "Runtime failover events: ${RUNTIME_FAILOVER_EVENTS_FILE:-<none>}"
log "Trigger kind: ${INJECT_KIND}"
log "Trigger thresholds: ${INJECT_THRESHOLDS_RAW}"
log "Inject delay: ${INJECT_DELAY}s"
if [[ -n "${INJECT_MATCH_TASK_LABEL}" ]]; then
    log "Inject match task_label: ${INJECT_MATCH_TASK_LABEL}"
fi
if [[ -n "${INJECT_MATCH_AGENT_ROLE}" ]]; then
    log "Inject match agent_role: ${INJECT_MATCH_AGENT_ROLE}"
fi
if [[ -n "${INJECT_MATCH_WORKER_ID}" ]]; then
    log "Inject match worker_id: ${INJECT_MATCH_WORKER_ID}"
fi
log "Fault filter: dp=${FAULT_DP_RANK:-any}, pp=${FAULT_PP_RANK}, tp=${FAULT_TP_RANK}"
log "Output dir: ${OUTPUT_DIR}"
log "Trace file: ${TRACE_FILE}"
log "Events file: ${EVENTS_FILE}"

(
    source "${CREWAI_VENV_DIR}/bin/activate"
    export PYTHONUNBUFFERED=1
    export CREWAI_SERVER_BASE_URL="${SERVER_URL}"
    export CREWAI_MODEL_PATH="${MODEL_PATH}"
    export CREWAI_EVENTS_FILE="${EVENTS_FILE}"
    # CrewAI's default SQLite location under ~/.local/share is unreliable on this
    # cluster filesystem. Keep its transient DB files on node-local tmp storage.
    CREWAI_TMP_BASE="${TMPDIR:-/tmp}/${USER}/crewai-storage"
    CREWAI_TMP_DB_NAME="job_${SLURM_JOB_ID_VALUE}_$(basename "${OUTPUT_DIR}")"
    export XDG_DATA_HOME="${CREWAI_TMP_BASE}/xdg"
    export SQLITE_TMPDIR="${CREWAI_TMP_BASE}/sqlite-tmp"
    export CREWAI_STORAGE_DIR="${CREWAI_TMP_DB_NAME}"
    mkdir -p "${XDG_DATA_HOME}" "${SQLITE_TMPDIR}"
    exec python "${CREWAI_SCRIPT}" \
        --csv "${JOBS_CSV}" \
        --limit "${JOB_LIMIT}" \
        --workers "${APP_WORKERS}" \
        --trace-file "${TRACE_FILE}" \
        --default-year "${DEFAULT_YEAR}"
) > "${APP_STDOUT_LOG}" 2>&1 &
APP_PID=$!

# ────────────────────────────────────────────────────────────
# Main monitoring loop
# ────────────────────────────────────────────────────────────

LAST_EVENT_LINE=0
while true; do
    if [[ -f "${EVENTS_FILE}" ]]; then
        CURRENT_EVENT_LINE=$(wc -l < "${EVENTS_FILE}" | tr -d ' ')
        if (( CURRENT_EVENT_LINE > LAST_EVENT_LINE )); then
            process_new_events $((LAST_EVENT_LINE + 1)) "${CURRENT_EVENT_LINE}"
            LAST_EVENT_LINE=${CURRENT_EVENT_LINE}
        fi
    fi

    if ! kill -0 "${APP_PID}" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [[ -f "${EVENTS_FILE}" ]]; then
    CURRENT_EVENT_LINE=$(wc -l < "${EVENTS_FILE}" | tr -d ' ')
    if (( CURRENT_EVENT_LINE > LAST_EVENT_LINE )); then
        process_new_events $((LAST_EVENT_LINE + 1)) "${CURRENT_EVENT_LINE}"
        LAST_EVENT_LINE=${CURRENT_EVENT_LINE}
    fi
fi

# ────────────────────────────────────────────────────────────
# Teardown and artifacts
# ────────────────────────────────────────────────────────────

set +e
wait "${APP_PID}"
APP_RC=$?
if [[ "${#FAULT_BG_PIDS[@]}" -gt 0 ]]; then
    for pid in "${FAULT_BG_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null
    done
fi
set -e

write_job_summary
write_task_summary
snapshot_server_logs

INTERNAL_FAILOVER_METRICS_JSON="${OUTPUT_DIR}/internal_failover_metrics.json"
if [[ -f "${FAILOVER_METRICS_SCRIPT}" ]]; then
    set +e
    (
        source "${CREWAI_VENV_DIR}/bin/activate"
        exec python "${FAILOVER_METRICS_SCRIPT}" \
            --trace-log "${TRACE_FILE}" \
            --events-file "${EVENTS_FILE}" \
            --runtime-events-file "${RUNTIME_FAILOVER_EVENTS_FILE}" \
            --output "${INTERNAL_FAILOVER_METRICS_JSON}"
    ) >> "${RUN_LOG}" 2>&1
    METRICS_RC=$?
    set -e
    if [[ "${METRICS_RC}" -ne 0 ]]; then
        log "WARNING: Failed to build internal failover metrics (rc=${METRICS_RC})"
    fi
fi

log "CrewAI run finished with exit code ${APP_RC}"
log "Stdout log: ${APP_STDOUT_LOG}"
if [[ -f "${JOB_SUMMARY_CSV}" ]]; then
    log "Job summary: ${JOB_SUMMARY_CSV}"
fi
if [[ -f "${TASK_SUMMARY_CSV}" ]]; then
    log "Task summary: ${TASK_SUMMARY_CSV}"
fi
if [[ -f "${INTERNAL_FAILOVER_METRICS_JSON}" ]]; then
    log "Internal failover metrics: ${INTERNAL_FAILOVER_METRICS_JSON}"
fi

exit "${APP_RC}"
