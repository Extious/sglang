#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_AGENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MULTI_AGENT_DIR}/../.." && pwd)"
CREWAI_APP_DIR="${MULTI_AGENT_DIR}/src/application/crewai"
CREWAI_VENV_DIR="${CREWAI_APP_DIR}/.venv"
CREWAI_SCRIPT="${CREWAI_APP_DIR}/crewai_collaboration.py"
FAILOVER_METRICS_SCRIPT="${CREWAI_APP_DIR}/build_internal_failover_metrics.py"

SERVER_URL="${SERVER_URL:-}"
STAGE_MANIFEST="${STAGE_MANIFEST:-${MULTI_AGENT_DIR}/logs/logs_qwen3_8b_gpuhome/stage_manifest.json}"
RUNTIME_FAILOVER_EVENTS_FILE="${RUNTIME_FAILOVER_EVENTS_FILE:-}"
SLURM_JOB_ID_VALUE="${SLURM_JOB_ID_VALUE:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
JOB_LIMIT="${JOB_LIMIT:-10}"
APP_WORKERS="${APP_WORKERS:-2}"
DEFAULT_YEAR="${DEFAULT_YEAR:-2025}"
JOBS_CSV="${JOBS_CSV:-${CREWAI_APP_DIR}/topics.csv}"
INJECT_AFTER_JOB="${INJECT_AFTER_JOB:-}"
INJECT_AFTER_TASK="${INJECT_AFTER_TASK:-}"
INJECT_DELAY="${INJECT_DELAY:-10}"
INJECT_MATCH_TASK_LABEL="${INJECT_MATCH_TASK_LABEL:-}"
INJECT_MATCH_AGENT_ROLE="${INJECT_MATCH_AGENT_ROLE:-}"
INJECT_MATCH_WORKER_ID="${INJECT_MATCH_WORKER_ID:-}"
FAULT_DP_RANK="${FAULT_DP_RANK:-}"
FAULT_PP_RANK="${FAULT_PP_RANK:-0}"
FAULT_TP_RANK="${FAULT_TP_RANK:-0}"
UNHEALTHY_TIMEOUT="${UNHEALTHY_TIMEOUT:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${MULTI_AGENT_DIR}/logs/logs_qwen3_8b_gpuhome/crewai_fault_$(date +%Y%m%d_%H%M%S)}"
TRACE_FILE=""

usage() {
    cat <<'EOF'
Usage:
  ./run_crewai_fault_injection_gpuhome.sh [options]

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
  --fault-pp-rank N           Pipeline stage rank to kill (default: 0)
  --fault-tp-rank N           Tensor rank to kill (default: 0)
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url) SERVER_URL="$2"; shift 2 ;;
        --stage-manifest) STAGE_MANIFEST="$2"; shift 2 ;;
        --slurm-job-id) SLURM_JOB_ID_VALUE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --jobs-csv|--topics-csv) JOBS_CSV="$2"; shift 2 ;;
        --job-limit|--limit) JOB_LIMIT="$2"; shift 2 ;;
        --workers) APP_WORKERS="$2"; shift 2 ;;
        --default-year) DEFAULT_YEAR="$2"; shift 2 ;;
        --inject-after-job|--inject-after-topic) INJECT_AFTER_JOB="$2"; shift 2 ;;
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

if [[ -n "${INJECT_AFTER_JOB}" && -n "${INJECT_AFTER_TASK}" ]]; then
    die "Use only one of --inject-after-job or --inject-after-task"
fi

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

if [[ -z "${SLURM_JOB_ID_VALUE}" ]]; then
    SLURM_JOB_ID_VALUE="$(
        python3 - "${STAGE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("slurm_job_id", ""))
PY
    )"
fi

if [[ -z "${SLURM_JOB_ID_VALUE}" ]]; then
    die "Missing --slurm-job-id and stage manifest does not contain slurm_job_id"
fi

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

snapshot_server_logs() {
    local log_dir=""
    if [[ -n "${RUNTIME_FAILOVER_EVENTS_FILE}" ]]; then
        log_dir="$(dirname "${RUNTIME_FAILOVER_EVENTS_FILE}")"
    fi
    if [[ -z "${log_dir}" || ! -d "${log_dir}" ]]; then
        return 0
    fi

    local snapshot_dir="${OUTPUT_DIR}/server_logs"
    mkdir -p "${snapshot_dir}"
    shopt -s nullglob
    local path=""
    for path in "${log_dir}"/node_*/server.log; do
        local node_name
        node_name="$(basename "$(dirname "${path}")")"
        mkdir -p "${snapshot_dir}/${node_name}"
        cp "${path}" "${snapshot_dir}/${node_name}/server.log"
    done
    shopt -u nullglob
}

append_events_file_record() {
    local event_type="$1"
    shift
    python3 - "${EVENTS_FILE}" "${event_type}" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

events_path = Path(sys.argv[1])
event_type = sys.argv[2]
pairs = sys.argv[3:]

record = {
    "event": event_type,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
for pair in pairs:
    key, value = pair.split("=", 1)
    if value.isdigit():
        record[key] = int(value)
    else:
        record[key] = value

events_path.parent.mkdir(parents=True, exist_ok=True)
with events_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

if [[ -z "${SERVER_URL}" ]]; then
    SERVER_URL="$(
        python3 - "${STAGE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("server_url", "")).rstrip("/"))
PY
    )"
fi

if [[ -z "${SERVER_URL}" ]]; then
    die "No head server URL found in stage manifest: ${STAGE_MANIFEST}"
fi

if [[ -z "${RUNTIME_FAILOVER_EVENTS_FILE}" ]]; then
    RUNTIME_FAILOVER_EVENTS_FILE="$(
        python3 - "${STAGE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("runtime_failover_events_file", ""))
PY
    )"
fi

FAULT_PP_MATCH_ALL=0
if [[ -z "${FAULT_PP_RANK}" || "${FAULT_PP_RANK}" == "all" || "${FAULT_PP_RANK}" == "*" ]]; then
    FAULT_PP_MATCH_ALL=1
fi

FAULT_TP_MATCH_ALL=0
if [[ -z "${FAULT_TP_RANK}" || "${FAULT_TP_RANK}" == "all" || "${FAULT_TP_RANK}" == "*" ]]; then
    FAULT_TP_MATCH_ALL=1
fi

mapfile -t ALL_STAGE_TARGETS < <(
    python3 - "${STAGE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for stage in payload.get("stages", []):
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

collect_replica_group_targets() {
    local selected_spec="$1"
    local -n out_specs_ref="$2"
    local selected_dp_rank selected_pp_rank selected_tp_rank selected_node_name selected_node_url selected_kill_pattern
    IFS=$'\t' read -r selected_dp_rank selected_pp_rank selected_tp_rank selected_node_name selected_node_url selected_kill_pattern <<< "${selected_spec}"

    local candidate_spec=""
    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    for candidate_spec in "${ALL_STAGE_TARGETS[@]}"; do
        IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${candidate_spec}"
        if [[ "${dp_rank}" != "${selected_dp_rank}" ]]; then
            continue
        fi
        out_specs_ref+=("${candidate_spec}")
    done
}

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

FAULT_BG_PIDS=()
JOB_DONE_COUNT=0
TASK_DONE_COUNT=0
NEXT_INJECT_IDX=0
APP_PID=""
_FI_CLEANUP_HANDLED=0

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

inject_stage_kill_fault() {
    local event_id="$1"
    local trigger_kind="$2"
    local trigger_count="$3"
    local stage_spec="$4"

    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${stage_spec}"

    log "Fault ${event_id}: killing dp=${dp_rank} pp=${pp_rank} tp=${tp_rank} on ${node_name} after ${trigger_kind}=${trigger_count}"
    append_events_file_record \
        "fault_injected" \
        "event_id=${event_id}" \
        "trigger_kind=${trigger_kind}" \
        "trigger_count=${trigger_count}" \
        "dp_rank=${dp_rank}" \
        "pp_rank=${pp_rank}" \
        "tp_rank=${tp_rank}" \
        "node=${node_name}" \
        "node_url=${node_url}" \
        "kill_pattern=${kill_pattern}" \
        "slurm_job_id=${SLURM_JOB_ID_VALUE}"

    local fault_start_epoch
    fault_start_epoch=$(date +%s)
    if kill_stage_via_slurm "${SLURM_JOB_ID_VALUE}" "${node_name}" "${kill_pattern}"; then
        log "Fault ${event_id}: kill command completed for ${kill_pattern}"
    else
        local rc=$?
        log "Fault ${event_id}: failed to kill ${kill_pattern} (rc=${rc})"
        append_events_file_record \
            "fault_injection_failed" \
            "event_id=${event_id}" \
            "trigger_kind=${trigger_kind}" \
            "trigger_count=${trigger_count}" \
            "dp_rank=${dp_rank}" \
            "pp_rank=${pp_rank}" \
            "tp_rank=${tp_rank}" \
            "node=${node_name}" \
            "node_url=${node_url}" \
            "kill_pattern=${kill_pattern}" \
            "status=kill_failed"
        return "${rc}"
    fi

    local runtime_event=""
    if runtime_event=$(wait_for_runtime_reaction "${dp_rank}" "${fault_start_epoch}" "${UNHEALTHY_TIMEOUT}" 2>/dev/null); then
        log "Fault ${event_id}: runtime observed failure reaction ${runtime_event}"
        append_events_file_record \
            "runtime_failover_reaction" \
            "event_id=${event_id}" \
            "dp_rank=${dp_rank}" \
            "pp_rank=${pp_rank}" \
            "tp_rank=${tp_rank}" \
            "node=${node_name}" \
            "node_url=${node_url}" \
            "status=runtime_reaction"
        return 0
    fi

    if wait_worker_unhealthy "${node_url}" "${UNHEALTHY_TIMEOUT}"; then
        log "Fault ${event_id}: node-local health became unhealthy at ${node_url}"
        append_events_file_record \
            "node_became_unhealthy" \
            "event_id=${event_id}" \
            "dp_rank=${dp_rank}" \
            "pp_rank=${pp_rank}" \
            "tp_rank=${tp_rank}" \
            "node=${node_name}" \
            "node_url=${node_url}" \
            "status=unhealthy"
        return 0
    fi

    log "Fault ${event_id}: no runtime reaction or health transition observed within ${UNHEALTHY_TIMEOUT}s"
    append_events_file_record \
        "runtime_reaction_timeout" \
        "event_id=${event_id}" \
        "dp_rank=${dp_rank}" \
        "pp_rank=${pp_rank}" \
        "tp_rank=${tp_rank}" \
        "node=${node_name}" \
        "node_url=${node_url}" \
        "status=timeout"
    return 1
}

delayed_inject_stage_kill_fault() {
    local event_id="$1"
    local trigger_kind="$2"
    local trigger_count="$3"
    local stage_spec="$4"
    log "Fault ${event_id}: waiting ${INJECT_DELAY}s after ${trigger_kind}=${trigger_count}"
    sleep "${INJECT_DELAY}"
    inject_stage_kill_fault "${event_id}" "${trigger_kind}" "${trigger_count}" "${stage_spec}"
}

delayed_inject_stage_group_kill_fault() {
    local event_id="$1"
    local trigger_kind="$2"
    local trigger_count="$3"
    shift 3
    local -a stage_specs=("$@")
    log "Fault ${event_id}: waiting ${INJECT_DELAY}s after ${trigger_kind}=${trigger_count}"
    sleep "${INJECT_DELAY}"

    local stage_spec=""
    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    local fault_start_epoch
    fault_start_epoch=$(date +%s)

    local -a kill_bg_pids=()
    local -a kill_stage_descs=()
    local first_dp_rank=""
    local first_node_url=""

    for stage_spec in "${stage_specs[@]}"; do
        IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${stage_spec}"
        if [[ -z "${first_dp_rank}" ]]; then
            first_dp_rank="${dp_rank}"
            first_node_url="${node_url}"
        fi

        log "Fault ${event_id}: killing dp=${dp_rank} pp=${pp_rank} tp=${tp_rank} on ${node_name} after ${trigger_kind}=${trigger_count}"
        append_events_file_record \
            "fault_injected" \
            "event_id=${event_id}" \
            "trigger_kind=${trigger_kind}" \
            "trigger_count=${trigger_count}" \
            "dp_rank=${dp_rank}" \
            "pp_rank=${pp_rank}" \
            "tp_rank=${tp_rank}" \
            "node=${node_name}" \
            "node_url=${node_url}" \
            "kill_pattern=${kill_pattern}" \
            "slurm_job_id=${SLURM_JOB_ID_VALUE}"

        (
            if kill_stage_via_slurm "${SLURM_JOB_ID_VALUE}" "${node_name}" "${kill_pattern}"; then
                log "Fault ${event_id}: kill command completed for ${kill_pattern}"
            else
                rc=$?
                log "Fault ${event_id}: failed to kill ${kill_pattern} (rc=${rc})"
                append_events_file_record \
                    "fault_injection_failed" \
                    "event_id=${event_id}" \
                    "trigger_kind=${trigger_kind}" \
                    "trigger_count=${trigger_count}" \
                    "dp_rank=${dp_rank}" \
                    "pp_rank=${pp_rank}" \
                    "tp_rank=${tp_rank}" \
                    "node=${node_name}" \
                    "node_url=${node_url}" \
                    "kill_pattern=${kill_pattern}" \
                    "status=kill_failed"
                exit "${rc}"
            fi
        ) &
        kill_bg_pids+=($!)
        kill_stage_descs+=("dp=${dp_rank} pp=${pp_rank} tp=${tp_rank} node=${node_name} pattern=${kill_pattern}")
    done

    local idx=0
    for pid in "${kill_bg_pids[@]}"; do
        wait "${pid}"
        local rc=$?
        if [[ "${rc}" -ne 0 ]]; then
            log "Fault ${event_id}: concurrent kill failed for ${kill_stage_descs[$idx]} (rc=${rc})"
            return "${rc}"
        fi
        idx=$((idx + 1))
    done

    local runtime_event=""
    if runtime_event=$(wait_for_runtime_reaction "${first_dp_rank}" "${fault_start_epoch}" "${UNHEALTHY_TIMEOUT}" 2>/dev/null); then
        log "Fault ${event_id}: runtime observed failure reaction ${runtime_event}"
        append_events_file_record \
            "runtime_failover_reaction" \
            "event_id=${event_id}" \
            "dp_rank=${first_dp_rank}" \
            "node_url=${first_node_url}" \
            "status=runtime_reaction"
        return 0
    fi

    if wait_worker_unhealthy "${first_node_url}" "${UNHEALTHY_TIMEOUT}"; then
        log "Fault ${event_id}: node-local health became unhealthy at ${first_node_url}"
        append_events_file_record \
            "node_became_unhealthy" \
            "event_id=${event_id}" \
            "dp_rank=${first_dp_rank}" \
            "node_url=${first_node_url}" \
            "status=unhealthy"
        return 0
    fi

    log "Fault ${event_id}: no runtime reaction or health transition observed within ${UNHEALTHY_TIMEOUT}s"
    append_events_file_record \
        "runtime_reaction_timeout" \
        "event_id=${event_id}" \
        "dp_rank=${first_dp_rank}" \
        "node_url=${first_node_url}" \
        "status=timeout"
    return 1
}

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
    local -a target_specs=()
    if [[ "${FAULT_PP_MATCH_ALL}" -eq 1 || "${FAULT_TP_MATCH_ALL}" -eq 1 ]]; then
        target_specs=("${STAGE_TARGETS[@]}")
    else
        local stage_index=$((NEXT_INJECT_IDX % ${#STAGE_TARGETS[@]}))
        collect_replica_group_targets "${STAGE_TARGETS[$stage_index]}" target_specs
    fi

    local first_spec="${target_specs[0]}"
    local dp_rank pp_rank tp_rank node_name node_url kill_pattern
    IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${first_spec}"

    log "Fault ${event_id}: trigger matched ${INJECT_KIND}=${current_count}, target-count=${#target_specs[@]}, primary=dp${dp_rank}/pp${pp_rank}/tp${tp_rank} on ${node_name}"
    local stage_spec
    for stage_spec in "${target_specs[@]}"; do
        IFS=$'\t' read -r dp_rank pp_rank tp_rank node_name node_url kill_pattern <<< "${stage_spec}"
        append_events_file_record \
            "fault_scheduled" \
            "event_id=${event_id}" \
            "trigger_kind=${INJECT_KIND}" \
            "trigger_count=${current_count}" \
            "dp_rank=${dp_rank}" \
            "pp_rank=${pp_rank}" \
            "tp_rank=${tp_rank}" \
            "node=${node_name}" \
            "node_url=${node_url}" \
            "inject_delay=${INJECT_DELAY}" \
            "task_label=${task_label}" \
            "agent_role=${agent_role}" \
            "worker_id=${worker_id}" \
            "job_id=${job_id}" \
            "job=${job_name}"
    done
    delayed_inject_stage_group_kill_fault "${event_id}" "${INJECT_KIND}" "${current_count}" "${target_specs[@]}" &
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

extract_event_field() {
    local event_line="$1"
    local field_name="$2"
    python3 - "${event_line}" "${field_name}" <<'PY'
import json
import sys

event = json.loads(sys.argv[1])
value = event.get(sys.argv[2], "")
if value is None:
    value = ""
print(value)
PY
}

process_new_events() {
    local start_line="$1"
    local end_line="$2"
    local event_line
    while IFS= read -r event_line || [[ -n "${event_line}" ]]; do
        [ -z "${event_line}" ] && continue
        local event_type
        event_type=$(extract_event_field "${event_line}" "event")
        case "${event_type}" in
            job_completed)
                JOB_DONE_COUNT=$((JOB_DONE_COUNT + 1))
                local job_id
                local job_name
                local job_worker_id
                job_id=$(extract_event_field "${event_line}" "job_id")
                job_name=$(extract_event_field "${event_line}" "job")
                job_worker_id=$(extract_event_field "${event_line}" "worker_id")
                log "Observed job completion ${JOB_DONE_COUNT}: job_id=${job_id} job=${job_name}"
                if [[ "${INJECT_KIND}" == "job" ]]; then
                    schedule_injection_if_needed "${JOB_DONE_COUNT}" "" "" "${job_worker_id}" "${job_id}" "${job_name}"
                fi
                ;;
            job_failed)
                local failed_job_id
                local failed_job_name
                failed_job_id=$(extract_event_field "${event_line}" "job_id")
                failed_job_name=$(extract_event_field "${event_line}" "job")
                log "Observed job failure: job_id=${failed_job_id} job=${failed_job_name}"
                ;;
            task_completed)
                local task_label
                local agent_role
                local worker_id
                local job_id
                local job_name
                task_label=$(extract_event_field "${event_line}" "task_label")
                agent_role=$(extract_event_field "${event_line}" "agent_role")
                worker_id=$(extract_event_field "${event_line}" "worker_id")
                job_id=$(extract_event_field "${event_line}" "job_id")
                job_name=$(extract_event_field "${event_line}" "job")
                if ! task_event_matches_filters "${task_label}" "${agent_role}" "${worker_id}"; then
                    log "Observed task completion (ignored by filter): ${task_label} -> ${agent_role}"
                    continue
                fi
                TASK_DONE_COUNT=$((TASK_DONE_COUNT + 1))
                log "Observed task completion ${TASK_DONE_COUNT}: ${task_label} -> ${agent_role}"
                if [[ "${INJECT_KIND}" == "task" ]]; then
                    schedule_injection_if_needed "${TASK_DONE_COUNT}" "${task_label}" "${agent_role}" "${worker_id}" "${job_id}" "${job_name}"
                fi
                ;;
            task_failed)
                local failed_task_label
                failed_task_label=$(extract_event_field "${event_line}" "task_label")
                log "Observed task failure: ${failed_task_label}"
                ;;
        esac
    done < <(sed -n "${start_line},${end_line}p" "${EVENTS_FILE}")
}

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
