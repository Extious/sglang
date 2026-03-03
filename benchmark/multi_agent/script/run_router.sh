#!/bin/bash
# Launch the router that load-balances across workers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29000}"
PROMETHEUS_HOST="${PROMETHEUS_HOST:-0.0.0.0}"

WORKER_BASE_PORT="${WORKER_BASE_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WORKER_URL_HOST="${WORKER_URL_HOST:-127.0.0.1}"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/../logs/worker_urls.txt}"
WORKER_URLS="${WORKER_URLS:-}"
AUTO_PORT_PICK="${AUTO_PORT_PICK:-1}"
KILL_EXISTING_ROUTER="${KILL_EXISTING_ROUTER:-0}"

usage() {
    cat <<EOF
Usage:
  ./run_router.sh [options]

Options:
  --worker-urls-file PATH     Read worker URLs from file (one URL per line)
  --worker-urls URLS          Worker URLs as a quoted string
  --policy NAME               Router policy (default: ${ROUTER_POLICY})
  --host HOST                 Router host (default: ${ROUTER_HOST})
  --port PORT                 Router port (default: ${ROUTER_PORT})
  --model-path PATH           Model path (default: ${MODEL_PATH})
  --prometheus-port PORT      Prometheus port (default: ${PROMETHEUS_PORT})
  --prometheus-host HOST      Prometheus host (default: ${PROMETHEUS_HOST})
  --num-workers N             Fallback worker count when file/urls not provided
  --worker-base-port PORT     Fallback worker base port
  --worker-url-host HOST      Fallback worker host
  --kill-existing-router      Kill existing sglang::router that owns conflicted port(s)
  --no-auto-port              Disable automatic free-port fallback
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker-urls-file)
            WORKER_URLS_FILE="$2"
            shift 2
            ;;
        --worker-urls)
            WORKER_URLS="$2"
            shift 2
            ;;
        --policy)
            ROUTER_POLICY="$2"
            shift 2
            ;;
        --host)
            ROUTER_HOST="$2"
            shift 2
            ;;
        --port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --prometheus-host)
            PROMETHEUS_HOST="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --worker-base-port)
            WORKER_BASE_PORT="$2"
            shift 2
            ;;
        --worker-url-host)
            WORKER_URL_HOST="$2"
            shift 2
            ;;
        --kill-existing-router)
            KILL_EXISTING_ROUTER=1
            shift
            ;;
        --no-auto-port)
            AUTO_PORT_PICK=0
            shift
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

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# Keep proxy env by default so outbound internet can use proxy.
# We only carve out local destinations via NO_PROXY/no_proxy.
EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"

NO_PROXY_COMBINED="localhost,127.0.0.1,0.0.0.0,::1"
append_no_proxy() {
    local item="$1"
    [ -z "${item}" ] && return
    case ",${NO_PROXY_COMBINED}," in
        *,"${item}",*) ;;
        *) NO_PROXY_COMBINED="${NO_PROXY_COMBINED},${item}" ;;
    esac
}

extract_url_host() {
    local url="$1"
    local authority="${url#*://}"
    authority="${authority%%/*}"
    if [[ "${authority}" == \[* ]]; then
        local host="${authority#\[}"
        echo "${host%%]*}"
    else
        echo "${authority%%:*}"
    fi
}

IFS=',' read -r -a EXISTING_NO_PROXY_ARR <<< "${EXISTING_NO_PROXY}"
for item in "${EXISTING_NO_PROXY_ARR[@]}"; do
    item="$(echo "${item}" | xargs)"
    append_no_proxy "${item}"
done

port_in_use() {
    local p="$1"
    ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN
}

port_owner_pids() {
    local p="$1"
    ss -ltnp "sport = :${p}" 2>/dev/null | awk -F'pid=' '/pid=/{split($2,a,","); print a[1]}' | sort -u
}

find_free_port() {
    local start="$1"
    local p="$start"
    while port_in_use "${p}"; do
        p=$((p + 1))
    done
    echo "${p}"
}

kill_router_on_port_if_requested() {
    local p="$1"
    if [ "${KILL_EXISTING_ROUTER}" -ne 1 ]; then
        return
    fi
    local pid
    for pid in $(port_owner_pids "${p}"); do
        if ps -p "${pid}" -o comm= 2>/dev/null | grep -q "sglang::router"; then
            echo "Killing existing router pid=${pid} on port ${p}"
            kill "${pid}" || true
        fi
    done
}

if [ -z "${WORKER_URLS}" ] && [ -f "${WORKER_URLS_FILE}" ]; then
    WORKER_URLS="$(grep -Eo 'https?://[^[:space:]]+' "${WORKER_URLS_FILE}" | xargs || true)"
fi

if [ -z "${WORKER_URLS}" ]; then
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        WORKER_URLS="${WORKER_URLS} http://${WORKER_URL_HOST}:${port}"
    done
    WORKER_URLS="${WORKER_URLS# }"
fi

if [ -z "${WORKER_URLS}" ]; then
    echo "ERROR: No worker URLs found." >&2
    exit 1
fi

kill_router_on_port_if_requested "${ROUTER_PORT}"
kill_router_on_port_if_requested "${PROMETHEUS_PORT}"
sleep 1

if port_in_use "${ROUTER_PORT}"; then
    if [ "${AUTO_PORT_PICK}" -eq 1 ]; then
        new_port="$(find_free_port "${ROUTER_PORT}")"
        echo "WARNING: Router port ${ROUTER_PORT} is in use, switching to ${new_port}."
        ROUTER_PORT="${new_port}"
    else
        echo "ERROR: Router port ${ROUTER_PORT} is already in use." >&2
        echo "Hint: rerun with --kill-existing-router or choose --port." >&2
        exit 1
    fi
fi

if port_in_use "${PROMETHEUS_PORT}"; then
    if [ "${AUTO_PORT_PICK}" -eq 1 ]; then
        new_prom_port="$(find_free_port "${PROMETHEUS_PORT}")"
        echo "WARNING: Prometheus port ${PROMETHEUS_PORT} is in use, switching to ${new_prom_port}."
        PROMETHEUS_PORT="${new_prom_port}"
    else
        echo "ERROR: Prometheus port ${PROMETHEUS_PORT} is already in use." >&2
        echo "Hint: rerun with --prometheus-port or --kill-existing-router." >&2
        exit 1
    fi
fi

read -r -a WORKER_URLS_ARR <<< "${WORKER_URLS}"
for worker_url in "${WORKER_URLS_ARR[@]}"; do
    append_no_proxy "$(extract_url_host "${worker_url}")"
done
if [ "${ROUTER_HOST}" != "0.0.0.0" ] && [ "${ROUTER_HOST}" != "::" ]; then
    append_no_proxy "${ROUTER_HOST}"
fi
append_no_proxy "$(hostname -s)"
append_no_proxy "$(hostname -f 2>/dev/null || true)"

export NO_PROXY="${NO_PROXY_COMBINED}"
export no_proxy="${NO_PROXY_COMBINED}"

LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"
ROUTER_URL_FILE="${ROUTER_URL_FILE:-${LOG_DIR}/router_url.txt}"
ROUTER_NODE_FILE="${ROUTER_NODE_FILE:-${LOG_DIR}/router_node.txt}"

router_record_host() {
    if [ "${ROUTER_HOST}" = "0.0.0.0" ] || [ "${ROUTER_HOST}" = "::" ]; then
        hostname -s
    else
        echo "${ROUTER_HOST}"
    fi
}

router_health_ok() {
    local targets=()
    targets+=("127.0.0.1")
    if [ "${ROUTER_HOST}" != "0.0.0.0" ] && [ "${ROUTER_HOST}" != "::" ]; then
        targets+=("${ROUTER_HOST}")
    fi
    targets+=("$(hostname -s)")

    local t
    for t in "${targets[@]}"; do
        [ -z "${t}" ] && continue
        if curl --noproxy '*' -fsS -m 2 "http://${t}:${ROUTER_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

echo "Starting router:"
echo "  host: ${ROUTER_HOST}"
echo "  port: ${ROUTER_PORT}"
echo "  prometheus: ${PROMETHEUS_HOST}:${PROMETHEUS_PORT}"
echo "  policy: ${ROUTER_POLICY}"
echo "  model: ${MODEL_PATH}"
echo "  workers: ${WORKER_URLS}"
echo "  NO_PROXY: ${NO_PROXY}"

python -m sglang_router.launch_router \
    --worker-urls "${WORKER_URLS_ARR[@]}" \
    --policy "${ROUTER_POLICY}" \
    --host "${ROUTER_HOST}" \
    --port "${ROUTER_PORT}" \
    --prometheus-host "${PROMETHEUS_HOST}" \
    --prometheus-port "${PROMETHEUS_PORT}" \
    --model-path "${MODEL_PATH}" &
ROUTER_PID=$!

READY=0
START_TS=$(date +%s)
while true; do
    if router_health_ok; then
        READY=1
        break
    fi

    if ! kill -0 "${ROUTER_PID}" >/dev/null 2>&1; then
        break
    fi

    if [ $(( $(date +%s) - START_TS )) -ge 60 ]; then
        break
    fi
    sleep 1
done

if [ "${READY}" -eq 1 ]; then
    ROUTER_RECORD_HOST="$(router_record_host)"
    printf "http://%s:%s\n" "${ROUTER_RECORD_HOST}" "${ROUTER_PORT}" > "${ROUTER_URL_FILE}"
    printf "%s\n" "${ROUTER_RECORD_HOST}" > "${ROUTER_NODE_FILE}"
    echo "Router is healthy. Recorded URL: $(cat "${ROUTER_URL_FILE}")"
else
    echo "WARNING: Router health check did not pass within startup window; router_url.txt not updated." >&2
fi

set +e
wait "${ROUTER_PID}"
RC=$?
set -e
exit "${RC}"
