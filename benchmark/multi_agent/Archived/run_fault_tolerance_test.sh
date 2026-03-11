#!/bin/bash
# Complete fault tolerance test with optimized configuration
#
# This script:
# 1. Stops existing router
# 2. Starts router with optimized timeout settings
# 3. Runs heavy_swarm with fault injection
#
# Expected results:
# - Normal task latency: ~10-15s
# - Fault period latency: ~15-20s (2x normal)
# - vs previous: 100+ seconds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/../logs/worker_urls.txt}"
SLURM_JOB_ID="${SLURM_JOB_ID:-$(squeue -u $USER -h -o "%i" | head -1)}"
FAILED_WORKER_URL="${FAILED_WORKER_URL:-http://hkbugpusrv10:8000}"
INJECT_AFTER="${INJECT_AFTER:-300}"
RECOVER_AFTER="${RECOVER_AFTER:-30}"

echo "=========================================="
echo "Fault Tolerance Test Setup"
echo "=========================================="
echo "Worker URLs file: $WORKER_URLS_FILE"
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Failed worker:    $FAILED_WORKER_URL"
echo "Inject after:     ${INJECT_AFTER}s"
echo "Recover after:    ${RECOVER_AFTER}s"
echo "=========================================="
echo ""

# Step 1: Stop existing router
echo "[1/3] Stopping existing router..."
pkill -f "sglang::router" || echo "No existing router found"
sleep 2

# Step 2: Start optimized router
echo "[2/3] Starting router with optimized settings..."
nohup "${SCRIPT_DIR}/run_router.sh" \
    --worker-urls-file "$WORKER_URLS_FILE" \
    > "${SCRIPT_DIR}/../logs/router_optimized.log" 2>&1 &

ROUTER_PID=$!
echo "Router started with PID: $ROUTER_PID"

# Wait for router to be ready
echo "Waiting for router to be ready..."
for i in {1..30}; do
    if curl --noproxy '*' -fsS -m 2 http://127.0.0.1:30000/health >/dev/null 2>&1; then
        echo "Router is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Router failed to start" >&2
        exit 1
    fi
    sleep 1
done

# Step 3: Run fault injection test
echo ""
echo "[3/3] Starting fault injection test..."
echo "Log file: ${SCRIPT_DIR}/../logs/fault_injection_optimized_$(date +%Y%m%d_%H%M%S).log"
echo ""

exec "${SCRIPT_DIR}/run_heavy_swarm_with_fault_injection.sh" \
    --failed-worker-url "$FAILED_WORKER_URL" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --inject-after "$INJECT_AFTER" \
    --recover-after "$RECOVER_AFTER" \
    --log-file "${SCRIPT_DIR}/../logs/fault_injection_optimized_$(date +%Y%m%d_%H%M%S).log"
