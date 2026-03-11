#!/bin/bash
# Complete deployment and fault tolerance testing pipeline
#
# This script:
# 1. Deploys workers via SLURM
# 2. Deploys router with optimized settings
# 3. Runs fault injection test
# 4. Monitors and reports results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-6}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-6}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
INJECT_AFTER="${INJECT_AFTER:-300}"
RECOVER_AFTER="${RECOVER_AFTER:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# ============================================================================
# Step 1: Deploy Workers via SLURM
# ============================================================================

deploy_workers() {
    log_info "Step 1: Deploying workers via SLURM..."

    # Cancel existing worker jobs
    log_info "Checking for existing worker jobs..."
    local existing_jobs=$(squeue -u $USER -n exp1_server -h -o "%i" || true)
    if [ -n "$existing_jobs" ]; then
        log_warning "Cancelling existing worker jobs: $existing_jobs"
        scancel $existing_jobs || true
        sleep 5
    fi

    # Submit new worker job
    log_info "Submitting SLURM job for workers..."
    cd "${REPO_ROOT}/benchmark/multi_agent/slurm"

    local job_output=$(sbatch --nodes=1 run_server.slurm \
        --num-workers ${NUM_WORKERS} \
        --workers-per-node ${WORKERS_PER_NODE} \
        --model-path "${MODEL_PATH}" 2>&1)

    if [ $? -ne 0 ]; then
        log_error "Failed to submit SLURM job"
        echo "$job_output"
        return 1
    fi

    local job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
    if [ -z "$job_id" ]; then
        log_error "Could not extract job ID from: $job_output"
        return 1
    fi

    log_success "SLURM job submitted: Job ID = $job_id"
    echo "$job_id" > "${SCRIPT_DIR}/../logs/current_slurm_job_id.txt"

    # Wait for workers to be ready
    log_info "Waiting for workers to be ready (timeout: 10 minutes)..."
    local worker_urls_file="${REPO_ROOT}/benchmark/multi_agent/logs/worker_urls.txt"
    local timeout=600
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if [ -f "$worker_urls_file" ] && [ -s "$worker_urls_file" ]; then
            local worker_count=$(wc -l < "$worker_urls_file")
            if [ "$worker_count" -ge "$NUM_WORKERS" ]; then
                log_success "Workers are ready! Found $worker_count workers"
                cat "$worker_urls_file"
                return 0
            fi
        fi

        # Check if job is still running
        if ! squeue -j $job_id -h >/dev/null 2>&1; then
            log_error "SLURM job $job_id is no longer running"
            log_info "Checking job logs..."
            local log_file="${REPO_ROOT}/benchmark/multi_agent/logs/run_server_${job_id}.out"
            if [ -f "$log_file" ]; then
                tail -50 "$log_file"
            fi
            return 1
        fi

        sleep 10
        elapsed=$((elapsed + 10))
        if [ $((elapsed % 60)) -eq 0 ]; then
            log_info "Still waiting... ${elapsed}s / ${timeout}s"
        fi
    done

    log_error "Timeout waiting for workers to be ready"
    return 1
}

# ============================================================================
# Step 2: Deploy Router
# ============================================================================

deploy_router() {
    log_info "Step 2: Deploying router with optimized settings..."

    # Stop existing router
    log_info "Stopping existing router..."
    pkill -f "sglang::router" || true
    sleep 2

    # Start router
    log_info "Starting router..."
    cd "${SCRIPT_DIR}"

    local worker_urls_file="${REPO_ROOT}/benchmark/multi_agent/logs/worker_urls.txt"
    if [ ! -f "$worker_urls_file" ]; then
        log_error "Worker URLs file not found: $worker_urls_file"
        return 1
    fi

    nohup ./run_router.sh --worker-urls-file "$worker_urls_file" \
        > "${REPO_ROOT}/benchmark/multi_agent/logs/router_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    local router_pid=$!
    log_info "Router started with PID: $router_pid"

    # Wait for router to be ready
    log_info "Waiting for router to be ready..."
    local timeout=60
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if curl --noproxy '*' -fsS -m 2 http://127.0.0.1:30000/health >/dev/null 2>&1; then
            log_success "Router is ready!"
            return 0
        fi

        if ! kill -0 $router_pid 2>/dev/null; then
            log_error "Router process died"
            return 1
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    log_error "Timeout waiting for router to be ready"
    return 1
}

# ============================================================================
# Step 3: Run Fault Injection Test
# ============================================================================

run_fault_injection_test() {
    log_info "Step 3: Running fault injection test..."

    # Get SLURM job ID
    local job_id_file="${SCRIPT_DIR}/../logs/current_slurm_job_id.txt"
    if [ ! -f "$job_id_file" ]; then
        log_error "SLURM job ID file not found"
        return 1
    fi

    local slurm_job_id=$(cat "$job_id_file")
    log_info "Using SLURM Job ID: $slurm_job_id"

    # Get first worker URL
    local worker_urls_file="${REPO_ROOT}/benchmark/multi_agent/logs/worker_urls.txt"
    local first_worker=$(head -1 "$worker_urls_file")
    log_info "Target worker for fault injection: $first_worker"

    # Run test
    log_info "Starting fault injection test..."
    log_info "  Inject after: ${INJECT_AFTER}s"
    log_info "  Recover after: ${RECOVER_AFTER}s"

    cd "${SCRIPT_DIR}"
    ./run_heavy_swarm_with_fault_injection.sh \
        --failed-worker-url "$first_worker" \
        --slurm-job-id "$slurm_job_id" \
        --inject-after "$INJECT_AFTER" \
        --recover-after "$RECOVER_AFTER" \
        --task-limit 80 \
        --task-processes 6

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Fault injection test completed successfully"
        return 0
    else
        log_error "Fault injection test failed with exit code: $exit_code"
        return 1
    fi
}

# ============================================================================
# Step 4: Analyze Results
# ============================================================================

analyze_results() {
    log_info "Step 4: Analyzing results..."

    local timing_dir="${HOME}/swarms/agent_workspace/timing_reports"

    if [ ! -d "$timing_dir" ]; then
        log_warning "Timing reports directory not found"
        return 1
    fi

    # Find latest timing report
    local latest_csv=$(ls -t "$timing_dir"/task_timing_summary_*.csv 2>/dev/null | head -1)

    if [ -z "$latest_csv" ]; then
        log_warning "No timing reports found"
        return 1
    fi

    log_success "Latest timing report: $latest_csv"

    # Calculate statistics
    log_info "Calculating task duration statistics..."

    python3 - <<EOF
import csv
import statistics

with open('$latest_csv', 'r') as f:
    reader = csv.DictReader(f)
    durations = [float(row['task_duration_seconds']) for row in reader]

if durations:
    print(f"Total tasks: {len(durations)}")
    print(f"Mean duration: {statistics.mean(durations):.2f}s")
    print(f"Median duration: {statistics.median(durations):.2f}s")
    print(f"Min duration: {min(durations):.2f}s")
    print(f"Max duration: {max(durations):.2f}s")
    print(f"Std deviation: {statistics.stdev(durations):.2f}s")

    # Identify outliers (> 2x median)
    median = statistics.median(durations)
    outliers = [(i, d) for i, d in enumerate(durations) if d > 2 * median]

    if outliers:
        print(f"\nOutliers (> 2x median = {2*median:.2f}s):")
        for task_idx, duration in outliers[:10]:  # Show first 10
            print(f"  Task {task_idx}: {duration:.2f}s")
else:
    print("No data found")
EOF

    log_info "Timing reports available at: $timing_dir"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "=========================================="
    echo "Fault Tolerance Testing Pipeline"
    echo "=========================================="
    echo "Configuration:"
    echo "  Workers: $NUM_WORKERS"
    echo "  Workers per node: $WORKERS_PER_NODE"
    echo "  Model: $MODEL_PATH"
    echo "  Inject after: ${INJECT_AFTER}s"
    echo "  Recover after: ${RECOVER_AFTER}s"
    echo "=========================================="
    echo ""

    # Step 1: Deploy workers
    if ! deploy_workers; then
        log_error "Failed to deploy workers"
        exit 1
    fi
    echo ""

    # Step 2: Deploy router
    if ! deploy_router; then
        log_error "Failed to deploy router"
        exit 1
    fi
    echo ""

    # Step 3: Run fault injection test
    if ! run_fault_injection_test; then
        log_error "Fault injection test failed"
        exit 1
    fi
    echo ""

    # Step 4: Analyze results
    analyze_results
    echo ""

    log_success "Pipeline completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Check timing reports: ls -lt ${HOME}/swarms/agent_workspace/timing_reports/"
    echo "  2. View logs: ls -lt ${REPO_ROOT}/benchmark/multi_agent/logs/"
    echo "  3. Stop workers: scancel \$(cat ${SCRIPT_DIR}/../logs/current_slurm_job_id.txt)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --inject-after)
            INJECT_AFTER="$2"
            shift 2
            ;;
        --recover-after)
            RECOVER_AFTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num-workers N       Number of workers (default: 6)"
            echo "  --inject-after N      Inject fault after N seconds (default: 300)"
            echo "  --recover-after N     Recover after N seconds (default: 30)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
