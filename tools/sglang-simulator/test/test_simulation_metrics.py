from sglang_simulator.simulation.types import RequestStats
from sglang_simulator.simulation.utils import calc_metrics


def test_calc_metrics_skips_completed_requests_without_token_latencies():
    requests = [
        RequestStats(
            rid="missing-latencies",
            status="completed",
            input_length=10,
            output_length=5,
            last_event_time=1.0,
            queue_start=0.0,
            queue_end=0.2,
            gen_token_latencies=[],
        ),
        RequestStats(
            rid="normal",
            status="completed",
            input_length=8,
            output_length=3,
            last_event_time=2.0,
            queue_start=0.0,
            queue_end=0.5,
            gen_token_latencies=[0.1, 0.2, 0.3],
        ),
    ]

    metrics = calc_metrics(requests)

    assert metrics["num_requests"] == 1
    assert metrics["completed"] == 1
    assert metrics["total_input"] == 8
    assert metrics["total_output"] == 3
    assert metrics["mean_ttft_ms"] == 100.0


def test_calc_metrics_skips_failed_over_attempts():
    requests = [
        RequestStats(
            rid="old-attempt",
            status="failed_over",
            input_length=10,
            output_length=5,
            last_event_time=1.0,
            queue_start=0.0,
            queue_end=0.2,
            gen_token_latencies=[0.1, 0.2],
        ),
        RequestStats(
            rid="retry-attempt",
            status="completed",
            input_length=12,
            output_length=3,
            last_event_time=2.0,
            queue_start=0.0,
            queue_end=0.5,
            gen_token_latencies=[0.4, 0.2, 0.1],
        ),
    ]

    metrics = calc_metrics(requests)

    assert metrics["num_requests"] == 1
    assert metrics["completed"] == 1
    assert metrics["request_throughput"] == 0.5
    assert metrics["total_input"] == 12
    assert metrics["total_output"] == 3
    assert metrics["mean_ttft_ms"] == 400.0
