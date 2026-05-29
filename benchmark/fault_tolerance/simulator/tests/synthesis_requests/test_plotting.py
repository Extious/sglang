import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.plot.backup_cache_hits import (
    build_request_matrix,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.plot.trace_log_profile import (
    build_runtime_series,
)


def test_build_request_matrix_uses_request_ids_without_task_or_agent_labels(tmp_path):
    cache_csv = tmp_path / "cache_hits.csv"
    cache_csv.write_text(
        "\n".join(
            [
                "job_id,rid,backup_policy,l1_match,l2_match,remote_match,remote_prefetch,reused_device,reused_host,reused_storage,is_failover_retried",
                "2,r2,host,4,3,0,0,4,3,0,true",
                "1,r1,remote_backup,0,0,6,5,0,0,6,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    request_ids, metrics, retried = build_request_matrix(cache_csv)

    assert request_ids == ["1", "2"]
    assert metrics["remote_match"] == [6, 0]
    assert metrics["reused_storage"] == [6, 0]
    assert metrics["l2_match"] == [0, 3]
    assert retried == [False, True]


def test_build_runtime_series_uses_request_detail_components(tmp_path):
    detail_csv = tmp_path / "request_detail.csv"
    detail_csv.write_text(
        "\n".join(
            [
                "job_id,rid,waiting_time_s,prefetch_time_s,backup_time_s,inference_time_s,failover_penalty_s,total_latency_s,is_failover_retried",
                "2,r2,0.2,0.1,0.3,1.4,0.5,2.5,true",
                "1,r1,0.1,0.0,0.2,1.0,0.0,1.3,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    request_ids, components, total_latency, retried = build_runtime_series(detail_csv)

    assert request_ids == ["1", "2"]
    assert components["waiting_time_s"] == [0.1, 0.2]
    assert components["backup_time_s"] == [0.2, 0.3]
    assert components["failover_penalty_s"] == [0.0, 0.5]
    assert total_latency == [1.3, 2.5]
    assert retried == [False, True]
