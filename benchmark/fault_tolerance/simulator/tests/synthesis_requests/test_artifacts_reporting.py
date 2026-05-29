import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.artifacts import (
    collect_jsonl_records,
    copy_simulator_artifacts,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.reporting import (
    write_outputs,
)


def _write_jsonl(path: Path, rows: list[dict], *, include_blank: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    if include_blank:
        lines.insert(1, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_collect_jsonl_records_reads_dp_dirs_in_order_and_ignores_blanks(tmp_path):
    _write_jsonl(
        tmp_path / "dp_0" / "request.jsonl",
        [{"rid": "dp0-a"}, {"rid": "dp0-b"}],
        include_blank=True,
    )
    _write_jsonl(tmp_path / "dp_1" / "request.jsonl", [{"rid": "dp1-a"}])
    _write_jsonl(tmp_path / "request.jsonl", [{"rid": "root"}])

    records = collect_jsonl_records(tmp_path, "request.jsonl")

    assert records == [{"rid": "dp0-a"}, {"rid": "dp0-b"}, {"rid": "dp1-a"}]


def test_collect_jsonl_records_falls_back_to_base_dir_when_no_dp_dirs(tmp_path):
    _write_jsonl(
        tmp_path / "request.jsonl",
        [{"rid": "root-a"}, {"rid": "root-b"}],
        include_blank=True,
    )

    assert collect_jsonl_records(tmp_path, "request.jsonl") == [
        {"rid": "root-a"},
        {"rid": "root-b"},
    ]


def test_copy_simulator_artifacts_flattens_dp_outputs_in_order(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    for dp_rank in (0, 1):
        (raw_dir / f"dp_{dp_rank}").mkdir(parents=True)

    (raw_dir / "dp_0" / "request.jsonl").write_text(
        '{"rid": "r0"}\n{"rid": "r1"}\n', encoding="utf-8"
    )
    (raw_dir / "dp_1" / "request.jsonl").write_text(
        '{"rid": "r2"}\n', encoding="utf-8"
    )
    (raw_dir / "dp_0" / "iteration.jsonl").write_text(
        '{"iter": 0}\n', encoding="utf-8"
    )
    (raw_dir / "dp_1" / "iteration.jsonl").write_text(
        '{"iter": 1}\n', encoding="utf-8"
    )
    (raw_dir / "dp_0" / "failure_events.jsonl").write_text(
        '{"event": "down"}\n', encoding="utf-8"
    )
    (raw_dir / "dp_1" / "failure_events.jsonl").write_text(
        '{"event": "up"}\n', encoding="utf-8"
    )

    copy_simulator_artifacts(raw_dir, output_dir)

    assert (output_dir / "simulator_request.jsonl").read_text(
        encoding="utf-8"
    ) == '{"rid": "r0"}\n{"rid": "r1"}\n{"rid": "r2"}\n'
    assert (output_dir / "simulator_iteration.jsonl").read_text(
        encoding="utf-8"
    ) == '{"iter": 0}\n{"iter": 1}\n'
    assert (output_dir / "sim_failure_events.jsonl").read_text(
        encoding="utf-8"
    ) == '{"event": "down"}\n{"event": "up"}\n'


def test_write_outputs_creates_csvs_with_reporting_fallbacks(tmp_path):
    request_stats = [
        {
            "rid": "r1",
            "job_id": "1",
            "worker_id": "1",
            "assigned_dp_rank": 0,
            "backup_policy": "remote_backup",
            "status": "completed",
            "created_time": 0.0,
            "queue_start": 0.0,
            "queue_end": 0.5,
            "finish_time": 1.7,
            "queue_time_s": 0.5,
            "prefetch_time_s": 0.2,
            "backup_time_s": 0.1,
            "inference_time_s": 1.0,
            "prefill_inference_time_s": 0.3,
            "decode_inference_time_s": 0.7,
            "is_failover_retried": True,
            "failure_impacted": True,
            "failure_time_s": 1.2,
            "recovery_time_s": -1,
            "pre_failover_output_tokens": 2,
            "pre_failover_backed_up_tokens": 2,
            "retry_prefill_tokens": 10,
            "failover_penalty_s": 0.2,
            "final_reused_tokens": 0,
            "prefetch_complete_tokens": 3,
        },
        {
            "rid": "r2",
            "job_id": "1",
            "worker_id": "2",
            "assigned_dp_rank": 1,
            "backup_policy": "host",
            "status": "failed_over",
            "created_time": 2.0,
            "last_event_time": 5.0,
            "queue_time_s": 0.1,
            "prefetch_time_s": 0.2,
            "backup_time_s": 0.3,
            "inference_time_s": 0.4,
            "failure_impacted": False,
            "pre_failover_backed_up_tokens": 4,
            "final_reused_tokens": 4,
        },
        {
            "rid": "r3",
            "job_id": "2",
            "backup_policy": "none",
            "status": "completed",
            "waiting_time_s": 0.8,
            "prefetch_time_s": 0.1,
            "backup_time_s": 0.1,
            "inference_time_s": 0.2,
            "failover_penalty_s": 0.9,
        },
    ]

    write_outputs(tmp_path, request_stats)

    for filename in (
        "job_summary.csv",
        "task_summary.csv",
        "cache_hits.csv",
        "request_detail.csv",
    ):
        assert (tmp_path / filename).is_file()

    detail = _read_csv(tmp_path / "request_detail.csv")
    assert detail[0]["waiting_time_s"] == "0.5"
    assert detail[0]["total_latency_s"] == "1.7"
    assert detail[1]["finish_time_s"] == "5.0"
    assert detail[1]["total_latency_s"] == "3.0"
    assert detail[2]["waiting_time_s"] == "0.8"
    assert detail[2]["finish_time_s"] == "-1.0"
    assert detail[2]["total_latency_s"] == "1.2"

    cache = _read_csv(tmp_path / "cache_hits.csv")
    assert "task_label" not in cache[0]
    assert "agent_role" not in cache[0]
    assert cache[0]["l1_match"] == "0"
    assert cache[0]["reused_device"] == "0"
    assert cache[0]["remote_match"] == "2"
    assert cache[0]["remote_prefetch"] == "3"
    assert cache[0]["reused_storage"] == "2"
    assert cache[0]["is_failover_retried"] == "True"
    assert cache[0]["pre_failover_output_tokens"] == "2"
    assert cache[0]["pre_failover_backed_up_tokens"] == "2"
    assert cache[1]["l1_match"] == "4"
    assert cache[1]["l2_match"] == "4"
    assert cache[1]["reused_device"] == "4"
    assert cache[1]["reused_host"] == "4"

    job_summary = _read_csv(tmp_path / "job_summary.csv")
    assert job_summary[0]["job_id"] == "1"
    assert job_summary[0]["request_count"] == "2"
    assert job_summary[0]["completed_count"] == "1"
    assert job_summary[0]["failed_over_attempts"] == "1"
    assert job_summary[0]["failure_impacted_count"] == "1"
    assert job_summary[0]["waiting_time_s"] == "0.6"
    assert job_summary[0]["total_latency_s"] == "4.7"
