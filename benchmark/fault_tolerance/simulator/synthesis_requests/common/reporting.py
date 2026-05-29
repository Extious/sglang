from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


REQUEST_DETAIL_COLUMNS = [
    "job_id",
    "rid",
    "worker_id",
    "assigned_dp_rank",
    "backup_policy",
    "status",
    "created_time_s",
    "queue_start_s",
    "queue_end_s",
    "finish_time_s",
    "waiting_time_s",
    "prefetch_time_s",
    "backup_time_s",
    "inference_time_s",
    "total_latency_s",
    "prefill_inference_time_s",
    "decode_inference_time_s",
    "failure_impacted",
    "is_failover_retried",
    "failure_time_s",
    "recovery_time_s",
    "pre_failover_output_tokens",
    "pre_failover_backed_up_tokens",
    "retry_prefill_tokens",
    "failover_penalty_s",
]

CACHE_HIT_COLUMNS = [
    "job_id",
    "rid",
    "worker_id",
    "assigned_dp_rank",
    "backup_policy",
    "l1_match",
    "l2_match",
    "reused_host",
    "remote_match",
    "remote_prefetch",
    "reused_device",
    "reused_storage",
    "is_failover_retried",
    "pre_failover_output_tokens",
    "pre_failover_backed_up_tokens",
    "final_reused_tokens",
    "prefetch_complete_tokens",
]

TASK_SUMMARY_COLUMNS = [
    "job_id",
    "rid",
    "worker_id",
    "status",
    "backup_policy",
    "waiting_time_s",
    "prefetch_time_s",
    "backup_time_s",
    "inference_time_s",
    "total_latency_s",
    "failure_impacted",
    "is_failover_retried",
]

JOB_SUMMARY_COLUMNS = [
    "job_id",
    "request_count",
    "completed_count",
    "failed_over_attempts",
    "failure_impacted_count",
    "waiting_time_s",
    "prefetch_time_s",
    "backup_time_s",
    "inference_time_s",
    "total_latency_s",
]


def write_outputs(output_dir: str | Path, request_stats: Iterable[Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = [_normalize_request(row) for row in request_stats]
    detail_rows = [_request_detail_row(row) for row in rows]

    _write_csv(output_path / "request_detail.csv", REQUEST_DETAIL_COLUMNS, detail_rows)
    _write_csv(
        output_path / "cache_hits.csv",
        CACHE_HIT_COLUMNS,
        [_cache_hit_row(row) for row in rows],
    )
    _write_csv(
        output_path / "task_summary.csv",
        TASK_SUMMARY_COLUMNS,
        [_task_summary_row(row) for row in detail_rows],
    )
    _write_csv(
        output_path / "job_summary.csv",
        JOB_SUMMARY_COLUMNS,
        _job_summary_rows(detail_rows),
    )


def _normalize_request(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    return dict(vars(row))


def _request_detail_row(row: dict[str, Any]) -> dict[str, Any]:
    created_time = _float_from_any(row, "created_time_s", "created_time")
    queue_start = _float_from_any(row, "queue_start_s", "queue_start")
    queue_end = _float_from_any(row, "queue_end_s", "queue_end")
    finish_time = _float_from_any(
        row, "finish_time_s", "finish_time", "last_event_time", default=-1.0
    )
    waiting_time = _float_from_any(row, "waiting_time_s", "queue_time_s")
    prefetch_time = _float_from_any(row, "prefetch_time_s")
    backup_time = _float_from_any(row, "backup_time_s")
    inference_time = _float_from_any(row, "inference_time_s")
    failover_penalty = _float_from_any(row, "failover_penalty_s")
    total_latency = _total_latency(
        created_time=created_time,
        finish_time=finish_time,
        waiting_time=waiting_time,
        prefetch_time=prefetch_time,
        backup_time=backup_time,
        inference_time=inference_time,
        failover_penalty=0.0,
    )

    return {
        "job_id": str(row.get("job_id", "")),
        "rid": str(row.get("rid", "")),
        "worker_id": str(row.get("worker_id", "")),
        "assigned_dp_rank": _int_from_any(row, "assigned_dp_rank"),
        "backup_policy": str(row.get("backup_policy", "none") or "none"),
        "status": str(row.get("status", "")),
        "created_time_s": created_time,
        "queue_start_s": queue_start,
        "queue_end_s": queue_end,
        "finish_time_s": finish_time,
        "waiting_time_s": waiting_time,
        "prefetch_time_s": prefetch_time,
        "backup_time_s": backup_time,
        "inference_time_s": inference_time,
        "total_latency_s": total_latency,
        "prefill_inference_time_s": _float_from_any(row, "prefill_inference_time_s"),
        "decode_inference_time_s": _float_from_any(row, "decode_inference_time_s"),
        "failure_impacted": _bool_from_any(row.get("failure_impacted", False)),
        "is_failover_retried": _bool_from_any(row.get("is_failover_retried", False)),
        "failure_time_s": _float_from_any(row, "failure_time_s", default=-1.0),
        "recovery_time_s": _float_from_any(row, "recovery_time_s", default=-1.0),
        "pre_failover_output_tokens": _int_from_any(
            row, "pre_failover_output_tokens"
        ),
        "pre_failover_backed_up_tokens": _int_from_any(
            row, "pre_failover_backed_up_tokens"
        ),
        "retry_prefill_tokens": _int_from_any(row, "retry_prefill_tokens"),
        "failover_penalty_s": failover_penalty,
    }


def _cache_hit_row(row: dict[str, Any]) -> dict[str, Any]:
    policy = str(row.get("backup_policy", "none") or "none")
    backed_up_tokens = _int_from_any(row, "pre_failover_backed_up_tokens")
    prefetch_tokens = _int_from_any(
        row, "prefetch_complete_tokens", default=backed_up_tokens
    )
    final_reused = _int_from_any(row, "final_reused_tokens")

    l2_match = reused_host = remote_match = remote_prefetch = reused_storage = 0
    if policy == "host":
        l2_match = backed_up_tokens
        reused_host = backed_up_tokens
    elif policy == "remote_backup":
        remote_match = backed_up_tokens
        remote_prefetch = prefetch_tokens
        reused_storage = backed_up_tokens

    return {
        "job_id": str(row.get("job_id", "")),
        "rid": str(row.get("rid", "")),
        "worker_id": str(row.get("worker_id", "")),
        "assigned_dp_rank": _int_from_any(row, "assigned_dp_rank"),
        "backup_policy": policy,
        "l1_match": final_reused,
        "l2_match": l2_match,
        "reused_host": reused_host,
        "remote_match": remote_match,
        "remote_prefetch": remote_prefetch,
        "reused_device": final_reused,
        "reused_storage": reused_storage,
        "is_failover_retried": _bool_from_any(row.get("is_failover_retried", False)),
        "pre_failover_output_tokens": _int_from_any(
            row, "pre_failover_output_tokens"
        ),
        "pre_failover_backed_up_tokens": backed_up_tokens,
        "final_reused_tokens": final_reused,
        "prefetch_complete_tokens": prefetch_tokens,
    }


def _task_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row[column] for column in TASK_SUMMARY_COLUMNS}


def _job_summary_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["job_id"])].append(row)

    summary_rows = []
    for job_id in sorted(grouped, key=_sort_key):
        rows = grouped[job_id]
        summary_rows.append(
            {
                "job_id": job_id,
                "request_count": len(rows),
                "completed_count": sum(
                    1 for row in rows if row["status"] == "completed"
                ),
                "failed_over_attempts": sum(
                    1 for row in rows if _bool_from_any(row["is_failover_retried"])
                ),
                "failure_impacted_count": sum(
                    1 for row in rows if _bool_from_any(row["failure_impacted"])
                ),
                "waiting_time_s": _sum(rows, "waiting_time_s"),
                "prefetch_time_s": _sum(rows, "prefetch_time_s"),
                "backup_time_s": _sum(rows, "backup_time_s"),
                "inference_time_s": _sum(rows, "inference_time_s"),
                "total_latency_s": _sum(rows, "total_latency_s"),
            }
        )
    return summary_rows


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float_from_any(
    row: dict[str, Any], *keys: str, default: float = 0.0
) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return default


def _int_from_any(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return int(row[key])
    return default


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _total_latency(
    *,
    created_time: float,
    finish_time: float,
    waiting_time: float,
    prefetch_time: float,
    backup_time: float,
    inference_time: float,
    failover_penalty: float,
) -> float:
    if finish_time >= created_time >= 0:
        return finish_time - created_time
    return (
        waiting_time
        + prefetch_time
        + backup_time
        + inference_time
        + failover_penalty
    )


def _sum(rows: Iterable[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)
