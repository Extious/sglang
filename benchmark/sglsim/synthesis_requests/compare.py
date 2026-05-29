from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_request_latencies(path: Path) -> list[float]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        latencies: list[float] = []
        for row in reader:
            if row.get("status") != "completed":
                continue
            for key in ("total_latency_s", "e2e_latency_s"):
                try:
                    value = float(row.get(key, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0:
                    latencies.append(value)
                    break
        return latencies


def _ratio(sim_value: float, real_value: float) -> float | None:
    if real_value <= 0:
        return None
    return sim_value / real_value


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def compare_metrics(sim_metrics: dict[str, Any], real_metrics: dict[str, Any]) -> list[str]:
    keys = [
        ("completed", "count"),
        ("duration", "s"),
        ("request_throughput", "req/s"),
        ("mean_e2e_latency_ms", "ms"),
        ("median_e2e_latency_ms", "ms"),
        ("p99_e2e_latency_ms", "ms"),
        ("output_throughput", "tok/s"),
    ]
    lines = [
        "metric,simulator,sglang,ratio_sim_over_real",
    ]
    for key, unit in keys:
        sim_value = float(sim_metrics.get(key, 0.0) or 0.0)
        real_value = float(real_metrics.get(key, 0.0) or 0.0)
        ratio = _ratio(sim_value, real_value)
        lines.append(
            f"{key} ({unit}),{sim_value:.6f},{real_value:.6f},{_format_ratio(ratio)}"
        )
    return lines


def _aggregate_from_request_details(path: Path) -> dict[str, Any]:
    latencies = _load_request_latencies(path)
    if not latencies:
        return {}
    import statistics

    completed = len(latencies)
    mean_s = statistics.mean(latencies)
    return {
        "completed": completed,
        "mean_e2e_latency_ms": mean_s * 1000.0,
        "duration": max(latencies),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare sglsim simulator results against SGLang GPU results.",
    )
    parser.add_argument(
        "--sim-metrics",
        type=Path,
        default=None,
        help="Optional sglsim metrics.json (derived from request_details if omitted).",
    )
    parser.add_argument(
        "--real-metrics",
        type=Path,
        default=None,
        help="Optional sglang metrics.json (derived from request_details if omitted).",
    )
    parser.add_argument(
        "--sim-request-details",
        type=Path,
        default=None,
        help="sglsim request_details.csv",
    )
    parser.add_argument(
        "--real-request-details",
        type=Path,
        default=None,
        help="sglang request_details.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    sim_metrics = (
        _load_metrics(args.sim_metrics)
        if args.sim_metrics is not None
        else _aggregate_from_request_details(args.sim_request_details)
        if args.sim_request_details is not None
        else {}
    )
    real_metrics = (
        _load_metrics(args.real_metrics)
        if args.real_metrics is not None
        else _aggregate_from_request_details(args.real_request_details)
        if args.real_request_details is not None
        else {}
    )
    if not sim_metrics or not real_metrics:
        raise SystemExit(
            "Provide --sim-metrics/--real-metrics or --sim-request-details/--real-request-details"
        )

    lines = compare_metrics(sim_metrics, real_metrics)

    if args.sim_request_details and args.real_request_details:
        sim_lat = _load_request_latencies(args.sim_request_details)
        real_lat = _load_request_latencies(args.real_request_details)
        if sim_lat and real_lat:
            import statistics

            sim_mean = statistics.mean(sim_lat)
            real_mean = statistics.mean(real_lat)
            lines.append("")
            lines.append(
                f"per_request_mean_latency_s,{sim_mean:.6f},{real_mean:.6f},"
                f"{_format_ratio(_ratio(sim_mean, real_mean))}"
            )

    report = "\n".join(lines)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
