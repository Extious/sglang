from __future__ import annotations

import json
from pathlib import Path


def collect_jsonl_records(base_dir: Path, filename: str) -> list[dict]:
    records: list[dict] = []
    paths = sorted(base_dir.glob(f"dp_*/{filename}"))
    if not paths:
        paths = [base_dir / filename]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _collect_jsonl_lines(base_dir: Path, filename: str) -> list[str]:
    lines: list[str] = []
    paths = sorted(base_dir.glob(f"dp_*/{filename}"))
    if not paths:
        paths = [base_dir / filename]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(line)
    return lines


def _write_jsonl(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )


def copy_simulator_artifacts(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "simulator_request.jsonl",
        _collect_jsonl_lines(raw_dir, "request.jsonl"),
    )
    _write_jsonl(
        output_dir / "simulator_iteration.jsonl",
        _collect_jsonl_lines(raw_dir, "iteration.jsonl"),
    )
    _write_jsonl(
        output_dir / "sim_failure_events.jsonl",
        _collect_jsonl_lines(raw_dir, "failure_events.jsonl"),
    )
