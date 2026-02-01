#!/usr/bin/env python3
"""
Parse DeepNLP Multi-Agents-Parallel-Orchestration-Dataset JSON.
Each line = one JSON object: { session_id: { trace_id: record, ... }, ... }.
One session can have multiple traces (parallel agent runs).
"""
import argparse
import json
import sys
from pathlib import Path


def load_json_lines(path):
    """Load file where each line is a JSON object (one session)."""
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(json.loads(line))
    return sessions


def summarize_record(rec):
    out = {
        "model": rec.get("model"),
        "session_id": rec.get("session_id"),
        "trace_id": rec.get("trace_id"),
        "has_plan": "plan" in rec,
        "num_function_calls": len(rec.get("function_calls", [])),
    }
    fc = rec.get("function_calls", [])
    if fc:
        m0 = fc[0].get("messages", [])
        out["first_prompt"] = None
        for m in m0:
            if m.get("role") == "user" and m.get("content"):
                out["first_prompt"] = (m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"])
                break
    return out


def parse(path, verbose=False):
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return None

    sessions = load_json_lines(path)
    stats = {
        "num_sessions": 0,
        "num_traces": 0,
        "models": set(),
        "function_calls_per_trace": [],
        "has_plan_count": 0,
    }
    records_summary = []

    for session_obj in sessions:
        for session_id, inner in session_obj.items():
            stats["num_sessions"] += 1
            for trace_id, rec in inner.items():
                stats["num_traces"] += 1
                if rec.get("model"):
                    stats["models"].add(rec["model"])
                fc = rec.get("function_calls", [])
                stats["function_calls_per_trace"].append(len(fc))
                if "plan" in rec:
                    stats["has_plan_count"] += 1
                records_summary.append(summarize_record(rec))

    stats["models"] = list(stats["models"])
    fc = stats["function_calls_per_trace"]
    stats["function_calls_min"] = min(fc) if fc else 0
    stats["function_calls_max"] = max(fc) if fc else 0
    stats["function_calls_avg"] = sum(fc) / len(fc) if fc else 0

    result = {"stats": stats, "records_summary": records_summary}
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Parse Multi-Agents-Parallel-Orchestration-Dataset JSON")
    parser.add_argument("path", nargs="?", default=None, help="Path to JSON file (default: example_*.json in script dir)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full summary including first_prompt per record")
    parser.add_argument("-o", "--output", help="Write summary JSON to file")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    if args.path:
        path = Path(args.path)
    else:
        candidates = list(base.glob("example_*.json"))
        path = candidates[0] if candidates else None

    if not path or not path.exists():
        print("No JSON file found. Specify path or run from Multi-Agents-Parallel-Orchestration-Dataset with example_*.json present.", file=sys.stderr)
        sys.exit(1)

    result = parse(path, verbose=args.verbose)
    if result is None:
        sys.exit(1)

    s = result["stats"]
    print("Sessions:", s["num_sessions"])
    print("Traces:", s["num_traces"])
    print("Models:", s["models"])
    print("function_calls per trace: min=%s max=%s avg=%.2f" % (s["function_calls_min"], s["function_calls_max"], s["function_calls_avg"]))
    print("Traces with plan (sequential):", s["has_plan_count"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Wrote", args.output)


if __name__ == "__main__":
    main()
