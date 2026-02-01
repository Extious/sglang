#!/usr/bin/env python3
"""
Extract prompt flow from MAST-Data and TRAIL datasets.
Output: per-dataset JSON files with trace_id and prompt(s).
"""

import ast
import json
import re
import argparse
from pathlib import Path


def extract_mast_full_trajectory(trajectory: str) -> str | None:
    """Extract task_prompt from MAD_full trajectory text (ChatDev style)."""
    if not trajectory or not isinstance(trajectory, str):
        return None
    # **task_prompt**: <content until next \n\n** or end of line block
    m = re.search(r"\*\*task_prompt\*\*:\s*(.+?)(?=\n\n\*\*|\n\n\[|\Z)", trajectory, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_mast_human_trace(trace: str) -> str | None:
    """Extract task description from MAD_human_labelled trace (AppWorld/Test-C style)."""
    if not trace or not isinstance(trace, str):
        return None
    # ******************** Task 5/16 (692c77d_1)  ********************\n<task line>
    m = re.search(
        r"\*+\s*Task\s+\d+/\d+\s+\([^)]+\)\s*\*+\s*\n\s*(.+?)(?=\n\n|\nResponse from)",
        trace,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def extract_mast_full(data_dir: Path) -> list[dict]:
    path = data_dir / "MAST-Data" / "MAD_full_dataset.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        trace = r.get("trace")
        if isinstance(trace, dict):
            traj = trace.get("trajectory", "")
        else:
            traj = ""
        prompt = extract_mast_full_trajectory(traj)
        out.append({
            "mas_name": r.get("mas_name"),
            "llm_name": r.get("llm_name"),
            "benchmark_name": r.get("benchmark_name"),
            "trace_id": r.get("trace_id"),
            "task_prompt": prompt,
        })
    return out


def extract_mast_human_labelled(data_dir: Path) -> list[dict]:
    path = data_dir / "MAST-Data" / "MAD_human_labelled_dataset.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        trace = r.get("trace", "")
        prompt = extract_mast_human_trace(trace)
        out.append({
            "round": r.get("round"),
            "mas_name": r.get("mas_name"),
            "benchmark_name": r.get("benchmark_name"),
            "trace_id": r.get("trace_id"),
            "task_prompt": prompt,
        })
    return out


def _collect_trail_prompts_from_spans(spans: list, source: str) -> list[dict]:
    """Recursively collect prompts from TRAIL spans (GAIA or SWE-Bench)."""
    collected = []

    def walk(s: dict):
        attrs = s.get("span_attributes") or {}
        if "input.value" in attrs:
            try:
                iv = json.loads(attrs["input.value"])
                if isinstance(iv, dict):
                    if "task" in iv and isinstance(iv["task"], str):
                        collected.append({
                            "source": source,
                            "span_name": s.get("span_name"),
                            "span_id": s.get("span_id"),
                            "type": "task",
                            "content": iv["task"],
                        })
                    if "messages" in iv:
                        for msg in iv["messages"]:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                content = msg.get("content")
                                if isinstance(content, str):
                                    collected.append({
                                        "source": source,
                                        "span_name": s.get("span_name"),
                                        "span_id": s.get("span_id"),
                                        "type": "user_message",
                                        "content": content[:2000],
                                    })
                                elif isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and c.get("type") == "text":
                                            text = c.get("text", "")
                                            collected.append({
                                                "source": source,
                                                "span_name": s.get("span_name"),
                                                "span_id": s.get("span_id"),
                                                "type": "user_message",
                                                "content": (text[:2000] if isinstance(text, str) else str(text)),
                                            })
                                            break
            except (json.JSONDecodeError, TypeError):
                pass

        for log in s.get("logs") or []:
            body = (log.get("body") or {})
            if source == "gaia":
                # function.output can be list of examples with question
                out = body.get("function.output")
                if isinstance(out, list) and out:
                    ex = out[0] if isinstance(out[0], dict) else None
                    if ex and "question" in ex:
                        collected.append({
                            "source": source,
                            "span_name": s.get("span_name"),
                            "span_id": s.get("span_id"),
                            "type": "question",
                            "content": ex["question"],
                        })
                args = body.get("function.arguments") or {}
                if isinstance(args, dict) and "example" in args:
                    ex = args["example"]
                    if isinstance(ex, dict) and "question" in ex:
                        collected.append({
                            "source": source,
                            "span_name": s.get("span_name"),
                            "span_id": s.get("span_id"),
                            "type": "question",
                            "content": ex["question"],
                        })
            else:
                # swe_bench: item with question, problem_statement
                args = body.get("function.arguments") or {}
                item = args.get("item") if isinstance(args, dict) else None
                if isinstance(item, dict):
                    if "question" in item:
                        collected.append({
                            "source": source,
                            "span_name": s.get("span_name"),
                            "span_id": s.get("span_id"),
                            "type": "question",
                            "content": item["question"],
                        })
                    if "problem_statement" in item:
                        collected.append({
                            "source": source,
                            "span_name": s.get("span_name"),
                            "span_id": s.get("span_id"),
                            "type": "problem_statement",
                            "content": item["problem_statement"],
                        })
                elif isinstance(item, str):
                    parsed = None
                    try:
                        parsed = json.loads(item.replace("'", '"'))
                    except (json.JSONDecodeError, TypeError):
                        try:
                            parsed = ast.literal_eval(item)
                        except (ValueError, SyntaxError, TypeError):
                            pass
                    if isinstance(parsed, dict):
                        if "question" in parsed:
                            collected.append({
                                "source": source,
                                "span_name": s.get("span_name"),
                                "span_id": s.get("span_id"),
                                "type": "question",
                                "content": parsed["question"],
                            })
                        if "problem_statement" in parsed:
                            collected.append({
                                "source": source,
                                "span_name": s.get("span_name"),
                                "span_id": s.get("span_id"),
                                "type": "problem_statement",
                                "content": parsed["problem_statement"],
                            })

        for ch in s.get("child_spans") or []:
            walk(ch)

    for span in spans:
        walk(span)
    return collected


def extract_trail_gaia(data_dir: Path) -> list[dict]:
    gaia_dir = data_dir / "TRAIL" / "GAIA"
    if not gaia_dir.exists():
        return []
    out = []
    for p in sorted(gaia_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        trace_id = doc.get("trace_id") or p.stem
        spans = doc.get("spans") or []
        prompts = _collect_trail_prompts_from_spans(spans, "gaia")
        # Prefer single main task: first "task" or first "question"
        main_task = None
        for x in prompts:
            if x["type"] == "task":
                main_task = x["content"]
                break
        if main_task is None:
            for x in prompts:
                if x["type"] == "question":
                    main_task = x["content"]
                    break
        out.append({
            "trace_id": trace_id,
            "main_task": main_task,
            "prompt_flow": prompts,
        })
    return out


def extract_trail_swe_bench(data_dir: Path) -> list[dict]:
    swe_dir = data_dir / "TRAIL" / "SWE Bench"
    if not swe_dir.exists():
        return []
    out = []
    for p in sorted(swe_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        trace_id = doc.get("trace_id") or p.stem
        spans = doc.get("spans") or []
        prompts = _collect_trail_prompts_from_spans(spans, "swe_bench")
        question = None
        problem_statement = None
        for x in prompts:
            if x["type"] == "question":
                question = x["content"]
                break
        for x in prompts:
            if x["type"] == "problem_statement":
                problem_statement = x["content"]
                break
        out.append({
            "trace_id": trace_id,
            "question": question,
            "problem_statement": problem_statement,
            "prompt_flow": prompts,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract prompt flow from multi_agent datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing MAST-Data/ and TRAIL/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as data-dir)",
    )
    parser.add_argument(
        "--dataset",
        choices=["mast_full", "mast_human", "trail_gaia", "trail_swe", "all"],
        default="all",
        help="Which dataset(s) to extract",
    )
    args = parser.parse_args()
    out_dir = args.output_dir or args.data_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("mast_full", "all"):
        data = extract_mast_full(args.data_dir)
        out_path = out_dir / "mast_full_prompt_flow.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"MAST-Data (full): {len(data)} records -> {out_path}")

    if args.dataset in ("mast_human", "all"):
        data = extract_mast_human_labelled(args.data_dir)
        out_path = out_dir / "mast_human_prompt_flow.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"MAST-Data (human): {len(data)} records -> {out_path}")

    if args.dataset in ("trail_gaia", "all"):
        data = extract_trail_gaia(args.data_dir)
        out_path = out_dir / "trail_gaia_prompt_flow.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"TRAIL GAIA: {len(data)} traces -> {out_path}")

    if args.dataset in ("trail_swe", "all"):
        data = extract_trail_swe_bench(args.data_dir)
        out_path = out_dir / "trail_swe_bench_prompt_flow.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"TRAIL SWE-Bench: {len(data)} traces -> {out_path}")


if __name__ == "__main__":
    main()
