#!/usr/bin/env python3
"""
Expand trajectory field in MAD_full_dataset.json into structured trajectory_expanded.
Output: same records with added trajectory_expanded (metadata, task, sections, evaluation).
"""

import json
import re
import argparse
from pathlib import Path


def expand_chatdev(traj: str) -> dict:
    """ChatDev: **key**: value / **key** | value in preprocessing, then phases/turns."""
    out = {"metadata": {}, "task_prompt": None, "sections": [], "evaluation": None}
    if not traj or not isinstance(traj, str):
        return out

    # Preprocessing block: **key**: value until next \n\n** or \n\n[
    for m in re.finditer(r"\*\*([^*\s|]+)\*\*\s*[:\|]\s*([^\n]*(?:\n(?!\n\*\*|\n\[)[^\n]*)*)", traj):
        key, val = m.group(1).strip(), m.group(2).strip()
        if key in ("Preprocessing", "chatting"):
            continue
        out["metadata"][key] = val
        if key == "task_prompt":
            out["task_prompt"] = val

    # Log/phase blocks: [YYYY-MM-DD HH:MM:SS INFO] ... or **phase_name** | X
    rest = traj
    phase_pattern = re.compile(
        r"(\[\d{4}[-\d]+\s+\d{2}:\d{2}:\d{2}\s+INFO\]|\*\*phase_name\*\*\s*\|\s*[^\n]+)\s*\n(.*?)(?=\[\d{4}[-\d]+\s+\d{2}:\d{2}:\d{2}\s+INFO\]|\*\*phase_name\*\*|\Z)",
        re.DOTALL,
    )
    for m in phase_pattern.finditer(traj):
        header, content = m.group(1).strip(), m.group(2).strip()
        if content and len(content) > 10:
            out["sections"].append({"type": "phase_or_log", "header": header, "content": content[:5000]})
    return out


def expand_appworld(traj: str) -> dict:
    """AppWorld: Task line, then Response from / Message to / Code Execution / Evaluation."""
    out = {"task": None, "sections": [], "evaluation": None}
    if not traj or not isinstance(traj, str):
        return out

    # Task: line after **** Task N/M (id) ****
    m = re.search(
        r"\*+\s*Task\s+\d+/\d+\s+\([^)]+\)\s*\*+\s*\n\s*(.+?)(?=\n\nResponse from|\n\nMessage to|\Z)",
        traj,
        re.DOTALL,
    )
    if m:
        out["task"] = m.group(1).strip()

    # Evaluation block at end
    eval_match = re.search(r"\nEvaluation\s*\n(\{[\s\S]*\})\s*$", traj)
    if eval_match:
        try:
            out["evaluation"] = json.loads(eval_match.group(1))
        except json.JSONDecodeError:
            out["evaluation"] = {"raw": eval_match.group(1)[:2000]}

    # Sections: header line then content until next header (allow single newline after header)
    section_pattern = re.compile(
        r"(Response from ([^\n]+)\n|Message to ([^\n]+)\n|Code Execution Output\n|Entering ([^\n]+) message loop\n|Exiting ([^\n]+) Agent message loop\n|Response from send_message API\n)\s*(.*?)(?=Response from |Message to |Code Execution Output\n|Entering |Exiting |Evaluation\n|\Z)",
        re.DOTALL,
    )
    for m in section_pattern.finditer(traj):
        header = m.group(1).strip()
        content = (m.group(6).strip() if m.lastindex and m.lastindex >= 6 else "") or ""
        agent = (m.group(2) or m.group(3) or m.group(4) or m.group(5) or "").strip()
        if "Response from" in header and "send_message" not in header:
            stype = "response_from_agent"
        elif "Message to" in header:
            stype = "message_to_agent"
        elif "Code Execution" in header:
            stype = "code_execution"
        elif "Entering" in header:
            stype = "entering_agent_loop"
        elif "Exiting" in header:
            stype = "exiting_agent_loop"
        elif "send_message API" in header:
            stype = "api_response"
        else:
            stype = "section"
        out["sections"].append({
            "type": stype,
            "header": header[:200],
            "agent": agent[:100] if agent else None,
            "content": content[:8000],
        })
    return out


def expand_ag2_or_hyperagent(traj: str) -> dict:
    """AG2/HyperAgent: instance_id, problem_statement, other_data, trajectory content (YAML-like)."""
    out = {"metadata": {}, "problem_statement": None, "sections": [], "evaluation": None}
    if not traj or not isinstance(traj, str):
        return out

    # key: value (single line)
    for m in re.finditer(r"^(\w[\w_]*)\s*:\s*(.+?)$", traj, re.MULTILINE):
        key, val = m.group(1), m.group(2).strip()
        out["metadata"][key] = val
        if key == "problem_statement":
            out["problem_statement"] = val

    # problem_statement: multi-line (indented)
    ps_match = re.search(r"problem_statement\s*:\s*\n((?:\s{2,}.+\n?)+)", traj)
    if ps_match:
        out["problem_statement"] = re.sub(r"^\s+", "", ps_match.group(1)).strip()
        out["metadata"]["problem_statement"] = out["problem_statement"]

    # trajectory:\n content: ... or rest as single section
    traj_match = re.search(r"trajectory\s*:\s*\n\s*content\s*:\s*\n(.*)", traj, re.DOTALL)
    if traj_match:
        out["sections"].append({"type": "trajectory_content", "content": traj_match.group(1).strip()[:15000]})
    return out


def expand_generic(traj: str) -> dict:
    """Generic: split by common headers into sections."""
    out = {"metadata": {}, "sections": [], "evaluation": None}
    if not traj or not isinstance(traj, str):
        return out

    # Split by RUN.SH STARTING, Processing, or double newline + uppercase/header line
    chunks = re.split(r"\n(?=(?:RUN\.SH STARTING|Processing |AUTOGEN_|Installing |\[[\d-]+ \d+:\d+:\d+))", traj)
    for i, c in enumerate(chunks):
        c = c.strip()
        if not c:
            continue
        first_line = c.split("\n")[0][:120] if c else ""
        out["sections"].append({"type": "block", "index": i, "header": first_line, "content": c[:6000]})
    if not out["sections"]:
        out["sections"].append({"type": "raw", "content": traj[:12000]})
    return out


def expand_trajectory(traj: str, mas_name: str) -> dict:
    """Dispatch by mas_name and return structured trajectory_expanded."""
    if mas_name == "ChatDev":
        return expand_chatdev(traj)
    if mas_name == "AppWorld":
        return expand_appworld(traj)
    if mas_name in ("AG2", "HyperAgent"):
        return expand_ag2_or_hyperagent(traj)
    return expand_generic(traj)


def main():
    parser = argparse.ArgumentParser(description="Expand MAD_full trajectory into trajectory_expanded")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "MAST-Data" / "MAD_full_dataset.json",
        help="Input MAD_full_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: MAST-Data/MAD_full_dataset_expanded.json)",
    )
    parser.add_argument(
        "--keep-trajectory",
        action="store_true",
        help="Keep original trajectory string in each record",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = args.output or (input_path.parent / "MAD_full_dataset_expanded.json")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    for r in data:
        trace = r.get("trace")
        if isinstance(trace, dict):
            traj = trace.get("trajectory", "") or ""
        else:
            traj = ""
        mas_name = r.get("mas_name", "")
        expanded = expand_trajectory(traj, mas_name)
        ev = expanded.get("evaluation")
        if ev is not None and not isinstance(ev, dict):
            expanded["evaluation"] = {"raw": str(ev)[:2000]}
        r["trajectory_expanded"] = expanded
        if not args.keep_trajectory and isinstance(trace, dict):
            trace.pop("trajectory", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Expanded {len(data)} records -> {output_path}")


if __name__ == "__main__":
    main()
