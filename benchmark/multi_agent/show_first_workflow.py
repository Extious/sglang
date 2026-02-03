"""
Show the workflow built from the first line of research_main.jsonl.
Run from benchmark/multi_agent: python show_first_workflow.py
"""
import json
from pathlib import Path

from marble_exp import build_workflows_from_traces

MARBLE_DIR = Path(__file__).resolve().parent / "MARBLE"
RESEARCH_JSONL = MARBLE_DIR / "multiagentbench" / "research" / "research_main.jsonl"

def main():
    with RESEARCH_JSONL.open() as f:
        first_line = f.readline()
    trace = json.loads(first_line)

    # Same logic as load_marble_traces + build_workflows_from_traces
    workflows = build_workflows_from_traces([trace])
    w = workflows[0]

    print("=== Trace (first line) key fields ===")
    print(f"  scenario: {trace.get('scenario')}")
    print(f"  task_id: {trace.get('task_id')}")
    print(f"  agents: {len(trace.get('agents', []))} agents")
    print(f"  environment.max_iterations: {repr(trace.get('environment', {}).get('max_iterations'))}")
    print(f"  task.content length: {len(trace.get('task', {}).get('content', ''))} chars")
    print()

    print("=== Built MarbleWorkflow ===")
    print(f"  scenario: {w.scenario}")
    print(f"  workflow_id: {w.workflow_id}")
    print(f"  steps: {len(w.steps)} (max_iter=3 for research, 3 * 5 agents = 15)")
    print()

    print("=== Steps (MarbleRequest) ===")
    for i, s in enumerate(w.steps):
        sys_len = len(s.messages[0]["content"]) if s.messages else 0
        user_len = len(s.messages[1]["content"]) if len(s.messages) > 1 else 0
        print(f"  [{i+1:2d}] step_id={s.step_id}, agent_id={s.agent_id}, "
              f"system_len={sys_len}, user_len={user_len}")

if __name__ == "__main__":
    main()
