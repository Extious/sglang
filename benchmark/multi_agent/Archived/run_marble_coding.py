"""
LangChain Multi-Agent Coding System for MARBLE Benchmark

This script implements a sequential pipeline multi-agent system for collaborative
software development based on MARBLE MultiAgentBench coding traces.

Architecture: Sequential Pipeline
    agent1 (Creator) -> agent2 (Reviser) -> agent3 (Optimizer)

Each agent has specific tools:
- agent1: create_code (creates initial code framework)
- agent2: give_advice_and_revise_code (adds missing functionality)
- agent3: give_advice_and_revise_code (fixes issues and optimizes)

Usage:
    python run_marble_coding.py --base-url http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.tools import tool
except ImportError:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    from langchain.tools import tool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_coding_jsonl() -> Path:
    return _repo_root() / "benchmark" / "multi_agent" / "dataset" / "MARBLE" / "multiagentbench" / "coding" / "coding_main.jsonl"


@dataclass
class CodeWorkspace:
    """Shared workspace for code artifacts."""
    code: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update_code(self, new_code: str, agent_id: str, action: str, advice: str = ""):
        """Update code and record history."""
        self.history.append({
            "agent_id": agent_id,
            "action": action,
            "advice": advice,
            "code_snapshot": new_code,
            "timestamp": time.time(),
        })
        self.code = new_code


@dataclass
class CodingAgent:
    """Represents a coding agent with specific capabilities."""
    agent_id: str
    profile: str
    agent_type: str
    allowed_actions: List[str]

    def can_create(self) -> bool:
        return "create_code" in self.allowed_actions

    def can_revise(self) -> bool:
        return "give_advice_and_revise_code" in self.allowed_actions


class CodingToolkit:
    """Tools available to coding agents."""

    def __init__(self, workspace: CodeWorkspace):
        self.workspace = workspace

    def create_code(self, agent_id: str, code: str) -> Dict[str, Any]:
        """
        Create initial code framework.
        Only agent1 should use this tool.
        """
        self.workspace.update_code(code, agent_id, "create_code")
        return {
            "success": True,
            "action": "create_code",
            "agent_id": agent_id,
            "message": "Code framework created successfully.",
            "code_length": len(code),
        }

    def give_advice_and_revise_code(
        self,
        agent_id: str,
        advice: str,
        revised_code: str
    ) -> Dict[str, Any]:
        """
        Give advice and revise existing code.
        agent2 and agent3 should use this tool.
        """
        self.workspace.update_code(revised_code, agent_id, "give_advice_and_revise_code", advice)
        return {
            "success": True,
            "action": "give_advice_and_revise_code",
            "agent_id": agent_id,
            "advice": advice,
            "message": "Code revised successfully.",
            "code_length": len(revised_code),
        }


class MultiAgentCodingSystem:
    """
    Multi-agent system for collaborative software development.

    Implements a sequential pipeline where:
    1. agent1 creates the initial code framework
    2. agent2 adds missing functionality
    3. agent3 optimizes and fixes issues
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        agents: List[CodingAgent],
        task_content: str,
        output_format: str,
        workspace: CodeWorkspace,
    ):
        self.llm = llm
        self.agents = agents
        self.task_content = task_content
        self.output_format = output_format
        self.workspace = workspace
        self.toolkit = CodingToolkit(workspace)

    def _build_system_prompt(self, agent: CodingAgent) -> str:
        """Build system prompt for an agent."""
        if agent.can_create():
            tool_instruction = """
I have access to the following tool:
- create_code: Use this to create the initial code framework

Output my code directly in a Python code block like this:
```python
# My complete Python code here
```
"""
        else:
            tool_instruction = """
I have access to the following tool:
- give_advice_and_revise_code: Use this to provide advice and revise the existing code

First briefly state my improvements (2-3 sentences), then output the COMPLETE revised code in a Python code block:
```python
# My complete revised Python code here
```
"""

        return f"""I am a coding agent participating in a collaborative software development project.

{agent.profile}

{tool_instruction}

IMPORTANT:
- Output complete, working Python code in a code block
- Do NOT output partial code or placeholders
- Follow software engineering best practices
- Keep explanations brief, focus on the code
- Do NOT use <think> tags or show my reasoning process
"""

    def _build_user_prompt(self, agent: CodingAgent, step: int) -> str:
        """Build user prompt based on current state."""
        if step == 0:
            # First agent creates code
            return f"""
## Software Development Task

{self.task_content}

## Your Task

I am the first developer. Please create the initial code framework for this project.
Create a complete, well-structured Python implementation in solution.py.

Use the create_code action to submit your code.
"""
        else:
            # Subsequent agents revise code
            current_code = self.workspace.code
            history_summary = self._format_history()

            return f"""
## Software Development Task

{self.task_content}

## Current Code

```python
{current_code}
```

## Development History

{history_summary}

## Your Task

Review the current code and improve it based on your expertise.
{"Add any missing functionality and ensure all requirements are met." if step == 1 else "Optimize the code, fix any issues, and ensure code quality."}

Use the give_advice_and_revise_code action to submit your improvements.
"""

    def _format_history(self) -> str:
        """Format development history."""
        if not self.workspace.history:
            return "No previous changes."

        lines = []
        for i, entry in enumerate(self.workspace.history):
            action = entry["action"]
            agent = entry["agent_id"]
            advice = entry.get("advice", "")
            lines.append(f"{i+1}. [{agent}] {action}")
            if advice:
                lines.append(f"   Advice: {advice[:200]}...")

        return "\n".join(lines)

    def _parse_agent_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse agent response to extract action and code."""
        # Try to find JSON block
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        try:
            # Find JSON-like structure
            start = response_text.find('{')
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i, c in enumerate(response_text[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response_text[start:start+i+1]
                            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract Python code from response if JSON parsing fails."""
        # Remove thinking tags if present (for models like Qwen3-Thinking)
        text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

        # Try to find Python code block
        python_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(python_pattern, text, re.DOTALL)

        if matches:
            # Return the longest code block (likely the complete code)
            return max(matches, key=len)

        return None

    def _extract_advice_from_response(self, response_text: str) -> str:
        """Extract advice/explanation from response."""
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

        # Look for advice patterns
        advice_patterns = [
            r'(?:advice|changes|improvements|modifications):\s*(.+?)(?=```|$)',
            r'(?:I have|I\'ve|Here are the)(.+?)(?=```|$)',
        ]

        for pattern in advice_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                advice = match.group(1).strip()
                if len(advice) > 20:
                    return advice[:500]

        # Fallback: use first paragraph
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            p = p.strip()
            if len(p) > 50 and '```' not in p:
                return p[:500]

        return "Code revised based on requirements."

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the sequential coding pipeline."""
        results = {
            "steps": [],
            "final_code": None,
            "total_time": 0,
            "success": False,
        }

        start_time = time.time()

        for step, agent in enumerate(self.agents):
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{len(self.agents)}: {agent.agent_id}")
            print(f"Role: {'Creator' if agent.can_create() else 'Reviser'}")
            print(f"{'='*60}")

            # Build prompts
            system_prompt = self._build_system_prompt(agent)
            user_prompt = self._build_user_prompt(agent, step)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            # Call LLM
            print(f"\n[{agent.agent_id}] Generating response...")

            try:
                t0 = time.time()
                response = self.llm.invoke(messages)
                latency = time.time() - t0

                response_text = response.content if hasattr(response, 'content') else str(response)

                print(f"[{agent.agent_id}] Response received ({latency:.2f}s)")

                # Parse response
                parsed = self._parse_agent_response(response_text)

                step_result = {
                    "agent_id": agent.agent_id,
                    "step": step + 1,
                    "latency_s": latency,
                    "success": False,
                    "action": None,
                    "advice": None,
                }

                if parsed:
                    action = parsed.get("action", "")

                    if action == "create_code" and agent.can_create():
                        code = parsed.get("code", "")
                        if code:
                            result = self.toolkit.create_code(agent.agent_id, code)
                            step_result["success"] = True
                            step_result["action"] = "create_code"
                            print(f"[{agent.agent_id}] Created code ({result['code_length']} chars)")

                    elif action == "give_advice_and_revise_code" and agent.can_revise():
                        advice = parsed.get("advice", "")
                        revised_code = parsed.get("revised_code", "")
                        if revised_code:
                            result = self.toolkit.give_advice_and_revise_code(
                                agent.agent_id, advice, revised_code
                            )
                            step_result["success"] = True
                            step_result["action"] = "give_advice_and_revise_code"
                            step_result["advice"] = advice
                            print(f"[{agent.agent_id}] Revised code ({result['code_length']} chars)")
                            print(f"[{agent.agent_id}] Advice: {advice[:200]}...")

                # Fallback: try to extract code directly
                if not step_result["success"]:
                    code = self._extract_code_from_response(response_text)
                    if code:
                        if agent.can_create() and step == 0:
                            self.toolkit.create_code(agent.agent_id, code)
                            step_result["success"] = True
                            step_result["action"] = "create_code (fallback)"
                            print(f"[{agent.agent_id}] Created code via fallback")
                        elif agent.can_revise():
                            advice = self._extract_advice_from_response(response_text)
                            self.toolkit.give_advice_and_revise_code(
                                agent.agent_id, advice, code
                            )
                            step_result["success"] = True
                            step_result["action"] = "give_advice_and_revise_code (fallback)"
                            step_result["advice"] = advice
                            print(f"[{agent.agent_id}] Revised code via fallback")
                            print(f"[{agent.agent_id}] Advice: {advice[:200]}...")

                if not step_result["success"]:
                    print(f"[{agent.agent_id}] Warning: Could not parse action from response")
                    # Print first 500 chars for debugging
                    print(f"Response preview: {response_text[:500]}...")

                step_result["response_preview"] = response_text[:1000]
                results["steps"].append(step_result)

            except Exception as e:
                print(f"[{agent.agent_id}] Error: {e}")
                results["steps"].append({
                    "agent_id": agent.agent_id,
                    "step": step + 1,
                    "success": False,
                    "error": str(e),
                })

        results["total_time"] = time.time() - start_time
        results["final_code"] = self.workspace.code
        results["success"] = bool(self.workspace.code)
        results["history"] = self.workspace.history

        return results


def load_trace(jsonl_path: Path, index: int = 0) -> Dict[str, Any]:
    """Load a single trace from the JSONL file."""
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
    raise ValueError(f"Trace index {index} not found in {jsonl_path}")


def create_agents_from_trace(trace: Dict[str, Any]) -> List[CodingAgent]:
    """Create CodingAgent instances from trace data."""
    agents = []
    for agent_data in trace.get("agents", []):
        agent_id = agent_data.get("agent_id", "unknown")
        profile = agent_data.get("profile", "")
        agent_type = agent_data.get("type", "CodingAgent")

        # Determine allowed actions based on profile
        if "create_code" in profile and "can't" not in profile.split("create_code")[0][-20:]:
            allowed_actions = ["create_code"]
        else:
            allowed_actions = ["give_advice_and_revise_code"]

        agent = CodingAgent(
            agent_id=agent_id,
            profile=profile,
            agent_type=agent_type,
            allowed_actions=allowed_actions,
        )
        agents.append(agent)

    return agents


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LangChain Multi-Agent Coding System")

    # Data arguments
    parser.add_argument("--jsonl", type=str, default=str(_default_coding_jsonl()),
                        help="Path to coding_main.jsonl")
    parser.add_argument("--trace-index", type=int, default=0,
                        help="Index of trace to use (0-based)")

    # LLM arguments
    parser.add_argument("--base-url", type=str,
                        default="http://127.0.0.1:30000/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model name")
    parser.add_argument("--api-key", type=str,
                        default="EMPTY",
                        help="API key")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum tokens per response")

    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--save-code", type=str, default=None,
                        help="Save final code to this file")

    args = parser.parse_args(argv)

    # Load trace
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}")
        return 1

    print(f"Loading trace {args.trace_index} from {jsonl_path}")
    trace = load_trace(jsonl_path, args.trace_index)

    print(f"Scenario: {trace.get('scenario', 'unknown')}")
    print(f"Task ID: {trace.get('task_id', 'unknown')}")

    # Create agents
    agents = create_agents_from_trace(trace)
    print(f"\nCreated {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.agent_id}: {agent.allowed_actions}")

    # Create LLM
    print(f"\nConnecting to LLM at {args.base_url}")
    print(f"Model: {args.model}")

    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=600,
    )

    # Get task content
    task = trace.get("task", {})
    task_content = task.get("content", "")
    output_format = task.get("output_format", "")

    # Create workspace and system
    workspace = CodeWorkspace()
    system = MultiAgentCodingSystem(
        llm=llm,
        agents=agents,
        task_content=task_content,
        output_format=output_format,
        workspace=workspace,
    )

    print("\nStarting multi-agent coding pipeline...")
    results = system.run_pipeline()

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Steps completed: {len(results['steps'])}")
    print(f"Success: {results['success']}")

    if results["final_code"]:
        print(f"\nFinal code length: {len(results['final_code'])} chars")
        print("\n" + "=" * 60)
        print("FINAL CODE (first 2000 chars)")
        print("=" * 60)
        print(results["final_code"][:2000])
        if len(results["final_code"]) > 2000:
            print("... (truncated)")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "logs"
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"coding_result_{timestamp}.json"

    # Remove large code from steps for JSON output
    results_for_json = {
        **results,
        "steps": [
            {k: v for k, v in step.items() if k != "response_preview"}
            for step in results["steps"]
        ],
    }

    with output_path.open("w") as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Save final code
    if args.save_code and results["final_code"]:
        code_path = Path(args.save_code)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(results["final_code"])
        print(f"Final code saved to: {code_path}")
    elif results["final_code"]:
        code_dir = Path(__file__).parent / "workspace"
        code_dir.mkdir(exist_ok=True)
        code_path = code_dir / "solution.py"
        code_path.write_text(results["final_code"])
        print(f"Final code saved to: {code_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
