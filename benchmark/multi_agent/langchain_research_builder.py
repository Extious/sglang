"""
LangChain prompt builder/runner for MARBLE MultiAgentBench (research scenario).

This script reads MARBLE traces from:
  benchmark/multi_agent/MARBLE/multiagentbench/research/research_main.jsonl

and expands each trace into per-agent, per-iteration chat prompts following the
same prompt fields in the JSONL:
  - system: agents[].profile
  - user:   task.content

It can either:
  1) Dry-run and print a short summary of the first built step, or
  2) Send all steps to a local OpenAI-compatible endpoint via LangChain.

Example (SGLang OpenAI-compatible server):
  python benchmark/multi_agent/langchain_research_builder.py \\
    --base-url http://127.0.0.1:30000/v1 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_research_jsonl() -> Path:
    return Path(__file__).resolve().parent / "MARBLE" / "multiagentbench" / "research" / "research_main.jsonl"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        s = str(value).strip()
        if not s:
            return default
        return int(s)
    except Exception:
        return default


def build_system_prompt(*, agent_id: str, profile: str) -> str:
    _ = agent_id
    return profile


def build_user_prompt(task: Dict[str, Any]) -> str:
    return str(task.get("content") or "")


@dataclass(frozen=True)
class BuiltStep:
    scenario: str
    task_id: str
    workflow_id: str
    step_id: str
    iteration: int
    agent_id: str
    messages: List[Dict[str, str]]  # OpenAI-style messages


def load_jsonl(path: Path, *, start: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_steps_from_traces(
    traces: Iterable[Dict[str, Any]],
    *,
    max_iterations_override: Optional[int] = None,
) -> List[BuiltStep]:
    steps: List[BuiltStep] = []
    for trace in traces:
        scenario = str(trace.get("scenario") or "unknown")
        task_id = str(trace.get("task_id") or trace.get("id") or "")
        workflow_id = task_id

        scenario_l = scenario.lower()
        default_max_iter = 3 if "research" in scenario_l else 1
        raw_max_iter = trace.get("environment", {}).get("max_iterations", default_max_iter)
        max_iter = _parse_int(raw_max_iter, default_max_iter)
        if max_iterations_override is not None:
            max_iter = max(1, int(max_iterations_override))
        else:
            max_iter = max(1, max_iter)

        user_prompt = build_user_prompt(trace.get("task", {}) or {})
        agents = trace.get("agents", []) or []
        for iteration in range(max_iter):
            for agent in agents:
                agent_id = str(agent.get("agent_id") or "")
                profile = str(agent.get("profile") or "")
                system_prompt = build_system_prompt(agent_id=agent_id, profile=profile)
                step_id = f"iter{iteration}_{agent_id}"
                steps.append(
                    BuiltStep(
                        scenario=scenario,
                        task_id=task_id,
                        workflow_id=workflow_id,
                        step_id=step_id,
                        iteration=iteration,
                        agent_id=agent_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                )
    return steps


def _import_chat_openai():
    # LangChain split packages over time; try the modern import first.
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI
    except Exception:
        pass
    try:
        from langchain_community.chat_models import ChatOpenAI  # type: ignore
        return ChatOpenAI
    except Exception as e:
        raise RuntimeError(
            "LangChain ChatOpenAI is not available. Install one of:\n"
            "  pip install langchain langchain-openai\n"
            "or (older)\n"
            "  pip install langchain langchain-community\n"
        ) from e


def _import_lc_messages():
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # type: ignore
        return SystemMessage, HumanMessage, AIMessage
    except Exception:
        from langchain.schema import AIMessage, HumanMessage, SystemMessage  # type: ignore
        return SystemMessage, HumanMessage, AIMessage


def make_local_chat_model(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
):
    ChatOpenAI = _import_chat_openai()

    kwargs: Dict[str, Any] = {}
    sig = inspect.signature(ChatOpenAI)

    # Model name parameter differs across versions.
    if "model" in sig.parameters:
        kwargs["model"] = model
    elif "model_name" in sig.parameters:
        kwargs["model_name"] = model
    else:
        kwargs["model"] = model  # best-effort

    # Base URL parameter differs across versions.
    if "base_url" in sig.parameters:
        kwargs["base_url"] = base_url
    elif "openai_api_base" in sig.parameters:
        kwargs["openai_api_base"] = base_url
    else:
        os.environ.setdefault("OPENAI_API_BASE", base_url)

    # API key parameter differs across versions.
    if "api_key" in sig.parameters:
        kwargs["api_key"] = api_key
    elif "openai_api_key" in sig.parameters:
        kwargs["openai_api_key"] = api_key
    else:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    if "temperature" in sig.parameters:
        kwargs["temperature"] = temperature
    if "max_tokens" in sig.parameters:
        kwargs["max_tokens"] = max_tokens
    if "timeout" in sig.parameters:
        kwargs["timeout"] = timeout_s
    elif "request_timeout" in sig.parameters:
        kwargs["request_timeout"] = timeout_s

    # Best-effort to avoid silent retries that skew latency benchmarking.
    if "max_retries" in sig.parameters:
        kwargs["max_retries"] = 0

    return ChatOpenAI(**kwargs)


def to_langchain_messages(messages: Sequence[Dict[str, str]]):
    SystemMessage, HumanMessage, AIMessage = _import_lc_messages()
    out = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = str(m.get("content") or "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
    # LangChain versions store token usage in different places.
    if resp is None:
        return None
    usage = None
    if hasattr(resp, "usage_metadata"):
        try:
            usage = dict(getattr(resp, "usage_metadata") or {})
        except Exception:
            usage = None
    if usage:
        return usage
    if hasattr(resp, "response_metadata"):
        md = getattr(resp, "response_metadata") or {}
        if isinstance(md, dict):
            u = md.get("token_usage") or md.get("usage")
            if isinstance(u, dict):
                return u
    return None


async def run_steps(
    steps: Sequence[BuiltStep],
    *,
    llm: Any,
    concurrency: int,
    out_path: Path,
    include_prompts: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(step: BuiltStep) -> None:
        async with sem:
            lc_msgs = to_langchain_messages(step.messages)
            t0 = time.time()
            try:
                if hasattr(llm, "ainvoke"):
                    resp = await llm.ainvoke(lc_msgs)
                else:
                    resp = await asyncio.to_thread(llm.invoke, lc_msgs)
                latency_s = time.time() - t0
                text = getattr(resp, "content", None)
                if text is None:
                    text = str(resp)
                rec: Dict[str, Any] = {
                    "scenario": step.scenario,
                    "task_id": step.task_id,
                    "workflow_id": step.workflow_id,
                    "step_id": step.step_id,
                    "iteration": step.iteration,
                    "agent_id": step.agent_id,
                    "success": True,
                    "latency_s": latency_s,
                    "response": text,
                    "usage": _extract_usage(resp),
                }
                if include_prompts:
                    rec["messages"] = step.messages
            except Exception as e:
                latency_s = time.time() - t0
                rec = {
                    "scenario": step.scenario,
                    "task_id": step.task_id,
                    "workflow_id": step.workflow_id,
                    "step_id": step.step_id,
                    "iteration": step.iteration,
                    "agent_id": step.agent_id,
                    "success": False,
                    "latency_s": latency_s,
                    "error": repr(e),
                }
                if include_prompts:
                    rec["messages"] = step.messages

            line = json.dumps(rec, ensure_ascii=False)
            async with lock:
                with out_path.open("a") as f:
                    f.write(line + "\n")

    await asyncio.gather(*(one(s) for s in steps))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build/run MARBLE research prompts via LangChain.")
    parser.add_argument("--jsonl", type=str, default=str(_default_research_jsonl()), help="Path to research_main.jsonl")
    parser.add_argument("--start", type=int, default=0, help="Start index in JSONL")
    parser.add_argument("--num-traces", type=int, default=1, help="Number of traces to load")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations (default: research=3)")

    parser.add_argument("--dry-run", action="store_true", help="Only build prompts and print summary")

    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "http://127.0.0.1:30000/v1")))
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "local-model"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=1)

    parser.add_argument("--include-prompts", action="store_true", help="Write built messages to output JSONL")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "langchain_runs" / f"{_now_ts()}_research.jsonl"),
        help="Output JSONL path",
    )

    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = (_repo_root() / jsonl_path).resolve()
    if not jsonl_path.exists():
        print(f"JSONL not found: {jsonl_path}", file=sys.stderr)
        return 2

    traces = load_jsonl(jsonl_path, start=args.start, limit=args.num_traces)
    steps = build_steps_from_traces(traces, max_iterations_override=args.max_iterations)

    if args.dry_run:
        if not steps:
            print("No steps built.")
            return 0
        s0 = steps[0]
        sys_len = len(s0.messages[0]["content"]) if s0.messages else 0
        user_len = len(s0.messages[1]["content"]) if len(s0.messages) > 1 else 0
        print("=== Built steps ===")
        print(f"  traces: {len(traces)}")
        print(f"  steps: {len(steps)}")
        print("=== First step ===")
        print(f"  task_id: {s0.task_id}")
        print(f"  step_id: {s0.step_id}")
        print(f"  agent_id: {s0.agent_id}")
        print(f"  system_len: {sys_len} chars")
        print(f"  user_len: {user_len} chars")
        return 0

    llm = make_local_chat_model(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (_repo_root() / out_path).resolve()
    print(f"Running {len(steps)} steps → {out_path}")
    asyncio.run(
        run_steps(
            steps,
            llm=llm,
            concurrency=args.concurrency,
            out_path=out_path,
            include_prompts=bool(args.include_prompts),
        )
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
