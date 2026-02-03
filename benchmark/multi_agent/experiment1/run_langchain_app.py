"""
LangChain Multi-Agent Application (Client)

Loads MARBLE dataset, builds multi-agent steps using LangChain,
and sends all requests to SGLang router (OpenAI-compatible API).

Usage:
    python run_langchain_app.py --router-url http://gpu20:30000 --num-traces 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from langchain_research_builder import (
    BuiltStep,
    build_steps_from_traces,
    load_jsonl,
    make_local_chat_model,
    run_steps,
    to_langchain_messages,
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_marble_dir() -> Path:
    return _script_dir().parent / "MARBLE"


def _default_research_jsonl() -> Path:
    return _default_marble_dir() / "multiagentbench" / "research" / "research_main.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LangChain multi-agent application against SGLang router"
    )
    parser.add_argument(
        "--router-url",
        type=str,
        default=None,
        help="Router base URL (e.g. http://gpu20:30000). Can also use --router-url-file or ROUTER_BASE_URL env",
    )
    parser.add_argument(
        "--router-url-file",
        type=Path,
        default=None,
        help="Path to router_url.txt (default: experiment1/logs/router_url.txt)",
    )
    parser.add_argument(
        "--marble-dir",
        type=Path,
        default=None,
        help="MARBLE dataset root (default: ../MARBLE)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Path to research_main.jsonl (default: auto-detect)",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=20,
        help="Number of traces to load",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Max iterations per trace (default: 1 for exp1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "local-model"),
        help="Model name (for OpenAI API compatibility)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key (not used for local router)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max completion tokens",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/<timestamp>_langchain.jsonl)",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include prompts in output JSONL",
    )
    args = parser.parse_args()

    script_dir = _script_dir()
    marble_dir = args.marble_dir or _default_marble_dir()
    jsonl_path = args.jsonl or _default_research_jsonl()

    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        return 2

    router_url = args.router_url
    if not router_url:
        router_url_file = args.router_url_file or (script_dir / "logs" / "router_url.txt")
        if router_url_file.exists():
            router_url = router_url_file.read_text().strip()
        else:
            router_url = os.getenv("ROUTER_BASE_URL")
            if not router_url:
                print(
                    f"ERROR: Router URL not found. Use --router-url, --router-url-file, or ROUTER_BASE_URL env",
                    file=sys.stderr,
                )
                return 2

    router_base_url = router_url.rstrip("/")
    if not router_base_url.endswith("/v1"):
        router_base_url = f"{router_base_url}/v1"

    print(f"Router base URL: {router_base_url}")
    print(f"Loading MARBLE traces from: {jsonl_path}")

    traces = load_jsonl(jsonl_path, start=0, limit=args.num_traces)
    steps = build_steps_from_traces(traces, max_iterations_override=args.max_iterations)

    if not steps:
        print("ERROR: No steps built from traces.", file=sys.stderr)
        return 2

    print(f"Built {len(steps)} steps from {len(traces)} traces")

    llm = make_local_chat_model(
        model=args.model,
        base_url=router_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )

    if args.output:
        out_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = script_dir / "results" / f"{ts}_langchain.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(steps)} steps → {out_path}")

    asyncio.run(
        run_steps(
            steps,
            llm=llm,
            concurrency=args.concurrency,
            out_path=out_path,
            include_prompts=args.include_prompts,
        )
    )

    print(f"Done. Results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
