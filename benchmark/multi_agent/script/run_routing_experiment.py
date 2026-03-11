"""
Unified Routing Experiment Script for SGLang Multi-Agent Benchmarks

This script runs experiments to compare different routing strategies:
1. Round Robin - Distribute requests evenly across workers
2. Agent Sticky - Route all requests from same agent to same worker
3. Trace Sticky - Route all requests from same trace to same worker

Experiment Flow:
1. Flush all worker caches
2. Run warmup traces (optional)
3. For each routing strategy:
   a. Reset metrics
   b. Run all coding traces
   c. Collect cache hit rate and TTFT per worker
   d. Visualize radix tree state
4. Generate comparison report

Usage:
    python run_routing_experiment.py --worker-urls-file logs/worker_urls.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Local imports - add paths for src modules
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR / "src" / "server") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR / "src" / "server"))
if str(_SCRIPT_DIR / "src" / "router") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR / "src" / "router"))
if str(_SCRIPT_DIR / "script") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR / "script"))
if str(_SCRIPT_DIR / "script" / "kvcache") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR / "script" / "kvcache"))

from metrics_collector import MetricsCollector, RequestMetrics, load_worker_urls
from radix_tree_visualizer import RadixTreeVisualizer
from prompt_tree_visualizer import PromptTreeVisualizer
from routing_strategies import (
    RoutingContext,
    RoutingStrategy,
    RoutedLLMClient,
    create_strategy,
)

# LangChain imports
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_coding_jsonl() -> Path:
    return _repo_root() / "benchmark" / "multi_agent" / "dataset" / "MARBLE" / "multiagentbench" / "coding" / "coding_main.jsonl"


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


@dataclass
class CodeWorkspace:
    """Shared workspace for code artifacts."""
    code: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update_code(self, new_code: str, agent_id: str, action: str, advice: str = ""):
        self.history.append({
            "agent_id": agent_id,
            "action": action,
            "advice": advice,
            "code_snapshot": new_code,
            "timestamp": time.time(),
        })
        self.code = new_code


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    worker_urls: List[str]
    jsonl_path: Path
    strategies: List[str]
    warmup_traces: int = 5
    warmup_scope: str = "per_strategy"  # per_strategy | once | none
    flush_before_strategy: bool = True
    model: str = "Qwen/Qwen3-4B-Thinking-2507"
    temperature: float = 0.7
    max_tokens: int = 8192
    output_dir: Path = Path("logs/experiments")
    use_streaming: bool = True  # Enable streaming to measure TTFT
    concurrency: int = 5  # Number of traces to run concurrently


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    strategy_name: str
    total_traces: int
    total_requests: int
    total_time: float
    overall_cache_hit_rate: float
    avg_ttft: float
    avg_latency: float
    worker_metrics: Dict[str, Dict[str, Any]]
    trace_results: List[Dict[str, Any]]


class RoutingExperiment:
    """Main experiment runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config.worker_urls)
        self.visualizer = RadixTreeVisualizer(
            config.worker_urls,
            output_dir=str(config.output_dir / "visualizations")
        )
        self.prompt_visualizer = PromptTreeVisualizer(
            config.worker_urls,
            output_dir=str(config.output_dir / "prompt_visualizations"),
        )
        self.results: Dict[str, ExperimentResult] = {}
        # Thread-safe locks for concurrent execution
        self._metrics_lock = threading.Lock()
        self._visualizer_lock = threading.Lock()
        self._print_lock = threading.Lock()

    def load_all_traces(self) -> List[Dict[str, Any]]:
        """Load all traces from the JSONL file."""
        traces = []
        with self.config.jsonl_path.open() as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line.strip()))
        return traces

    def create_agents_from_trace(self, trace: Dict[str, Any]) -> List[CodingAgent]:
        """Create CodingAgent instances from trace data."""
        agents = []
        for agent_data in trace.get("agents", []):
            agent_id = agent_data.get("agent_id", "unknown")
            profile = agent_data.get("profile", "")
            agent_type = agent_data.get("type", "CodingAgent")

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

    def _build_user_prompt(self, agent: CodingAgent, step: int, task_content: str, workspace: CodeWorkspace) -> str:
        """Build user prompt based on current state."""
        if step == 0:
            return f"""
## Software Development Task

{task_content}

## Your Task

I am the first developer. Please create the initial code framework for this project.
Create a complete, well-structured Python implementation in solution.py.

Use the create_code action to submit your code.
"""
        else:
            current_code = workspace.code
            history_lines = []
            for i, entry in enumerate(workspace.history):
                action = entry["action"]
                agent_id = entry["agent_id"]
                advice = entry.get("advice", "")
                history_lines.append(f"{i+1}. [{agent_id}] {action}")
                if advice:
                    history_lines.append(f"   Advice: {advice[:200]}...")

            history_summary = "\n".join(history_lines) if history_lines else "No previous changes."

            return f"""
## Software Development Task

{task_content}

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

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract Python code from response."""
        text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        python_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(python_pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len)
        return None

    def _extract_advice_from_response(self, response_text: str) -> str:
        """Extract advice from response."""
        text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            p = p.strip()
            if len(p) > 50 and '```' not in p:
                return p[:500]
        return "Code revised based on requirements."

    def run_single_trace(
        self,
        trace: Dict[str, Any],
        trace_idx: int,
        client: RoutedLLMClient,
        strategy: RoutingStrategy,
    ) -> Dict[str, Any]:
        """Run a single trace and collect metrics (thread-safe)."""
        trace_id = trace.get("task_id", f"trace_{trace_idx}")
        scenario = trace.get("scenario", "unknown")

        with self._print_lock:
            print(f"\n  [Trace {trace_idx}] Starting: {trace_id} ({scenario})")

        agents = self.create_agents_from_trace(trace)
        task = trace.get("task", {})
        task_content = task.get("content", "")

        workspace = CodeWorkspace()
        trace_result = {
            "trace_id": trace_id,
            "scenario": scenario,
            "steps": [],
            "success": False,
            "total_time": 0,
        }

        start_time = time.time()

        for step, agent in enumerate(agents):
            system_prompt = self._build_system_prompt(agent)
            user_prompt = self._build_user_prompt(agent, step, task_content, workspace)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            context = RoutingContext(
                trace_id=trace_id,
                agent_id=agent.agent_id,
            )

            try:
                t0 = time.time()
                response = client.chat_completion(
                    messages=messages,
                    context=context,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=self.config.use_streaming,
                )
                latency = time.time() - t0

                # Extract response content
                content = ""
                if "choices" in response and len(response["choices"]) > 0:
                    msg = response["choices"][0].get("message", {})
                    content = msg.get("content", "")

                # Extract code
                code = self._extract_code_from_response(content)
                success = False

                if code:
                    if agent.can_create() and step == 0:
                        workspace.update_code(code, agent.agent_id, "create_code")
                        success = True
                    elif agent.can_revise():
                        advice = self._extract_advice_from_response(content)
                        workspace.update_code(code, agent.agent_id, "give_advice_and_revise_code", advice)
                        success = True

                # Record metrics (thread-safe)
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cache_hit_tokens = 0

                request_metrics = RequestMetrics(
                    request_id=f"{trace_id}_{step}",
                    trace_id=trace_id,
                    agent_id=agent.agent_id,
                    worker_url=client.last_worker_url,
                    ttft=client.last_ttft,
                    total_latency=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_hit_tokens=cache_hit_tokens,
                )

                with self._metrics_lock:
                    self.metrics_collector.record_request(request_metrics)

                # Record for visualization (thread-safe)
                with self._visualizer_lock:
                    self.visualizer.record_request(
                        worker_url=client.last_worker_url,
                        trace_id=trace_id,
                        agent_id=agent.agent_id,
                        input_tokens=input_tokens,
                        cache_hit_tokens=cache_hit_tokens,
                        output_tokens=output_tokens,
                    )

                trace_result["steps"].append({
                    "agent_id": agent.agent_id,
                    "step": step + 1,
                    "success": success,
                    "latency": latency,
                    "ttft": client.last_ttft,
                    "worker_url": client.last_worker_url,
                })

                with self._print_lock:
                    ttft_str = f"{client.last_ttft:.2f}s" if client.last_ttft else "N/A"
                    print(f"    [Trace {trace_idx}] Step {step + 1}/{len(agents)}: {agent.agent_id} -> {client.last_worker_url} "
                          f"(TTFT: {ttft_str}, Latency: {latency:.2f}s)")

            except Exception as e:
                with self._print_lock:
                    print(f"    [Trace {trace_idx}] Step {step + 1}/{len(agents)}: {agent.agent_id} - ERROR: {e}")
                trace_result["steps"].append({
                    "agent_id": agent.agent_id,
                    "step": step + 1,
                    "success": False,
                    "error": str(e),
                })

        trace_result["total_time"] = time.time() - start_time
        trace_result["success"] = bool(workspace.code)
        trace_result["final_code_length"] = len(workspace.code)

        with self._print_lock:
            print(f"  [Trace {trace_idx}] Completed: {trace_id} (success={trace_result['success']}, time={trace_result['total_time']:.2f}s)")

        return trace_result

    def run_traces_concurrent(
        self,
        traces: List[Dict[str, Any]],
        strategy: RoutingStrategy,
        concurrency: int,
    ) -> List[Dict[str, Any]]:
        """Run multiple traces concurrently."""
        trace_results = [None] * len(traces)

        def run_trace_wrapper(idx: int, trace: Dict[str, Any]) -> tuple:
            # Each thread gets its own client instance
            client = RoutedLLMClient(
                worker_urls=self.config.worker_urls,
                strategy=strategy,
                model=self.config.model,
            )
            result = self.run_single_trace(trace, idx, client, strategy)
            return idx, result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run_trace_wrapper, idx, trace): idx
                for idx, trace in enumerate(traces)
            }

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    trace_results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    with self._print_lock:
                        print(f"  [Trace {idx}] Failed with exception: {e}")
                    trace_results[idx] = {
                        "trace_id": f"trace_{idx}",
                        "success": False,
                        "error": str(e),
                        "steps": [],
                        "total_time": 0,
                    }

        return trace_results

    def run_experiment_for_strategy(
        self,
        strategy_name: str,
        traces: List[Dict[str, Any]],
        warmup: bool = False,
    ) -> ExperimentResult:
        """Run experiment for a single routing strategy."""
        print(f"\n{'=' * 80}")
        print(f"RUNNING EXPERIMENT: {strategy_name.upper()}")
        print(f"Concurrency: {self.config.concurrency} traces")
        print(f"{'=' * 80}")

        # Create strategy
        strategy = create_strategy(strategy_name, self.config.worker_urls)

        # Reset metrics
        self.metrics_collector.reset_metrics()
        self.visualizer.reset()

        # Capture baseline
        self.metrics_collector.capture_baseline_stats()

        start_time = time.time()

        # Run traces concurrently
        if self.config.concurrency > 1:
            trace_results = self.run_traces_concurrent(traces, strategy, self.config.concurrency)
        else:
            # Sequential execution (for debugging)
            trace_results = []
            client = RoutedLLMClient(
                worker_urls=self.config.worker_urls,
                strategy=strategy,
                model=self.config.model,
            )
            for idx, trace in enumerate(traces):
                result = self.run_single_trace(trace, idx, client, strategy)
                trace_results.append(result)

        total_time = time.time() - start_time

        # Final snapshot
        self.visualizer.capture_all_snapshots()

        # Capture prompt radix trees per worker (best effort)
        prompt_captured = self.prompt_visualizer.capture_all(
            strategy_name=strategy_name,
            include_prefix=True,
            include_segment=True,
            include_text=True,
            max_nodes=3000,
            max_depth=128,
            max_tokens_per_node=8192,
            timeout_s=30,
        )
        self.prompt_visualizer.export_to_json(
            prompt_captured,
            filename=f"{strategy_name}_prompt_trees.json",
        )
        self.prompt_visualizer.generate_html_report(
            prompt_captured,
            filename=f"{strategy_name}_prompt_trees.html",
            title=f"Prompt Trees - {strategy_name}",
        )

        # Get server-side cache hit rates
        server_stats = self.metrics_collector.get_current_stats()

        # Get summary
        summary = self.metrics_collector.get_summary()

        # Build worker metrics with server-side stats
        worker_metrics = {}
        for url in self.config.worker_urls:
            client_metrics = summary["worker_summaries"].get(url, {})
            server_metric = server_stats.get(url, {})

            worker_metrics[url] = {
                "requests": client_metrics.get("total_requests", 0),
                "client_cache_hit_rate": client_metrics.get("cache_hit_rate", 0.0),
                "server_cache_hit_rate": server_metric.get("cache_hit_rate", 0.0),
                "avg_ttft": client_metrics.get("avg_ttft", 0.0),
                "avg_latency": client_metrics.get("avg_latency", 0.0),
                "token_usage": server_metric.get("token_usage", 0.0),
            }

        result = ExperimentResult(
            strategy_name=strategy_name,
            total_traces=len(traces),
            total_requests=summary["total_requests"],
            total_time=total_time,
            overall_cache_hit_rate=summary["overall_cache_hit_rate"],
            avg_ttft=summary["avg_ttft"],
            avg_latency=summary["avg_latency"],
            worker_metrics=worker_metrics,
            trace_results=trace_results,
        )

        # Print summary
        self.metrics_collector.print_summary()

        # Print server-side cache hit rates (from Prometheus)
        print("\n" + "-" * 80)
        print("SERVER-SIDE CACHE HIT RATES (from Prometheus metrics)")
        print("-" * 80)
        total_prompt_tokens = 0
        total_cached_tokens = 0
        for url, stats in server_stats.items():
            cache_hit_rate = stats.get('cache_hit_rate', 0.0)
            prompt_tokens = stats.get('prompt_tokens', 0)
            cached_tokens = stats.get('cached_tokens', 0)
            total_prompt_tokens += prompt_tokens
            total_cached_tokens += cached_tokens
            print(f"  {url}:")
            print(f"    Cache Hit Rate: {cache_hit_rate:.4f} ({cache_hit_rate*100:.2f}%)")
            print(f"    Prompt Tokens: {prompt_tokens:,}")
            print(f"    Cached Tokens: {cached_tokens:,}")

        # Overall cache hit rate
        overall_total = total_prompt_tokens + total_cached_tokens
        overall_cache_hit = total_cached_tokens / overall_total if overall_total > 0 else 0.0
        print(f"\n  OVERALL:")
        print(f"    Total Prompt Tokens: {total_prompt_tokens:,}")
        print(f"    Total Cached Tokens: {total_cached_tokens:,}")
        print(f"    Overall Cache Hit Rate: {overall_cache_hit:.4f} ({overall_cache_hit*100:.2f}%)")

        # Generate visualizations
        self.visualizer.print_all_workers()
        self.visualizer.export_to_json(f"{strategy_name}_tree_data.json")
        self.visualizer.generate_html_report(f"{strategy_name}_report.html")

        return result

    def run_all_experiments(self, max_traces: Optional[int] = None) -> Dict[str, ExperimentResult]:
        """Run experiments for all strategies."""
        all_traces = self.load_all_traces()
        print(f"Loaded {len(all_traces)} traces from {self.config.jsonl_path}")

        # Limit traces if specified
        if max_traces is not None:
            # warmup traces + experiment traces
            total_needed = self.config.warmup_traces + max_traces
            traces = all_traces[:total_needed]
            print(f"Using {len(traces)} traces total (warmup: {self.config.warmup_traces}, experiment: {max_traces})")
        else:
            traces = all_traces

        warmup_traces = traces[: self.config.warmup_traces] if self.config.warmup_traces > 0 else []
        experiment_traces = traces[self.config.warmup_traces :] if self.config.warmup_traces > 0 else traces

        print(f"\nWarmup traces: {len(warmup_traces)}")
        print(f"Experiment traces: {len(experiment_traces)}")

        warmup_scope = (self.config.warmup_scope or "none").lower()
        if warmup_scope not in ("per_strategy", "once", "none"):
            print(f"Warning: unknown warmup_scope={warmup_scope}, fallback to none")
            warmup_scope = "none"

        # Run experiments for each strategy
        for strategy_name in self.config.strategies:
            strategy = create_strategy(strategy_name, self.config.worker_urls)

            if self.config.flush_before_strategy:
                print(f"\nFlushing caches before {strategy_name} experiment...")
                self.metrics_collector.flush_all_caches()
                time.sleep(2)

            if warmup_traces and warmup_scope == "per_strategy":
                print(f"\n{'=' * 80}")
                print(f"WARMUP ({strategy_name}) - {len(warmup_traces)} traces")
                print(f"{'=' * 80}")
                self.run_traces_concurrent(warmup_traces, strategy, self.config.concurrency)
                print("\nWarmup complete. Keeping KV cache for experiment.")

            if warmup_traces and warmup_scope == "once":
                print(f"\n{'=' * 80}")
                print(f"WARMUP (once) - {len(warmup_traces)} traces")
                print(f"{'=' * 80}")
                warmup_strategy = create_strategy("round_robin", self.config.worker_urls)
                self.run_traces_concurrent(warmup_traces, warmup_strategy, self.config.concurrency)
                print("\nWarmup complete. Keeping KV cache for experiment.")
                # Prevent running warmup multiple times
                warmup_traces = []

            result = self.run_experiment_for_strategy(strategy_name, experiment_traces)
            self.results[strategy_name] = result

        return self.results

    def generate_comparison_report(self) -> str:
        """Generate a comparison report of all strategies."""
        if not self.results:
            return "No results to compare."

        report_lines = []
        report_lines.append("\n" + "=" * 100)
        report_lines.append("EXPERIMENT COMPARISON REPORT")
        report_lines.append("=" * 100)

        # Summary table
        report_lines.append("\n" + "-" * 100)
        report_lines.append(f"{'Strategy':<20} {'Traces':<10} {'Requests':<10} {'Time (s)':<12} "
                           f"{'Cache Hit':<12} {'Avg TTFT':<12} {'Avg Latency':<12}")
        report_lines.append("-" * 100)

        for name, result in self.results.items():
            report_lines.append(
                f"{name:<20} {result.total_traces:<10} {result.total_requests:<10} "
                f"{result.total_time:<12.2f} {result.overall_cache_hit_rate:<12.4f} "
                f"{result.avg_ttft:<12.4f} {result.avg_latency:<12.4f}"
            )

        # Per-worker comparison
        report_lines.append("\n" + "-" * 100)
        report_lines.append("PER-WORKER CACHE HIT RATES")
        report_lines.append("-" * 100)

        header = f"{'Worker':<25}"
        for name in self.results.keys():
            header += f" {name:<15}"
        report_lines.append(header)

        for url in self.config.worker_urls:
            line = f"{url:<25}"
            for name, result in self.results.items():
                wm = result.worker_metrics.get(url, {})
                cache_rate = wm.get("server_cache_hit_rate", 0.0)
                line += f" {cache_rate:<15.4f}"
            report_lines.append(line)

        # Per-worker request distribution
        report_lines.append("\n" + "-" * 100)
        report_lines.append("PER-WORKER REQUEST DISTRIBUTION")
        report_lines.append("-" * 100)

        header = f"{'Worker':<25}"
        for name in self.results.keys():
            header += f" {name:<15}"
        report_lines.append(header)

        for url in self.config.worker_urls:
            line = f"{url:<25}"
            for name, result in self.results.items():
                wm = result.worker_metrics.get(url, {})
                requests = wm.get("requests", 0)
                line += f" {requests:<15}"
            report_lines.append(line)

        report_lines.append("\n" + "=" * 100)

        report = "\n".join(report_lines)
        print(report)

        # Save report
        report_path = self.config.output_dir / "comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        return report

    def save_results(self):
        """Save all results to JSON."""
        results_data = {}
        for name, result in self.results.items():
            results_data[name] = {
                "strategy_name": result.strategy_name,
                "total_traces": result.total_traces,
                "total_requests": result.total_requests,
                "total_time": result.total_time,
                "overall_cache_hit_rate": result.overall_cache_hit_rate,
                "avg_ttft": result.avg_ttft,
                "avg_latency": result.avg_latency,
                "worker_metrics": result.worker_metrics,
                "trace_results": result.trace_results,
            }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = self.config.output_dir / f"experiment_results_{timestamp}.json"

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {results_path}")
        return str(results_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run routing strategy experiments for SGLang multi-agent benchmarks"
    )

    # Data arguments
    parser.add_argument("--jsonl", type=str, default=str(_default_coding_jsonl()),
                        help="Path to coding_main.jsonl")
    parser.add_argument("--worker-urls-file", type=str, default="logs/worker_urls.txt",
                        help="Path to worker_urls.txt")

    # Experiment arguments
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["round_robin", "agent_sticky", "trace_sticky"],
                        help="Routing strategies to test")
    parser.add_argument("--warmup-traces", type=int, default=0,
                        help="Number of traces for warmup (0 to disable)")
    parser.add_argument(
        "--warmup-scope",
        type=str,
        choices=["per_strategy", "once", "none"],
        default="per_strategy",
        help="Warmup scope: per_strategy|once|none (default: per_strategy)",
    )
    parser.add_argument("--max-traces", type=int, default=20,
                        help="Maximum number of traces to run (default: 20)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of traces to run concurrently (default: 5)")
    parser.add_argument(
        "--no-flush-before-strategy",
        action="store_true",
        help="Do not flush KV cache before each strategy",
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum tokens per response")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable streaming (TTFT will be less accurate)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="logs/experiments",
                        help="Output directory for results")

    args = parser.parse_args(argv)

    # Load worker URLs
    script_dir = Path(__file__).parent
    worker_urls_path = script_dir / args.worker_urls_file
    if not worker_urls_path.exists():
        print(f"Error: Worker URLs file not found: {worker_urls_path}")
        return 1

    worker_urls = load_worker_urls(str(worker_urls_path))
    print(f"Loaded {len(worker_urls)} worker URLs")

    # Check JSONL file
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}")
        return 1

    # Create output directory
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = ExperimentConfig(
        worker_urls=worker_urls,
        jsonl_path=jsonl_path,
        strategies=args.strategies,
        warmup_traces=args.warmup_traces,
        warmup_scope=args.warmup_scope,
        flush_before_strategy=not args.no_flush_before_strategy,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        use_streaming=not args.no_streaming,
        concurrency=args.concurrency,
    )

    # Run experiment
    experiment = RoutingExperiment(config)

    try:
        experiment.run_all_experiments(max_traces=args.max_traces)
        experiment.generate_comparison_report()
        experiment.save_results()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        experiment.generate_comparison_report()
        experiment.save_results()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
