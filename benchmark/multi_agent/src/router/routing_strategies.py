"""
Routing Strategies for SGLang Multi-Agent Experiments

Implements three routing strategies:
1. Round Robin - Distribute requests evenly across workers
2. Agent Sticky - Route all requests from same agent to same worker
3. Trace Sticky - Route all requests from same trace to same worker
"""

from __future__ import annotations

import hashlib
import re
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def _get_no_proxy_session() -> requests.Session:
    """Create a requests session that bypasses proxy for internal hosts."""
    session = requests.Session()
    # Disable proxy for this session
    session.trust_env = False
    return session


# Global session for all requests
_session = _get_no_proxy_session()


@dataclass
class RoutingContext:
    """Context for routing decisions."""
    trace_id: str
    agent_id: str
    request_text: Optional[str] = None
    extra: Dict[str, Any] = None


class RoutingStrategy(ABC):
    """Base class for routing strategies."""

    def __init__(self, worker_urls: List[str]):
        self.worker_urls = worker_urls
        self.request_count = 0

    @abstractmethod
    def select_worker(self, context: RoutingContext) -> str:
        """Select a worker URL for the given context."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    def reset(self):
        """Reset any internal state."""
        self.request_count = 0


class RoundRobinStrategy(RoutingStrategy):
    """Round Robin routing - distribute requests evenly."""

    def __init__(self, worker_urls: List[str]):
        super().__init__(worker_urls)
        self._counter = 0

    @property
    def name(self) -> str:
        return "round_robin"

    def select_worker(self, context: RoutingContext) -> str:
        idx = self._counter % len(self.worker_urls)
        self._counter += 1
        self.request_count += 1
        return self.worker_urls[idx]

    def reset(self):
        super().reset()
        self._counter = 0


class AgentStickyStrategy(RoutingStrategy):
    """Agent Sticky routing with explicit worker pools per agent.

    Default behavior (when worker_urls has at least 5 entries):
    - agent 1 -> worker 0
    - agent 2 -> workers 1,2 (round-robin per request)
    - agent 3 -> workers 3,4 (round-robin per request)

    For other agent ids, they will be routed to workers 1..N-1 (round-robin).
    """

    def __init__(self, worker_urls: List[str]):
        super().__init__(worker_urls)
        self._agent_to_worker: Dict[str, str] = {}
        self._agent_rr_counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "agent_sticky"

    def select_worker(self, context: RoutingContext) -> str:
        agent_id = context.agent_id or "unknown"

        def parse_agent_num(aid: str) -> Optional[int]:
            m = re.search(r"(\d+)", aid)
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        def pick_pool(aid: str) -> List[int]:
            n = len(self.worker_urls)
            if n <= 0:
                return [0]

            agent_num = parse_agent_num(aid)
            # Use 5-instance layout when available.
            if n >= 5 and agent_num in (1, 2, 3):
                if agent_num == 1:
                    return [0]
                if agent_num == 2:
                    return [1, 2]
                return [3, 4]

            if n == 1:
                return [0]

            # Fallback: avoid worker 0 being overloaded by always including all remaining workers.
            return list(range(1, n))

        pool = pick_pool(agent_id)
        pool = [i for i in pool if 0 <= i < len(self.worker_urls)]
        if not pool:
            pool = [0]

        with self._lock:
            # Round-robin within the pool.
            c = self._agent_rr_counters.get(agent_id, 0)
            idx = pool[c % len(pool)]
            self._agent_rr_counters[agent_id] = c + 1
            self._agent_to_worker[agent_id] = self.worker_urls[idx]
            self.request_count += 1
            return self.worker_urls[idx]

    def reset(self):
        super().reset()
        self._agent_to_worker = {}
        self._agent_rr_counters = {}

    def get_agent_mapping(self) -> Dict[str, str]:
        """Return the current agent to worker mapping."""
        return self._agent_to_worker.copy()


class TraceStickyStrategy(RoutingStrategy):
    """Trace Sticky routing - same trace always goes to same worker."""

    def __init__(self, worker_urls: List[str]):
        super().__init__(worker_urls)
        self._trace_to_worker: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "trace_sticky"

    def select_worker(self, context: RoutingContext) -> str:
        trace_id = context.trace_id

        if trace_id not in self._trace_to_worker:
            # Hash trace_id to get consistent worker assignment
            hash_val = int(hashlib.md5(trace_id.encode()).hexdigest(), 16)
            idx = hash_val % len(self.worker_urls)
            self._trace_to_worker[trace_id] = self.worker_urls[idx]

        self.request_count += 1
        return self._trace_to_worker[trace_id]

    def reset(self):
        super().reset()
        self._trace_to_worker = {}

    def get_trace_mapping(self) -> Dict[str, str]:
        """Return the current trace to worker mapping."""
        return self._trace_to_worker.copy()


class RoutedLLMClient:
    """LLM client that uses a routing strategy to select workers."""

    def __init__(
        self,
        worker_urls: List[str],
        strategy: RoutingStrategy,
        model: str = "Qwen/Qwen3-4B-Thinking-2507",
        api_key: str = "EMPTY",
        timeout: int = 600,
    ):
        self.worker_urls = worker_urls
        self.strategy = strategy
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

        # Track per-request metrics
        self.last_worker_url: Optional[str] = None
        self.last_ttft: Optional[float] = None
        self.last_latency: Optional[float] = None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        context: RoutingContext,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the selected worker."""
        worker_url = self.strategy.select_worker(context)
        self.last_worker_url = worker_url

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_time = time.time()
        self.last_ttft = None

        try:
            if stream:
                return self._stream_request(worker_url, payload, headers, start_time)
            else:
                resp = _session.post(
                    f"{worker_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                self.last_latency = time.time() - start_time

                if resp.status_code == 200:
                    result = resp.json()
                    # For non-streaming, TTFT is approximately the full latency
                    # (since we get all tokens at once)
                    self.last_ttft = self.last_latency
                    return result
                else:
                    raise Exception(f"Request failed: {resp.status_code} - {resp.text[:200]}")

        except Exception as e:
            self.last_latency = time.time() - start_time
            raise

    def _stream_request(
        self,
        worker_url: str,
        payload: Dict,
        headers: Dict,
        start_time: float,
    ) -> Dict[str, Any]:
        """Handle streaming request and measure TTFT."""
        payload["stream"] = True

        resp = _session.post(
            f"{worker_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise Exception(f"Request failed: {resp.status_code}")

        content_parts = []
        first_token_received = False

        for line in resp.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                try:
                    import json
                    data = json.loads(data_str)

                    if not first_token_received:
                        self.last_ttft = time.time() - start_time
                        first_token_received = True

                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"] is not None:
                            content_parts.append(delta["content"])

                except json.JSONDecodeError:
                    continue

        self.last_latency = time.time() - start_time

        # Filter out any None values and construct response
        valid_parts = [p for p in content_parts if p is not None]

        # Construct a response similar to non-streaming
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "".join(valid_parts),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,  # Not available in streaming
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def create_strategy(name: str, worker_urls: List[str]) -> RoutingStrategy:
    """Factory function to create routing strategies."""
    strategies = {
        "round_robin": RoundRobinStrategy,
        "agent_sticky": AgentStickyStrategy,
        "trace_sticky": TraceStickyStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](worker_urls)


if __name__ == "__main__":
    # Test routing strategies
    worker_urls = [
        "http://gpu17:8000",
        "http://gpu17:8001",
        "http://gpu18:8002",
        "http://gpu18:8003",
        "http://gpu19:8004",
    ]

    print("Testing Round Robin Strategy:")
    rr = RoundRobinStrategy(worker_urls)
    for i in range(10):
        ctx = RoutingContext(trace_id=f"trace_{i % 3}", agent_id=f"agent_{i % 2}")
        print(f"  Request {i}: {rr.select_worker(ctx)}")

    print("\nTesting Agent Sticky Strategy:")
    agent_sticky = AgentStickyStrategy(worker_urls)
    for i in range(10):
        ctx = RoutingContext(trace_id=f"trace_{i % 3}", agent_id=f"agent_{i % 3}")
        print(f"  Request {i} (agent_{i % 3}): {agent_sticky.select_worker(ctx)}")
    print(f"  Agent mapping: {agent_sticky.get_agent_mapping()}")

    print("\nTesting Trace Sticky Strategy:")
    trace_sticky = TraceStickyStrategy(worker_urls)
    for i in range(10):
        ctx = RoutingContext(trace_id=f"trace_{i % 3}", agent_id=f"agent_{i % 2}")
        print(f"  Request {i} (trace_{i % 3}): {trace_sticky.select_worker(ctx)}")
    print(f"  Trace mapping: {trace_sticky.get_trace_mapping()}")
