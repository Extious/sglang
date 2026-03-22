"""
Fault-aware router for multi-agent experiments.

Features:
- Agent-aware / cache-aware / round-robin worker ordering
- Runtime failure injection (down/up/toggle) via admin endpoint
- Automatic failover when a worker is marked down
- Optional temporary down with auto-recovery timer
- Router-side per-worker/per-agent statistics
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from aiohttp import ClientSession, ClientTimeout, web


LOG = logging.getLogger(__name__)

AGENT_ID_MARKER_RE = re.compile(r"\[agent_id\s*:\s*([a-f])\]", re.IGNORECASE)
PROMPT_TOKENS_METRIC = "sglang:prompt_tokens_total"
CACHED_TOKENS_METRIC = "sglang:cached_tokens_total"


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


@dataclass
class RouterStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    failover_requests: int = 0
    down_reject_requests: int = 0
    per_worker_requests: Dict[str, int] = field(default_factory=dict)
    per_agent_requests: Dict[str, int] = field(default_factory=dict)
    per_agent_worker_requests: Dict[str, Dict[str, int]] = field(default_factory=dict)
    prefill_prompt_tokens_total: float = 0.0
    prefill_cached_tokens_total: float = 0.0

    def note_success(
        self,
        worker: str,
        agent: str,
        attempts: int,
        prefill_prompt_tokens: float = 0.0,
        prefill_cached_tokens: float = 0.0,
    ) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        if attempts > 1:
            self.failover_requests += 1

        self.per_worker_requests[worker] = self.per_worker_requests.get(worker, 0) + 1
        self.per_agent_requests[agent] = self.per_agent_requests.get(agent, 0) + 1
        by_agent = self.per_agent_worker_requests.setdefault(agent, {})
        by_agent[worker] = by_agent.get(worker, 0) + 1

        self.prefill_prompt_tokens_total += float(prefill_prompt_tokens or 0.0)
        self.prefill_cached_tokens_total += float(prefill_cached_tokens or 0.0)

    def note_failure(self, down_reject: bool) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        if down_reject:
            self.down_reject_requests += 1


class AgentFaultRouter:
    def __init__(
        self,
        worker_urls: List[str],
        policy: str = "cache_aware",
        request_timeout_s: float = 180.0,
        max_attempts: int = 6,
        prefill_stats_mode: str = "non_blocking",
        metrics_timeout_s: float = 1.0,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        auto_down_cooldown_s: float = 0.0,
    ):
        if not worker_urls:
            raise ValueError("worker_urls cannot be empty")

        if policy not in {"agent_aware", "cache_aware", "round_robin"}:
            raise ValueError(f"Unsupported policy: {policy}")

        self.worker_urls = worker_urls
        self.policy = policy
        self.request_timeout_s = request_timeout_s
        self.max_attempts = max_attempts
        self.prefill_stats_mode = str(prefill_stats_mode or "non_blocking").strip().lower()
        if self.prefill_stats_mode not in {"blocking", "non_blocking", "disabled"}:
            raise ValueError(f"Unsupported prefill_stats_mode: {self.prefill_stats_mode}")

        self.metrics_timeout_s = max(0.1, float(metrics_timeout_s))
        self.model_path = model_path
        self.auto_down_cooldown_s = max(0.0, float(auto_down_cooldown_s))

        self.down_workers: Set[str] = set()
        self.down_since: Dict[str, float] = {}
        self.down_reason: Dict[str, str] = {}
        self._recover_tasks: Dict[str, asyncio.Task] = {}

        self.inflight = {u: 0 for u in worker_urls}
        self.stats = RouterStats()
        self.agent_rr_cursor: Dict[str, int] = {}
        self.worker_measure_locks = {u: asyncio.Lock() for u in worker_urls}
        self._session: Optional[ClientSession] = None
        self._global_rr_cursor = 0

        self.agent_worker_indices = {
            "A": [0, 1, 2, 3, 4, 5],
            "B": [0, 1],
            "C": [0, 1, 2, 3, 4, 5],
            "D": [2, 3],
            "E": [4, 5],
            "F": [0, 1, 2, 3, 4, 5],
        }
        self.agent_alias_to_id = {
            "A": "A",
            "QUESTION": "A",
            "QUESTION-AGENT": "A",
            "B": "B",
            "RESEARCH": "B",
            "RESEARCH-AGENT": "B",
            "C": "C",
            "ANALYSIS": "C",
            "ANALYSIS-AGENT": "C",
            "D": "D",
            "ALTERNATIVES": "D",
            "ALTERNATIVES-AGENT": "D",
            "E": "E",
            "VERIFICATION": "E",
            "VERIFICATION-AGENT": "E",
            "F": "F",
            "SYNTHESIS": "F",
            "SYNTHESIS-AGENT": "F",
        }

    def _normalize_agent_id(self, raw: str) -> str:
        s = str(raw or "").strip().upper().replace("_", "-")
        if not s:
            return "UNKNOWN"
        return self.agent_alias_to_id.get(s, s)

    async def startup(self) -> None:
        self._session = ClientSession(
            timeout=ClientTimeout(total=self.request_timeout_s),
            trust_env=False,
        )

    async def shutdown(self) -> None:
        tasks = list(self._recover_tasks.values())
        self._recover_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._session is not None:
            await self._session.close()
            self._session = None

    def _all_candidates(self) -> List[str]:
        return list(self.worker_urls)

    def _rotate(self, items: List[str], seed: int) -> List[str]:
        if not items:
            return items
        start = seed % len(items)
        return items[start:] + items[:start]

    def _extract_agent_id(self, request: web.Request, payload: Dict[str, Any]) -> str:
        h = request.headers.get("X-Agent-Id")
        if h:
            return self._normalize_agent_id(h)

        meta = payload.get("metadata")
        if isinstance(meta, dict) and meta.get("agent_id"):
            return self._normalize_agent_id(str(meta["agent_id"]))

        step_id = request.headers.get("X-Step-Id", "")
        if "_" in step_id:
            tail = step_id.rsplit("_", 1)[-1].strip().upper()
            norm = self._normalize_agent_id(tail)
            if norm in self.agent_worker_indices:
                return norm

        inferred = self._infer_agent_id_from_system_prompt(payload)
        if inferred:
            return self._normalize_agent_id(inferred)

        return "UNKNOWN"

    def _infer_agent_id_from_system_prompt(
        self, payload: Dict[str, Any]
    ) -> Optional[str]:
        msgs = payload.get("messages") or []
        if not isinstance(msgs, list):
            return None

        system_parts: List[str] = []
        for one in msgs:
            if not isinstance(one, dict):
                continue
            role = str(one.get("role", "")).strip().lower()
            if role == "system":
                system_parts.append(str(one.get("content", "")))

        if not system_parts and msgs and isinstance(msgs[0], dict):
            system_parts.append(str(msgs[0].get("content", "")))

        text = "\n".join(system_parts).strip().lower()
        if not text:
            return None

        marker = AGENT_ID_MARKER_RE.search(text)
        if marker:
            return marker.group(1).upper()

        rules = [
            (
                "A",
                ["question-agent", "question agent", "generate_specialized_questions"],
            ),
            ("B", ["research-agent", "research agent"]),
            ("C", ["analysis-agent", "analysis agent"]),
            ("D", ["alternatives-agent", "alternatives agent"]),
            ("E", ["verification-agent", "verification agent"]),
            ("F", ["synthesis-agent", "synthesis agent"]),
        ]
        for agent_id, pats in rules:
            if any(p in text for p in pats):
                return agent_id
        return None

    def _extract_workflow_id(
        self, request: web.Request, payload: Dict[str, Any]
    ) -> str:
        h = request.headers.get("X-Workflow-Id")
        if h:
            return h
        meta = payload.get("metadata")
        if isinstance(meta, dict) and meta.get("workflow_id"):
            return str(meta["workflow_id"])
        return "wf-unknown"

    def _cache_aware_order(self, payload: Dict[str, Any]) -> List[str]:
        msgs = payload.get("messages") or []
        system_prompt = ""
        if msgs and isinstance(msgs, list):
            first = msgs[0]
            if isinstance(first, dict):
                system_prompt = str(first.get("content", ""))
        seed = stable_hash(system_prompt or "")
        return self._rotate(self._all_candidates(), seed)

    def _agent_aware_order(self, agent_id: str, workflow_id: str) -> List[str]:
        _ = workflow_id
        candidates_idx = self.agent_worker_indices.get(
            agent_id,
            list(range(len(self.worker_urls))),
        )
        candidates = [
            self.worker_urls[i]
            for i in candidates_idx
            if 0 <= i < len(self.worker_urls)
        ]
        if not candidates:
            candidates = self._all_candidates()

        cursor = self.agent_rr_cursor.get(agent_id, 0)
        ordered = self._rotate(candidates, cursor)
        self.agent_rr_cursor[agent_id] = (cursor + 1) % max(1, len(candidates))

        seen = set(ordered)
        for one in self.worker_urls:
            if one not in seen:
                ordered.append(one)
        return ordered

    def _round_robin_order(self) -> List[str]:
        ordered = self._rotate(self._all_candidates(), self._global_rr_cursor)
        self._global_rr_cursor = (self._global_rr_cursor + 1) % max(1, len(self.worker_urls))
        return ordered

    def _ordered_workers(
        self, payload: Dict[str, Any], agent_id: str, workflow_id: str
    ) -> List[str]:
        if self.policy == "cache_aware":
            return self._cache_aware_order(payload)
        if self.policy == "round_robin":
            return self._round_robin_order()
        return self._agent_aware_order(agent_id, workflow_id)

    def _eligible_workers(self, ordered: List[str]) -> List[str]:
        return [u for u in ordered if u not in self.down_workers]

    def _parse_metric_value(self, text: str, metric_name: str) -> float:
        for line in text.splitlines():
            if not line.startswith(metric_name):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                return float(parts[-1])
            except ValueError:
                continue
        return 0.0

    async def _collect_worker_prefill_counters(
        self, worker_url: str
    ) -> Dict[str, float]:
        if self._session is None:
            raise AssertionError

        if self.prefill_stats_mode == "disabled":
            return {
                "prompt_tokens_total": 0.0,
                "cached_tokens_total": 0.0,
            }

        try:
            async with self._session.get(
                f"{worker_url}/metrics",
                timeout=ClientTimeout(total=self.metrics_timeout_s),
            ) as resp:
                txt = await resp.text()
        except Exception:
            return {
                "prompt_tokens_total": 0.0,
                "cached_tokens_total": 0.0,
            }

        return {
            "prompt_tokens_total": self._parse_metric_value(
                txt, PROMPT_TOKENS_METRIC
            ),
            "cached_tokens_total": self._parse_metric_value(
                txt, CACHED_TOKENS_METRIC
            ),
        }

    def _extract_usage_prefill_tokens(
        self, body: Dict[str, Any]
    ) -> Dict[str, float]:
        if not isinstance(body, dict):
            return {"prompt_tokens": 0.0, "cached_tokens": 0.0}

        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0.0, "cached_tokens": 0.0}

        prompt_tokens = float(usage.get("prompt_tokens", 0.0) or 0.0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = 0.0
        if isinstance(prompt_details, dict):
            cached_tokens = float(
                prompt_details.get("cached_tokens", 0.0)
                or prompt_details.get("cache_tokens", 0.0)
                or 0.0
            )

        cached_tokens = max(0.0, min(cached_tokens, prompt_tokens))
        return {
            "prompt_tokens": max(0.0, prompt_tokens),
            "cached_tokens": cached_tokens,
        }

    def _normalize_worker_target(
        self, worker_url: Optional[str], worker_index: Optional[Any]
    ) -> Optional[str]:
        if worker_url is None and worker_index is None:
            return None
        if worker_url is None:
            try:
                idx = int(worker_index)
                return self.worker_urls[idx]
            except Exception:
                return None
        worker = str(worker_url)
        if worker not in self.worker_urls:
            return None
        return worker

    def _cancel_recover_task(self, worker_url: str) -> None:
        task = self._recover_tasks.pop(worker_url, None)
        if task is not None:
            task.cancel()

    async def _recover_worker_after(self, worker_url: str, duration_s: float) -> None:
        try:
            await asyncio.sleep(max(0.0, duration_s))
            if worker_url in self.down_workers:
                self.down_workers.discard(worker_url)
                self.down_since.pop(worker_url, None)
                self.down_reason.pop(worker_url, None)
                LOG.warning("Auto-recovered worker=%s after %.3fs", worker_url, duration_s)
        except asyncio.CancelledError:
            pass
        finally:
            current = self._recover_tasks.get(worker_url)
            if current is not None and current.done():
                self._recover_tasks.pop(worker_url, None)

    def _mark_worker_down(
        self,
        worker_url: str,
        reason: str,
        duration_s: Optional[float] = None,
    ) -> None:
        self.down_workers.add(worker_url)
        self.down_since[worker_url] = time.time()
        self.down_reason[worker_url] = reason

        self._cancel_recover_task(worker_url)
        if duration_s is not None and duration_s > 0:
            self._recover_tasks[worker_url] = asyncio.create_task(
                self._recover_worker_after(worker_url, duration_s)
            )

    def _mark_worker_up(self, worker_url: str) -> None:
        self.down_workers.discard(worker_url)
        self.down_since.pop(worker_url, None)
        self.down_reason.pop(worker_url, None)
        self._cancel_recover_task(worker_url)

    async def _proxy_one(
        self,
        worker: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        attempt_started_at: float,
    ) -> Dict[str, Any]:
        if self._session is None:
            raise AssertionError

        self.inflight[worker] = self.inflight.get(worker, 0) + 1
        try:
            async with self._session.post(
                f"{worker}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(
                        f"worker={worker} status={resp.status} body={txt[:300]}"
                    )
                try:
                    body = json.loads(txt)
                except json.JSONDecodeError:
                    raise RuntimeError(f"worker={worker} returned non-JSON payload")

                down_ts = self.down_since.get(worker)
                if down_ts is not None and down_ts >= attempt_started_at:
                    raise RuntimeError(
                        "worker={} disconnected during in-flight request "
                        "(down_ts={}, attempt_started_at={})".format(
                            worker,
                            down_ts,
                            attempt_started_at,
                        )
                    )
                return body
        finally:
            self.inflight[worker] = max(0, self.inflight.get(worker, 1) - 1)

    async def handle_completion(self, request: web.Request) -> web.Response:
        started_at = time.time()

        try:
            payload = await request.json()
        except Exception:
            self.stats.note_failure(down_reject=False)
            return web.json_response(
                {"error": "Invalid JSON payload"},
                status=400,
            )

        if payload.get("stream", False):
            self.stats.note_failure(down_reject=False)
            return web.json_response(
                {
                    "error": (
                        "This experiment router currently supports "
                        "stream=false only."
                    )
                },
                status=400,
            )

        agent_id = self._extract_agent_id(request, payload)
        workflow_id = self._extract_workflow_id(request, payload)
        ordered = self._ordered_workers(payload, agent_id, workflow_id)
        eligible = self._eligible_workers(ordered)

        if not eligible:
            self.stats.note_failure(down_reject=True)
            return web.json_response(
                {
                    "error": "No available workers",
                    "router_meta": {
                        "policy": self.policy,
                        "agent_id": agent_id,
                        "workflow_id": workflow_id,
                        "down_workers": sorted(self.down_workers),
                    },
                },
                status=503,
            )

        fwd_headers = {"Content-Type": "application/json"}
        auth = request.headers.get("Authorization")
        if auth:
            fwd_headers["Authorization"] = auth

        tried: List[str] = []
        errors: List[str] = []
        max_attempts = min(self.max_attempts, len(eligible))

        for worker in eligible[:max_attempts]:
            tried.append(worker)
            attempt_started_at = time.time()

            if worker in self.down_workers:
                errors.append(f"worker={worker} already down before attempt")
                continue

            try:
                if self.prefill_stats_mode == "blocking":
                    async with self.worker_measure_locks[worker]:
                        before = await self._collect_worker_prefill_counters(worker)
                        body = await self._proxy_one(
                            worker,
                            payload,
                            fwd_headers,
                            attempt_started_at=attempt_started_at,
                        )
                        after = await self._collect_worker_prefill_counters(worker)
                elif self.prefill_stats_mode == "non_blocking":
                    before = await self._collect_worker_prefill_counters(worker)
                    body = await self._proxy_one(
                        worker,
                        payload,
                        fwd_headers,
                        attempt_started_at=attempt_started_at,
                    )
                    after = await self._collect_worker_prefill_counters(worker)
                else:
                    before = {
                        "prompt_tokens_total": 0.0,
                        "cached_tokens_total": 0.0,
                    }
                    body = await self._proxy_one(
                        worker,
                        payload,
                        fwd_headers,
                        attempt_started_at=attempt_started_at,
                    )
                    after = {
                        "prompt_tokens_total": 0.0,
                        "cached_tokens_total": 0.0,
                    }

                prefill_prompt_delta = max(
                    0.0,
                    float(after.get("prompt_tokens_total", 0.0))
                    - float(before.get("prompt_tokens_total", 0.0)),
                )
                prefill_cached_delta = max(
                    0.0,
                    float(after.get("cached_tokens_total", 0.0))
                    - float(before.get("cached_tokens_total", 0.0)),
                )

                if prefill_prompt_delta <= 0.0:
                    usage_tokens = self._extract_usage_prefill_tokens(body)
                    prefill_prompt_delta = float(usage_tokens["prompt_tokens"])
                    prefill_cached_delta = float(usage_tokens["cached_tokens"])

                prefill_cached_delta = max(
                    0.0, min(prefill_cached_delta, prefill_prompt_delta)
                )
                prefill_hit_rate = (
                    prefill_cached_delta / prefill_prompt_delta
                    if prefill_prompt_delta > 0
                    else 0.0
                )

                body["router_meta"] = {
                    "policy": self.policy,
                    "agent_id": agent_id,
                    "workflow_id": workflow_id,
                    "selected_worker": worker,
                    "attempts": len(tried),
                    "tried_workers": tried,
                    "request_started_at": started_at,
                    "request_finished_at": time.time(),
                    "latency_s": time.time() - started_at,
                    "down_workers": sorted(self.down_workers),
                    "prefill_prompt_tokens_delta": prefill_prompt_delta,
                    "prefill_cached_tokens_delta": prefill_cached_delta,
                    "prefill_cache_hit_rate": prefill_hit_rate,
                    "prefill_stats_mode": self.prefill_stats_mode,
                }

                self.stats.note_success(
                    worker=worker,
                    agent=agent_id,
                    attempts=len(tried),
                    prefill_prompt_tokens=prefill_prompt_delta,
                    prefill_cached_tokens=prefill_cached_delta,
                )

                return web.json_response(body)
            except Exception as exc:
                errors.append(str(exc))
                if self.auto_down_cooldown_s > 0.0:
                    self._mark_worker_down(
                        worker,
                        reason="auto_error",
                        duration_s=self.auto_down_cooldown_s,
                    )
                continue

        self.stats.note_failure(down_reject=False)
        return web.json_response(
            {
                "error": "All worker attempts failed",
                "router_meta": {
                    "policy": self.policy,
                    "agent_id": agent_id,
                    "workflow_id": workflow_id,
                    "attempts": len(tried),
                    "tried_workers": tried,
                    "errors": errors[-5:],
                    "down_workers": sorted(self.down_workers),
                },
            },
            status=502,
        )

    async def handle_models(self, _request: web.Request) -> web.Response:
        now = int(time.time())
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_path,
                        "object": "model",
                        "created": now,
                        "owned_by": "sglang",
                    }
                ],
            }
        )

    async def handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "policy": self.policy,
            }
        )

    async def handle_state(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "policy": self.policy,
                "worker_urls": self.worker_urls,
                "down_workers": sorted(self.down_workers),
                "down_since": self.down_since,
                "down_reason": self.down_reason,
                "inflight": self.inflight,
                "agent_rr_cursor": self.agent_rr_cursor,
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "failover_requests": self.stats.failover_requests,
                    "down_reject_requests": self.stats.down_reject_requests,
                    "per_worker_requests": self.stats.per_worker_requests,
                    "per_agent_requests": self.stats.per_agent_requests,
                    "per_agent_worker_requests": self.stats.per_agent_worker_requests,
                    "prefill_prompt_tokens_total": self.stats.prefill_prompt_tokens_total,
                    "prefill_cached_tokens_total": self.stats.prefill_cached_tokens_total,
                },
                "prefill_stats_mode": self.prefill_stats_mode,
                "metrics_timeout_s": self.metrics_timeout_s,
            }
        )

    async def handle_failure(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        action = str(body.get("action", "down")).lower()
        worker_url = body.get("worker_url")
        worker_index = body.get("worker_index")
        duration_s = body.get("duration_s")
        reason = str(body.get("reason", "manual"))

        worker = self._normalize_worker_target(worker_url, worker_index)
        if worker is None:
            return web.json_response(
                {"error": "worker_url or valid worker_index is required"},
                status=400,
            )

        duration_value: Optional[float]
        if duration_s is None:
            duration_value = None
        else:
            try:
                duration_value = max(0.0, float(duration_s))
            except Exception:
                return web.json_response(
                    {"error": "duration_s must be a number"},
                    status=400,
                )

        if action == "down":
            self._mark_worker_down(worker, reason=f"manual:{reason}", duration_s=duration_value)
        elif action == "up":
            self._mark_worker_up(worker)
        elif action == "toggle":
            if worker in self.down_workers:
                self._mark_worker_up(worker)
            else:
                self._mark_worker_down(worker, reason=f"manual:{reason}", duration_s=duration_value)
        else:
            return web.json_response(
                {"error": "action must be one of down/up/toggle"},
                status=400,
            )

        LOG.warning(
            "Failure injection action=%s worker=%s duration_s=%s down_workers=%s",
            action,
            worker,
            duration_value,
            sorted(self.down_workers),
        )

        return web.json_response(
            {
                "ok": True,
                "action": action,
                "worker_url": worker,
                "duration_s": duration_value,
                "down_workers": sorted(self.down_workers),
            }
        )

    async def handle_reset(self, _request: web.Request) -> web.Response:
        for worker in list(self._recover_tasks.keys()):
            self._cancel_recover_task(worker)

        self.down_workers.clear()
        self.down_since.clear()
        self.down_reason.clear()
        self.stats = RouterStats()
        self.inflight = {u: 0 for u in self.worker_urls}
        self.agent_rr_cursor = {}
        self._global_rr_cursor = 0
        return web.json_response({"ok": True})


async def create_app(args: argparse.Namespace) -> web.Application:
    worker_urls = [u.strip() for u in args.worker_urls if u.strip()]
    router = AgentFaultRouter(
        worker_urls=worker_urls,
        policy=args.policy,
        request_timeout_s=args.request_timeout,
        max_attempts=args.max_attempts,
        prefill_stats_mode=args.prefill_stats_mode,
        metrics_timeout_s=args.metrics_timeout,
        model_path=args.model_path,
        auto_down_cooldown_s=args.auto_down_cooldown,
    )

    app = web.Application(client_max_size=32 * 1024 * 1024)

    async def on_startup(_app: web.Application) -> None:
        await router.startup()
        LOG.info(
            "Router started policy=%s workers=%s prefill_stats_mode=%s metrics_timeout_s=%.3f auto_down_cooldown_s=%.3f",
            args.policy,
            worker_urls,
            args.prefill_stats_mode,
            float(args.metrics_timeout),
            float(args.auto_down_cooldown),
        )

    async def on_cleanup(_app: web.Application) -> None:
        await router.shutdown()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.add_routes(
        [
            web.get("/health", router.handle_health),
            web.get("/v1/models", router.handle_models),
            web.get("/admin/state", router.handle_state),
            web.post("/admin/failure", router.handle_failure),
            web.post("/admin/reset", router.handle_reset),
            web.post("/v1/chat/completions", router.handle_completion),
        ]
    )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent-aware fault-tolerance experiment router"
    )
    parser.add_argument(
        "--worker-urls",
        nargs="+",
        required=True,
        help="Worker URLs, e.g. http://host:8000",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument(
        "--policy",
        choices=["cache_aware", "agent_aware", "round_robin"],
        default="cache_aware",
    )
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument(
        "--prefill-stats-mode",
        choices=["blocking", "non_blocking", "disabled"],
        default="non_blocking",
        help=(
            "Prefill token accounting mode. 'blocking' serializes by worker "
            "for higher precision; 'non_blocking' avoids serialization for "
            "throughput; 'disabled' skips metrics probing."
        ),
    )
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for worker /metrics probing.",
    )
    parser.add_argument(
        "--auto-down-cooldown",
        type=float,
        default=0.0,
        help=(
            "Auto mark-down duration (seconds) when a worker request fails. "
            "0 disables auto mark-down."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model id exposed at /v1/models",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(create_app(args))
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
