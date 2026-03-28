"""
Lightweight Python router with cache-aware scheduling and per-chunk idle timeout.

Provides the same core features as the Rust sglang_router but adds chunk-level
idle timeout detection for streaming responses: if no new chunk arrives within
``chunk_timeout_secs`` the connection is severed and the request retried on
another worker.

Features replicated from the Rust router:
  - Cache-aware routing via an in-process radix-tree prefix matcher
  - Shortest-queue fallback when load is imbalanced
  - Background health checks with failure/success thresholds
  - Automatic retries with exponential backoff + jitter
  - Per-worker load (in-flight request) tracking
  - OpenAI-compatible /v1/chat/completions, /v1/completions, /v1/models
  - SSE streaming pass-through with per-chunk idle timeout

New vs Rust router:
  - ``--chunk-timeout-secs`` (default 60): idle timeout between consecutive
    streaming chunks.  If a worker stops sending data for this long the stream
    is aborted and the request retried on another worker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("python_router")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Radix tree for prefix matching (mirrors Rust tree.rs — multi-tenant)
# ---------------------------------------------------------------------------

class _RadixNode:
    """Each node stores per-tenant last-access timestamps (multi-tenant)."""
    __slots__ = ("children", "tenant_last_access", "parent")

    def __init__(self) -> None:
        self.children: Dict[str, Tuple[str, _RadixNode]] = {}
        self.tenant_last_access: Dict[str, float] = {}
        self.parent: Optional[_RadixNode] = None


class RadixTree:
    """Character-level multi-tenant radix tree mirroring the Rust ``tree.rs``.

    Every node along the insertion path records the tenant, so
    ``prefix_match`` can return a tenant even from intermediate nodes.
    """

    def __init__(self, max_size: int = 2 ** 20) -> None:
        self._root = _RadixNode()
        self._tenant_char_counts: Dict[str, int] = defaultdict(int)

    @property
    def size(self) -> int:
        return sum(self._tenant_char_counts.values())

    # ---- insert (mirrors Rust Tree::insert) ----

    def insert(self, text: str, tenant: str) -> None:
        now = time.monotonic()
        node = self._root
        node.tenant_last_access[tenant] = now
        if tenant not in self._tenant_char_counts:
            self._tenant_char_counts[tenant] = 0

        i = 0
        while i < len(text):
            ch = text[i]
            if ch not in node.children:
                remaining = text[i:]
                leaf = _RadixNode()
                leaf.parent = node
                leaf.tenant_last_access[tenant] = now
                node.children[ch] = (remaining, leaf)
                self._tenant_char_counts[tenant] += len(remaining)
                return

            edge_label, child = node.children[ch]
            j = 0
            while j < len(edge_label) and i < len(text) and edge_label[j] == text[i]:
                j += 1
                i += 1

            if j < len(edge_label):
                # Split: create a new mid-node
                mid = _RadixNode()
                mid.parent = node
                mid.tenant_last_access = dict(child.tenant_last_access)
                mid.children[edge_label[j]] = (edge_label[j:], child)
                child.parent = mid
                node.children[ch] = (edge_label[:j], mid)

                if tenant not in mid.tenant_last_access:
                    self._tenant_char_counts[tenant] += j
                mid.tenant_last_access[tenant] = now

                if i < len(text):
                    leaf = _RadixNode()
                    leaf.parent = mid
                    leaf.tenant_last_access[tenant] = now
                    remaining = text[i:]
                    mid.children[text[i]] = (remaining, leaf)
                    self._tenant_char_counts[tenant] += len(remaining)
                return

            # Full edge match — move to child
            node = child
            if tenant not in node.tenant_last_access:
                self._tenant_char_counts[tenant] += len(edge_label)
            node.tenant_last_access[tenant] = now

    # ---- prefix_match (mirrors Rust Tree::prefix_match) ----

    def prefix_match(self, text: str) -> Tuple[int, Optional[str]]:
        """Return ``(matched_length, tenant)`` for the longest prefix match.

        Picks the first tenant from the deepest matched node (same as
        Rust's ``iter().next()``), then walks up to root updating timestamps.
        """
        if not text:
            return 0, None

        node = self._root
        matched = 0
        i = 0

        while i < len(text):
            ch = text[i]
            if ch not in node.children:
                break
            edge_label, child = node.children[ch]
            j = 0
            while j < len(edge_label) and i < len(text) and edge_label[j] == text[i]:
                j += 1
                i += 1
                matched += 1
            if j < len(edge_label):
                node = child
                break
            node = child

        tenant: Optional[str] = None
        if node.tenant_last_access:
            tenant = next(iter(node.tenant_last_access))

        if tenant is not None:
            now = time.monotonic()
            walk: Optional[_RadixNode] = node
            while walk is not None:
                walk.tenant_last_access[tenant] = now
                walk = walk.parent

        return matched, tenant

    # ---- reset ----

    def reset(self) -> None:
        self._root = _RadixNode()
        self._tenant_char_counts.clear()

    # ---- eviction (mirrors Rust Tree::evict_tenant_by_size) ----

    def evict_tenant_by_size(self, max_size: int) -> None:
        """Per-tenant LRU leaf eviction matching the Rust implementation."""
        import heapq

        leaves: List[Tuple[float, int, str, _RadixNode]] = []
        counter = 0

        def _collect(nd: _RadixNode) -> None:
            nonlocal counter
            for t in self._leaf_tenants(nd):
                ts = nd.tenant_last_access.get(t, 0.0)
                heapq.heappush(leaves, (ts, counter, t, nd))
                counter += 1
            for _, (_, ch) in nd.children.items():
                _collect(ch)

        _collect(self._root)

        while leaves:
            ts, _, tenant, nd = heapq.heappop(leaves)
            cur_size = self._tenant_char_counts.get(tenant, 0)
            if cur_size <= max_size:
                continue

            if tenant in nd.tenant_last_access:
                edge_len = 0
                if nd.parent is not None:
                    for _, (el, c) in nd.parent.children.items():
                        if c is nd:
                            edge_len = len(el)
                            break
                self._tenant_char_counts[tenant] = max(
                    self._tenant_char_counts.get(tenant, 0) - edge_len, 0
                )

            nd.tenant_last_access.pop(tenant, None)

            if not nd.children and not nd.tenant_last_access and nd.parent is not None:
                parent = nd.parent
                for ch_key, (_, ch_node) in list(parent.children.items()):
                    if ch_node is nd:
                        del parent.children[ch_key]
                        break

            if nd.parent is not None:
                parent = nd.parent
                parent_leaves = self._leaf_tenants(parent)
                if tenant in parent_leaves:
                    pts = parent.tenant_last_access.get(tenant, 0.0)
                    heapq.heappush(leaves, (pts, counter, tenant, parent))
                    counter += 1

    def evict_if_needed(self, max_size: Optional[int] = None) -> None:
        """Called by the background eviction loop."""
        if max_size is None:
            max_size = 2 ** 20
        for tenant, sz in list(self._tenant_char_counts.items()):
            if sz > max_size:
                self.evict_tenant_by_size(max_size)
                return

    # ---- helpers ----

    @staticmethod
    def _leaf_tenants(node: _RadixNode) -> List[str]:
        """Return tenants for which *node* is a leaf (tenant present in
        node but not in any child)."""
        child_tenants: Set[str] = set()
        for _, (_, ch) in node.children.items():
            child_tenants.update(ch.tenant_last_access.keys())
        return [t for t in node.tenant_last_access if t not in child_tenants]

    def remove_tenant(self, tenant: str) -> None:
        """Remove *tenant* from the entire tree (mirrors Rust ``remove_tenant``)."""
        stack = [self._root]
        leaves: List[_RadixNode] = []
        while stack:
            nd = stack.pop()
            lt = self._leaf_tenants(nd)
            if tenant in lt:
                leaves.append(nd)
            for _, (_, ch) in nd.children.items():
                stack.append(ch)

        from collections import deque
        queue: deque[_RadixNode] = deque(leaves)
        while queue:
            nd = queue.popleft()
            nd.tenant_last_access.pop(tenant, None)
            if not nd.children and not nd.tenant_last_access and nd.parent is not None:
                parent = nd.parent
                for ch_key, (_, ch_node) in list(parent.children.items()):
                    if ch_node is nd:
                        del parent.children[ch_key]
                        break
            if nd.parent is not None:
                parent = nd.parent
                if tenant in self._leaf_tenants(parent):
                    queue.append(parent)

        self._tenant_char_counts.pop(tenant, None)


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    url: str
    load: int = 0
    healthy: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    inflight_abort_events: Set[asyncio.Event] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Router configuration
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    worker_urls: List[str] = field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 30000
    model_path: str = ""
    # Cache-aware policy
    cache_threshold: float = 0.5
    balance_abs_threshold: int = 32
    balance_rel_threshold: float = 1.1
    eviction_interval_secs: int = 30
    max_tree_size: int = 2 ** 20
    # Timeouts
    request_timeout_secs: int = 300
    chunk_timeout_secs: float = 60.0
    prefill_timeout_secs: float = 0
    connect_timeout_secs: int = 10
    # Health check
    health_check_interval_secs: int = 10
    health_check_timeout_secs: int = 5
    health_failure_threshold: int = 3
    health_success_threshold: int = 2
    # Retry
    retry_max_retries: int = 3
    retry_initial_backoff_ms: int = 50
    retry_max_backoff_ms: int = 5000
    retry_backoff_multiplier: float = 1.5
    retry_jitter_factor: float = 0.2
    # Fault tolerance
    enable_fault_tolerance: bool = False
    # Prometheus
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 29000

    RETRYABLE_STATUSES: Set[int] = field(
        default_factory=lambda: {408, 429, 500, 502, 503, 504}
    )


# ---------------------------------------------------------------------------
# Python Router
# ---------------------------------------------------------------------------

class PythonRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.workers = [WorkerState(url=u) for u in config.worker_urls]
        self.tree = RadixTree()
        for w in self.workers:
            self.tree.insert("", w.url)
        self._session: Optional[aiohttp.ClientSession] = None
        self._health_task: Optional[asyncio.Task] = None
        self._eviction_task: Optional[asyncio.Task] = None
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "requests_total": 0,
            "retries_total": 0,
            "chunk_timeouts": 0,
            "stream_attempts_total": 0,
            "stream_attempts_with_cache_details": 0,
            "stream_peer_cache_hits": 0,
        }
        self._retry_records: List[Dict[str, Any]] = []
        self._retry_impacted_requests: List[Dict[str, Any]] = []
        self._stream_attempt_records: List[Dict[str, Any]] = []
        self._normal_latencies: List[float] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=self.config.connect_timeout_secs,
                sock_read=self.config.chunk_timeout_secs,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _register_abort_event(self, worker: WorkerState) -> asyncio.Event:
        event = asyncio.Event()
        worker.inflight_abort_events.add(event)
        return event

    def _unregister_abort_event(
        self, worker: WorkerState, event: Optional[asyncio.Event]
    ) -> None:
        if event is not None:
            worker.inflight_abort_events.discard(event)

    def _invalidate_inflight_requests(self, worker: WorkerState) -> None:
        for event in list(worker.inflight_abort_events):
            event.set()

    # ---- Cache-aware worker selection ----

    def _healthy_indices(self) -> List[int]:
        return [i for i, w in enumerate(self.workers) if w.healthy]

    def select_worker(
        self, request_text: str, failed_worker_idx: Optional[int] = None
    ) -> Optional[int]:
        healthy = self._healthy_indices()
        if not healthy:
            return None

        if (
            self.config.enable_fault_tolerance
            and failed_worker_idx is not None
        ):
            ring_target = (failed_worker_idx + 1) % len(self.workers)
            if ring_target in healthy:
                self.tree.insert(request_text, self.workers[ring_target].url)
                return ring_target

        loads = [(i, self.workers[i].load) for i in healthy]
        min_load = min(l for _, l in loads)
        max_load = max(l for _, l in loads)

        is_imbalanced = (
            (max_load - min_load) > self.config.balance_abs_threshold
            and max_load > min_load * self.config.balance_rel_threshold
        )

        if is_imbalanced:
            idx = min(healthy, key=lambda i: self.workers[i].load)
            self.tree.insert(request_text, self.workers[idx].url)
            return idx

        matched_len, matched_url = self.tree.prefix_match(request_text)
        text_len = len(request_text) if request_text else 1
        match_rate = matched_len / text_len if text_len > 0 else 0.0

        if match_rate > self.config.cache_threshold and matched_url:
            self._stats["cache_hits"] += 1
            url_to_idx = {w.url: i for i, w in enumerate(self.workers)}
            idx = url_to_idx.get(matched_url)
            if idx is not None and idx in healthy:
                self.tree.insert(request_text, self.workers[idx].url)
                return idx

        self._stats["cache_misses"] += 1
        idx = min(healthy, key=lambda i: self.workers[i].load)
        self.tree.insert(request_text, self.workers[idx].url)
        return idx

    # ---- Request text extraction ----

    @staticmethod
    def extract_request_text(body: Dict[str, Any]) -> str:
        messages = body.get("messages")
        if messages and isinstance(messages, list):
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
            return " ".join(parts)
        prompt = body.get("prompt")
        if prompt:
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list):
                return " ".join(str(p) for p in prompt)
        return ""

    # ---- Retry with backoff ----

    def _backoff(self, attempt: int) -> float:
        base = self.config.retry_initial_backoff_ms / 1000.0
        delay = base * (self.config.retry_backoff_multiplier ** attempt)
        cap = self.config.retry_max_backoff_ms / 1000.0
        delay = min(delay, cap)
        jitter = delay * self.config.retry_jitter_factor * random.random()
        return delay + jitter

    def _extract_request_context(
        self,
        user_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _parse_context(source: Any) -> Dict[str, Any]:
            if source is None:
                return {}
            if isinstance(source, str):
                if not source:
                    return {}
                try:
                    source = json.loads(source)
                except Exception:
                    return {}
            if not isinstance(source, dict):
                return {}
            context = {}
            for key in (
                "job_id",
                "job",
                "year",
                "task_label",
                "agent_role",
                "worker_id",
            ):
                value = source.get(key)
                if value is not None:
                    context[key] = value
            return context

        context: Dict[str, Any] = {}
        request_metadata = metadata if isinstance(metadata, dict) else {}
        for source in (
            request_metadata.get("request_context"),
            request_metadata,
            user_tag,
        ):
            for key, value in _parse_context(source).items():
                context.setdefault(key, value)
        if context:
            return context
        if user_tag:
            return {"user": user_tag}
        return {}

    @staticmethod
    def _context_with_user(
        request_context: Dict[str, Any], user_tag: str
    ) -> Dict[str, Any]:
        record = dict(request_context)
        if user_tag:
            record["user"] = user_tag
        return record

    def _mark_retry_needed(
        self,
        request_context: Dict[str, Any],
        user_tag: str,
        mode: str,
        worker_url: str,
    ) -> None:
        record = {
            **self._context_with_user(request_context, user_tag),
            "mode": mode,
            "from_worker": worker_url,
        }
        self._retry_impacted_requests.append(record)

    def _build_retry_record(
        self,
        request_context: Dict[str, Any],
        user_tag: str,
        mode: str,
        from_worker: str,
        to_worker: str,
        total_ms: float,
        retry_ms: float,
        tokens_before_fault: int,
        tokens_total: int,
    ) -> Dict[str, Any]:
        return {
            **self._context_with_user(request_context, user_tag),
            "tokens_before_fault": tokens_before_fault,
            "tokens_total": tokens_total,
            "total_ms": round(total_ms, 1),
            "retry_ms": round(retry_ms, 1),
            "from_worker": from_worker,
            "to_worker": to_worker,
            "mode": mode,
        }

    @staticmethod
    def _summarize_cached_tokens_details(
        details: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        details = details or {}
        device = int(details.get("device", 0) or 0)
        host = int(details.get("host", 0) or 0)
        storage = int(details.get("storage", 0) or 0)
        storage_backend = details.get("storage_backend")
        peer_cache_hit = storage_backend == "PeerCacheStorage" and storage > 0
        return {
            "cached_tokens_details": {
                "device": device,
                "host": host,
                "storage": storage,
                "storage_backend": storage_backend,
            },
            "cached_tokens_total": device + host + storage,
            "peer_cache_hit": peer_cache_hit,
            "peer_cache_tokens": storage if peer_cache_hit else 0,
        }

    def _build_stream_attempt_record(
        self,
        *,
        request_context: Dict[str, Any],
        user_tag: str,
        attempt_idx: int,
        worker_url: str,
        request_path: str,
        retry_from_worker: Optional[str],
        resume_chars: int,
        generated_tokens_before_attempt: int,
        resume_mode: str,
        resume_prompt_token_count: int,
        resume_token_count: int,
    ) -> Dict[str, Any]:
        return {
            **self._context_with_user(request_context, user_tag),
            "mode": "stream",
            "attempt_idx": attempt_idx,
            "request_path": request_path,
            "worker": worker_url,
            "retry_from_worker": retry_from_worker,
            "is_retry": attempt_idx > 1,
            "resume_chars": resume_chars,
            "resume_mode": resume_mode,
            "resume_prompt_token_count": resume_prompt_token_count,
            "resume_token_count": resume_token_count,
            "captured_input_token_count": 0,
            "captured_output_token_count": 0,
            "captured_resume_token_ready": False,
            "captured_output_token_style": "unknown",
            "next_resume_mode": "none",
            "generated_tokens_before_attempt": generated_tokens_before_attempt,
            "generated_tokens_this_attempt": 0,
            "generated_tokens_total": generated_tokens_before_attempt,
            "cached_tokens_details": None,
            "cached_tokens_total": 0,
            "peer_cache_hit": False,
            "peer_cache_tokens": 0,
            "status": "in_progress",
            "error": None,
        }

    def _attach_stream_attempt_cache_details(
        self, attempt_record: Dict[str, Any], details: Dict[str, Any]
    ) -> None:
        if attempt_record.get("cached_tokens_details") is not None:
            return
        summary = self._summarize_cached_tokens_details(details)
        attempt_record.update(summary)
        self._stats["stream_attempts_with_cache_details"] += 1
        if summary["peer_cache_hit"]:
            self._stats["stream_peer_cache_hits"] += 1

    def _finalize_stream_attempt_record(
        self,
        attempt_record: Dict[str, Any],
        *,
        status: str,
        error: Optional[str],
        generated_tokens_this_attempt: int,
        generated_tokens_total: int,
        next_resume_mode: str = "none",
    ) -> None:
        attempt_record["status"] = status
        attempt_record["error"] = error
        attempt_record["generated_tokens_this_attempt"] = generated_tokens_this_attempt
        attempt_record["generated_tokens_total"] = generated_tokens_total
        attempt_record["next_resume_mode"] = next_resume_mode
        if attempt_record.get("is_retry"):
            self._stream_attempt_records.append(dict(attempt_record))

    # ---- Proxying (non-streaming) ----

    async def _proxy_non_stream(
        self,
        body: Dict[str, Any],
        request_text: str,
        path: str,
        excluded: Optional[Set[int]] = None,
    ) -> JSONResponse:
        self._stats["requests_total"] += 1
        excluded = excluded or set()
        last_failed_idx: Optional[int] = None
        req_start = time.monotonic()
        fault_detected_at: Optional[float] = None
        failed_worker_url = ""
        user_tag = body.get("user", "")
        request_metadata = body.get("metadata")
        if not isinstance(request_metadata, dict):
            request_metadata = None
        request_context = self._extract_request_context(user_tag, request_metadata)
        retry_marked = False

        def _mark_fault(worker_url: str) -> None:
            nonlocal fault_detected_at, failed_worker_url, retry_marked
            if fault_detected_at is None:
                fault_detected_at = time.monotonic()
                failed_worker_url = worker_url
            if not retry_marked:
                self._mark_retry_needed(
                    request_context, user_tag, "non_stream", worker_url
                )
                retry_marked = True

        for attempt in range(1 + self.config.retry_max_retries):
            healthy_available = [
                i for i in self._healthy_indices() if i not in excluded
            ]
            if not healthy_available:
                healthy_available = self._healthy_indices()
                excluded.clear()
            if not healthy_available:
                return JSONResponse(
                    {"error": "No healthy workers available"}, status_code=503
                )

            idx = self.select_worker(request_text, failed_worker_idx=last_failed_idx)
            if idx is None or idx in excluded:
                idx = min(healthy_available, key=lambda i: self.workers[i].load)
            worker = self.workers[idx]
            worker.load += 1
            abort_event = self._register_abort_event(worker)
            resp: Optional[aiohttp.ClientResponse] = None
            try:
                session = await self._get_session()
                post_task = asyncio.create_task(
                    session.post(
                        f"{worker.url}{path}",
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_secs),
                    )
                )
                abort_wait_task = asyncio.create_task(abort_event.wait())
                done, _ = await asyncio.wait(
                    {post_task, abort_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_wait_task in done:
                    post_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await post_task
                    raise RuntimeError(f"Worker {worker.url} removed mid-request")

                abort_wait_task.cancel()
                with suppress(asyncio.CancelledError):
                    await abort_wait_task
                resp = await post_task

                if abort_event.is_set() or not worker.healthy:
                    resp.close()
                    raise RuntimeError(f"Worker {worker.url} removed mid-request")

                read_task = asyncio.create_task(resp.read())
                abort_wait_task = asyncio.create_task(abort_event.wait())
                done, _ = await asyncio.wait(
                    {read_task, abort_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_wait_task in done:
                    resp.close()
                    read_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await read_task
                    raise RuntimeError(f"Worker {worker.url} removed mid-request")

                abort_wait_task.cancel()
                with suppress(asyncio.CancelledError):
                    await abort_wait_task
                resp_body = await read_task

                if abort_event.is_set() or not worker.healthy:
                    resp.close()
                    raise RuntimeError(f"Worker {worker.url} removed mid-request")

                if resp.status in self.config.RETRYABLE_STATUSES:
                    _mark_fault(worker.url)
                    last_failed_idx = idx
                    excluded.add(idx)
                    self._stats["retries_total"] += 1
                    if attempt < self.config.retry_max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                elapsed_ms = (time.monotonic() - req_start) * 1000
                if fault_detected_at is not None:
                    retry_ms = (time.monotonic() - fault_detected_at) * 1000
                    record = self._build_retry_record(
                        request_context=request_context,
                        user_tag=user_tag,
                        mode="non_stream",
                        from_worker=failed_worker_url,
                        to_worker=worker.url,
                        total_ms=elapsed_ms,
                        retry_ms=retry_ms,
                        tokens_before_fault=0,
                        tokens_total=0,
                    )
                    self._retry_records.append(record)
                    logger.info(
                        "REQUEST_METRIC retried=true mode=non_stream "
                        "total_ms=%.0f retry_ms=%.0f from=%s to=%s user=%s",
                        elapsed_ms, retry_ms,
                        failed_worker_url, worker.url, user_tag,
                    )
                else:
                    self._normal_latencies.append(elapsed_ms)
                return JSONResponse(
                    content=None,
                    status_code=resp.status,
                    headers=dict(resp.headers),
                    media_type=resp.content_type,
                ) if not resp_body else Response(
                    content=resp_body,
                    status_code=resp.status,
                    media_type=resp.content_type,
                )
            except Exception as exc:
                logger.warning("Worker %s failed: %s", worker.url, exc)
                _mark_fault(worker.url)
                last_failed_idx = idx
                excluded.add(idx)
                self._stats["retries_total"] += 1
                if attempt < self.config.retry_max_retries:
                    await asyncio.sleep(self._backoff(attempt))
            finally:
                self._unregister_abort_event(worker, abort_event)
                worker.load -= 1
                if resp is not None:
                    try:
                        await resp.release()
                    except Exception:
                        pass
        return JSONResponse({"error": "All retries exhausted"}, status_code=502)

    # ---- Proxying (streaming with chunk idle timeout) ----

    async def _proxy_stream(
        self,
        body: Dict[str, Any],
        request_text: str,
        path: str,
    ) -> StreamingResponse:
        self._stats["requests_total"] += 1
        excluded: Set[int] = set()
        last_failed_idx: Optional[int] = None
        user_tag = body.get("user", "")
        request_metadata = body.get("metadata")
        if not isinstance(request_metadata, dict):
            request_metadata = None
        request_context = self._extract_request_context(user_tag, request_metadata)
        request_trace_id = hashlib.sha1(
            f"{time.time_ns()}|{random.random()}|{path}|{user_tag}".encode()
        ).hexdigest()[:12]

        async def _try_stream():
            nonlocal last_failed_idx

            accumulated_text = ""
            generated_tokens_total = 0
            request_attempt_traces: List[Dict[str, Any]] = []
            orig_max_tokens = body.get("max_tokens") or body.get(
                "max_completion_tokens"
            )
            current_body = body

            req_start = time.monotonic()
            fault_detected_at: Optional[float] = None
            tokens_at_fault = 0
            failed_worker_url = ""
            retry_marked = False

            def _mark_fault(w_url: str) -> None:
                nonlocal fault_detected_at, tokens_at_fault, failed_worker_url, retry_marked
                if fault_detected_at is None:
                    fault_detected_at = time.monotonic()
                    tokens_at_fault = generated_tokens_total
                    failed_worker_url = w_url
                if not retry_marked:
                    self._mark_retry_needed(request_context, user_tag, "stream", w_url)
                    retry_marked = True

            def _get_prompt_token_ids_from_body(
                request_body: Dict[str, Any],
            ) -> Optional[List[int]]:
                input_ids = request_body.get("input_ids")
                if isinstance(input_ids, list) and all(
                    isinstance(tok, int) for tok in input_ids
                ):
                    return list(input_ids)
                prompt = request_body.get("prompt")
                if isinstance(prompt, list) and all(
                    isinstance(tok, int) for tok in prompt
                ):
                    return list(prompt)
                return None

            def _get_resume_mode_from_body(
                request_body: Dict[str, Any], is_retry: bool
            ) -> str:
                if not is_retry:
                    return "none"
                if _get_prompt_token_ids_from_body(request_body) is not None:
                    return "token"
                messages = request_body.get("messages")
                if (
                    isinstance(messages, list)
                    and messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("role") == "assistant"
                    and request_body.get("continue_final_message")
                ):
                    return "text"
                return "unknown"

            for attempt in range(1 + self.config.retry_max_retries):
                healthy_available = [
                    i for i in self._healthy_indices() if i not in excluded
                ]
                if not healthy_available:
                    healthy_available = self._healthy_indices()
                    excluded.clear()
                if not healthy_available:
                    yield _sse_error("No healthy workers available")
                    return

                if (
                    orig_max_tokens is not None
                    and generated_tokens_total >= orig_max_tokens
                ):
                    yield b"data: [DONE]\n\n"
                    return

                idx = self.select_worker(request_text, failed_worker_idx=last_failed_idx)
                if idx is None or idx in excluded:
                    idx = min(healthy_available, key=lambda i: self.workers[i].load)
                worker = self.workers[idx]
                worker.load += 1
                self._stats["stream_attempts_total"] += 1
                generated_tokens_before_attempt = generated_tokens_total
                attempt_generated_tokens_estimate = 0
                attempt_generated_tokens_usage = 0
                attempt_resume_mode = _get_resume_mode_from_body(
                    current_body, attempt > 0
                )
                attempt_prompt_token_ids = _get_prompt_token_ids_from_body(current_body)
                attempt_output_token_ids: List[int] = []
                attempt_record = self._build_stream_attempt_record(
                    request_context=request_context,
                    user_tag=user_tag,
                    attempt_idx=attempt + 1,
                    worker_url=worker.url,
                    request_path=path,
                    retry_from_worker=failed_worker_url or None,
                    resume_chars=len(accumulated_text),
                    generated_tokens_before_attempt=generated_tokens_before_attempt,
                    resume_mode=attempt_resume_mode,
                    resume_prompt_token_count=(
                        len(attempt_prompt_token_ids) if attempt_prompt_token_ids else 0
                    ),
                    resume_token_count=(
                        len(attempt_prompt_token_ids) if attempt_prompt_token_ids else 0
                    ),
                )
                attempt_trace = {
                    **request_context,
                    "request_trace_id": request_trace_id,
                    "attempt_idx": attempt + 1,
                    "worker": worker.url,
                    "retry_from_worker": failed_worker_url or None,
                    "is_retry": attempt > 0,
                    "resume_chars": len(accumulated_text),
                    "resume_mode": attempt_resume_mode,
                    "resume_prompt_token_count": (
                        len(attempt_prompt_token_ids) if attempt_prompt_token_ids else 0
                    ),
                    "resume_token_count": (
                        len(attempt_prompt_token_ids) if attempt_prompt_token_ids else 0
                    ),
                    "captured_input_token_count": 0,
                    "captured_output_token_count": 0,
                    "captured_resume_token_ready": False,
                    "captured_output_token_style": "unknown",
                    "next_resume_mode": "none",
                    "generated_tokens_before_attempt": generated_tokens_before_attempt,
                    "cached_tokens_details": None,
                    "cached_tokens_total": 0,
                    "peer_cache_hit": False,
                    "peer_cache_tokens": 0,
                    "status": "in_progress",
                    "error": None,
                }

                def _current_attempt_tokens() -> int:
                    return max(
                        attempt_generated_tokens_estimate,
                        attempt_generated_tokens_usage,
                    )

                def _current_total_tokens() -> int:
                    return generated_tokens_before_attempt + _current_attempt_tokens()

                def _attach_attempt_trace_cache_details(details: Dict[str, Any]) -> None:
                    if attempt_trace.get("cached_tokens_details") is not None:
                        return
                    attempt_trace.update(self._summarize_cached_tokens_details(details))

                def _finalize_attempt_trace(status: str, error: Optional[str]) -> None:
                    attempt_trace["status"] = status
                    attempt_trace["error"] = error
                    attempt_trace["generated_tokens_this_attempt"] = (
                        _current_attempt_tokens()
                    )
                    attempt_trace["generated_tokens_total"] = generated_tokens_total
                    request_attempt_traces.append(dict(attempt_trace))

                def _update_resume_capture_fields() -> None:
                    captured_input = (
                        len(attempt_prompt_token_ids) if attempt_prompt_token_ids else 0
                    )
                    captured_output = len(attempt_output_token_ids)
                    ready = captured_input > 0 and captured_output > 0
                    attempt_record["captured_input_token_count"] = captured_input
                    attempt_record["captured_output_token_count"] = captured_output
                    attempt_record["captured_resume_token_ready"] = ready
                    attempt_trace["captured_input_token_count"] = captured_input
                    attempt_trace["captured_output_token_count"] = captured_output
                    attempt_trace["captured_resume_token_ready"] = ready

                def _merge_output_token_ids(new_output_token_ids: List[int]) -> None:
                    nonlocal attempt_output_token_ids
                    if not new_output_token_ids:
                        return
                    if not attempt_output_token_ids:
                        attempt_output_token_ids = list(new_output_token_ids)
                        attempt_record["captured_output_token_style"] = "initial"
                        attempt_trace["captured_output_token_style"] = "initial"
                        return
                    if (
                        len(new_output_token_ids) >= len(attempt_output_token_ids)
                        and new_output_token_ids[: len(attempt_output_token_ids)]
                        == attempt_output_token_ids
                    ):
                        # Most streaming backends expose cumulative output ids.
                        attempt_output_token_ids = list(new_output_token_ids)
                        attempt_record["captured_output_token_style"] = "cumulative"
                        attempt_trace["captured_output_token_style"] = "cumulative"
                        return
                    if (
                        len(attempt_output_token_ids) >= len(new_output_token_ids)
                        and attempt_output_token_ids[: len(new_output_token_ids)]
                        == new_output_token_ids
                    ):
                        return
                    # Fallback for true delta-style chunks.
                    attempt_output_token_ids.extend(new_output_token_ids)
                    attempt_record["captured_output_token_style"] = "delta_or_mixed"
                    attempt_trace["captured_output_token_style"] = "delta_or_mixed"

                def _choose_next_resume_mode() -> str:
                    if attempt_prompt_token_ids and attempt_output_token_ids:
                        return "token"
                    if accumulated_text:
                        return "text"
                    return "none"

                try:
                    prefill_to = self.config.prefill_timeout_secs or self.config.chunk_timeout_secs
                    session = await self._get_session()
                    resp = await session.post(
                        f"{worker.url}{path}",
                        json=current_body,
                        timeout=aiohttp.ClientTimeout(
                            total=None,
                            connect=self.config.connect_timeout_secs,
                            sock_read=prefill_to,
                        ),
                    )
                    if resp.status >= 400 and resp.status in self.config.RETRYABLE_STATUSES:
                        await resp.release()
                        worker.load -= 1
                        generated_tokens_total = _current_total_tokens()
                        _mark_fault(worker.url)
                        self._finalize_stream_attempt_record(
                            attempt_record,
                            status="retried"
                            if attempt < self.config.retry_max_retries
                            else "failed",
                            error=f"worker returned {resp.status}",
                            generated_tokens_this_attempt=_current_attempt_tokens(),
                            generated_tokens_total=generated_tokens_total,
                            next_resume_mode=_choose_next_resume_mode(),
                        )
                        attempt_trace["next_resume_mode"] = _choose_next_resume_mode()
                        _finalize_attempt_trace(
                            "retried" if attempt < self.config.retry_max_retries else "failed",
                            f"worker returned {resp.status}",
                        )
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        yield _sse_error(f"Worker returned {resp.status}")
                        return

                    try:
                        chunk_to = self.config.chunk_timeout_secs
                        cur_timeout = prefill_to
                        while not resp.content.at_eof():
                            chunk = await asyncio.wait_for(
                                resp.content.readany(), timeout=cur_timeout
                            )
                            if chunk:
                                if not worker.healthy:
                                    logger.warning(
                                        "Worker %s removed mid-stream, triggering retry "
                                        "(accumulated %d tokens)",
                                        worker.url,
                                        _current_total_tokens(),
                                    )
                                    raise RuntimeError(
                                        f"Worker {worker.url} removed mid-stream"
                                    )
                                cur_timeout = chunk_to
                                chunk_info = _extract_sse_chunk_info(chunk)
                                if chunk_info.cached_tokens_details is not None:
                                    self._attach_stream_attempt_cache_details(
                                        attempt_record, chunk_info.cached_tokens_details
                                    )
                                    _attach_attempt_trace_cache_details(
                                        chunk_info.cached_tokens_details
                                    )
                                if (
                                    chunk_info.input_token_ids is not None
                                    and attempt_prompt_token_ids is None
                                ):
                                    attempt_prompt_token_ids = list(
                                        chunk_info.input_token_ids
                                    )
                                if chunk_info.output_token_ids:
                                    _merge_output_token_ids(
                                        chunk_info.output_token_ids
                                    )
                                _update_resume_capture_fields()
                                if chunk_info.text:
                                    accumulated_text += chunk_info.text
                                    attempt_generated_tokens_estimate += (
                                        chunk_info.text_token_estimate
                                    )
                                if chunk_info.completion_tokens_total is not None:
                                    attempt_generated_tokens_usage = max(
                                        attempt_generated_tokens_usage,
                                        chunk_info.completion_tokens_total,
                                    )
                                generated_tokens_total = _current_total_tokens()
                                yield chunk
                    except asyncio.TimeoutError:
                        self._stats["chunk_timeouts"] += 1
                        generated_tokens_total = _current_total_tokens()
                        _mark_fault(worker.url)
                        logger.warning(
                            "Chunk idle timeout (%.0fs) on worker %s, attempt %d/%d "
                            "(accumulated %d tokens so far)",
                            self.config.chunk_timeout_secs,
                            worker.url,
                            attempt + 1,
                            1 + self.config.retry_max_retries,
                            generated_tokens_total,
                        )
                        self._finalize_stream_attempt_record(
                            attempt_record,
                            status="retried"
                            if attempt < self.config.retry_max_retries
                            else "failed",
                            error=(
                                f"chunk idle timeout after "
                                f"{self.config.chunk_timeout_secs}s"
                            ),
                            generated_tokens_this_attempt=_current_attempt_tokens(),
                            generated_tokens_total=generated_tokens_total,
                            next_resume_mode=_choose_next_resume_mode(),
                        )
                        attempt_trace["next_resume_mode"] = _choose_next_resume_mode()
                        _finalize_attempt_trace(
                            "retried" if attempt < self.config.retry_max_retries else "failed",
                            f"chunk idle timeout after {self.config.chunk_timeout_secs}s",
                        )
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            if accumulated_text or attempt_output_token_ids:
                                current_body = _patch_body_for_resume(
                                    body,
                                    accumulated_text,
                                    generated_tokens_total,
                                    prompt_token_ids=attempt_prompt_token_ids,
                                    output_token_ids=attempt_output_token_ids,
                                )
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        yield _sse_error(
                            f"Chunk idle timeout after {self.config.chunk_timeout_secs}s"
                        )
                        return
                    except Exception as exc:
                        generated_tokens_total = _current_total_tokens()
                        _mark_fault(worker.url)
                        logger.warning(
                            "Stream error on worker %s: %s", worker.url, exc
                        )
                        self._finalize_stream_attempt_record(
                            attempt_record,
                            status="retried"
                            if attempt < self.config.retry_max_retries
                            else "failed",
                            error=str(exc),
                            generated_tokens_this_attempt=_current_attempt_tokens(),
                            generated_tokens_total=generated_tokens_total,
                            next_resume_mode=_choose_next_resume_mode(),
                        )
                        attempt_trace["next_resume_mode"] = _choose_next_resume_mode()
                        _finalize_attempt_trace(
                            "retried" if attempt < self.config.retry_max_retries else "failed",
                            str(exc),
                        )
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            if accumulated_text or attempt_output_token_ids:
                                current_body = _patch_body_for_resume(
                                    body,
                                    accumulated_text,
                                    generated_tokens_total,
                                    prompt_token_ids=attempt_prompt_token_ids,
                                    output_token_ids=attempt_output_token_ids,
                                )
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        yield _sse_error(str(exc))
                        return
                    else:
                        generated_tokens_total = _current_total_tokens()
                        self._finalize_stream_attempt_record(
                            attempt_record,
                            status="completed",
                            error=None,
                            generated_tokens_this_attempt=_current_attempt_tokens(),
                            generated_tokens_total=generated_tokens_total,
                            next_resume_mode="none",
                        )
                        attempt_trace["next_resume_mode"] = "none"
                        _finalize_attempt_trace("completed", None)
                        elapsed_ms = (time.monotonic() - req_start) * 1000
                        if fault_detected_at is not None:
                            retry_ms = (time.monotonic() - fault_detected_at) * 1000
                            record = self._build_retry_record(
                                request_context=request_context,
                                user_tag=user_tag,
                                mode="stream",
                                from_worker=failed_worker_url,
                                to_worker=worker.url,
                                total_ms=elapsed_ms,
                                retry_ms=retry_ms,
                                tokens_before_fault=tokens_at_fault,
                                tokens_total=generated_tokens_total,
                            )
                            record["request_trace_id"] = request_trace_id
                            record["attempt_traces"] = [
                                t for t in request_attempt_traces
                                if t.get("is_retry")
                            ]
                            self._retry_records.append(record)
                            logger.info(
                                "REQUEST_METRIC retried=true mode=stream "
                                "tokens_before_fault=%d tokens_after_retry=%d "
                                "tokens_total=%d total_ms=%.0f retry_ms=%.0f "
                                "from=%s to=%s user=%s",
                                tokens_at_fault,
                                generated_tokens_total - tokens_at_fault,
                                generated_tokens_total,
                                elapsed_ms,
                                retry_ms,
                                failed_worker_url,
                                worker.url,
                                user_tag,
                            )
                        else:
                            self._normal_latencies.append(elapsed_ms)
                            if len(self._normal_latencies) % 50 == 0:
                                logger.info(
                                    "REQUEST_METRIC retried=false "
                                    "tokens=%d total_ms=%.0f worker=%s "
                                    "(sample, every 50th)",
                                    generated_tokens_total,
                                    elapsed_ms,
                                    worker.url,
                                )
                        return
                    finally:
                        worker.load -= 1
                        try:
                            await resp.release()
                        except Exception:
                            pass
                except asyncio.TimeoutError:
                    worker.load -= 1
                    generated_tokens_total = _current_total_tokens()
                    _mark_fault(worker.url)
                    self._stats["chunk_timeouts"] += 1
                    self._finalize_stream_attempt_record(
                        attempt_record,
                        status="retried"
                        if attempt < self.config.retry_max_retries
                        else "failed",
                        error="connection timeout",
                        generated_tokens_this_attempt=_current_attempt_tokens(),
                        generated_tokens_total=generated_tokens_total,
                        next_resume_mode=_choose_next_resume_mode(),
                    )
                    attempt_trace["next_resume_mode"] = _choose_next_resume_mode()
                    _finalize_attempt_trace(
                        "retried" if attempt < self.config.retry_max_retries else "failed",
                        "connection timeout",
                    )
                    last_failed_idx = idx
                    excluded.add(idx)
                    self._stats["retries_total"] += 1
                    if attempt < self.config.retry_max_retries:
                        if accumulated_text or attempt_output_token_ids:
                            current_body = _patch_body_for_resume(
                                body,
                                accumulated_text,
                                generated_tokens_total,
                                prompt_token_ids=attempt_prompt_token_ids,
                                output_token_ids=attempt_output_token_ids,
                            )
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    yield _sse_error("Connection timeout")
                    return
                except Exception as exc:
                    worker.load -= 1
                    generated_tokens_total = _current_total_tokens()
                    _mark_fault(worker.url)
                    logger.warning("Worker %s stream failed: %s", worker.url, exc)
                    self._finalize_stream_attempt_record(
                        attempt_record,
                        status="retried"
                        if attempt < self.config.retry_max_retries
                        else "failed",
                        error=str(exc),
                        generated_tokens_this_attempt=_current_attempt_tokens(),
                        generated_tokens_total=generated_tokens_total,
                        next_resume_mode=_choose_next_resume_mode(),
                    )
                    attempt_trace["next_resume_mode"] = _choose_next_resume_mode()
                    _finalize_attempt_trace(
                        "retried" if attempt < self.config.retry_max_retries else "failed",
                        str(exc),
                    )
                    last_failed_idx = idx
                    excluded.add(idx)
                    self._stats["retries_total"] += 1
                    if attempt < self.config.retry_max_retries:
                        if accumulated_text or attempt_output_token_ids:
                            current_body = _patch_body_for_resume(
                                body,
                                accumulated_text,
                                generated_tokens_total,
                                prompt_token_ids=attempt_prompt_token_ids,
                                output_token_ids=attempt_output_token_ids,
                            )
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    yield _sse_error(str(exc))
                    return

        return StreamingResponse(_try_stream(), media_type="text/event-stream")

    # ---- Health checks ----

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.health_check_interval_secs)
            session = await self._get_session()
            for worker in self.workers:
                try:
                    async with session.get(
                        f"{worker.url}/health",
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.health_check_timeout_secs
                        ),
                    ) as resp:
                        if resp.status == 200:
                            worker.consecutive_failures = 0
                            worker.consecutive_successes += 1
                            if (
                                not worker.healthy
                                and worker.consecutive_successes
                                >= self.config.health_success_threshold
                            ):
                                worker.healthy = True
                                logger.info("Worker %s is now healthy", worker.url)
                        else:
                            self._mark_health_failure(worker)
                except Exception:
                    self._mark_health_failure(worker)

    def _mark_health_failure(self, worker: WorkerState) -> None:
        worker.consecutive_successes = 0
        worker.consecutive_failures += 1
        if (
            worker.healthy
            and worker.consecutive_failures >= self.config.health_failure_threshold
        ):
            worker.healthy = False
            self._invalidate_inflight_requests(worker)
            logger.warning("Worker %s marked unhealthy", worker.url)

    # ---- Tree eviction ----

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.eviction_interval_secs)
            self.tree.evict_tenant_by_size(self.config.max_tree_size)

    # ---- Lifecycle ----

    async def start_background_tasks(self) -> None:
        self._health_task = asyncio.create_task(self._health_loop())
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info(
            "Background tasks started (health_check=%ds, eviction=%ds)",
            self.config.health_check_interval_secs,
            self.config.eviction_interval_secs,
        )

    async def shutdown(self) -> None:
        for task in (self._health_task, self._eviction_task):
            if task:
                task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_error(msg: str) -> bytes:
    import json

    payload = json.dumps({"error": {"message": msg, "type": "router_error"}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


@dataclass
class StreamChunkInfo:
    text: str = ""
    text_token_estimate: int = 0
    completion_tokens_total: Optional[int] = None
    cached_tokens_details: Optional[Dict[str, Any]] = None
    input_token_ids: Optional[List[int]] = None
    output_token_ids: Optional[List[int]] = None


def _make_stream_chunk_record(
    chunk_idx: int, raw_chunk: bytes, chunk_info: StreamChunkInfo
) -> Dict[str, Any]:
    return {
        "chunk_idx": chunk_idx,
        "raw_sse": raw_chunk.decode("utf-8", errors="replace"),
        "text": chunk_info.text,
        "text_token_estimate": chunk_info.text_token_estimate,
        "completion_tokens_total": chunk_info.completion_tokens_total,
        "cached_tokens_details": chunk_info.cached_tokens_details,
        "input_token_ids": chunk_info.input_token_ids,
        "output_token_ids": chunk_info.output_token_ids,
    }


def _extract_sse_chunk_info(raw: bytes) -> StreamChunkInfo:
    """Extract text, usage totals, and cache metadata from SSE bytes."""
    import json as _json

    text_parts: List[str] = []
    text_token_estimate = 0
    completion_tokens_total: Optional[int] = None
    cached_tokens_details: Optional[Dict[str, Any]] = None
    input_token_ids: Optional[List[int]] = None
    output_token_ids: Optional[List[int]] = None
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            continue
        try:
            obj = _json.loads(payload)
        except Exception:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                text_parts.append(content)
                text_token_estimate += 1
            text = choice.get("text")
            if text:
                text_parts.append(text)
                text_token_estimate += 1
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens"):
            completion_tokens_total = max(
                completion_tokens_total or 0,
                int(usage["completion_tokens"]),
            )
        sglext = obj.get("sglext") or {}
        details = sglext.get("cached_tokens_details")
        if details is not None:
            cached_tokens_details = details
        chunk_input_token_ids = sglext.get("input_token_ids")
        if chunk_input_token_ids is not None and input_token_ids is None:
            input_token_ids = [int(tok) for tok in chunk_input_token_ids]
        chunk_output_token_ids = sglext.get("output_token_ids")
        if chunk_output_token_ids:
            output_token_ids = [int(tok) for tok in chunk_output_token_ids]
    return StreamChunkInfo(
        text="".join(text_parts),
        text_token_estimate=text_token_estimate,
        completion_tokens_total=completion_tokens_total,
        cached_tokens_details=cached_tokens_details,
        input_token_ids=input_token_ids,
        output_token_ids=output_token_ids,
    )


def _set_resume_max_tokens(body: Dict[str, Any], generated_tokens: int) -> None:
    orig_max = body.get("max_tokens") or body.get("max_completion_tokens")
    if orig_max is None or generated_tokens <= 0:
        return
    remaining = max(1, int(orig_max) - generated_tokens)
    if "max_tokens" in body:
        body["max_tokens"] = remaining
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = remaining
    if "min_tokens" in body:
        try:
            body["min_tokens"] = min(int(body["min_tokens"]), remaining)
        except (TypeError, ValueError):
            body["min_tokens"] = remaining


def _patch_body_for_text_resume(
    body: Dict[str, Any],
    partial_text: str,
    generated_tokens: int,
) -> Dict[str, Any]:
    import copy

    new_body = copy.deepcopy(body)
    messages = new_body.get("messages")
    if messages is not None and partial_text:
        messages.append({"role": "assistant", "content": partial_text})
        new_body["continue_final_message"] = True

    _set_resume_max_tokens(new_body, generated_tokens)
    return new_body


def _patch_body_for_token_resume(
    body: Dict[str, Any],
    prompt_token_ids: List[int],
    output_token_ids: List[int],
    generated_tokens: int,
) -> Dict[str, Any]:
    import copy

    new_body = copy.deepcopy(body)
    combined_ids = list(prompt_token_ids) + list(output_token_ids)

    if "messages" in new_body:
        new_body["input_ids"] = combined_ids
        new_body.pop("continue_final_message", None)
    elif "prompt" in new_body:
        new_body["prompt"] = combined_ids
    else:
        return new_body

    _set_resume_max_tokens(new_body, generated_tokens)
    return new_body


def _patch_body_for_resume(
    body: Dict[str, Any],
    partial_text: str,
    generated_tokens: int,
    prompt_token_ids: Optional[List[int]] = None,
    output_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if prompt_token_ids and output_token_ids:
        return _patch_body_for_token_resume(
            body, prompt_token_ids, output_token_ids, generated_tokens
        )
    return _patch_body_for_text_resume(body, partial_text, generated_tokens)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_app(config: RouterConfig) -> FastAPI:
    app = FastAPI(title="SGLang Python Router")
    router = PythonRouter(config)

    @app.on_event("startup")
    async def _startup():
        await router.start_background_tasks()
        logger.info(
            "Python router ready — workers=%s, policy=cache_aware, "
            "prefill_timeout=%.0fs, chunk_timeout=%.0fs, request_timeout=%ds",
            [w.url for w in router.workers],
            config.prefill_timeout_secs or config.chunk_timeout_secs,
            config.chunk_timeout_secs,
            config.request_timeout_secs,
        )

    @app.on_event("shutdown")
    async def _shutdown():
        await router.shutdown()

    @app.get("/health")
    async def health():
        healthy_count = sum(1 for w in router.workers if w.healthy)
        if healthy_count == 0:
            return JSONResponse({"status": "unhealthy"}, status_code=503)
        return JSONResponse({"status": "healthy", "healthy_workers": healthy_count})

    @app.get("/v1/models")
    async def models():
        for worker in router.workers:
            if not worker.healthy:
                continue
            try:
                session = await router._get_session()
                async with session.get(
                    f"{worker.url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    return JSONResponse(content=data)
            except Exception:
                continue
        return JSONResponse({"error": "No workers available"}, status_code=503)

    @app.get("/stats")
    async def stats():
        worker_info = [
            {"url": w.url, "healthy": w.healthy, "load": w.load}
            for w in router.workers
        ]
        normal_lats = router._normal_latencies
        retry_recs = router._retry_records
        impacted_reqs = router._retry_impacted_requests
        stream_attempt_recs = router._stream_attempt_records
        retry_summary = {}
        if retry_recs:
            retry_summary = {
                "retried_requests": len(retry_recs),
                "avg_tokens_before_fault": round(
                    sum(r["tokens_before_fault"] for r in retry_recs) / len(retry_recs)
                ),
                "avg_tokens_total": round(
                    sum(r["tokens_total"] for r in retry_recs) / len(retry_recs)
                ),
                "avg_total_ms": round(
                    sum(r["total_ms"] for r in retry_recs) / len(retry_recs), 1
                ),
                "avg_retry_ms": round(
                    sum(r["retry_ms"] for r in retry_recs) / len(retry_recs), 1
                ),
                "retry_records": retry_recs,
            }
        impacted_tasks = []
        if impacted_reqs:
            grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            for rec in impacted_reqs:
                key = (
                    rec.get("job_id"),
                    rec.get("task_label"),
                    rec.get("agent_role"),
                    rec.get("worker_id"),
                    rec.get("mode"),
                )
                if key not in grouped:
                    grouped[key] = {
                        "job_id": rec.get("job_id"),
                        "task_label": rec.get("task_label"),
                        "agent_role": rec.get("agent_role"),
                        "worker_id": rec.get("worker_id"),
                        "mode": rec.get("mode"),
                        "retry_needed_count": 0,
                        "from_workers": [],
                    }
                grouped[key]["retry_needed_count"] += 1
                from_worker = rec.get("from_worker")
                if from_worker and from_worker not in grouped[key]["from_workers"]:
                    grouped[key]["from_workers"].append(from_worker)
            impacted_tasks = list(grouped.values())
        normal_summary = {}
        if normal_lats:
            normal_summary = {
                "normal_requests": len(normal_lats),
                "avg_latency_ms": round(sum(normal_lats) / len(normal_lats), 1),
                "p50_latency_ms": round(sorted(normal_lats)[len(normal_lats) // 2], 1),
                "p99_latency_ms": round(
                    sorted(normal_lats)[int(len(normal_lats) * 0.99)], 1
                ),
            }
        stream_attempt_summary = {}
        grouped_stream_attempts = []
        if stream_attempt_recs:
            grouped_attempts: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            peer_hits = 0
            attempts_with_cache = 0
            for rec in stream_attempt_recs:
                if rec.get("cached_tokens_details") is not None:
                    attempts_with_cache += 1
                if rec.get("peer_cache_hit"):
                    peer_hits += 1
                key = (
                    rec.get("job_id"),
                    rec.get("task_label"),
                    rec.get("agent_role"),
                    rec.get("worker_id"),
                    rec.get("request_path"),
                )
                if key not in grouped_attempts:
                    grouped_attempts[key] = {
                        "job_id": rec.get("job_id"),
                        "task_label": rec.get("task_label"),
                        "agent_role": rec.get("agent_role"),
                        "worker_id": rec.get("worker_id"),
                        "request_path": rec.get("request_path"),
                        "attempts": [],
                    }
                grouped_attempts[key]["attempts"].append(rec)
            grouped_stream_attempts = list(grouped_attempts.values())
            stream_attempt_summary = {
                "attempts_total": len(stream_attempt_recs),
                "attempts_with_cache_details": attempts_with_cache,
                "peer_cache_hit_attempts": peer_hits,
                "attempt_records": stream_attempt_recs,
                "grouped_requests": grouped_stream_attempts,
            }
        return JSONResponse(
            {
                "workers": worker_info,
                "tree_size": router.tree.size,
                **router._stats,
                "normal_request_stats": normal_summary,
                "retry_request_stats": retry_summary,
                "failure_impacted_tasks": impacted_tasks,
                "stream_attempt_stats": stream_attempt_summary,
            }
        )

    @app.post("/flush_tree")
    async def flush_tree():
        old_size = router.tree.size
        router.tree.reset()
        logger.info("Radix tree flushed (old_size=%d)", old_size)
        return JSONResponse({"status": "ok", "evicted_size": old_size})

    @app.post("/remove_worker")
    async def remove_worker(request: Request):
        body = await request.json()
        url = body.get("url", "").rstrip("/")
        for w in router.workers:
            if w.url.rstrip("/") == url:
                w.healthy = False
                w.consecutive_failures = router.config.health_failure_threshold
                w.consecutive_successes = 0
                router.tree.remove_tenant(w.url)
                router._invalidate_inflight_requests(w)
                logger.info("Worker %s removed (unhealthy + radix cleared)", url)
                return JSONResponse({"status": "ok", "url": url})
        return JSONResponse({"status": "error", "message": f"worker {url} not found"}, status_code=404)

    @app.post("/add_worker")
    async def add_worker(request: Request):
        body = await request.json()
        url = body.get("url", "").rstrip("/")
        for w in router.workers:
            if w.url.rstrip("/") == url:
                w.healthy = True
                w.consecutive_failures = 0
                w.consecutive_successes = router.config.health_success_threshold
                router.tree.insert("", w.url)
                logger.info("Worker %s added (healthy + radix registered)", url)
                return JSONResponse({"status": "ok", "url": url})
        return JSONResponse({"status": "error", "message": f"worker {url} not found"}, status_code=404)

    async def _handle_openai(request: Request, path: str):
        body = await request.json()
        request_text = PythonRouter.extract_request_text(body)
        is_stream = body.get("stream", False)
        if is_stream:
            return await router._proxy_stream(body, request_text, path)
        return await router._proxy_non_stream(body, request_text, path)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_openai(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _handle_openai(request, "/v1/completions")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(request: Request, path: str):
        full_path = f"/{path}"
        if request.method == "GET":
            for worker in router.workers:
                if not worker.healthy:
                    continue
                try:
                    session = await router._get_session()
                    async with session.get(
                        f"{worker.url}{full_path}",
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        data = await resp.read()
                        return Response(
                            content=data,
                            status_code=resp.status,
                            media_type=resp.content_type,
                        )
                except Exception:
                    continue
            return JSONResponse({"error": "No workers available"}, status_code=503)

        try:
            body = await request.json()
        except Exception:
            body = {}
        request_text = PythonRouter.extract_request_text(body)
        is_stream = body.get("stream", False)
        if is_stream:
            return await router._proxy_stream(body, request_text, full_path)
        return await router._proxy_non_stream(body, request_text, full_path)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> RouterConfig:
    parser = argparse.ArgumentParser(
        description="SGLang Python Router with cache-aware scheduling and chunk idle timeout"
    )
    parser.add_argument(
        "--worker-urls", nargs="+", required=True, help="Backend worker URLs"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--cache-threshold", type=float, default=0.5)
    parser.add_argument("--balance-abs-threshold", type=int, default=32)
    parser.add_argument("--balance-rel-threshold", type=float, default=1.1)
    parser.add_argument("--eviction-interval-secs", type=int, default=30)
    parser.add_argument("--max-tree-size", type=int, default=2 ** 20)
    parser.add_argument("--request-timeout-secs", type=int, default=300)
    parser.add_argument(
        "--chunk-timeout-secs",
        type=float,
        default=60.0,
        help="Idle timeout between consecutive streaming chunks (seconds)",
    )
    parser.add_argument(
        "--prefill-timeout-secs",
        type=float,
        default=0,
        help="Timeout for the first chunk / prefill (seconds). "
        "0 means use chunk-timeout-secs for both.",
    )
    parser.add_argument("--connect-timeout-secs", type=int, default=10)
    parser.add_argument("--health-check-interval-secs", type=int, default=10)
    parser.add_argument("--health-check-timeout-secs", type=int, default=5)
    parser.add_argument("--health-failure-threshold", type=int, default=3)
    parser.add_argument("--health-success-threshold", type=int, default=2)
    parser.add_argument("--retry-max-retries", type=int, default=3)
    parser.add_argument("--retry-initial-backoff-ms", type=int, default=50)
    parser.add_argument("--retry-max-backoff-ms", type=int, default=5000)
    parser.add_argument("--retry-backoff-multiplier", type=float, default=1.5)
    parser.add_argument("--retry-jitter-factor", type=float, default=0.2)
    parser.add_argument(
        "--enable-fault-tolerance",
        action="store_true",
        default=False,
        help="Enable ring failover for peer KV cache replication. "
        "When a worker fails, the router prefers routing to the ring "
        "neighbor (i+1)%%N which holds the failed worker's KV replicas.",
    )
    parser.add_argument("--prometheus-host", default="0.0.0.0")
    parser.add_argument("--prometheus-port", type=int, default=29000)
    args = parser.parse_args(argv)
    return RouterConfig(
        worker_urls=args.worker_urls,
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        cache_threshold=args.cache_threshold,
        balance_abs_threshold=args.balance_abs_threshold,
        balance_rel_threshold=args.balance_rel_threshold,
        eviction_interval_secs=args.eviction_interval_secs,
        max_tree_size=args.max_tree_size,
        request_timeout_secs=args.request_timeout_secs,
        chunk_timeout_secs=args.chunk_timeout_secs,
        prefill_timeout_secs=args.prefill_timeout_secs,
        connect_timeout_secs=args.connect_timeout_secs,
        health_check_interval_secs=args.health_check_interval_secs,
        health_check_timeout_secs=args.health_check_timeout_secs,
        health_failure_threshold=args.health_failure_threshold,
        health_success_threshold=args.health_success_threshold,
        retry_max_retries=args.retry_max_retries,
        retry_initial_backoff_ms=args.retry_initial_backoff_ms,
        retry_max_backoff_ms=args.retry_max_backoff_ms,
        retry_backoff_multiplier=args.retry_backoff_multiplier,
        retry_jitter_factor=args.retry_jitter_factor,
        enable_fault_tolerance=args.enable_fault_tolerance,
        prometheus_host=args.prometheus_host,
        prometheus_port=args.prometheus_port,
    )


_config: Optional[RouterConfig] = None


def get_app() -> FastAPI:
    global _config
    if _config is None:
        _config = parse_args()
    return create_app(_config)


def main() -> None:
    config = parse_args()
    app = create_app(config)
    logger.info(
        "Starting Python router on %s:%d  workers=%s  chunk_timeout=%.0fs",
        config.host,
        config.port,
        config.worker_urls,
        config.chunk_timeout_secs,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
