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
import logging
import random
import sys
import time
from collections import defaultdict
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
        }
        self._retry_records: List[Dict[str, Any]] = []
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
            try:
                session = await self._get_session()
                async with session.post(
                    f"{worker.url}{path}",
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_secs),
                ) as resp:
                    resp_body = await resp.read()
                    if resp.status in self.config.RETRYABLE_STATUSES:
                        if fault_detected_at is None:
                            fault_detected_at = time.monotonic()
                            failed_worker_url = worker.url
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                    elapsed_ms = (time.monotonic() - req_start) * 1000
                    if fault_detected_at is not None:
                        retry_ms = (time.monotonic() - fault_detected_at) * 1000
                        record = {
                            "tokens_before_fault": 0,
                            "tokens_total": 0,
                            "total_ms": round(elapsed_ms, 1),
                            "retry_ms": round(retry_ms, 1),
                            "from_worker": failed_worker_url,
                            "to_worker": worker.url,
                            "mode": "non_stream",
                            "user": user_tag,
                        }
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
                if fault_detected_at is None:
                    fault_detected_at = time.monotonic()
                    failed_worker_url = worker.url
                last_failed_idx = idx
                excluded.add(idx)
                self._stats["retries_total"] += 1
                if attempt < self.config.retry_max_retries:
                    await asyncio.sleep(self._backoff(attempt))
            finally:
                worker.load -= 1
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

        async def _try_stream():
            nonlocal last_failed_idx

            accumulated_text = ""
            accumulated_tokens = 0
            orig_max_tokens = body.get("max_tokens") or body.get(
                "max_completion_tokens"
            )
            current_body = body

            req_start = time.monotonic()
            fault_detected_at: Optional[float] = None
            tokens_at_fault = 0
            failed_worker_url = ""

            def _mark_fault(w_url: str) -> None:
                nonlocal fault_detected_at, tokens_at_fault, failed_worker_url
                if fault_detected_at is None:
                    fault_detected_at = time.monotonic()
                    tokens_at_fault = accumulated_tokens
                    failed_worker_url = w_url

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
                    and accumulated_tokens >= orig_max_tokens
                ):
                    yield b"data: [DONE]\n\n"
                    return

                idx = self.select_worker(request_text, failed_worker_idx=last_failed_idx)
                if idx is None or idx in excluded:
                    idx = min(healthy_available, key=lambda i: self.workers[i].load)
                worker = self.workers[idx]
                worker.load += 1
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
                        _mark_fault(worker.url)
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
                                        accumulated_tokens,
                                    )
                                    raise RuntimeError(
                                        f"Worker {worker.url} removed mid-stream"
                                    )
                                cur_timeout = chunk_to
                                text, tokens = _extract_sse_content(chunk)
                                accumulated_text += text
                                accumulated_tokens += tokens
                                yield chunk
                    except asyncio.TimeoutError:
                        self._stats["chunk_timeouts"] += 1
                        _mark_fault(worker.url)
                        logger.warning(
                            "Chunk idle timeout (%.0fs) on worker %s, attempt %d/%d "
                            "(accumulated %d tokens so far)",
                            self.config.chunk_timeout_secs,
                            worker.url,
                            attempt + 1,
                            1 + self.config.retry_max_retries,
                            accumulated_tokens,
                        )
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            if accumulated_text:
                                current_body = _patch_body_for_resume(
                                    body, accumulated_text, accumulated_tokens,
                                )
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        yield _sse_error(
                            f"Chunk idle timeout after {self.config.chunk_timeout_secs}s"
                        )
                        return
                    except Exception as exc:
                        _mark_fault(worker.url)
                        logger.warning(
                            "Stream error on worker %s: %s", worker.url, exc
                        )
                        last_failed_idx = idx
                        excluded.add(idx)
                        self._stats["retries_total"] += 1
                        if attempt < self.config.retry_max_retries:
                            if accumulated_text:
                                current_body = _patch_body_for_resume(
                                    body, accumulated_text, accumulated_tokens,
                                )
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        yield _sse_error(str(exc))
                        return
                    else:
                        elapsed_ms = (time.monotonic() - req_start) * 1000
                        if fault_detected_at is not None:
                            retry_ms = (time.monotonic() - fault_detected_at) * 1000
                            record = {
                                "tokens_before_fault": tokens_at_fault,
                                "tokens_total": accumulated_tokens,
                                "total_ms": round(elapsed_ms, 1),
                                "retry_ms": round(retry_ms, 1),
                                "from_worker": failed_worker_url,
                                "to_worker": worker.url,
                                "mode": "stream",
                                "user": user_tag,
                            }
                            self._retry_records.append(record)
                            logger.info(
                                "REQUEST_METRIC retried=true mode=stream "
                                "tokens_before_fault=%d tokens_after_retry=%d "
                                "tokens_total=%d total_ms=%.0f retry_ms=%.0f "
                                "from=%s to=%s user=%s",
                                tokens_at_fault,
                                accumulated_tokens - tokens_at_fault,
                                accumulated_tokens,
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
                                    accumulated_tokens,
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
                    _mark_fault(worker.url)
                    self._stats["chunk_timeouts"] += 1
                    last_failed_idx = idx
                    excluded.add(idx)
                    self._stats["retries_total"] += 1
                    if attempt < self.config.retry_max_retries:
                        if accumulated_text:
                            current_body = _patch_body_for_resume(
                                body, accumulated_text, accumulated_tokens,
                            )
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    yield _sse_error("Connection timeout")
                    return
                except Exception as exc:
                    worker.load -= 1
                    _mark_fault(worker.url)
                    logger.warning("Worker %s stream failed: %s", worker.url, exc)
                    last_failed_idx = idx
                    excluded.add(idx)
                    self._stats["retries_total"] += 1
                    if attempt < self.config.retry_max_retries:
                        if accumulated_text:
                            current_body = _patch_body_for_resume(
                                body, accumulated_text, accumulated_tokens,
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


def _extract_sse_content(raw: bytes) -> Tuple[str, int]:
    """Extract accumulated delta text and completion token count from SSE bytes.

    Parses ``data: {...}`` lines from an SSE chunk (which may contain
    multiple events) and returns ``(text, token_count)``.
    """
    import json as _json

    text_parts: List[str] = []
    token_count = 0
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
                token_count += 1
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens"):
            token_count = max(token_count, int(usage["completion_tokens"]))
    return "".join(text_parts), token_count


def _patch_body_for_resume(
    body: Dict[str, Any],
    partial_text: str,
    generated_tokens: int,
) -> Dict[str, Any]:
    """Return a copy of *body* adjusted for resume after partial generation.

    - Appends the partial output as an assistant message with
      ``continue_final_message=True`` so the next worker continues
      seamlessly.
    - Reduces ``max_tokens`` by the number of tokens already generated
      (they become prefill tokens in the retry).
    """
    import copy

    new_body = copy.deepcopy(body)
    messages = new_body.get("messages")
    if messages is not None and partial_text:
        messages.append({"role": "assistant", "content": partial_text})
        new_body["continue_final_message"] = True

    orig_max = new_body.get("max_tokens") or new_body.get("max_completion_tokens")
    if orig_max is not None and generated_tokens > 0:
        remaining = max(1, int(orig_max) - generated_tokens)
        if "max_tokens" in new_body:
            new_body["max_tokens"] = remaining
        if "max_completion_tokens" in new_body:
            new_body["max_completion_tokens"] = remaining

    return new_body


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
        return JSONResponse(
            {
                "workers": worker_info,
                "tree_size": router.tree.size,
                **router._stats,
                "normal_request_stats": normal_summary,
                "retry_request_stats": retry_summary,
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
