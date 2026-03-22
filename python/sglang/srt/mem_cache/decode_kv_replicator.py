"""
DecodeKVReplicator: real-time page-granularity backup of decode KV cache
to a peer worker's DRAM for fault tolerance.

During streaming decode, KV cache lives only in the GPU KV pool and is NOT
inserted into the radix tree until the request finishes. This module bypasses
the radix tree and backs up decode KV directly from the GPU KV pool to the
peer storage backend at page granularity.

Flow (per decode step):
  1. Scheduler calls ``on_decode_step(batch)`` after each decode forward.
  2. For each request in the batch, the new token's device KV index is recorded.
  3. When ``page_size`` tokens accumulate for a request, a page-aligned backup
     is initiated:  GPU → Host (async DMA) → Peer buffer (async TCP).
  4. Page hashes are computed with the same chaining algorithm used by
     ``prefetch_from_storage``, so the peer can look them up on failover.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


@dataclass
class _PendingPage:
    """A batch of decode KV pages ready for async GPU→Host DMA."""

    page_hash: List[str]
    host_indices: torch.Tensor
    node_id: int
    rid: str = ""


@dataclass
class _ReqState:
    """Per-request accumulator for decode tokens awaiting page flush."""

    token_ids: List[int] = field(default_factory=list)
    device_indices: List[int] = field(default_factory=list)
    parent_hash: Optional[str] = None
    pages_flushed: int = 0


class DecodeKVReplicator:
    """Backs up decode KV to peer storage at page granularity.

    Lifecycle:
      - Created by ``HiRadixCache.__init__`` when peer storage is enabled.
      - ``on_decode_step(batch)`` called by the scheduler after each decode.
      - ``poll_completions()`` called from ``check_hicache_events()`` to
        drive the DMA-complete → storage-send pipeline.
      - ``on_request_finished(req)`` cleans up per-request state.
    """

    MIN_FLUSH_SIZE = 64

    def __init__(
        self,
        cache_controller: HiCacheController,
        req_to_token_pool: ReqToTokenPool,
        page_size: int,
        tree_cache: HiRadixCache,
    ):
        self._cc = cache_controller
        self._req_to_token_pool = req_to_token_pool
        self._page_size = page_size
        self._flush_size = max(page_size, self.MIN_FLUSH_SIZE)
        self._tree_cache = tree_cache
        self._states: Dict[str, _ReqState] = {}
        self._pending_dma: Dict[int, _PendingPage] = {}
        self._node_id_counter = -(10 ** 9)

    def _next_node_id(self) -> int:
        """Negative IDs avoid collision with real TreeNode IDs."""
        self._node_id_counter -= 1
        return self._node_id_counter

    def set_prefill_parent_hash(
        self, req: Req, parent_hash: Optional[str]
    ) -> None:
        """Record the last page hash from the prefill phase.

        This provides the chaining seed so decode page hashes are compatible
        with the prefetch restore path on the peer worker.
        """
        state = self._states.setdefault(req.rid, _ReqState())
        state.parent_hash = parent_hash

    def on_decode_step(self, batch: ScheduleBatch) -> None:
        """Called after each decode forward pass.

        Collects the newly generated token and its KV device index for every
        request in *batch*.  Flushes a page whenever ``page_size`` tokens
        have accumulated.
        """
        for req in batch.reqs:
            if req.finished():
                continue

            rid = req.rid
            state = self._states.get(rid)

            output_ids = req.output_ids
            if not output_ids:
                continue
            new_token_id = output_ids[-1]

            if state is None:
                state = _ReqState()
                state.parent_hash = self._tree_cache.get_last_hash_for_req(
                    req.origin_input_ids
                )
                self._states[rid] = state

            seq_pos = len(req.origin_input_ids) + len(output_ids) - 1
            if req.req_pool_idx is None:
                continue
            device_idx = self._req_to_token_pool.req_to_token[
                req.req_pool_idx, seq_pos
            ].item()

            state.token_ids.append(new_token_id)
            state.device_indices.append(device_idx)

            if len(state.token_ids) >= self._flush_size:
                self._flush_page(rid, state)

    def _flush_page(self, rid: str, state: _ReqState) -> None:
        flush_len = (len(state.token_ids) // self._page_size) * self._page_size
        if flush_len < self._page_size:
            return

        flush_token_ids = state.token_ids[:flush_len]
        flush_device_indices = state.device_indices[:flush_len]

        page_hashes = []
        parent_hash = state.parent_hash
        for start in range(0, flush_len, self._page_size):
            page_tok = flush_token_ids[start : start + self._page_size]
            h = get_hash_str(page_tok, prior_hash=parent_hash)
            page_hashes.append(h)
            parent_hash = h

        gpu_device = self._req_to_token_pool.device
        device_tensor = torch.tensor(
            flush_device_indices, dtype=torch.int64, device=gpu_device
        )
        node_id = self._next_node_id()
        host_indices = self._cc.write(device_indices=device_tensor, node_id=node_id)
        if host_indices is None:
            logger.debug("DecodeKVReplicator: host alloc failed for %s, skip", rid)
            state.token_ids = state.token_ids[flush_len:]
            state.device_indices = state.device_indices[flush_len:]
            state.parent_hash = parent_hash
            state.pages_flushed += len(page_hashes)
            return

        self._pending_dma[node_id] = _PendingPage(
            page_hash=page_hashes,
            host_indices=host_indices,
            node_id=node_id,
            rid=rid,
        )

        state.parent_hash = parent_hash
        state.pages_flushed += len(page_hashes)
        state.token_ids = state.token_ids[flush_len:]
        state.device_indices = state.device_indices[flush_len:]

    def on_write_complete(self, node_id: int) -> bool:
        """Called when a DMA write completes for *node_id*.

        Returns True if this node_id belongs to the replicator (and was
        handled), False otherwise (let the normal HiCache path handle it).
        """
        pending = self._pending_dma.pop(node_id, None)
        if pending is None:
            return False

        if not self._cc.enable_storage or self._cc.storage_backend is None:
            return True

        self._cc.write_storage(
            pending.host_indices,
            [],
            pending.page_hash,
        )
        logger.info(
            "DecodeKVReplicator: backed up %d pages to peer for req %s",
            len(pending.page_hash),
            pending.rid,
        )
        return True

    def on_request_finished(self, req: Req) -> None:
        """Clean up state for a finished request."""
        self._states.pop(req.rid, None)

    @property
    def active_requests(self) -> int:
        return len(self._states)

    @property
    def pending_dma_count(self) -> int:
        return len(self._pending_dma)
