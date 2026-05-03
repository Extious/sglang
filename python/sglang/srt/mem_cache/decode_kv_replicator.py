"""
DecodeKVReplicator: real-time page-granularity backup of decode KV cache
to a remote backup server for fault tolerance.

During streaming decode, KV cache lives only in the GPU KV pool and is NOT
inserted into the radix tree until the request finishes. This module bypasses
the radix tree and backs up decode KV directly from the GPU KV pool to the
storage backend at page granularity.

Flow (per decode step):
  1. Scheduler calls ``on_decode_step(batch)`` after each decode forward.
  2. For each request in the batch, the new token's device KV index is recorded.
  3. When ``page_size`` tokens accumulate for a request, a page-aligned backup
     is initiated:  GPU -> Host (async DMA) -> Remote backup server.
  4. Page hashes are still computed for generic HiCache bookkeeping, while
     failover restore uses ``full_token_ids`` + ``page_start`` to address
     the token-indexed radix buffer on the backup server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.managers.cache_controller import StorageOperationKind
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
    full_token_ids: Optional[List[int]] = None
    page_start: int = 0
    checkpoint_output_len: int = 0
    generation: int = 0  # remote_backup_generation for lease tracking


@dataclass
class _PendingStorage:
    rid: str
    host_indices: torch.Tensor
    checkpoint_output_len: int


@dataclass
class _ReqState:
    """Per-request accumulator for decode tokens awaiting page flush."""

    token_ids: List[int] = field(default_factory=list)
    device_indices: List[int] = field(default_factory=list)
    parent_hash: Optional[str] = None
    pages_flushed: int = 0
    origin_input_ids: Optional[List[int]] = None
    flushed_token_ids: List[int] = field(default_factory=list)
    generation: int = 0  # remote_backup_generation for lease tracking


class DecodeKVReplicator:
    """Backs up decode KV to remote storage at page granularity.

    Lifecycle:
      - Created by ``HiRadixCache.__init__`` when remote backup storage is enabled.
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
        self._pending_storage: Dict[int, _PendingStorage] = {}
        self._finished_rids: set[str] = set()
        self._node_id_counter = -(10 ** 9)

    def _next_node_id(self) -> int:
        """Negative IDs avoid collision with real TreeNode IDs."""
        self._node_id_counter -= 1
        return self._node_id_counter

    def _has_pending_for_rid(self, rid: str) -> bool:
        return any(pending.rid == rid for pending in self._pending_dma.values()) or any(
            pending.rid == rid for pending in self._pending_storage.values()
        )

    def _maybe_forget_finished_rid(self, rid: str) -> None:
        if rid in self._states or self._has_pending_for_rid(rid):
            return
        self._finished_rids.discard(rid)

    def set_prefill_parent_hash(
        self, req: Req, parent_hash: Optional[str]
    ) -> None:
        """Record the last page hash from the prefill phase.

        This provides the chaining seed so decode page hashes are compatible
        with the prefetch restore path on the backup server.
        Also stores ``origin_input_ids`` for building the full token context
        needed by the token-indexed radix buffer on the backup server.

        Critically, this also captures the first output token generated during
        the prefill forward pass (stored in ``req.output_ids`` before this
        method is called).  Without this, the decode backup's token sequence
        would be off-by-one relative to the request's visible output prefix,
        causing a permanent mismatch in the backup radix trie.
        """
        state = self._states.setdefault(req.rid, _ReqState())
        state.parent_hash = parent_hash
        state.origin_input_ids = list(req.origin_input_ids)
        state.generation = getattr(req, "remote_backup_generation", 0)

        if req.output_ids and req.req_pool_idx is not None:
            n_origin = len(req.origin_input_ids)
            for i, tok_id in enumerate(req.output_ids):
                state.token_ids.append(tok_id)
                seq_pos = n_origin + i
                device_idx = self._req_to_token_pool.req_to_token[
                    req.req_pool_idx, seq_pos
                ].item()
                state.device_indices.append(device_idx)

        logger.debug(
            "DecodeKVReplicator.set_prefill_parent_hash: rid=%s, "
            "parent_hash=%s, origin_input_len=%d, prefill_output_tokens=%d",
            req.rid,
            parent_hash[:16] if parent_hash else "None",
            len(req.origin_input_ids),
            len(req.output_ids) if req.output_ids else 0,
        )

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

            if req.req_pool_idx is None:
                continue

            if state is None:
                state = _ReqState()
                state.parent_hash = self._tree_cache.get_last_hash_for_req(
                    req.origin_input_ids
                )
                state.origin_input_ids = list(req.origin_input_ids)
                n_origin = len(req.origin_input_ids)
                for i, tok_id in enumerate(output_ids):
                    state.token_ids.append(tok_id)
                    device_idx = self._req_to_token_pool.req_to_token[
                        req.req_pool_idx, n_origin + i
                    ].item()
                    state.device_indices.append(device_idx)
                self._states[rid] = state

                if len(state.token_ids) >= self._flush_size:
                    self._flush_page(rid, state)
                continue

            new_token_id = output_ids[-1]
            seq_pos = len(req.origin_input_ids) + len(output_ids) - 1
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

        origin = state.origin_input_ids or []
        full_token_ids = origin + state.flushed_token_ids + flush_token_ids
        page_start = (len(origin) + len(state.flushed_token_ids)) // self._page_size
        checkpoint_output_len = len(state.flushed_token_ids) + flush_len

        gpu_device = self._req_to_token_pool.device
        device_tensor = torch.tensor(
            flush_device_indices, dtype=torch.int64, device=gpu_device
        )
        node_id = self._next_node_id()
        host_indices = self._cc.write(device_indices=device_tensor, node_id=node_id)
        if host_indices is None:
            logger.debug("DecodeKVReplicator: host alloc failed for %s, skip", rid)
            state.flushed_token_ids.extend(flush_token_ids)
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
            full_token_ids=full_token_ids,
            page_start=page_start,
            checkpoint_output_len=checkpoint_output_len,
            generation=state.generation,
        )

        if state.pages_flushed == 0:
            logger.debug(
                "DecodeKVReplicator._flush_page FIRST flush: rid=%s, "
                "anchor_hash=%s, first_token=%d, first_page_hash=%s, "
                "flush_len=%d, pages=%d, page_start=%d",
                rid,
                (state.parent_hash or "None")[:16],
                flush_token_ids[0] if flush_token_ids else -1,
                page_hashes[0][:16] if page_hashes else "None",
                flush_len,
                len(page_hashes),
                page_start,
            )

        state.flushed_token_ids.extend(flush_token_ids)
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
            self._cc.append_host_mem_release(pending.host_indices)
            self._maybe_forget_finished_rid(pending.rid)
            return True

        operation_id = self._cc.write_storage(
            pending.host_indices,
            [],
            pending.page_hash,
            full_token_ids=pending.full_token_ids,
            page_start=pending.page_start,
            operation_kind=StorageOperationKind.DECODE_STREAM_BACKUP,
            request_id=pending.rid,
            request_generation=pending.generation,
        )
        self._pending_storage[operation_id] = _PendingStorage(
            rid=pending.rid,
            host_indices=pending.host_indices,
            checkpoint_output_len=pending.checkpoint_output_len,
        )
        logger.debug(
            "DecodeKVReplicator: backed up %d pages to remote storage for req %s",
            len(pending.page_hash),
            pending.rid,
        )
        return True

    def on_backup_complete(self, operation_id: int) -> Optional[torch.Tensor]:
        """Called when a storage backup completes for *operation_id*.

        Returns the host indices to be freed if this belongs to the
        replicator, None otherwise.
        """
        pending = self._pending_storage.pop(operation_id, None)
        if pending is None:
            return None
        if pending.rid not in self._finished_rids:
            self._tree_cache.update_local_checkpointed_output_len(
                pending.rid, pending.checkpoint_output_len
            )
        self._maybe_forget_finished_rid(pending.rid)
        return pending.host_indices

    def on_request_finished(self, req: Req) -> None:
        """Clean up state for a finished request."""
        self._states.pop(req.rid, None)
        self._finished_rids.add(req.rid)
        self._tree_cache.clear_request_checkpoint_state(req.rid)

    def force_release_pending_storage(self) -> None:
        """Free all tracked host indices from pending storage backups.

        Safety net for shutdown/detach paths.
        """
        for pending in self._pending_storage.values():
            self._cc.append_host_mem_release(pending.host_indices)
        self._pending_storage.clear()
        self._finished_rids.clear()

    @property
    def active_requests(self) -> int:
        return len(self._states)

    @property
    def pending_dma_count(self) -> int:
        return len(self._pending_dma)

    @property
    def pending_storage_count(self) -> int:
        return len(self._pending_storage)
