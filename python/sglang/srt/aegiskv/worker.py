"""Scheduler collaborator that executes AegisKV data-plane commands.

The master decides placement; this worker only moves bytes and publishes KV:

* `protect`: copy a running request's committed KV, every `interval` tokens,
  into a peer-owned DRAM arena (slot `slot_offset + token`, layers
  `[layer_lo, layer_hi)`), never past `max_tokens`.
* `reserve_region` / `stage_region` / `free_region`: hold the first `L` layers
  of a peer's request in this worker's own KV pool (peer HBM). Bytes arrive
  through a staging arena because the GPUs may lack peer-to-peer access.
* `import`: rebuild a committed prefix from a region and/or an arena into fresh
  pool slots, insert it into the radix cache and lock it until `release_import`.
* `poll`: report commit frontiers, finished stages/imports and pool usage.
"""

from __future__ import annotations

import logging
import time
from array import array
from typing import Any, Callable, Dict, List, Optional

import msgspec
import torch

from sglang.srt.aegiskv.arena import ArenaRegistry
from sglang.srt.aegiskv.kv_copy import (
    PackedRegion,
    arena_to_device,
    arena_to_region,
    device_to_arena,
    region_to_device,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class CheckpointTarget(msgspec.Struct, kw_only=True):
    target_id: str
    rid: str
    arena: str
    slot_offset: int
    capacity: int
    layer_lo: int
    layer_hi: int
    max_tokens: int
    launched: int = 0
    committed: int = 0
    seen: bool = False
    ended: bool = False


class ImportState(msgspec.Struct, kw_only=True):
    status: str
    num_tokens: int
    prefix_hit: int = 0
    copy_seconds: float = 0.0
    error: str = ""


class _Job(msgspec.Struct, kw_only=True):
    kind: str
    key: str
    event: Any
    upto: int = 0
    slots: Any = None
    token_ids: Any = None
    free_region: Optional[str] = None
    started: float = 0.0


class _ImportLock(msgspec.Struct, kw_only=True):
    node: Any
    receipt: Any


class AegisKVWorker:
    def __init__(
        self,
        *,
        tree_cache: Any,
        token_to_kv_pool_allocator: Any,
        req_to_token_pool: Any,
        max_total_num_tokens: int,
        page_size: int,
        forward_stream: torch.cuda.Stream,
        get_running_batch: Callable,
        get_last_batch: Callable,
        get_chunked_req: Callable,
        get_waiting_queue_len: Callable[[], int],
        interval_tokens: int = 64,
    ):
        kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self._validate(tree_cache=tree_cache, kv_pool=kv_pool, page_size=page_size)
        self._tree_cache = tree_cache
        self._allocator = token_to_kv_pool_allocator
        self._req_to_token = req_to_token_pool.req_to_token
        self._max_total_num_tokens = max_total_num_tokens
        self._forward_stream = forward_stream
        self._get_running_batch = get_running_batch
        self._get_last_batch = get_last_batch
        self._get_chunked_req = get_chunked_req
        self._get_waiting_queue_len = get_waiting_queue_len
        self._interval_tokens = interval_tokens

        self._k_buffers = kv_pool.k_buffer
        self._v_buffers = kv_pool.v_buffer
        self._num_layers = kv_pool.layer_num
        self._device = self._k_buffers[0].device
        self._arenas = ArenaRegistry(
            num_layers=kv_pool.layer_num,
            head_num=kv_pool.head_num,
            head_dim=kv_pool.head_dim,
            dtype=kv_pool.dtype,
        )
        self._stream = torch.cuda.Stream(device=self._device)
        # Imports must not queue behind checkpoint copies, which wait for the
        # in-flight forward; they get their own stream.
        self._import_stream = torch.cuda.Stream(device=self._device)

        self._targets: Dict[str, CheckpointTarget] = {}
        self._regions: Dict[str, PackedRegion] = {}
        self._imports: Dict[str, ImportState] = {}
        self._import_locks: Dict[str, _ImportLock] = {}
        self._finished_stages: List[str] = []
        self._jobs: List[_Job] = []
        self._ops = {
            "protect": self._op_protect,
            "unprotect": self._op_unprotect,
            "reserve_region": self._op_reserve_region,
            "stage_region": self._op_stage_region,
            "free_region": self._op_free_region,
            "import": self._op_import,
            "release_import": self._op_release_import,
            "map_arenas": self._op_map_arenas,
            "drop_arena": self._op_drop_arena,
            "poll": self._op_poll,
            "reset": self._op_reset,
        }

    @staticmethod
    def _validate(*, tree_cache: Any, kv_pool: Any, page_size: int) -> None:
        if not isinstance(kv_pool, MHATokenToKVPool) or kv_pool.k_buffer is None:
            raise ValueError(f"AegisKV needs an MHA KV pool, got {type(kv_pool)}")
        if kv_pool.store_dtype != kv_pool.dtype or kv_pool.k_buffer[0].dim() != 3:
            raise ValueError("AegisKV needs an unquantized NHD KV pool")
        if kv_pool.start_layer != 0:
            raise ValueError("AegisKV does not support pipeline parallelism")
        if page_size != 1:
            raise ValueError("AegisKV needs --page-size 1")
        if not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
            raise ValueError("AegisKV needs the unified radix cache enabled")

    # ------------------------------------------------------------------ #
    # Scheduler entry points
    # ------------------------------------------------------------------ #

    def handle_control(self, *, op: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if op == "batch":
            # Each control message costs one scheduler iteration; a batch runs
            # many ops in one, and one failing op does not abort the others.
            return {"results": [self._run_one(item) for item in args["ops"]]}
        handler = self._ops.get(op)
        if handler is None:
            raise ValueError(f"Unknown AegisKV op {op!r}")
        return handler(args)

    def _run_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "ok": True,
                "result": self.handle_control(op=item["op"], args=item["args"]),
            }
        except Exception as e:
            logger.exception("AegisKV op %s failed", item["op"])
            return {"ok": False, "error": repr(e)}

    def on_step(self) -> None:
        """Retire finished copies, then launch due checkpoint increments."""
        if not self._jobs and not self._targets:
            return
        # Checkpoint copies read request-owned slots; finishing them before this
        # step's scheduling guarantees the slots are not freed and reused mid-copy.
        self._retire_jobs(block_checkpoints=True)
        if self._targets:
            self._advance_targets()

    def held_tokens(self) -> int:
        """Pool slots owned by AegisKV, outside both the allocator and the tree."""
        held = sum(region.slots.numel() for region in self._regions.values())
        for job in self._jobs:
            if job.kind == "free":
                held += job.slots.numel()
        return held

    # ------------------------------------------------------------------ #
    # Checkpoint capture
    # ------------------------------------------------------------------ #

    def _op_protect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for spec in args["targets"]:
            old = self._targets.get(spec["target_id"])
            target = CheckpointTarget(rid=args["rid"], **spec)
            if old is not None:
                target.launched, target.committed = old.launched, old.committed
                target.seen, target.ended = old.seen, old.ended
            self._arenas.get(target.arena)
            self._targets[target.target_id] = target
        return {}

    def _op_unprotect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for target_id in args["target_ids"]:
            self._targets.pop(target_id, None)
        return {}

    def _live_reqs(self) -> Dict[str, Any]:
        reqs: Dict[str, Any] = {}
        for batch in (self._get_running_batch(), self._get_last_batch()):
            if batch is not None:
                for req in batch.reqs:
                    reqs[req.rid] = req
        chunked = self._get_chunked_req()
        if chunked is not None:
            reqs[chunked.rid] = chunked
        return reqs

    def _advance_targets(self) -> None:
        live = self._live_reqs()
        due: List[tuple] = []
        for target in self._targets.values():
            req = live.get(target.rid)
            if req is None or req.finished() or req.kv.req_pool_idx is None:
                target.ended = target.ended or target.seen
                continue
            target.seen = True
            limit = min(target.max_tokens, target.capacity)
            hi = min(req.kv.kv_committed_len, limit)
            if hi > target.launched and (
                hi - target.launched >= self._interval_tokens or hi == limit
            ):
                due.append((target, req.kv.req_pool_idx, hi))
        if due:
            self._launch_checkpoints(due)

    def _launch_checkpoints(self, due: List[tuple]) -> None:
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(self._forward_stream)
            self._stream.wait_stream(torch.cuda.default_stream(self._device))
            for target, req_pool_idx, hi in due:
                lo = target.launched
                src_slots = self._req_to_token[req_pool_idx, lo:hi].to(torch.int64)
                device_to_arena(
                    k_buffers=self._k_buffers,
                    v_buffers=self._v_buffers,
                    src_slots=src_slots,
                    arena=self._arenas.get(target.arena).tensor,
                    dst_start=target.slot_offset + lo,
                    layer_lo=target.layer_lo,
                    layer_hi=target.layer_hi,
                )
                target.launched = hi
                event = torch.cuda.Event()
                event.record(self._stream)
                self._jobs.append(
                    _Job(kind="ckpt", key=target.target_id, event=event, upto=hi)
                )

    # ------------------------------------------------------------------ #
    # Peer-HBM regions held for other workers
    # ------------------------------------------------------------------ #

    def _op_reserve_region(self, args: Dict[str, Any]) -> Dict[str, Any]:
        num_slots = PackedRegion.num_slots_needed(
            token_capacity=args["token_capacity"],
            resident_layers=args["resident_layers"],
            num_layers=self._num_layers,
        )
        slots = self._alloc(num_slots)
        if slots is None:
            return {"reserved": False}
        self._regions[args["region_id"]] = PackedRegion(
            slots=slots,
            token_capacity=args["token_capacity"],
            resident_layers=args["resident_layers"],
            num_layers=self._num_layers,
        )
        return {"reserved": True, "num_slots": num_slots}

    def _op_stage_region(self, args: Dict[str, Any]) -> Dict[str, Any]:
        region = self._regions[args["region_id"]]
        with torch.cuda.stream(self._stream):
            arena_to_region(
                k_buffers=self._k_buffers,
                v_buffers=self._v_buffers,
                region=region,
                arena=self._arenas.get(args["arena"]).tensor,
                src_start=args["slot_offset"],
                token_lo=args["token_lo"],
                token_hi=args["token_hi"],
            )
            event = torch.cuda.Event()
            event.record(self._stream)
        self._jobs.append(_Job(kind="stage", key=args["task_id"], event=event))
        return {}

    def _op_free_region(self, args: Dict[str, Any]) -> Dict[str, Any]:
        region = self._regions.pop(args["region_id"], None)
        if region is not None:
            # Stages into the region may still be queued; free once they drain.
            with torch.cuda.stream(self._stream):
                event = torch.cuda.Event()
                event.record(self._stream)
            self._jobs.append(
                _Job(
                    kind="free", key=args["region_id"], event=event, slots=region.slots
                )
            )
        return {}

    # ------------------------------------------------------------------ #
    # Takeover import
    # ------------------------------------------------------------------ #

    def _op_import(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild a committed prefix and publish it before replying.

        The copy (tens of ms for a 16K-token prefix) runs inside this control
        op, so the takeover can resubmit as soon as the reply arrives instead
        of waiting another scheduler iteration plus a poll.
        """
        import_id = args["import_id"]
        token_ids = array("q", args["token_ids"])
        num_tokens = len(token_ids)
        region = self._regions.get(args["region_id"]) if args.get("region_id") else None
        arena = args.get("arena")
        self._check_import_coverage(region=region, arena=arena)
        started = time.perf_counter()

        free_region = None
        if region is not None and region.is_identity and arena is None:
            # A full-layer replica is already laid out as pool KV: adopt its slots.
            self._regions.pop(args["region_id"])
            slots = region.slots[:num_tokens]
            self._allocator.free(region.slots[num_tokens:])
        else:
            slots = self._alloc(num_tokens)
            if slots is None:
                self._imports[import_id] = ImportState(
                    status="failed", num_tokens=num_tokens, error="no KV capacity"
                )
                return {"accepted": False}
            self._copy_import(slots=slots, region=region, arena=arena)
            if region is not None:
                free_region = args["region_id"]
        self._imports[import_id] = ImportState(status="copying", num_tokens=num_tokens)
        self._publish_import(
            _Job(
                kind="import",
                key=import_id,
                event=None,
                slots=slots,
                token_ids=token_ids,
                free_region=free_region,
                started=started,
            )
        )
        return {
            "accepted": True,
            "state": msgspec.to_builtins(self._imports[import_id]),
        }

    def _check_import_coverage(
        self, *, region: Optional[PackedRegion], arena: Optional[Dict[str, Any]]
    ) -> None:
        covered = region.resident_layers if region is not None else 0
        if arena is not None:
            if arena["layer_lo"] > covered:
                raise ValueError("Import sources leave a gap in the layer range")
            covered = max(covered, arena["layer_hi"])
        if covered != self._num_layers:
            raise ValueError(
                f"Import sources cover {covered}/{self._num_layers} layers"
            )

    def _copy_import(
        self,
        *,
        slots: torch.Tensor,
        region: Optional[PackedRegion],
        arena: Optional[Dict[str, Any]],
    ) -> None:
        with torch.cuda.stream(self._import_stream):
            if region is not None:
                # Region bytes arrive through stage copies on the backup stream.
                self._import_stream.wait_stream(self._stream)
                region_to_device(
                    k_buffers=self._k_buffers,
                    v_buffers=self._v_buffers,
                    region=region,
                    dst_slots=slots,
                )
            if arena is not None:
                layer_lo = region.resident_layers if region is not None else 0
                arena_to_device(
                    k_buffers=self._k_buffers,
                    v_buffers=self._v_buffers,
                    arena=self._arenas.get(arena["path"]).tensor,
                    src_start=arena["slot_offset"],
                    dst_slots=slots,
                    layer_lo=max(layer_lo, arena["layer_lo"]),
                    layer_hi=arena["layer_hi"],
                )
        self._import_stream.synchronize()

    def _publish_import(self, job: _Job) -> None:
        state = self._imports[job.key]
        key = RadixKey(job.token_ids)
        self._tree_cache.insert(InsertParams(key=key, value=job.slots))
        match = self._tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(job.token_ids))
        )
        node = match.last_device_node
        receipt = self._tree_cache.inc_lock_ref(node).to_dec_params()
        self._import_locks[job.key] = _ImportLock(node=node, receipt=receipt)
        state.prefix_hit = len(match.device_indices)
        state.copy_seconds = time.perf_counter() - job.started
        state.status = "done"
        if job.free_region is not None:
            region = self._regions.pop(job.free_region, None)
            if region is not None:
                self._allocator.free(region.slots)

    def _op_release_import(self, args: Dict[str, Any]) -> Dict[str, Any]:
        lock = self._import_locks.pop(args["import_id"], None)
        if lock is not None:
            self._tree_cache.dec_lock_ref(lock.node, lock.receipt)
        self._imports.pop(args["import_id"], None)
        return {}

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _alloc(self, num_slots: int) -> Optional[torch.Tensor]:
        shortfall = num_slots - self._allocator.available_size()
        if shortfall > 0:
            self._tree_cache.evict(EvictParams(num_tokens=shortfall))
        return self._allocator.alloc(num_slots)

    def _retire_jobs(self, *, block_checkpoints: bool) -> None:
        pending: List[_Job] = []
        for job in self._jobs:
            if job.kind == "ckpt" and block_checkpoints:
                job.event.synchronize()
            elif not job.event.query():
                pending.append(job)
                continue
            self._complete(job)
        self._jobs = pending

    def _complete(self, job: _Job) -> None:
        if job.kind == "ckpt":
            target = self._targets.get(job.key)
            if target is not None:
                target.committed = max(target.committed, job.upto)
        elif job.kind == "stage":
            self._finished_stages.append(job.key)
        elif job.kind == "free":
            self._allocator.free(job.slots)

    def _op_poll(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._retire_jobs(block_checkpoints=False)
        stages, self._finished_stages = self._finished_stages, []
        return {
            "targets": {
                tid: [t.committed, t.seen, t.ended] for tid, t in self._targets.items()
            },
            "stages": stages,
            "imports": {k: msgspec.to_builtins(v) for k, v in self._imports.items()},
            "stats": self._stats(),
        }

    def _stats(self) -> Dict[str, int]:
        running = self._get_running_batch()
        return {
            "total": self._max_total_num_tokens,
            "available": int(self._allocator.available_size()),
            "evictable": int(self._tree_cache.evictable_size()),
            "protected": int(self._tree_cache.protected_size()),
            "held": self.held_tokens(),
            "running_reqs": len(running.reqs) if running is not None else 0,
            "waiting_reqs": self._get_waiting_queue_len(),
        }

    def _op_map_arenas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map and pin arenas ahead of use; pinning takes about 1 s per GiB."""
        for path in args["paths"]:
            self._arenas.get(path)
        return {}

    def _op_drop_arena(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._stream.synchronize()
        self._retire_jobs(block_checkpoints=True)
        path = args["path"]
        for target_id in [t for t, v in self._targets.items() if v.arena == path]:
            del self._targets[target_id]
        self._arenas.drop(path)
        return {}

    def _op_reset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._stream.synchronize()
        self._retire_jobs(block_checkpoints=True)
        for import_id in list(self._import_locks):
            self._op_release_import({"import_id": import_id})
        for region_id in list(self._regions):
            self._op_free_region({"region_id": region_id})
        self._stream.synchronize()
        self._retire_jobs(block_checkpoints=True)
        self._targets.clear()
        self._imports.clear()
        self._finished_stages = []
        return {}
