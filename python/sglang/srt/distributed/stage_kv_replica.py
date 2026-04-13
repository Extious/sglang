# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Stage-replica KV sync: broadcast primary (batch_dp_rank==0) KV rows to peers in the same PP stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import get_stage_replica_group
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def maybe_sync_stage_kv_replica(
    scheduler: "Scheduler",
    batch: Optional["ScheduleBatch"],
) -> None:
    """
    Synchronously broadcast KV cache from batch_dp_rank=0 to peer DP ranks
    in the same PP stage.

    When enable_async_kv_replica is True, delegates to the AsyncKVReplicator
    for non-blocking chunked replication. Otherwise, uses the original
    synchronous broadcast.
    """
    if batch is None:
        return
    sa = scheduler.server_args
    if not getattr(sa, "enable_stage_kv_replica", False):
        return
    mr = scheduler.tp_worker.model_runner
    if mr.batch_dp_rank is None:
        return
    sg = get_stage_replica_group()
    if sg is None:
        return

    local_nel = 0
    if batch.out_cache_loc is not None:
        local_nel = int(batch.out_cache_loc.numel())
    max_nel = local_nel
    if sg.world_size > 1:
        n_t = torch.tensor([local_nel], dtype=torch.int64)
        dist.all_reduce(n_t, op=dist.ReduceOp.MAX, group=sg.cpu_group)
        max_nel = int(n_t.item())
    if max_nel == 0:
        return

    # Delegate to async replicator if available (sync path required when peer has
    # cache rows but this rank has no local out_cache_loc to drive replication).
    _force_sync = sg.world_size > 1 and local_nel == 0 and max_nel > 0
    if getattr(sa, "enable_async_kv_replica", False) and not _force_sync:
        _async_sync_stage_kv_replica(scheduler, batch)
        return

    if batch.forward_mode.is_extend():
        scheduler._stage_kv_decode_counter = 0

    if sa.stage_kv_sync_prefill_only and not batch.forward_mode.is_extend():
        return

    if not batch.forward_mode.is_extend():
        scheduler._stage_kv_decode_counter += 1
        n = sa.stage_kv_sync_every_n_steps
        if n > 1 and sg.world_size > 1:
            cnt = torch.tensor(
                [scheduler._stage_kv_decode_counter], dtype=torch.int64
            )
            dist.all_reduce(cnt, op=dist.ReduceOp.MAX, group=sg.cpu_group)
            scheduler._stage_kv_decode_counter = int(cnt.item())
        if n > 1 and (scheduler._stage_kv_decode_counter % n) != 0:
            return

    pool = mr.token_to_kv_pool
    if not isinstance(pool, MHATokenToKVPool):
        return

    dev = pool.k_buffer[0].device
    # GroupCoordinator.broadcast src=0 is batch_dp_rank 0 within this stage (maps to global rank via sg.ranks[0]).
    if sg.world_size > 1:
        nel_canon = torch.zeros(1, dtype=torch.long, device=dev)
        if mr.batch_dp_rank == 0:
            nel_canon[0] = local_nel
        sg.broadcast(nel_canon, src=0)
        cn = int(nel_canon.item())
        if cn == 0:
            return
        if mr.batch_dp_rank == 0:
            indices = batch.out_cache_loc.to(dtype=torch.int64, device=dev)
        else:
            indices = torch.empty(cn, dtype=torch.int64, device=dev)
        sg.broadcast(indices, src=0)
    else:
        indices = batch.out_cache_loc.to(dtype=torch.int64, device=dev)

    for li in range(len(pool.k_buffer)):
        sg.broadcast(pool.k_buffer[li][indices], src=0)
        sg.broadcast(pool.v_buffer[li][indices], src=0)


def _async_sync_stage_kv_replica(
    scheduler: "Scheduler",
    batch: "ScheduleBatch",
) -> None:
    """
    Delegated async KV replication path.

    Enqueues replication tasks to the AsyncKVReplicator instead of
    blocking on the synchronous broadcast. This allows the inference pipeline
    to continue without waiting for NCCL communication to complete.
    """
    from sglang.srt.fault_tolerance.async_kv_replicator import (
        AsyncKVReplicator,
        ReplicationPriority,
    )

    replicator: Optional[AsyncKVReplicator] = getattr(
        scheduler, "_async_kv_replicator", None
    )
    if replicator is None:
        # Fall back to sync path
        maybe_sync_stage_kv_replica(scheduler, batch)
        return

    mr = scheduler.tp_worker.model_runner
    pool = mr.token_to_kv_pool

    if not isinstance(pool, MHATokenToKVPool):
        # Fall back to sync path
        maybe_sync_stage_kv_replica(scheduler, batch)
        return

    # Determine priority based on forward mode
    if batch.forward_mode.is_extend():
        priority = ReplicationPriority.CRITICAL
        scheduler._stage_kv_decode_counter = 0
        is_prefill = True
    else:
        priority = ReplicationPriority.NORMAL
        scheduler._stage_kv_decode_counter += 1
        n = scheduler.server_args.stage_kv_sync_every_n_steps
        if n > 1 and (scheduler._stage_kv_decode_counter % n) != 0:
            return
        is_prefill = False

    # Enqueue replication for each request in the batch
    indices = batch.out_cache_loc
    num_layers = len(pool.k_buffer)
    num_tokens = indices.numel()

    # Group by request ID
    req_to_reqs = getattr(batch, "reqs", [])
    for req in req_to_reqs:
        if req is None:
            continue
        replicator.enqueue_replication(
            req_id=req.rid,
            indices=indices,
            layer_indices=list(range(num_layers)),
            num_tokens=num_tokens,
            priority=priority,
            source_rank=0,
            is_prefill=is_prefill,
        )
