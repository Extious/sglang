# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Stage-replica KV sync: broadcast primary (batch_dp_rank==0) KV rows to peers in the same PP stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed.parallel_state import get_stage_replica_group
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def maybe_sync_stage_kv_replica(
    scheduler: "Scheduler",
    batch: Optional["ScheduleBatch"],
) -> None:
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
    if batch.out_cache_loc is None or batch.out_cache_loc.numel() == 0:
        return

    if batch.forward_mode.is_extend():
        scheduler._stage_kv_decode_counter = 0

    if sa.stage_kv_sync_prefill_only and not batch.forward_mode.is_extend():
        return

    if not batch.forward_mode.is_extend():
        scheduler._stage_kv_decode_counter += 1
        n = sa.stage_kv_sync_every_n_steps
        if n > 1 and (scheduler._stage_kv_decode_counter % n) != 0:
            return

    pool = mr.token_to_kv_pool
    if not isinstance(pool, MHATokenToKVPool):
        return

    dev = pool.k_buffer[0].device
    indices = batch.out_cache_loc.to(dtype=torch.int64, device=dev)
    group = sg.device_group
    src = 0
    for li in range(len(pool.k_buffer)):
        if mr.batch_dp_rank == 0:
            torch.distributed.broadcast(
                pool.k_buffer[li][indices], src=src, group=group
            )
            torch.distributed.broadcast(
                pool.v_buffer[li][indices], src=src, group=group
            )
        else:
            torch.distributed.broadcast(
                pool.k_buffer[li][indices], src=src, group=group
            )
            torch.distributed.broadcast(
                pool.v_buffer[li][indices], src=src, group=group
            )
