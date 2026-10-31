"""KV copies between the device pool, peer-DRAM arenas and packed HBM regions.

All functions enqueue work on the current CUDA stream; callers own ordering.
`k_buffers[layer]` / `v_buffers[layer]` are the MHA pool tensors of shape
`[num_slots, head_num, head_dim]`.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch


def device_to_arena(
    *,
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    src_slots: torch.Tensor,
    arena: torch.Tensor,
    dst_start: int,
    layer_lo: int,
    layer_hi: int,
) -> None:
    n = src_slots.numel()
    for layer in range(layer_lo, layer_hi):
        arena[layer, 0, dst_start : dst_start + n].copy_(
            k_buffers[layer].index_select(0, src_slots), non_blocking=True
        )
        arena[layer, 1, dst_start : dst_start + n].copy_(
            v_buffers[layer].index_select(0, src_slots), non_blocking=True
        )


def arena_to_device(
    *,
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    arena: torch.Tensor,
    src_start: int,
    dst_slots: torch.Tensor,
    layer_lo: int,
    layer_hi: int,
) -> None:
    n = dst_slots.numel()
    device = dst_slots.device
    for layer in range(layer_lo, layer_hi):
        k_buffers[layer].index_copy_(
            0,
            dst_slots,
            arena[layer, 0, src_start : src_start + n].to(device, non_blocking=True),
        )
        v_buffers[layer].index_copy_(
            0,
            dst_slots,
            arena[layer, 1, src_start : src_start + n].to(device, non_blocking=True),
        )


class PackedRegion:
    """The first `resident_layers` layers of up to `token_capacity` tokens, packed
    into pool slots so a partial-layer replica costs only its own bytes.

    Cell `c = layer * token_capacity + token` lives in pool layer `c // m` at slot
    `slots[c % m]`, with `m = ceil(resident_layers * token_capacity / num_layers)`.
    With `resident_layers == num_layers` this is the identity layout.
    """

    def __init__(
        self,
        *,
        slots: torch.Tensor,
        token_capacity: int,
        resident_layers: int,
        num_layers: int,
    ):
        self.slots = slots
        self.token_capacity = token_capacity
        self.resident_layers = resident_layers
        self.num_layers = num_layers

    @staticmethod
    def num_slots_needed(
        *, token_capacity: int, resident_layers: int, num_layers: int
    ) -> int:
        return -(-resident_layers * token_capacity // num_layers)

    @property
    def is_identity(self) -> bool:
        return self.resident_layers == self.num_layers

    def runs(
        self, *, layer: int, token_lo: int, token_hi: int
    ) -> Iterator[Tuple[int, int, int, int, int]]:
        """Yield `(pool_layer, slot_lo, slot_hi, token_lo, token_hi)` pieces."""
        m = self.slots.numel()
        base = layer * self.token_capacity
        cell, end = base + token_lo, base + token_hi
        while cell < end:
            pool_layer, pos = divmod(cell, m)
            take = min(end - cell, m - pos)
            yield pool_layer, pos, pos + take, cell - base, cell - base + take
            cell += take


def arena_to_region(
    *,
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    region: PackedRegion,
    arena: torch.Tensor,
    src_start: int,
    token_lo: int,
    token_hi: int,
) -> None:
    # The staging arena holds token t at slot src_start + t.
    device = region.slots.device
    for layer in range(region.resident_layers):
        for pool_layer, s_lo, s_hi, t_lo, t_hi in region.runs(
            layer=layer, token_lo=token_lo, token_hi=token_hi
        ):
            dst = region.slots[s_lo:s_hi]
            src = slice(src_start + t_lo, src_start + t_hi)
            k_buffers[pool_layer].index_copy_(
                0, dst, arena[layer, 0, src].to(device, non_blocking=True)
            )
            v_buffers[pool_layer].index_copy_(
                0, dst, arena[layer, 1, src].to(device, non_blocking=True)
            )


def region_to_device(
    *,
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    region: PackedRegion,
    dst_slots: torch.Tensor,
) -> None:
    n = dst_slots.numel()
    for layer in range(region.resident_layers):
        for pool_layer, s_lo, s_hi, t_lo, t_hi in region.runs(
            layer=layer, token_lo=0, token_hi=n
        ):
            src = region.slots[s_lo:s_hi]
            dst = dst_slots[t_lo:t_hi]
            k_buffers[layer].index_copy_(
                0, dst, k_buffers[pool_layer].index_select(0, src)
            )
            v_buffers[layer].index_copy_(
                0, dst, v_buffers[pool_layer].index_select(0, src)
            )
