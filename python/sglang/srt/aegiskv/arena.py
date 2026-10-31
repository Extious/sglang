"""Peer-DRAM backup arenas shared between worker processes through /dev/shm."""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class HostArena:
    """A /dev/shm file mapped as a page-locked KV backup tensor.

    The master creates the file; every worker that reads or writes it maps the
    same pages. The layout is shared with the master's slot allocator:
    `[num_layers, 2 (K/V), num_slots, head_num, head_dim]`, one slot per token.
    """

    def __init__(
        self,
        *,
        path: str,
        num_layers: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        slot_bytes = 2 * num_layers * head_num * head_dim * dtype.itemsize
        nbytes = os.path.getsize(path)
        if nbytes == 0 or nbytes % slot_bytes != 0:
            raise ValueError(
                f"Arena {path} has {nbytes} bytes, not a multiple of {slot_bytes}"
            )
        self.path = path
        self.num_slots = nbytes // slot_bytes
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
        flat = torch.empty(0, dtype=torch.uint8).set_(storage)
        self.tensor = flat.view(dtype).view(
            num_layers, 2, self.num_slots, head_num, head_dim
        )
        self._registered_ptr = flat.data_ptr()
        err = torch.cuda.cudart().cudaHostRegister(self._registered_ptr, nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({path}) failed with {err}")
        logger.info(
            "AegisKV arena mapped: %s, %d slots, %.2f GiB",
            path,
            self.num_slots,
            nbytes / (1 << 30),
        )

    def close(self) -> None:
        if self._registered_ptr:
            torch.cuda.cudart().cudaHostUnregister(self._registered_ptr)
            self._registered_ptr = 0


class ArenaRegistry:
    """Maps arenas on first use and keeps them mapped for the process lifetime."""

    def __init__(
        self, *, num_layers: int, head_num: int, head_dim: int, dtype: torch.dtype
    ):
        self._num_layers = num_layers
        self._head_num = head_num
        self._head_dim = head_dim
        self._dtype = dtype
        self._arenas: Dict[str, HostArena] = {}

    def get(self, path: str) -> HostArena:
        arena = self._arenas.get(path)
        if arena is None:
            arena = HostArena(
                path=path,
                num_layers=self._num_layers,
                head_num=self._head_num,
                head_dim=self._head_dim,
                dtype=self._dtype,
            )
            self._arenas[path] = arena
        return arena

    def drop(self, path: str) -> None:
        arena = self._arenas.pop(path, None)
        if arena is not None:
            arena.close()
