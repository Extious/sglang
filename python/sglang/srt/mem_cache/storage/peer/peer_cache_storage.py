# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
PeerCacheStorage: HiCacheStorage backend that replicates KV Cache pages
to a neighbor worker's DRAM via TCP for cross-worker fault tolerance.

Key asymmetry:
  - batch_set_v1 (write path): sends pages to REMOTE neighbor's PeerCacheServer
  - batch_get_v1 (read path): reads from LOCAL PeerReplicaBuffer (failover)
  - batch_exists: checks LOCAL PeerReplicaBuffer
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.storage.peer.peer_cache_server import (
    PeerCacheClient,
    PeerCacheServer,
)

logger = logging.getLogger(__name__)


class PeerCacheStorage(HiCacheStorage):
    """L3 storage backend that replicates KV Cache to a peer worker's DRAM.

    Each worker runs a PeerCacheServer (TCP listener + local buffer) and
    a PeerCacheClient (TCP sender to the ring neighbor).

    Write path: host L2 → serialize → TCP → neighbor's buffer
    Read path:  local buffer → deserialize → host L2 (failover only)
    """

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host):
        cfg = storage_config.extra_config or {}
        peer_port_base = cfg.get(
            "peer_port",
            int(os.environ.get("SGLANG_PEER_CACHE_PORT", "9000")),
        )
        peer_target = cfg.get(
            "peer_url",
            os.environ.get("SGLANG_PEER_TARGET_URL", ""),
        )
        buffer_gb = cfg.get(
            "peer_buffer_size_gb",
            float(os.environ.get("SGLANG_PEER_BUFFER_SIZE_GB", "8")),
        )

        tp_rank = storage_config.tp_rank

        # Each TP rank binds its own port so that rank-to-rank replication
        # works in parallel without data collision (MHA models shard KV
        # cache across ranks, so each rank holds different attention heads).
        peer_port = peer_port_base + tp_rank

        self.mem_pool_host = mem_pool_host
        self.page_size = mem_pool_host.page_size

        self.local_server = PeerCacheServer(peer_port, buffer_gb)
        self.local_server.start()

        if peer_target:
            host, port_str = peer_target.rsplit(":", 1)
            target_port = int(port_str) + tp_rank
            actual_target = f"{host}:{target_port}"
            self.peer_client = PeerCacheClient(actual_target)
            logger.info(
                "PeerCacheStorage[tp_rank=%d]: port %d, replicating to %s, buffer %.1f GB",
                tp_rank, peer_port, actual_target, buffer_gb,
            )
        else:
            self.peer_client = None
            logger.info(
                "PeerCacheStorage[tp_rank=%d]: port %d, no peer_url configured, receive-only mode",
                tp_rank, peer_port,
            )

    # ------------------------------------------------------------------
    # v1 interface (used by HiCacheController backup/prefetch threads)
    # ------------------------------------------------------------------

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Write path: read pages from host L2, TCP send to neighbor."""
        if self.peer_client is None:
            return [False] * len(keys)

        pages: List[bytes] = []
        for i in range(len(keys)):
            idx = host_indices[i * self.page_size].item()
            page_tensor = self.mem_pool_host.get_data_page(idx, flat=True)
            raw = page_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            pages.append(raw)

        results = self.peer_client.put_pages(keys, pages)
        ok_count = sum(1 for r in results if r)
        if ok_count < len(keys):
            logger.warning(
                "PeerCacheStorage.batch_set_v1: only %d/%d pages sent successfully",
                ok_count,
                len(keys),
            )
        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Read path: restore pages from local buffer into host L2."""
        raw_pages = self.local_server.get_pages(keys)

        dummy = self.mem_pool_host.get_dummy_flat_data_page()
        dtype = dummy.dtype

        results: List[bool] = []
        for i, raw in enumerate(raw_pages):
            if raw is not None:
                try:
                    page_tensor = torch.frombuffer(
                        bytearray(raw), dtype=dtype
                    ).reshape(dummy.shape)
                    idx = host_indices[i * self.page_size].item()
                    self.mem_pool_host.set_from_flat_data_page(idx, page_tensor)
                    results.append(True)
                except Exception as exc:
                    logger.warning("PeerCacheStorage.batch_get_v1 page %d: %s", i, exc)
                    results.append(False)
            else:
                results.append(False)
        return results

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        """Check local buffer for consecutive hits."""
        hit_count = self.local_server.exists(keys)
        buf_size = len(self.local_server.buffer)
        if hit_count > 0:
            logger.info(
                "PeerCacheStorage.batch_exists: %d/%d pages found in peer buffer "
                "(buffer has %d entries)",
                hit_count,
                len(keys),
                buf_size,
            )
        elif buf_size > 0 and len(keys) > 0:
            first_key = keys[0]
            has_first = first_key in self.local_server.buffer
            logger.info(
                "PeerCacheStorage.batch_exists: 0/%d hits, buffer has %d entries, "
                "first_key_present=%s, first_key=%s",
                len(keys),
                buf_size,
                has_first,
                first_key[:32],
            )
        return hit_count

    # ------------------------------------------------------------------
    # Legacy abstract methods (required by HiCacheStorage ABC)
    # ------------------------------------------------------------------

    def get(
        self, key: str, target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        return None

    def batch_get(
        self, keys: List[str], target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Optional[torch.Tensor]]:
        return [None] * len(keys)

    def set(
        self, key: str, value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False

    def batch_set(
        self, keys: List[str], values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return key in self.local_server.buffer

    def clear(self) -> None:
        self.local_server.clear()

    def get_stats(self):
        return None
