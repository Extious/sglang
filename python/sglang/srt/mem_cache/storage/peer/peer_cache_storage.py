# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
PeerCacheStorage: HiCacheStorage backend that replicates KV cache pages
to a peer worker's DRAM via TCP for cross-worker fault tolerance.

The peer buffer uses a token-indexed radix trie instead of hash keys, so
failover restore is driven by ``full_token_ids`` + ``page_start`` rather than
by requiring both workers to share the same hash-chain anchors.

Key asymmetry:
  - ``batch_set_v1`` sends pages + token context to the remote peer
  - ``match_prefix_from`` / ``get_pages_tokens`` read from the local peer buffer
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


def _stage_key(dp_rank: int, pp_rank: int, tp_rank: int) -> str:
    return f"{dp_rank}:{pp_rank}:{tp_rank}"


def _stage_shard_rank(
    dp_rank: int, pp_rank: int, tp_rank: int, pp_size: int, tp_size: int
) -> int:
    return (dp_rank * pp_size + pp_rank) * tp_size + tp_rank


def _parse_target_spec(
    target_spec: Any, default_port_base: int
) -> tuple[Optional[str], Optional[int]]:
    if not target_spec:
        return None, None
    if isinstance(target_spec, str):
        host, port_str = target_spec.rsplit(":", 1)
        return host, int(port_str)
    if isinstance(target_spec, dict):
        host = target_spec.get("host")
        if not host:
            return None, None
        return host, int(target_spec.get("port", default_port_base))
    raise TypeError(f"Unsupported peer target spec: {target_spec!r}")


class PeerCacheStorage(HiCacheStorage):
    """L3 storage backend that replicates KV cache to a peer worker's DRAM.

    Each worker runs a PeerCacheServer (TCP listener + radix buffer) and
    a PeerCacheClient (TCP sender to the ring neighbor).

    Write path: host L2 → serialize → TCP (with token context) → neighbor's trie
    Read path:  local trie → deserialize → host L2 (failover only)
    """

    supports_token_matching = True

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host):
        cfg = storage_config.extra_config or {}
        peer_port_base = cfg.get(
            "peer_port",
            int(os.environ.get("SGLANG_PEER_CACHE_PORT", "9000")),
        )
        peer_targets = cfg.get("peer_targets") or {}
        peer_target = cfg.get(
            "peer_url",
            os.environ.get("SGLANG_PEER_TARGET_URL", ""),
        )
        buffer_gb = cfg.get(
            "peer_buffer_size_gb",
            float(os.environ.get("SGLANG_PEER_BUFFER_SIZE_GB", "8")),
        )

        dp_rank = storage_config.dp_rank
        dp_size = storage_config.dp_size
        tp_rank = storage_config.tp_rank
        tp_size = storage_config.tp_size
        pp_rank = storage_config.pp_rank
        pp_size = storage_config.pp_size
        local_shard_rank = _stage_shard_rank(dp_rank, pp_rank, tp_rank, pp_size, tp_size)

        peer_port = peer_port_base + local_shard_rank

        self.mem_pool_host = mem_pool_host
        self.page_size = mem_pool_host.page_size

        self.local_server = PeerCacheServer(
            peer_port, buffer_gb, page_size=self.page_size
        )
        self.local_server.start()

        backup_dp_rank = (dp_rank + 1) % dp_size if dp_size > 1 else None
        actual_target = ""
        if backup_dp_rank is not None:
            target_stage_key = _stage_key(backup_dp_rank, pp_rank, tp_rank)
            target_spec = peer_targets.get(target_stage_key)
            if target_spec is None:
                target_spec = peer_targets.get(str(backup_dp_rank))
            if target_spec is not None:
                host, target_port_base = _parse_target_spec(target_spec, peer_port_base)
            elif peer_target:
                host, target_port_base = _parse_target_spec(peer_target, peer_port_base)
            else:
                host, target_port_base = (None, None)

            if host and target_port_base is not None:
                target_shard_rank = _stage_shard_rank(
                    backup_dp_rank, pp_rank, tp_rank, pp_size, tp_size
                )
                target_port = target_port_base + target_shard_rank
                actual_target = f"{host}:{target_port}"

        if actual_target:
            self.peer_client = PeerCacheClient(actual_target)
            logger.info(
                "PeerCacheStorage[dp_rank=%d,pp_rank=%d,tp_rank=%d,local_shard=%d]: "
                "port %d, replicating to %s, buffer %.1f GB, page_size=%d",
                dp_rank,
                pp_rank,
                tp_rank,
                local_shard_rank,
                peer_port,
                actual_target,
                buffer_gb,
                self.page_size,
            )
        else:
            self.peer_client = None
            logger.info(
                "PeerCacheStorage[dp_rank=%d,pp_rank=%d,tp_rank=%d,local_shard=%d]: "
                "port %d, no peer_url configured, receive-only mode",
                dp_rank,
                pp_rank,
                tp_rank,
                local_shard_rank,
                peer_port,
            )

    # ------------------------------------------------------------------
    # Token-oriented API used by HiCacheController for peer storage.
    # ------------------------------------------------------------------

    def match_prefix_from(
        self, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        """Return ``(data_count, navigable_range)`` from *start_page*.

        This queries the LOCAL radix buffer (populated by the remote peer's
        backup TCP writes).  *data_count* excludes evicted pages while
        *navigable_range* includes them (use as *count* in :meth:`get_pages_tokens`).
        """
        data_count, navigable = self.local_server.match_prefix_from(
            token_ids, start_page
        )
        buf_pages = self.local_server.buffer.page_count
        logger.debug(
            "PeerCacheStorage.match_prefix_from: start_page=%d, data=%d, "
            "range=%d, buffer_pages=%d, query_tokens=%d",
            start_page,
            data_count,
            navigable,
            buf_pages,
            len(token_ids),
        )
        return data_count, navigable

    def get_pages_tokens(
        self,
        token_ids: List[int],
        start_page: int,
        count: int,
        host_indices: torch.Tensor,
    ) -> List[bool]:
        """Retrieve pages from local radix buffer and write to host memory."""
        raw_pages = self.local_server.get_pages_by_tokens(
            token_ids, start_page, count
        )

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
                    logger.warning(
                        "PeerCacheStorage.get_pages_tokens page %d: %s", i, exc
                    )
                    results.append(False)
            else:
                results.append(False)
        return results

    # ------------------------------------------------------------------
    # Token-aware v1 interface used by the generic HiCache backup path.
    # ------------------------------------------------------------------

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Write path: serialize host pages + token context, TCP send to peer."""
        if self.peer_client is None:
            return [False] * len(keys)

        if extra_info is None or extra_info.full_token_ids is None:
            logger.warning(
                "PeerCacheStorage.batch_set_v1: no token context, skipping %d pages",
                len(keys),
            )
            return [False] * len(keys)

        pages: List[bytes] = []
        for i in range(len(keys)):
            idx = host_indices[i * self.page_size].item()
            page_tensor = self.mem_pool_host.get_data_page(idx, flat=True)
            raw = page_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            pages.append(raw)

        results = self.peer_client.put_pages_tokens(
            extra_info.full_token_ids,
            extra_info.page_start,
            pages,
        )
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
        """Read path (token-based): restore pages from local trie into host."""
        if extra_info is None or extra_info.full_token_ids is None:
            logger.warning("PeerCacheStorage.batch_get_v1: no token context")
            return [False] * len(keys)

        return self.get_pages_tokens(
            extra_info.full_token_ids,
            extra_info.page_start,
            len(keys),
            host_indices,
        )

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        """Token-based prefix match on local radix buffer."""
        if extra_info is None or extra_info.full_token_ids is None:
            return 0

        data_count, _navigable = self.match_prefix_from(
            extra_info.full_token_ids, extra_info.page_start
        )
        return data_count

    # ------------------------------------------------------------------
    # Legacy abstract methods (required by HiCacheStorage ABC)
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Optional[torch.Tensor]]:
        return [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return False

    def clear(self) -> None:
        self.local_server.clear()

    def get_stats(self):
        return None
