# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
RemoteBackupStorage: HiCacheStorage backend that uses a centralized remote backup server
for decode KV cache backup and restore.

Each worker connects via RemoteBackupClient to the remote backup server, which maintains
a global radix buffer written by (dp_rank, token_ids). Read-side match and prefetch
scan all DP-rank roots, allowing any worker to restore KV cache backed up by any
rank after a failure.

Write path (backup): host L2 → serialize → TCP → remote backup server radix buffer
Read path (restore): local query → TCP → remote backup server → TCP response → host L2
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.storage.remote_backup.remote_backup_server import (
    LEASE_REASON_ABORT,
    LEASE_REASON_FAILOVER_OLD,
    LEASE_REASON_FAILOVER_SUPERSEDE,
    LEASE_REASON_NORMAL,
    REMOTE_BACKUP_ALL_DP_RANKS,
    REMOTE_BACKUP_NO_SOURCE_DP_RANK,
    RemoteBackupClient,
    RemoteBackupServer,
)

# When True, writes and reads use the request-namespace API as primary path.
_USE_REQUEST_NAMESPACE = True

logger = logging.getLogger(__name__)

_REASON_STR_TO_CODE = {
    "normal": LEASE_REASON_NORMAL,
    "abort": LEASE_REASON_ABORT,
    "failover_old": LEASE_REASON_FAILOVER_OLD,
    "failover_supersede": LEASE_REASON_FAILOVER_SUPERSEDE,
}


def _dp_rank_label(dp_rank: Optional[int]) -> str:
    if dp_rank is None:
        return "none"
    if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
        return "all"
    if dp_rank == REMOTE_BACKUP_NO_SOURCE_DP_RANK:
        return "none"
    return str(dp_rank)


class RemoteBackupStorage(HiCacheStorage):
    """Remote host backup storage backend using a centralized server for KV cache.

    Each worker:
      - Runs a local RemoteBackupServer (TCP listener + radix buffer) so the
        server process can be co-located with workers (for zero-copy host
        memory access via shared memory, or a separate dedicated server).
      - Connects via RemoteBackupClient to the remote backup server to back up decode KV pages.

    The remote backup server stores pages by dp_rank and token IDs, while read-side
    lookup scans every dp_rank root so any worker can prefetch KV cache after a failure.

    Mode selection:
      - "server": this instance is the server (listens, serves prefetch, no backup client)
      - "client": this instance is a worker (backs up to remote server)
      - "combined": this instance is both (starts server + connects to self as client)

    Default mode is "combined" — the recommended configuration where the server
    is co-located with one of the workers.
    """

    supports_token_matching = True

    @staticmethod
    def _parse_remote_backup_urls(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("remote_backup_urls JSON must be a list")
                return [str(x) for x in parsed if str(x).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        raise TypeError(
            f"remote_backup_urls must be list or str, got {type(value).__name__}"
        )

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host):
        cfg = storage_config.extra_config or {}
        self.mem_pool_host = mem_pool_host
        self.page_size = mem_pool_host.page_size
        self.local_server: Optional[RemoteBackupServer] = None
        self.clients_by_rank: dict[int, RemoteBackupClient] = {}

        remote_backup_port = cfg.get(
            "remote_backup_port",
            int(os.environ.get("SGLANG_REMOTE_BACKUP_PORT", "30000")),
        )
        remote_backup_url = cfg.get(
            "remote_backup_url",
            os.environ.get("SGLANG_REMOTE_BACKUP_URL", ""),
        )
        remote_backup_urls = self._parse_remote_backup_urls(
            cfg.get(
                "remote_backup_urls",
                os.environ.get("SGLANG_REMOTE_BACKUP_URLS", ""),
            )
        )
        buffer_gb = cfg.get(
            "remote_backup_buffer_size_gb",
            float(os.environ.get("SGLANG_REMOTE_BACKUP_BUFFER_SIZE_GB", "32")),
        )
        mode = cfg.get("remote_backup_mode", os.environ.get("SGLANG_REMOTE_BACKUP_MODE", "combined"))
        self._dp_rank = storage_config.dp_rank

        if mode in ("server", "combined"):
            self.local_server = RemoteBackupServer(
                remote_backup_port, buffer_gb, page_size=self.page_size
            )
            self.local_server.start()

        # Client connection to remote server
        if mode == "client":
            if not remote_backup_urls and not remote_backup_url:
                raise ValueError(
                    "remote_backup_mode=client requires remote_backup_url(s) in extra_config or env"
                )
            if remote_backup_urls:
                self.clients_by_rank = {
                    rank: RemoteBackupClient(url)
                    for rank, url in enumerate(remote_backup_urls)
                }
                default_rank = int(self._dp_rank or 0)
                self.client = self.clients_by_rank.get(default_rank) or next(
                    iter(self.clients_by_rank.values())
                )
                logger.info(
                    "RemoteBackupStorage[dp_rank=%d]: sharded client mode, %d backup shards, "
                    "local rank routes to %s, buffer %.1f GB, page_size=%d",
                    self._dp_rank,
                    len(self.clients_by_rank),
                    remote_backup_urls[min(default_rank, len(remote_backup_urls) - 1)],
                    buffer_gb,
                    self.page_size,
                )
            else:
                self.client = RemoteBackupClient(remote_backup_url)
                logger.info(
                    "RemoteBackupStorage[dp_rank=%d]: client mode, connected to remote backup server %s, "
                    "local port %d, buffer %.1f GB, page_size=%d",
                    self._dp_rank,
                    remote_backup_url,
                    remote_backup_port,
                    buffer_gb,
                    self.page_size,
                )
        else:
            # server or combined mode: connect back to self
            self_url = f"localhost:{remote_backup_port}"
            self.client = RemoteBackupClient(self_url)
            logger.info(
                "RemoteBackupStorage[dp_rank=%d]: %s mode, local port %d, "
                "client connecting to %s, buffer %.1f GB, page_size=%d",
                self._dp_rank,
                mode,
                remote_backup_port,
                self_url,
                buffer_gb,
                self.page_size,
            )

    # ------------------------------------------------------------------
    # Token-oriented API used by HiCacheController for prefetch/restore.
    # ------------------------------------------------------------------

    def _resolve_lookup_dp_rank(self, lookup_dp_rank: Optional[int]) -> int:
        # In sharded mode, a specific source rank maps to a specific remote
        # backup shard, so preserve the hint when we have one.
        if self.clients_by_rank and lookup_dp_rank not in (None, REMOTE_BACKUP_ALL_DP_RANKS):
            return int(lookup_dp_rank)
        return REMOTE_BACKUP_ALL_DP_RANKS

    def _get_client_for_rank(self, dp_rank: int) -> Optional[RemoteBackupClient]:
        if self.clients_by_rank:
            return self.clients_by_rank.get(int(dp_rank))
        return self.client

    def _select_best_shard_for_tokens(
        self, token_ids: List[int], start_page: int
    ) -> tuple[int, int, int]:
        best_source = REMOTE_BACKUP_NO_SOURCE_DP_RANK
        best_data = 0
        best_navigable = 0
        for rank, client in self.clients_by_rank.items():
            source_dp_rank, data_count, navigable = client.match_prefix_from_with_source(
                rank, token_ids, start_page
            )
            if data_count == 0 and navigable == 0:
                continue
            if (
                data_count > best_data
                or (data_count == best_data and navigable > best_navigable)
                or (
                    data_count == best_data
                    and navigable == best_navigable
                    and (
                        best_source == REMOTE_BACKUP_NO_SOURCE_DP_RANK
                        or source_dp_rank < best_source
                    )
                )
            ):
                best_source = source_dp_rank
                best_data = data_count
                best_navigable = navigable
        return best_source, best_data, best_navigable

    def match_prefix_from(
        self,
        token_ids: List[int],
        start_page: int = 0,
        lookup_dp_rank: Optional[int] = None,
    ) -> tuple:
        """Return ``(data_count, navigable_range)`` from *start_page*.

        This queries the LOCAL radix buffer (populated by remote worker backup TCP writes).
        """
        query_dp_rank = self._resolve_lookup_dp_rank(lookup_dp_rank)
        source_dp_rank = REMOTE_BACKUP_NO_SOURCE_DP_RANK
        if self.local_server is not None:
            source_dp_rank, data_count, navigable = self.local_server.match_prefix_from_with_source(
                query_dp_rank, token_ids, start_page
            )
            buf_pages = self.local_server.buffer.page_count
        elif self.clients_by_rank:
            if query_dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
                source_dp_rank, data_count, navigable = self._select_best_shard_for_tokens(
                    token_ids, start_page
                )
            else:
                client = self._get_client_for_rank(query_dp_rank)
                if client is None:
                    return 0, 0
                source_dp_rank, data_count, navigable = client.match_prefix_from_with_source(
                    query_dp_rank, token_ids, start_page
                )
            buf_pages = -1
        elif self.client is not None:
            source_dp_rank, data_count, navigable = self.client.match_prefix_from_with_source(
                query_dp_rank, token_ids, start_page
            )
            buf_pages = -1
        else:
            return 0, 0
        logger.debug(
            "RemoteBackupStorage.match_prefix_from: requested_dp_rank=%s, query_dp_rank=%s, "
            "source_dp_rank=%s, local_dp_rank=%d, start_page=%d, data=%d, range=%d, "
            "buffer_pages=%d, query_tokens=%d",
            _dp_rank_label(lookup_dp_rank),
            _dp_rank_label(query_dp_rank),
            _dp_rank_label(source_dp_rank),
            self._dp_rank,
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
        lookup_dp_rank: Optional[int] = None,
    ) -> List[bool]:
        """Retrieve pages from local radix buffer and write to host memory."""
        query_dp_rank = self._resolve_lookup_dp_rank(lookup_dp_rank)
        if self.local_server is not None:
            raw_pages = self.local_server.get_pages_by_tokens(
                query_dp_rank, token_ids, start_page, count
            )
        elif self.clients_by_rank:
            if query_dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
                source_dp_rank, _data_count, _navigable = self._select_best_shard_for_tokens(
                    token_ids, start_page
                )
                if source_dp_rank == REMOTE_BACKUP_NO_SOURCE_DP_RANK:
                    raw_pages = [None] * count
                else:
                    client = self._get_client_for_rank(source_dp_rank)
                    raw_pages = (
                        client.get_pages_by_tokens(
                            source_dp_rank, token_ids, start_page, count
                        )
                        if client is not None
                        else [None] * count
                    )
            else:
                client = self._get_client_for_rank(query_dp_rank)
                raw_pages = (
                    client.get_pages_by_tokens(
                        query_dp_rank, token_ids, start_page, count
                    )
                    if client is not None
                    else [None] * count
                )
        elif self.client is not None:
            raw_pages = self.client.get_pages_by_tokens(
                query_dp_rank, token_ids, start_page, count
            )
        else:
            return [False] * count

        none_count = sum(1 for r in raw_pages if r is None)
        if none_count > 0:
            logger.warning(
                "RemoteBackupStorage.get_pages_tokens: requested_dp_rank=%s, query_dp_rank=%s, local_dp_rank=%d, "
                "start_page=%d, count=%d, none=%d/%d",
                _dp_rank_label(lookup_dp_rank),
                _dp_rank_label(query_dp_rank),
                self._dp_rank,
                start_page,
                count,
                none_count,
                len(raw_pages),
            )

        return self._deserialize_pages_to_host(raw_pages, host_indices)

    # ------------------------------------------------------------------
    # Request-namespace read API (for failover/retry restore)
    # ------------------------------------------------------------------

    def match_prefix_by_request(
        self,
        dp_rank: int,
        rid: str,
        generation: int,
    ) -> tuple:
        """Query request namespace for available page range.

        Returns ``(contiguous_pages, total_pages)`` — contiguous is the max
        range starting from page 0 that can be restored without gaps.
        """
        if self.local_server is not None:
            return self.local_server.query_request_range(dp_rank, rid, generation)
        client = self._get_client_for_rank(dp_rank)
        if client is not None:
            return client.query_request_range(dp_rank, rid, generation)
        return 0, 0

    def get_pages_by_request(
        self,
        dp_rank: int,
        rid: str,
        generation: int,
        page_start: int,
        count: int,
        host_indices: torch.Tensor,
    ) -> List[bool]:
        """Retrieve pages from the request namespace and write to host memory."""
        if self.local_server is not None:
            raw_pages = self.local_server.get_pages_by_request(
                dp_rank, rid, generation, page_start, count
            )
        else:
            client = self._get_client_for_rank(dp_rank)
            if client is None:
                return [False] * count
            raw_pages = client.get_pages_by_request(
                dp_rank, rid, generation, page_start, count
            )

        none_count = sum(1 for r in raw_pages if r is None)
        if none_count > 0:
            logger.warning(
                "RemoteBackupStorage.get_pages_by_request: dp_rank=%d, rid=%s, "
                "gen=%d, page_start=%d, count=%d, none=%d/%d",
                dp_rank,
                rid[:16] if rid else "",
                generation,
                page_start,
                count,
                none_count,
                len(raw_pages),
            )

        return self._deserialize_pages_to_host(raw_pages, host_indices)

    # ------------------------------------------------------------------
    # Request lifecycle hooks (request-aware retention)
    # ------------------------------------------------------------------

    def start_request(self, request_id: str, dp_rank: int, generation: int) -> None:
        """Notify server that a request is starting; creates a lease.

        Called by HiCacheController.notify_request_start via the storage backend
        hook. Uses in-process call when local_server is set, TCP otherwise.
        """
        if self.local_server is not None and self.local_server.request_aware:
            self.local_server.start_request(dp_rank, request_id, generation)
        else:
            client = self._get_client_for_rank(dp_rank)
            if client is not None:
                client.start_request(request_id, dp_rank, generation)

    def finish_request(
        self, request_id: str, dp_rank: int, generation: int, reason: str
    ) -> None:
        """Notify server that a request is done; enqueues lease for GC.

        ``reason`` is a string: "normal", "abort", "failover_old", "failover_supersede".
        """
        reason_code = _REASON_STR_TO_CODE.get(reason, LEASE_REASON_NORMAL)
        if self.local_server is not None and self.local_server.request_aware:
            self.local_server.finish_request(dp_rank, request_id, generation, reason_code)
        else:
            client = self._get_client_for_rank(dp_rank)
            if client is not None:
                client.finish_request(request_id, dp_rank, generation, reason_code)

    def _serialize_pages(
        self, keys: List[str], host_indices: torch.Tensor
    ) -> List[bytes]:
        """Serialize host KV pages to raw bytes."""
        pages: List[bytes] = []
        for i in range(len(keys)):
            idx = host_indices[i * self.page_size].item()
            page_tensor = self.mem_pool_host.get_data_page(idx, flat=True)
            raw = page_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            pages.append(raw)
        return pages

    def _deserialize_pages_to_host(
        self, raw_pages: List[Optional[bytes]], host_indices: torch.Tensor
    ) -> List[bool]:
        """Write raw page bytes into host memory pool. Returns per-page success."""
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
                        "RemoteBackupStorage._deserialize_pages_to_host page %d: %s",
                        i,
                        exc,
                    )
                    results.append(False)
            else:
                results.append(False)
        return results

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Write path: serialize host pages and send to remote backup server.

        When ``_USE_REQUEST_NAMESPACE`` is enabled and ``extra_info.request_id``
        is set, pages are written to the request namespace (primary path).
        Otherwise falls back to the legacy token-trie v1/v2 write.
        """
        if self.client is None and self.local_server is None:
            return [False] * len(keys)

        rid = getattr(extra_info, "request_id", None) if extra_info else None
        generation = getattr(extra_info, "request_generation", 0) if extra_info else 0
        page_start = getattr(extra_info, "page_start", 0) if extra_info else 0

        # Request-namespace primary write path
        if _USE_REQUEST_NAMESPACE and rid:
            pages = self._serialize_pages(keys, host_indices)
            if self.local_server is not None:
                results = self.local_server.put_pages_by_request(
                    self._dp_rank, rid, generation, page_start, pages
                )
            else:
                client = self._get_client_for_rank(int(self._dp_rank or 0))
                if client is None:
                    return [False] * len(keys)
                results = client.put_pages_by_request(
                    self._dp_rank, rid, generation, page_start, pages
                )
            ok_count = sum(1 for r in results if r)
            if ok_count < len(keys):
                logger.warning(
                    "RemoteBackupStorage.batch_set_v1(ns): only %d/%d pages sent",
                    ok_count,
                    len(keys),
                )
            return results

        # Legacy token-trie fallback
        if extra_info is None or extra_info.full_token_ids is None:
            logger.warning(
                "RemoteBackupStorage.batch_set_v1: no token context, skipping %d pages",
                len(keys),
            )
            return [False] * len(keys)

        pages = self._serialize_pages(keys, host_indices)

        if rid:
            if self.local_server is not None and self.local_server.request_aware:
                results = self.local_server.insert_pages_v2(
                    self._dp_rank,
                    extra_info.full_token_ids,
                    extra_info.page_start,
                    pages,
                    rid,
                    generation,
                )
            else:
                client = self._get_client_for_rank(int(self._dp_rank or 0))
                if client is None:
                    return [False] * len(keys)
                results = client.put_pages_tokens_v2(
                    self._dp_rank,
                    extra_info.full_token_ids,
                    extra_info.page_start,
                    pages,
                    rid,
                    generation,
                )
        else:
            client = self._get_client_for_rank(int(self._dp_rank or 0))
            if client is None:
                return [False] * len(keys)
            results = client.put_pages_tokens(
                self._dp_rank,
                extra_info.full_token_ids,
                extra_info.page_start,
                pages,
            )

        ok_count = sum(1 for r in results if r)
        if ok_count < len(keys):
            logger.warning(
                "RemoteBackupStorage.batch_set_v1: only %d/%d pages sent successfully",
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
            logger.warning("RemoteBackupStorage.batch_get_v1: no token context")
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
        if self.local_server is not None:
            self.local_server.clear()

    def get_stats(self):
        return None
