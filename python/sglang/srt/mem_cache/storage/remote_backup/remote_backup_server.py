# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
RemoteBackupServer: centralized TCP service that accepts KV cache pages from
all worker nodes and stores them in a shared DRAM radix buffer.

RemoteBackupClient: TCP client that sends KV cache pages to the remote backup server.

RemoteBackupStorage: HiCacheStorage backend that uses the remote backup server for
decode KV cache backup and restore.

Architecture
------------
Remote host backup uses a centralized 1:N topology:

    Worker-0 ──┐
    Worker-1 ──┼── TCP ──► RemoteBackupServer (radix buffer in DRAM)
    Worker-N ──┘         Port: remote_backup_port_base + shard_rank

The remote backup server maintains a global radix buffer written by
(dp_rank, token_ids). Prefetch queries can scan all DP-rank roots so any worker
can restore KV cache backed up by any other rank after a failure. Prefetch uses
the standard HiCacheStorage token-based API (match_prefix_from / get_pages_tokens).

Request-aware retention
-----------------------
When ``remote_backup_request_aware=True`` (default), the server tracks per-request
lease metadata so that pages owned by a running request are protected from LRU
eviction.  Two separate LRU queues are maintained:

  * ``_inactive_lru``: pages with ``active_refcnt == 0`` — evicted first.
  * ``_active_lru``: pages held by at least one active request — evicted only as
    a last resort (with a warning).

Async GC is performed by a background daemon thread that processes finished leases
in small lock-held chunks to minimise tail latency impact on the hot write path.

Binary protocol over TCP (token-based):
  CMD_PUT_TOKENS (4):  [legacy – no lease metadata]
    Send: [cmd:1B][seq_len:4B][page_start:4B][page_count:4B][dp_rank:1B]
          [tokens: seq_len * 4B little-endian uint32]
          Per page: [data_len:4B][data]
    Recv: [status:1B][count:4B][ok:1B * count]

  CMD_MATCH_PREFIX (5):
    Send: [cmd:1B][seq_len:4B][start_page:4B][dp_rank:1B]
          [tokens: seq_len * 4B little-endian uint32]
    Recv: [status:1B][source_dp_rank:4B][data_count:4B][navigable:4B]

  CMD_GET_PAGES (6):
    Send: [cmd:1B][seq_len:4B][start_page:4B][count:4B][dp_rank:1B]
          [tokens: seq_len * 4B little-endian uint32]
    Recv: [status:1B][count:4B] per page: [data_len:4B][data]

  CMD_HEALTH (255): [cmd:1B]  Recv: [0x01:1B]

  CMD_PUT_TOKENS_V2 (7):  [with lease metadata]
    Send: [cmd:1B][seq_len:4B][page_start:4B][page_count:4B][dp_rank:1B]
          [rid_len:2B][rid: rid_len bytes][generation:8B little-endian]
          [tokens: seq_len * 4B little-endian uint32]
          Per page: [data_len:4B][data]
    Recv: [status:1B][count:4B][ok:1B * count]

  CMD_REQUEST_START (8):
    Send: [cmd:1B][dp_rank:1B][rid_len:2B][rid: rid_len bytes][generation:8B]
    Recv: [status:1B]  (0=ok, 1=stale, 2=superseded)

  CMD_REQUEST_FINISH (9):
    Send: [cmd:1B][dp_rank:1B][rid_len:2B][rid: rid_len bytes]
          [generation:8B][reason:1B]
    Recv: [status:1B]
    reason: 0=normal, 1=abort, 2=failover_old, 3=failover_supersede
"""

from __future__ import annotations

import array
import logging
import queue
import socket
import socketserver
import struct
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

CMD_PUT_TOKENS = 4
CMD_MATCH_PREFIX = 5
CMD_GET_PAGES = 6
CMD_HEALTH = 255

# Request-aware commands (new)
CMD_PUT_TOKENS_V2 = 7
CMD_REQUEST_START = 8
CMD_REQUEST_FINISH = 9

DECODE_DEBUG_PAGE_START_THRESHOLD = 100
REMOTE_BACKUP_ALL_DP_RANKS = 255
REMOTE_BACKUP_NO_SOURCE_DP_RANK = -1
_REMOTE_BACKUP_NO_SOURCE_DP_RANK_WIRE = 0xFFFFFFFF

_ITEM_HDR_FMT = "!I"
_ITEM_HDR_SIZE = struct.calcsize(_ITEM_HDR_FMT)

# ---------------------------------------------------------------------------
# Lease management constants
# ---------------------------------------------------------------------------

_LEASE_STATE_ACTIVE = "active"
_LEASE_STATE_FINISHED = "finished"
_LEASE_STATE_ABORTED = "aborted"
_LEASE_STATE_SUPERSEDED = "superseded"

# CMD_REQUEST_START / CMD_REQUEST_FINISH response status codes
_LEASE_STATUS_OK = 0
_LEASE_STATUS_STALE = 1
_LEASE_STATUS_SUPERSEDED_CODE = 2

# CMD_REQUEST_FINISH reason codes
LEASE_REASON_NORMAL = 0
LEASE_REASON_ABORT = 1
LEASE_REASON_FAILOVER_OLD = 2
LEASE_REASON_FAILOVER_SUPERSEDE = 3

# GC tuning
_GC_CHUNK_PAGES = 4096
_GC_HIGH_WATER_LEASE_QUEUE = 1024
_GC_CHUNK_PAGES_HIGH_WATER = 16384
_LEASE_IDLE_TTL_S = 600.0
_LEASE_TTL_CHECK_INTERVAL_S = 60.0


# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------


def _recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly *n* bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def _pack_tokens(token_ids: List[int]) -> bytes:
    """Serialize token IDs to little-endian uint32 bytes."""
    arr = array.array("I", token_ids)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def _unpack_tokens(data: bytes) -> List[int]:
    """Deserialize little-endian uint32 bytes to token IDs."""
    arr = array.array("I")
    arr.frombytes(data)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist()


def _looks_like_decode_page_start(page_start: int) -> bool:
    """Best-effort heuristic used only for debug logging."""
    return page_start >= DECODE_DEBUG_PAGE_START_THRESHOLD


def _dp_rank_label(dp_rank: int) -> str:
    if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
        return "all"
    if dp_rank == REMOTE_BACKUP_NO_SOURCE_DP_RANK:
        return "none"
    return str(dp_rank)


# ---------------------------------------------------------------------------
# Lease entry dataclass
# ---------------------------------------------------------------------------


@dataclass
class RequestLeaseEntry:
    """Tracks KV pages written by a single request generation.

    ``pages`` holds the page_ids that were written under this lease.
    When the lease is finished/aborted, the GC worker decrements
    ``active_refcnt`` on each referenced node and moves pages from
    ``_active_lru`` to ``_inactive_lru`` when the count reaches zero.
    """

    lease_id: int
    rid: str
    dp_rank: int
    generation: int
    state: str = _LEASE_STATE_ACTIVE
    pages: Set[int] = field(default_factory=set)
    last_touch_ts: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Trie data structure for the remote backup radix buffer
# ---------------------------------------------------------------------------


class _TrieNode:
    """A node in the remote backup radix buffer trie.

    Each level corresponds to one KV page (``page_size`` tokens).
    For page_size == 1 the edge key is a single token ID (int);
    for page_size > 1 it is a tuple of token IDs.

    Lease tracking fields
    ---------------------
    lease_refs   : set of lease_ids that currently reference this node's KV data.
                   None when no leases have ever touched this node (lazy init).
    active_refcnt: mirrors len(lease_refs) for O(1) check whether this page is
                   protected by at least one running request.
    parent       : parent _TrieNode (None for root nodes).
    edge_key     : key used in parent.children to reach this node.
    """

    __slots__ = (
        "children",
        "kv_data",
        "data_size",
        "page_id",
        "lease_refs",
        "active_refcnt",
        "parent",
        "edge_key",
    )

    def __init__(self):
        self.children: Dict = {}
        self.kv_data: Optional[bytes] = None
        self.data_size: int = 0
        self.page_id: int = -1
        self.lease_refs: Optional[Set[int]] = None
        self.active_refcnt: int = 0
        self.parent: Optional[_TrieNode] = None
        self.edge_key = None


class RemoteBackupRadixBuffer:
    """Token-indexed radix buffer for centralized remote backup KV cache pages.

    A trie where each level corresponds to one page (``page_size`` tokens).
    Indexed by (dp_rank, full_token_ids) for multi-worker access.

    Two LRU queues are maintained when request-aware retention is active:
    - ``_inactive_lru``: pages with active_refcnt == 0 — evicted first.
    - ``_active_lru``  : pages held by running requests — evicted last.
    ``_node_by_page_id`` provides O(1) lookup for the GC worker.
    """

    def __init__(self, page_size: int, max_size_bytes: int):
        self.root: Dict[int, _TrieNode] = {}  # dp_rank -> root trie node
        self.page_size = page_size
        self.max_size = max_size_bytes
        self.current_size: int = 0
        self._page_counter: int = 0
        # Inactive (no active lease) — evicted first
        self._inactive_lru: OrderedDict[int, _TrieNode] = OrderedDict()
        # Active (held by ≥1 running request) — last-resort eviction
        self._active_lru: OrderedDict[int, _TrieNode] = OrderedDict()
        # Reverse map for O(1) GC node lookup
        self._node_by_page_id: Dict[int, _TrieNode] = {}
        # Set by insert_pages; used by server to update lease.pages
        self._last_page_ids: List[int] = []
        # Counters
        self._active_evict_count: int = 0

    @property
    def page_count(self) -> int:
        return len(self._inactive_lru) + len(self._active_lru)

    def _edge_key(self, token_ids: List[int], page_idx: int):
        start = page_idx * self.page_size
        if self.page_size == 1:
            return token_ids[start]
        return tuple(token_ids[start : start + self.page_size])

    def _get_root(self, dp_rank: int) -> _TrieNode:
        if dp_rank not in self.root:
            self.root[dp_rank] = _TrieNode()
        return self.root[dp_rank]

    def _iter_query_roots(self, dp_rank: int):
        if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
            return sorted(self.root.items())
        node = self.root.get(dp_rank)
        if node is None:
            return []
        return [(dp_rank, node)]

    def _collapse_source_ranks(self, sources) -> int:
        if not sources:
            return REMOTE_BACKUP_NO_SOURCE_DP_RANK
        if len(sources) == 1:
            return next(iter(sources))
        return REMOTE_BACKUP_ALL_DP_RANKS

    # ------------------------------------------------------------------
    # LRU helpers
    # ------------------------------------------------------------------

    def _touch_page(self, node: _TrieNode) -> None:
        """Move a page to the MRU end of its respective LRU queue."""
        if node.page_id < 0:
            return
        if node.active_refcnt > 0:
            if node.page_id in self._active_lru:
                self._active_lru.move_to_end(node.page_id)
        else:
            if node.page_id in self._inactive_lru:
                self._inactive_lru.move_to_end(node.page_id)

    def _remove_from_lrus(self, node: _TrieNode) -> None:
        """Remove a node from whichever LRU it currently occupies."""
        if node.page_id < 0:
            return
        self._inactive_lru.pop(node.page_id, None)
        self._active_lru.pop(node.page_id, None)
        self._node_by_page_id.pop(node.page_id, None)

    # ------------------------------------------------------------------
    # Trie traversal helpers
    # ------------------------------------------------------------------

    def _nodes_after_prefix(
        self, dp_rank: int, token_ids: List[int], start_page: int
    ) -> List[tuple]:
        nodes = []
        for source_dp_rank, root in self._iter_query_roots(dp_rank):
            node = root
            prefix_found = True
            for p in range(start_page):
                key = self._edge_key(token_ids, p)
                child = node.children.get(key)
                if child is None:
                    prefix_found = False
                    break
                node = child
            if prefix_found:
                nodes.append((source_dp_rank, node))
        return nodes

    def _match_prefix_from_node(
        self, node: _TrieNode, token_ids: List[int], start_page: int
    ) -> tuple:
        total_pages = len(token_ids) // self.page_size

        for p in range(start_page):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                return 0, 0
            node = child

        data_count = 0
        navigable = 0
        data_prefix_open = True
        for p in range(start_page, total_pages):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                break
            navigable += 1
            if child.kv_data is not None:
                self._touch_page(child)
                if data_prefix_open:
                    data_count += 1
            else:
                data_prefix_open = False
            node = child

        return data_count, navigable

    def _match_prefix_from_all_sources(
        self, token_ids: List[int], start_page: int
    ) -> tuple:
        total_pages = len(token_ids) // self.page_size
        candidates = self._nodes_after_prefix(
            REMOTE_BACKUP_ALL_DP_RANKS, token_ids, start_page
        )
        data_count = 0
        navigable = 0
        data_prefix_open = True
        data_sources = set()
        structural_sources = set()

        for p in range(start_page, total_pages):
            key = self._edge_key(token_ids, p)
            next_candidates = []
            page_has_data = False

            for source_dp_rank, node in candidates:
                child = node.children.get(key)
                if child is None:
                    continue
                next_candidates.append((source_dp_rank, child))
                structural_sources.add(source_dp_rank)
                if child.kv_data is not None:
                    self._touch_page(child)
                    page_has_data = True
                    data_sources.add(source_dp_rank)

            if not next_candidates:
                break
            navigable += 1
            if page_has_data and data_prefix_open:
                data_count += 1
            elif not page_has_data:
                data_prefix_open = False
            candidates = next_candidates

        source_dp_rank = self._collapse_source_ranks(data_sources or structural_sources)
        return source_dp_rank, data_count, navigable

    def _get_pages_from_all_sources(
        self, token_ids: List[int], start_page: int, count: int
    ) -> List[Optional[bytes]]:
        total_pages = len(token_ids) // self.page_size
        candidates = self._nodes_after_prefix(
            REMOTE_BACKUP_ALL_DP_RANKS, token_ids, start_page
        )
        if not candidates:
            return [None] * count

        results: List[Optional[bytes]] = []
        for i in range(count):
            p = start_page + i
            if p >= total_pages:
                results.extend([None] * (count - len(results)))
                break

            key = self._edge_key(token_ids, p)
            next_candidates = []
            page_data: Optional[bytes] = None

            for source_dp_rank, node in candidates:
                child = node.children.get(key)
                if child is None:
                    continue
                next_candidates.append((source_dp_rank, child))
                if page_data is None and child.kv_data is not None:
                    self._touch_page(child)
                    page_data = child.kv_data

            if not next_candidates:
                results.extend([None] * (count - len(results)))
                break

            results.append(page_data)
            candidates = next_candidates

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _insert_total: int = 0
    _insert_decode_total: int = 0

    def insert_pages(
        self,
        dp_rank: int,
        full_token_ids: List[int],
        page_start: int,
        kv_pages: List[bytes],
        lease_id: Optional[int] = None,
    ) -> List[bool]:
        """Insert KV pages into the trie.

        ``full_token_ids`` is the complete token sequence up to (and including)
        the tokens covered by the last page in *kv_pages*.  Intermediate nodes
        for the prefix ``[0, page_start)`` are created automatically.

        When ``lease_id`` is provided, each inserted page is associated with
        that lease: ``node.lease_refs.add(lease_id)`` and the page is placed in
        ``_active_lru`` instead of ``_inactive_lru``.  The server populates
        ``lease.pages`` by reading ``self._last_page_ids`` after this call.
        """
        RemoteBackupRadixBuffer._insert_total += 1
        is_decode_debug = _looks_like_decode_page_start(page_start)
        if is_decode_debug:
            RemoteBackupRadixBuffer._insert_decode_total += 1
            if (
                logger.isEnabledFor(logging.DEBUG)
                and RemoteBackupRadixBuffer._insert_decode_total % 50 == 1
            ):
                first_key = self._edge_key(full_token_ids, page_start)
                logger.debug(
                    "RemoteBackupRadixBuffer.insert_pages DECODE #%d (total=%d): "
                    "dp_rank=%d, page_start=%d, pages=%d, tokens=%d, "
                    "first_key=%s, page_count=%d, data_bytes=%d, lease_id=%s",
                    RemoteBackupRadixBuffer._insert_decode_total,
                    RemoteBackupRadixBuffer._insert_total,
                    dp_rank,
                    page_start,
                    len(kv_pages),
                    len(full_token_ids),
                    first_key,
                    self.page_count,
                    self.current_size,
                    lease_id,
                )

        self._last_page_ids = []

        results: List[bool] = []
        node = self._get_root(dp_rank)

        # Navigate / create prefix nodes
        _prefix_existing = 0
        _prefix_created = 0
        for p in range(page_start):
            key = self._edge_key(full_token_ids, p)
            child = node.children.get(key)
            if child is None:
                child = _TrieNode()
                child.parent = node
                child.edge_key = key
                node.children[key] = child
                _prefix_created += 1
            else:
                _prefix_existing += 1
                self._touch_page(child)
            node = child

        # Insert payload pages
        for i, kv_data in enumerate(kv_pages):
            target_page = page_start + i
            key = self._edge_key(full_token_ids, target_page)
            child = node.children.get(key)
            if child is None:
                child = _TrieNode()
                child.parent = node
                child.edge_key = key
                node.children[key] = child

            # Evict old version if present
            if child.page_id >= 0:
                self._remove_from_lrus(child)
                self.current_size -= child.data_size
                # Clear stale lease refs (old page_id is now gone; GC will
                # find nothing in _node_by_page_id for old page_ids in lease.pages)
                child.lease_refs = None
                child.active_refcnt = 0

            child.kv_data = kv_data
            child.data_size = len(kv_data)
            self.current_size += child.data_size

            self._page_counter += 1
            child.page_id = self._page_counter
            self._node_by_page_id[child.page_id] = child
            self._last_page_ids.append(child.page_id)

            if lease_id is not None:
                # Associate with the lease and put in active LRU
                if child.lease_refs is None:
                    child.lease_refs = set()
                child.lease_refs.add(lease_id)
                child.active_refcnt = len(child.lease_refs)
                self._active_lru[child.page_id] = child
            else:
                # No lease: goes to inactive LRU
                self._inactive_lru[child.page_id] = child

            node = child
            results.append(True)

        self._evict_if_needed()
        return results

    def match_prefix_from(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        _source_dp_rank, data_count, navigable = self.match_prefix_from_with_source(
            dp_rank, token_ids, start_page
        )
        return data_count, navigable

    def match_prefix_from_with_source(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
            return self._match_prefix_from_all_sources(token_ids, start_page)

        best_source = REMOTE_BACKUP_NO_SOURCE_DP_RANK
        best_data = 0
        best_navigable = 0

        for source_dp_rank, root in self._iter_query_roots(dp_rank):
            data_count, navigable = self._match_prefix_from_node(
                root, token_ids, start_page
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

    def get_pages(
        self,
        dp_rank: int,
        token_ids: List[int],
        start_page: int,
        count: int,
    ) -> List[Optional[bytes]]:
        if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
            pages = self._get_pages_from_all_sources(token_ids, start_page, count)
            if all(page is None for page in pages):
                logger.warning(
                    "get_pages: no page data found for all-rank query "
                    "(start_page=%d, count=%d, total_pages=%d, roots=%s)",
                    start_page,
                    count,
                    len(token_ids) // self.page_size,
                    list(self.root.keys()),
                )
            return pages

        total_pages = len(token_ids) // self.page_size
        node = self.root.get(dp_rank)
        if node is None:
            logger.warning(
                "get_pages: dp_rank=%d not in root (keys=%s), returning None*%d",
                dp_rank,
                list(self.root.keys()),
                count,
            )
            return [None] * count

        for p in range(start_page):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                logger.warning(
                    "get_pages: prefix nav failed at page %d/%d "
                    "(dp_rank=%d, start_page=%d, count=%d, total_pages=%d, "
                    "node_children=%d, key=%s)",
                    p,
                    start_page,
                    dp_rank,
                    start_page,
                    count,
                    total_pages,
                    len(node.children),
                    key,
                )
                return [None] * count
            node = child

        results: List[Optional[bytes]] = []
        _none_count = 0
        _data_count = 0
        for i in range(count):
            p = start_page + i
            if p >= total_pages:
                results.extend([None] * (count - len(results)))
                break
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                results.extend([None] * (count - len(results)))
                break
            if child.kv_data is not None:
                self._touch_page(child)
                results.append(child.kv_data)
                _data_count += 1
            else:
                results.append(None)
                _none_count += 1
            node = child

        if _none_count > 0:
            logger.warning(
                "get_pages: dp_rank=%d, start_page=%d, count=%d, "
                "data=%d, evicted=%d, total_pages=%d, page_count=%d, "
                "current_size_mb=%.1f, max_size_mb=%.1f",
                dp_rank,
                start_page,
                count,
                _data_count,
                _none_count,
                total_pages,
                self.page_count,
                self.current_size / (1024 * 1024),
                self.max_size / (1024 * 1024),
            )
        return results

    def clear(self) -> None:
        self.root = {}
        self._inactive_lru.clear()
        self._active_lru.clear()
        self._node_by_page_id.clear()
        self._last_page_ids = []
        self.current_size = 0
        self._page_counter = 0

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict pages to stay within max_size.

        Eviction order:
          1. Inactive pages (active_refcnt == 0) — normal path.
          2. Active pages (active_refcnt > 0) — last resort, with warning.
        """
        while self.current_size > self.max_size:
            if self._inactive_lru:
                page_id, victim = self._inactive_lru.popitem(last=False)
            elif self._active_lru:
                page_id, victim = self._active_lru.popitem(last=False)
                self._active_evict_count += 1
                logger.warning(
                    "remote_backup: evicting active page page_id=%d "
                    "(no inactive pages available); active_evict_total=%d",
                    page_id,
                    self._active_evict_count,
                )
            else:
                break

            self._node_by_page_id.pop(page_id, None)
            self.current_size -= victim.data_size
            victim.kv_data = None
            victim.data_size = 0
            victim.page_id = -1
            # Note: lease_refs / active_refcnt intentionally not cleared here.
            # GC will find page_id == -1 or missing from _node_by_page_id and skip.

            # Opportunistically prune dead leaf nodes to keep trie compact.
            self._try_prune_node(victim)

    def _try_prune_node(self, node: _TrieNode) -> None:
        """Walk up from *node* and remove dead leaf nodes.

        A node is prunable when it has no KV data, no active lease references,
        and no children.  We stop when we hit a node that still has data,
        active leases, or children (it is useful as a prefix navigator).
        """
        while (
            node.parent is not None  # do not remove dp_rank root nodes
            and node.kv_data is None
            and node.active_refcnt == 0
            and len(node.children) == 0
        ):
            parent = node.parent
            edge = node.edge_key
            # Guard against concurrent structural changes (shouldn't happen
            # under the server lock, but be safe)
            if edge is not None and parent.children.get(edge) is node:
                del parent.children[edge]
            node.parent = None  # Detach to prevent re-processing
            node = parent

    # ------------------------------------------------------------------
    # GC helpers (called by RemoteBackupServer._gc_worker under lock chunks)
    # ------------------------------------------------------------------

    def gc_decrement_lease(self, lease_id: int, page_id: int) -> None:
        """Remove *lease_id* from a page node and move it to inactive if
        active_refcnt drops to zero.  Called by the GC worker under self.lock.
        """
        node = self._node_by_page_id.get(page_id)
        if node is None:
            return  # Already evicted or page_id stale (page was overwritten)

        if node.lease_refs is None:
            return

        node.lease_refs.discard(lease_id)
        node.active_refcnt = len(node.lease_refs)

        if node.active_refcnt == 0:
            # Move from _active_lru to _inactive_lru
            if page_id in self._active_lru:
                del self._active_lru[page_id]
                if node.kv_data is not None:
                    self._inactive_lru[page_id] = node
                else:
                    # No data: just prune
                    self._node_by_page_id.pop(page_id, None)

            # Prune dead leaf
            if node.kv_data is None and len(node.children) == 0:
                self._try_prune_node(node)


# ---------------------------------------------------------------------------
# RemoteBackupServer – TCP service + in-process API
# ---------------------------------------------------------------------------


class RemoteBackupServer:
    """Centralized TCP service that accepts KV cache pages from all workers.

    Access modes:
    - Via TCP from RemoteBackupClient (insert_pages during backup)
    - In-process from RemoteBackupStorage (match_prefix_from / get_pages during prefetch)

    When ``request_aware=True`` (default), request lifecycle commands
    (CMD_REQUEST_START / CMD_REQUEST_FINISH) enable per-request retention
    and async GC so finished-request pages are evicted before active ones.
    """

    def __init__(
        self,
        port: int,
        max_buffer_size_gb: float = 32.0,
        page_size: int = 1,
        request_aware: bool = True,
    ):
        self.port = port
        self.page_size = page_size
        self.request_aware = request_aware
        self.buffer = RemoteBackupRadixBuffer(
            page_size=page_size,
            max_size_bytes=int(max_buffer_size_gb * (1024**3)),
        )
        self.lock = threading.Lock()
        self._tcp_server: Optional[socketserver.ThreadingTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        # Lease management (request-aware retention)
        self._lease_seq: int = 0
        # (dp_rank, rid) -> active lease entry
        self._active_leases: Dict[tuple, RequestLeaseEntry] = {}
        # lease_id -> lease entry (includes both active and GC-pending leases)
        self._lease_by_id: Dict[int, RequestLeaseEntry] = {}
        self._gc_queue: queue.Queue = queue.Queue()
        self._stop_gc: threading.Event = threading.Event()
        self._gc_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the TCP listener and GC daemon threads."""
        server = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self):
                sock = self.request
                try:
                    while True:
                        cmd_byte = _recvall(sock, 1)[0]

                        if cmd_byte == CMD_PUT_TOKENS:
                            # ----- Legacy v1 write (no lease metadata) -----
                            meta = _recvall(sock, 12)
                            seq_len, page_start, page_count = struct.unpack(
                                "!III", meta
                            )
                            dp_rank_byte = _recvall(sock, 1)
                            dp_rank = dp_rank_byte[0] if dp_rank_byte else 0
                            token_bytes = _recvall(sock, seq_len * 4)
                            token_ids = _unpack_tokens(token_bytes)
                            pages: List[bytes] = []
                            for _ in range(page_count):
                                dlen = struct.unpack(
                                    _ITEM_HDR_FMT,
                                    _recvall(sock, _ITEM_HDR_SIZE),
                                )[0]
                                pages.append(_recvall(sock, dlen))
                            results = server.insert_pages(
                                dp_rank, token_ids, page_start, pages
                            )
                            if (
                                logger.isEnabledFor(logging.DEBUG)
                                and _looks_like_decode_page_start(page_start)
                                and page_count <= 128
                            ):
                                logger.debug(
                                    "TCP_RECV decode v1: dp_rank=%d, page_start=%d, "
                                    "count=%d, tokens=%d, ok=%d/%d",
                                    dp_rank,
                                    page_start,
                                    page_count,
                                    seq_len,
                                    sum(results),
                                    len(results),
                                )
                            resp = struct.pack("!BI", 0, len(results))
                            resp += b"".join(
                                struct.pack("!?", ok) for ok in results
                            )
                            sock.sendall(resp)

                        elif cmd_byte == CMD_PUT_TOKENS_V2:
                            # ----- v2 write (with rid + generation) -----
                            meta = _recvall(sock, 12)
                            seq_len, page_start, page_count = struct.unpack(
                                "!III", meta
                            )
                            dp_rank_byte = _recvall(sock, 1)
                            dp_rank = dp_rank_byte[0] if dp_rank_byte else 0

                            rid_len = struct.unpack("!H", _recvall(sock, 2))[0]
                            rid = _recvall(sock, rid_len).decode("utf-8", errors="replace")
                            generation = struct.unpack("<Q", _recvall(sock, 8))[0]

                            token_bytes = _recvall(sock, seq_len * 4)
                            token_ids = _unpack_tokens(token_bytes)
                            pages = []
                            for _ in range(page_count):
                                dlen = struct.unpack(
                                    _ITEM_HDR_FMT,
                                    _recvall(sock, _ITEM_HDR_SIZE),
                                )[0]
                                pages.append(_recvall(sock, dlen))

                            results = server.insert_pages_v2(
                                dp_rank, token_ids, page_start, pages, rid, generation
                            )
                            resp = struct.pack("!BI", 0, len(results))
                            resp += b"".join(
                                struct.pack("!?", ok) for ok in results
                            )
                            sock.sendall(resp)

                        elif cmd_byte == CMD_REQUEST_START:
                            # ----- Lease start -----
                            dp_rank = _recvall(sock, 1)[0]
                            rid_len = struct.unpack("!H", _recvall(sock, 2))[0]
                            rid = _recvall(sock, rid_len).decode("utf-8", errors="replace")
                            generation = struct.unpack("<Q", _recvall(sock, 8))[0]
                            status = server.start_request(dp_rank, rid, generation)
                            sock.sendall(struct.pack("!B", status))

                        elif cmd_byte == CMD_REQUEST_FINISH:
                            # ----- Lease finish -----
                            dp_rank = _recvall(sock, 1)[0]
                            rid_len = struct.unpack("!H", _recvall(sock, 2))[0]
                            rid = _recvall(sock, rid_len).decode("utf-8", errors="replace")
                            generation = struct.unpack("<Q", _recvall(sock, 8))[0]
                            reason = _recvall(sock, 1)[0]
                            status = server.finish_request(dp_rank, rid, generation, reason)
                            sock.sendall(struct.pack("!B", status))

                        elif cmd_byte == CMD_MATCH_PREFIX:
                            meta = _recvall(sock, 9)
                            seq_len, start_page = struct.unpack("!II", meta[:8])
                            dp_rank = meta[8]
                            token_ids = _unpack_tokens(_recvall(sock, seq_len * 4))
                            (
                                source_dp_rank,
                                data_count,
                                navigable,
                            ) = server.match_prefix_from_with_source(
                                dp_rank, token_ids, start_page
                            )
                            source_wire = (
                                _REMOTE_BACKUP_NO_SOURCE_DP_RANK_WIRE
                                if source_dp_rank == REMOTE_BACKUP_NO_SOURCE_DP_RANK
                                else source_dp_rank
                            )
                            sock.sendall(
                                struct.pack(
                                    "!BIII", 0, source_wire, data_count, navigable
                                )
                            )

                        elif cmd_byte == CMD_GET_PAGES:
                            meta = _recvall(sock, 13)
                            seq_len, start_page, count = struct.unpack("!III", meta[:12])
                            dp_rank = meta[12]
                            token_ids = _unpack_tokens(_recvall(sock, seq_len * 4))
                            pages = server.get_pages_by_tokens(
                                dp_rank, token_ids, start_page, count
                            )
                            resp = bytearray(struct.pack("!BI", 0, len(pages)))
                            for page in pages:
                                if page is None:
                                    resp.extend(struct.pack("!I", 0))
                                else:
                                    resp.extend(struct.pack("!I", len(page)))
                                    resp.extend(page)
                            sock.sendall(resp)

                        elif cmd_byte == CMD_HEALTH:
                            sock.sendall(b"\x01")

                        else:
                            break

                except (ConnectionError, struct.error):
                    pass

        class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self._tcp_server = _ThreadingTCPServer(
            ("0.0.0.0", self.port), _Handler
        )
        self._tcp_server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._tcp_server.serve_forever, daemon=True
        )
        self._server_thread.start()

        if self.request_aware:
            self._gc_thread = threading.Thread(
                target=self._gc_worker, daemon=True, name="remote-backup-gc"
            )
            self._gc_thread.start()

        logger.info(
            "RemoteBackupServer listening on port %d "
            "(buffer %.1f GB, page_size=%d, request_aware=%s)",
            self.port,
            self.buffer.max_size / (1024**3),
            self.page_size,
            self.request_aware,
        )

    def stop(self):
        self._stop_gc.set()
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=5)
        self._gc_thread = None

        tcp_server = self._tcp_server
        if tcp_server:
            try:
                tcp_server.shutdown()
            finally:
                tcp_server.server_close()
                self._tcp_server = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        self._server_thread = None

    # ------------------------------------------------------------------
    # Lease management
    # ------------------------------------------------------------------

    def start_request(self, dp_rank: int, rid: str, generation: int) -> int:
        """Register or supersede a request lease.

        Returns _LEASE_STATUS_OK / _LEASE_STATUS_STALE / _LEASE_STATUS_SUPERSEDED_CODE.
        Thread-safe under self.lock.
        """
        if not self.request_aware:
            return _LEASE_STATUS_OK

        with self.lock:
            key = (dp_rank, rid)
            existing = self._active_leases.get(key)
            if existing is not None:
                if existing.generation == generation:
                    existing.last_touch_ts = time.monotonic()
                    return _LEASE_STATUS_OK  # Idempotent
                elif existing.generation < generation:
                    # Supersede old generation
                    existing.state = _LEASE_STATE_SUPERSEDED
                    del self._active_leases[key]
                    # Keep in _lease_by_id until GC processes it
                    self._gc_queue.put(existing.lease_id)
                else:
                    # Stale start (old generation arriving late)
                    return _LEASE_STATUS_STALE

            self._lease_seq += 1
            lease_id = self._lease_seq
            entry = RequestLeaseEntry(
                lease_id=lease_id,
                rid=rid,
                dp_rank=dp_rank,
                generation=generation,
            )
            self._active_leases[key] = entry
            self._lease_by_id[lease_id] = entry
            return _LEASE_STATUS_OK

    def finish_request(
        self, dp_rank: int, rid: str, generation: int, reason: int
    ) -> int:
        """Mark a request lease as finished and enqueue it for GC.

        Returns _LEASE_STATUS_OK or _LEASE_STATUS_STALE.
        Thread-safe under self.lock.
        """
        if not self.request_aware:
            return _LEASE_STATUS_OK

        with self.lock:
            key = (dp_rank, rid)
            existing = self._active_leases.get(key)
            if existing is None or existing.generation != generation:
                return _LEASE_STATUS_STALE

            if reason == LEASE_REASON_NORMAL:
                existing.state = _LEASE_STATE_FINISHED
            elif reason == LEASE_REASON_ABORT:
                existing.state = _LEASE_STATE_ABORTED
            else:
                existing.state = _LEASE_STATE_SUPERSEDED

            del self._active_leases[key]
            # Keep in _lease_by_id until GC processes and cleans up
            self._gc_queue.put(existing.lease_id)
            return _LEASE_STATUS_OK

    # ------------------------------------------------------------------
    # Async GC
    # ------------------------------------------------------------------

    def _gc_worker(self) -> None:
        """Background daemon: process finished leases in small lock-held chunks."""
        last_ttl_check = time.monotonic()
        while not self._stop_gc.is_set():
            try:
                lease_id = self._gc_queue.get(timeout=1.0)
            except queue.Empty:
                now = time.monotonic()
                if now - last_ttl_check >= _LEASE_TTL_CHECK_INTERVAL_S:
                    self._gc_expire_ttl()
                    last_ttl_check = now
                continue

            entry = self._lease_by_id.pop(lease_id, None)
            if entry is None:
                continue
            self._gc_process_lease(entry)

    def _gc_process_lease(self, entry: RequestLeaseEntry) -> None:
        """Decrement active_refcnt for every page in the lease, in chunks."""
        pages_list = list(entry.pages)
        if not pages_list:
            return

        gc_chunk = (
            _GC_CHUNK_PAGES_HIGH_WATER
            if self._gc_queue.qsize() > _GC_HIGH_WATER_LEASE_QUEUE
            else _GC_CHUNK_PAGES
        )
        for chunk_start in range(0, len(pages_list), gc_chunk):
            chunk = pages_list[chunk_start : chunk_start + gc_chunk]
            with self.lock:
                for page_id in chunk:
                    self.buffer.gc_decrement_lease(entry.lease_id, page_id)
            if len(pages_list) > gc_chunk:
                time.sleep(0)  # Yield CPU between chunks

    def _gc_expire_ttl(self) -> None:
        """Expire leases that have been idle longer than _LEASE_IDLE_TTL_S."""
        now = time.monotonic()
        to_expire: List[tuple] = []
        with self.lock:
            for key, entry in list(self._active_leases.items()):
                if now - entry.last_touch_ts > _LEASE_IDLE_TTL_S:
                    to_expire.append((key, entry))

        for key, entry in to_expire:
            with self.lock:
                current = self._active_leases.get(key)
                if current is not None and current.lease_id == entry.lease_id:
                    entry.state = _LEASE_STATE_ABORTED
                    del self._active_leases[key]
                    self._gc_queue.put(entry.lease_id)
            logger.warning(
                "remote_backup: TTL-expired idle lease rid=%s dp_rank=%d gen=%d",
                entry.rid,
                entry.dp_rank,
                entry.generation,
            )

    # ------------------------------------------------------------------
    # Token-based API (thread-safe via self.lock)
    # ------------------------------------------------------------------

    def insert_pages(
        self,
        dp_rank: int,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
    ) -> List[bool]:
        """V1 insert (no lease metadata). Pages go to inactive LRU."""
        with self.lock:
            results = self.buffer.insert_pages(dp_rank, token_ids, page_start, pages)
            ok_count = sum(1 for ok in results if ok)
            logger.info(
                "REMOTE_BACKUP backup_put dp_rank=%d page_start=%d pages=%d "
                "ok=%d/%d seq_tokens=%d bytes=%d "
                "inactive_pages=%d active_pages=%d buffer_mb=%.1f",
                dp_rank,
                page_start,
                len(pages),
                ok_count,
                len(results),
                len(token_ids),
                sum(len(page) for page in pages),
                len(self.buffer._inactive_lru),
                len(self.buffer._active_lru),
                self.buffer.current_size / (1024 * 1024),
            )
            return results

    def insert_pages_v2(
        self,
        dp_rank: int,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
        rid: str,
        generation: int,
    ) -> List[bool]:
        """V2 insert (with rid + generation). Pages go to active LRU under the lease."""
        with self.lock:
            lease_id = self._get_or_create_lease_locked(dp_rank, rid, generation)
            results = self.buffer.insert_pages(
                dp_rank, token_ids, page_start, pages, lease_id=lease_id
            )
            # Track page_ids in the lease for GC
            if lease_id is not None:
                entry = self._lease_by_id.get(lease_id)
                if entry is not None:
                    entry.pages.update(self.buffer._last_page_ids)
                    entry.last_touch_ts = time.monotonic()

            ok_count = sum(1 for ok in results if ok)
            logger.info(
                "REMOTE_BACKUP backup_put_v2 dp_rank=%d page_start=%d pages=%d "
                "ok=%d/%d seq_tokens=%d bytes=%d rid=%s gen=%d lease_id=%s "
                "inactive_pages=%d active_pages=%d buffer_mb=%.1f",
                dp_rank,
                page_start,
                len(pages),
                ok_count,
                len(results),
                len(token_ids),
                sum(len(page) for page in pages),
                rid[:16] if rid else "",
                generation,
                lease_id,
                len(self.buffer._inactive_lru),
                len(self.buffer._active_lru),
                self.buffer.current_size / (1024 * 1024),
            )
            return results

    def _get_or_create_lease_locked(
        self, dp_rank: int, rid: str, generation: int
    ) -> Optional[int]:
        """Find or lazily create a lease under self.lock. Returns lease_id or None."""
        if not self.request_aware or not rid:
            return None
        key = (dp_rank, rid)
        existing = self._active_leases.get(key)
        if existing is not None:
            if existing.generation == generation:
                return existing.lease_id
            elif existing.generation < generation:
                # Supersede old, create new (handles out-of-order start packets)
                existing.state = _LEASE_STATE_SUPERSEDED
                del self._active_leases[key]
                self._gc_queue.put(existing.lease_id)
            else:
                # Stale write from an old generation: no lease association
                return None

        # Lazy-create lease (handles lost CMD_REQUEST_START packets)
        self._lease_seq += 1
        lease_id = self._lease_seq
        entry = RequestLeaseEntry(
            lease_id=lease_id,
            rid=rid,
            dp_rank=dp_rank,
            generation=generation,
        )
        self._active_leases[key] = entry
        self._lease_by_id[lease_id] = entry
        return lease_id

    def match_prefix_from(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        _source_dp_rank, data_count, navigable = self.match_prefix_from_with_source(
            dp_rank, token_ids, start_page
        )
        return data_count, navigable

    def match_prefix_from_with_source(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        with self.lock:
            source_dp_rank, data_count, navigable = (
                self.buffer.match_prefix_from_with_source(dp_rank, token_ids, start_page)
            )
            logger.info(
                "REMOTE_BACKUP prefetch_match query_dp_rank=%s source_dp_rank=%s "
                "start_page=%d query_pages=%d data_pages=%d navigable_pages=%d "
                "query_tokens=%d inactive_pages=%d active_pages=%d buffer_mb=%.1f",
                _dp_rank_label(dp_rank),
                _dp_rank_label(source_dp_rank),
                start_page,
                max(0, len(token_ids) // self.page_size - start_page),
                data_count,
                navigable,
                len(token_ids),
                len(self.buffer._inactive_lru),
                len(self.buffer._active_lru),
                self.buffer.current_size / (1024 * 1024),
            )
            return source_dp_rank, data_count, navigable

    def get_pages_by_tokens(
        self,
        dp_rank: int,
        token_ids: List[int],
        start_page: int,
        count: int,
    ) -> List[Optional[bytes]]:
        with self.lock:
            source_dp_rank = dp_rank
            if dp_rank == REMOTE_BACKUP_ALL_DP_RANKS:
                source_dp_rank, _data_count, _navigable = (
                    self.buffer.match_prefix_from_with_source(
                        dp_rank, token_ids, start_page
                    )
                )
            pages = self.buffer.get_pages(dp_rank, token_ids, start_page, count)
            hit_count = sum(1 for page in pages if page is not None)
            logger.info(
                "REMOTE_BACKUP prefetch_get query_dp_rank=%s source_dp_rank=%s "
                "start_page=%d count=%d hit_pages=%d/%d bytes=%d "
                "query_tokens=%d inactive_pages=%d active_pages=%d buffer_mb=%.1f",
                _dp_rank_label(dp_rank),
                _dp_rank_label(source_dp_rank),
                start_page,
                count,
                hit_count,
                len(pages),
                sum(len(page) for page in pages if page is not None),
                len(token_ids),
                len(self.buffer._inactive_lru),
                len(self.buffer._active_lru),
                self.buffer.current_size / (1024 * 1024),
            )
            return pages

    def clear(self):
        with self.lock:
            self.buffer.clear()


# ---------------------------------------------------------------------------
# RemoteBackupClient – TCP sender (used by workers to back up decode KV)
# ---------------------------------------------------------------------------


class RemoteBackupClient:
    """TCP client that sends KV cache pages to the remote backup server."""

    CONNECT_TIMEOUT_S = 2.0
    IO_TIMEOUT_S = 2.0
    FAILURE_COOLDOWN_S = 15.0
    FAILURE_THRESHOLD = 2

    def __init__(self, remote_backup_url: str):
        parts = remote_backup_url.split(":")
        self.host = parts[0]
        self.port = int(parts[1])
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def _is_in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def _record_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.FAILURE_THRESHOLD:
            self._cooldown_until = time.monotonic() + self.FAILURE_COOLDOWN_S
            logger.warning(
                "RemoteBackupClient entering cooldown for %s:%d after %d failures: %s "
                "(cooldown_until=%.1f)",
                self.host,
                self.port,
                self._consecutive_failures,
                exc,
                self._cooldown_until,
            )
        else:
            logger.warning("RemoteBackupClient.put_pages_tokens failed: %s", exc)
        self._reset()

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        if self._is_in_cooldown():
            raise TimeoutError(
                f"Remote backup server {self.host}:{self.port} is in cooldown after repeated failures"
            )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.CONNECT_TIMEOUT_S)
        sock.connect((self.host, self.port))
        sock.settimeout(self.IO_TIMEOUT_S)
        self._sock = sock
        return sock

    def _reset(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    # V1 API (no lease metadata)
    # ------------------------------------------------------------------

    def put_pages_tokens(
        self,
        dp_rank: int,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
    ) -> List[bool]:
        """Send pages with token context to the remote backup server (v1)."""
        with self._lock:
            if self._is_in_cooldown():
                return [False] * len(pages)
            try:
                sock = self._connect()
                seq_len = len(token_ids)
                page_count = len(pages)
                hdr = struct.pack(
                    "!BIII", CMD_PUT_TOKENS, seq_len, page_start, page_count
                )
                sock.sendall(hdr)
                sock.sendall(bytes([dp_rank & 0xFF]))
                sock.sendall(_pack_tokens(token_ids))
                for page in pages:
                    sock.sendall(struct.pack(_ITEM_HDR_FMT, len(page)))
                    sock.sendall(page)
                resp_hdr = _recvall(sock, 5)
                _, count = struct.unpack("!BI", resp_hdr)
                resp_data = _recvall(sock, count)
                self._record_success()
                return [bool(b) for b in resp_data]
            except Exception as exc:
                self._record_failure(exc)
                return [False] * len(pages)

    # ------------------------------------------------------------------
    # V2 API (with lease metadata)
    # ------------------------------------------------------------------

    def put_pages_tokens_v2(
        self,
        dp_rank: int,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
        rid: str,
        generation: int,
    ) -> List[bool]:
        """Send pages with token context AND lease metadata to the server (v2).

        Protocol: CMD_PUT_TOKENS_V2 header + rid + generation + tokens + page data.
        Falls back to v1 behavior if in cooldown.
        """
        with self._lock:
            if self._is_in_cooldown():
                return [False] * len(pages)
            try:
                sock = self._connect()
                rid_bytes = rid.encode("utf-8")
                rid_len = len(rid_bytes)
                seq_len = len(token_ids)
                page_count = len(pages)

                hdr = struct.pack(
                    "!BIII", CMD_PUT_TOKENS_V2, seq_len, page_start, page_count
                )
                sock.sendall(hdr)
                sock.sendall(bytes([dp_rank & 0xFF]))
                sock.sendall(struct.pack("!H", rid_len))
                sock.sendall(rid_bytes)
                sock.sendall(struct.pack("<Q", generation))
                sock.sendall(_pack_tokens(token_ids))
                for page in pages:
                    sock.sendall(struct.pack(_ITEM_HDR_FMT, len(page)))
                    sock.sendall(page)

                resp_hdr = _recvall(sock, 5)
                _, count = struct.unpack("!BI", resp_hdr)
                resp_data = _recvall(sock, count)
                self._record_success()
                return [bool(b) for b in resp_data]
            except Exception as exc:
                self._record_failure(exc)
                return [False] * len(pages)

    def start_request(
        self, rid: str, dp_rank: int, generation: int
    ) -> bool:
        """Send CMD_REQUEST_START to register a lease on the server.

        Returns True on success (ok or already-registered idempotent),
        False on failure / cooldown.
        """
        with self._lock:
            if self._is_in_cooldown():
                return False
            try:
                sock = self._connect()
                rid_bytes = rid.encode("utf-8")
                rid_len = len(rid_bytes)
                hdr = struct.pack("!BB", CMD_REQUEST_START, dp_rank & 0xFF)
                sock.sendall(hdr)
                sock.sendall(struct.pack("!H", rid_len))
                sock.sendall(rid_bytes)
                sock.sendall(struct.pack("<Q", generation))
                resp = _recvall(sock, 1)
                self._record_success()
                status = resp[0]
                return status in (_LEASE_STATUS_OK, _LEASE_STATUS_SUPERSEDED_CODE)
            except Exception as exc:
                self._record_failure(exc)
                return False

    def finish_request(
        self, rid: str, dp_rank: int, generation: int, reason: int = LEASE_REASON_NORMAL
    ) -> bool:
        """Send CMD_REQUEST_FINISH to mark a lease as done on the server.

        Returns True on success, False on failure / cooldown.
        """
        with self._lock:
            if self._is_in_cooldown():
                return False
            try:
                sock = self._connect()
                rid_bytes = rid.encode("utf-8")
                rid_len = len(rid_bytes)
                hdr = struct.pack("!BB", CMD_REQUEST_FINISH, dp_rank & 0xFF)
                sock.sendall(hdr)
                sock.sendall(struct.pack("!H", rid_len))
                sock.sendall(rid_bytes)
                sock.sendall(struct.pack("<Q", generation))
                sock.sendall(struct.pack("!B", reason & 0xFF))
                resp = _recvall(sock, 1)
                self._record_success()
                return resp[0] == _LEASE_STATUS_OK
            except Exception as exc:
                self._record_failure(exc)
                return False

    def match_prefix_from(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple[int, int]:
        _source_dp_rank, data_count, navigable = self.match_prefix_from_with_source(
            dp_rank, token_ids, start_page
        )
        return data_count, navigable

    def match_prefix_from_with_source(
        self, dp_rank: int, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        with self._lock:
            if self._is_in_cooldown():
                return REMOTE_BACKUP_NO_SOURCE_DP_RANK, 0, 0
            try:
                sock = self._connect()
                hdr = struct.pack(
                    "!BIIB", CMD_MATCH_PREFIX, len(token_ids), start_page, dp_rank & 0xFF
                )
                sock.sendall(hdr)
                sock.sendall(_pack_tokens(token_ids))
                resp = _recvall(sock, 13)
                _, source_wire, data_count, navigable = struct.unpack("!BIII", resp)
                self._record_success()
                source_dp_rank = (
                    REMOTE_BACKUP_NO_SOURCE_DP_RANK
                    if source_wire == _REMOTE_BACKUP_NO_SOURCE_DP_RANK_WIRE
                    else source_wire
                )
                return source_dp_rank, data_count, navigable
            except Exception as exc:
                self._record_failure(exc)
                return REMOTE_BACKUP_NO_SOURCE_DP_RANK, 0, 0

    def get_pages_by_tokens(
        self,
        dp_rank: int,
        token_ids: List[int],
        start_page: int,
        count: int,
    ) -> List[Optional[bytes]]:
        with self._lock:
            if self._is_in_cooldown():
                logger.warning(
                    "RemoteBackupClient.get_pages_by_tokens: in cooldown, "
                    "returning None*%d (dp_rank=%d, start_page=%d)",
                    count,
                    dp_rank,
                    start_page,
                )
                return [None] * count
            try:
                sock = self._connect()
                hdr = struct.pack(
                    "!BIIIB",
                    CMD_GET_PAGES,
                    len(token_ids),
                    start_page,
                    count,
                    dp_rank & 0xFF,
                )
                sock.sendall(hdr)
                sock.sendall(_pack_tokens(token_ids))
                resp_hdr = _recvall(sock, 5)
                _, recv_count = struct.unpack("!BI", resp_hdr)
                pages: List[Optional[bytes]] = []
                for _ in range(recv_count):
                    dlen = struct.unpack("!I", _recvall(sock, 4))[0]
                    if dlen == 0:
                        pages.append(None)
                    else:
                        pages.append(_recvall(sock, dlen))
                self._record_success()
                if len(pages) < count:
                    pages.extend([None] * (count - len(pages)))
                return pages[:count]
            except Exception as exc:
                self._record_failure(exc)
                return [None] * count

    def close(self):
        with self._lock:
            self._reset()
