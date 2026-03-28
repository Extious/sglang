# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
PeerCacheServer: TCP service that accepts KV Cache pages from a peer worker
and stores them in local DRAM using a token-indexed radix buffer (trie).

PeerCacheClient: TCP client that sends KV Cache pages to a remote
PeerCacheServer for replication.

The radix buffer indexes KV pages by the full token sequence, eliminating
hash-chain fragility.  Prefix matching is inherent to the trie structure,
so cache hits no longer depend on identical hash anchors across workers.

Binary protocol over TCP (token-based):
  CMD_PUT_TOKENS (4):
    Send: [cmd:1B][seq_len:4B][page_start:4B][page_count:4B]
          [tokens: seq_len * 4B little-endian uint32]
          Per page: [data_len:4B][data]
    Recv: [status:1B][count:4B][ok:1B * count]
"""

from __future__ import annotations

import array
import logging
import socket
import socketserver
import struct
import sys
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CMD_PUT_TOKENS = 4
DECODE_DEBUG_PAGE_START_THRESHOLD = 100

_ITEM_HDR_FMT = "!I"
_ITEM_HDR_SIZE = struct.calcsize(_ITEM_HDR_FMT)


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


# ---------------------------------------------------------------------------
# Trie data structure for the peer radix buffer
# ---------------------------------------------------------------------------


class _TrieNode:
    """A node in the peer radix buffer trie.

    Each level corresponds to one KV page (``page_size`` tokens).
    For page_size == 1 the edge key is a single token ID (int);
    for page_size > 1 it is a tuple of token IDs.
    """

    __slots__ = ("children", "kv_data", "data_size", "page_id")

    def __init__(self):
        self.children: Dict = {}
        self.kv_data: Optional[bytes] = None
        self.data_size: int = 0
        self.page_id: int = -1


class PeerRadixBuffer:
    """Token-indexed radix buffer for peer KV cache pages.

    A trie where each level corresponds to one page (``page_size`` tokens).
    """

    def __init__(self, page_size: int, max_size_bytes: int):
        self.root = _TrieNode()
        self.page_size = page_size
        self.max_size = max_size_bytes
        self.current_size: int = 0
        self._page_counter: int = 0
        self._lru: OrderedDict[int, _TrieNode] = OrderedDict()

    @property
    def page_count(self) -> int:
        return len(self._lru)

    def _edge_key(self, token_ids: List[int], page_idx: int):
        start = page_idx * self.page_size
        if self.page_size == 1:
            return token_ids[start]
        return tuple(token_ids[start : start + self.page_size])

    # -- public API ---------------------------------------------------------

    _insert_total: int = 0
    _insert_decode_total: int = 0

    def insert_pages(
        self,
        full_token_ids: List[int],
        page_start: int,
        kv_pages: List[bytes],
    ) -> List[bool]:
        """Insert KV pages into the trie at *page_start*.

        ``full_token_ids`` is the complete token sequence up to (and including)
        the tokens covered by the last page in *kv_pages*.  Intermediate nodes
        for the prefix ``[0, page_start)`` are created automatically.
        """
        PeerRadixBuffer._insert_total += 1
        is_decode_debug = _looks_like_decode_page_start(page_start)
        if is_decode_debug:
            PeerRadixBuffer._insert_decode_total += 1
            if (
                logger.isEnabledFor(logging.DEBUG)
                and PeerRadixBuffer._insert_decode_total % 50 == 1
            ):
                first_key = self._edge_key(full_token_ids, page_start)
                logger.debug(
                    "PeerRadixBuffer.insert_pages DECODE #%d (total=%d): "
                    "page_start=%d, pages=%d, tokens=%d, "
                    "first_key=%s, lru_size=%d, data_bytes=%d",
                    PeerRadixBuffer._insert_decode_total,
                    PeerRadixBuffer._insert_total,
                    page_start,
                    len(kv_pages),
                    len(full_token_ids),
                    first_key,
                    len(self._lru),
                    self.current_size,
                )

        results: List[bool] = []
        node = self.root

        _prefix_existing = 0
        _prefix_created = 0
        for p in range(page_start):
            key = self._edge_key(full_token_ids, p)
            child = node.children.get(key)
            if child is None:
                child = _TrieNode()
                node.children[key] = child
                _prefix_created += 1
            else:
                _prefix_existing += 1
                if child.page_id >= 0:
                    self._lru.move_to_end(child.page_id)
            node = child

        if (
            is_decode_debug
            and logger.isEnabledFor(logging.DEBUG)
            and (
                PeerRadixBuffer._insert_decode_total % 50 == 1
                or _prefix_created > 0
            )
        ):
            first_key = self._edge_key(full_token_ids, page_start)
            logger.debug(
                "insert_pages PREFIX: page_start=%d, existing=%d, "
                "created=%d, first_key=%s, node_children=%d",
                page_start,
                _prefix_existing,
                _prefix_created,
                first_key,
                len(node.children),
            )

        for i, kv_data in enumerate(kv_pages):
            target_page = page_start + i
            key = self._edge_key(full_token_ids, target_page)
            child = node.children.get(key)
            if child is None:
                child = _TrieNode()
                node.children[key] = child

            if child.page_id >= 0:
                self._lru.pop(child.page_id, None)
                self.current_size -= child.data_size

            child.kv_data = kv_data
            child.data_size = len(kv_data)
            self.current_size += child.data_size

            self._page_counter += 1
            child.page_id = self._page_counter
            self._lru[child.page_id] = child

            node = child
            results.append(True)

        self._evict_if_needed()
        return results

    def match_prefix_from(
        self, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        """Return ``(data_count, navigable_range)`` starting from *start_page*.

        Evicted nodes (kv_data is None but structurally present) are traversed
        through but not counted in *data_count*.  *navigable_range* is the total
        number of trie levels traversed (including evicted nodes), which should
        be used as the *count* argument for :meth:`get_pages`.
        """
        total_pages = len(token_ids) // self.page_size
        node = self.root

        for p in range(start_page):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                return 0, 0
            node = child

        data_count = 0
        navigable = 0
        for p in range(start_page, total_pages):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                break
            navigable += 1
            if child.kv_data is not None:
                if child.page_id >= 0:
                    self._lru.move_to_end(child.page_id)
                data_count += 1
            node = child

        if start_page > 0 and logger.isEnabledFor(logging.DEBUG):
            stop_page = start_page + navigable
            num_children_at_stop = len(node.children) if node else 0
            if stop_page < total_pages and num_children_at_stop > 0:
                expected_key = self._edge_key(token_ids, stop_page)
                actual_keys = list(node.children.keys())[:5]
                logger.debug(
                    "match_prefix_from MISMATCH: start=%d, data=%d, nav=%d, "
                    "stop_page=%d, total_pages=%d, "
                    "expected_key=%s, actual_keys=%s, "
                    "lru_size=%d, decode_inserts=%d",
                    start_page,
                    data_count,
                    navigable,
                    stop_page,
                    total_pages,
                    expected_key,
                    actual_keys,
                    len(self._lru),
                    PeerRadixBuffer._insert_decode_total,
                )
            elif navigable > 0:
                logger.debug(
                    "match_prefix_from: start=%d, data=%d, nav=%d, "
                    "stop_page=%d, children_at_stop=%d, total_pages=%d, "
                    "lru_size=%d, decode_inserts=%d",
                    start_page,
                    data_count,
                    navigable,
                    stop_page,
                    num_children_at_stop,
                    total_pages,
                    len(self._lru),
                    PeerRadixBuffer._insert_decode_total,
                )

        return data_count, navigable

    def get_pages(
        self,
        token_ids: List[int],
        start_page: int,
        count: int,
    ) -> List[Optional[bytes]]:
        """Get KV page data for *count* consecutive pages from *start_page*."""
        total_pages = len(token_ids) // self.page_size
        node = self.root

        for p in range(start_page):
            key = self._edge_key(token_ids, p)
            child = node.children.get(key)
            if child is None:
                return [None] * count
            node = child

        results: List[Optional[bytes]] = []
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
                if child.page_id >= 0:
                    self._lru.move_to_end(child.page_id)
                results.append(child.kv_data)
            else:
                results.append(None)
            node = child
        return results

    def clear(self):
        self.root = _TrieNode()
        self._lru.clear()
        self.current_size = 0
        self._page_counter = 0

    def _evict_if_needed(self):
        while self.current_size > self.max_size and self._lru:
            _, victim = self._lru.popitem(last=False)
            self.current_size -= victim.data_size
            victim.kv_data = None
            victim.data_size = 0
            victim.page_id = -1


# ---------------------------------------------------------------------------
# PeerCacheServer – TCP service + in-process API
# ---------------------------------------------------------------------------


class PeerCacheServer:
    """Accepts KV Cache pages from a peer worker and stores in a radix buffer.

    Access modes:
    - Via TCP from a remote ``PeerCacheClient`` (``insert_tokens`` during backup)
    - In-process from ``PeerCacheStorage`` (``match_prefix_from`` / ``get_pages``
      during failover prefetch)
    """

    def __init__(
        self,
        port: int,
        max_buffer_size_gb: float = 8.0,
        page_size: int = 1,
    ):
        self.port = port
        self.page_size = page_size
        self.buffer = PeerRadixBuffer(
            page_size=page_size,
            max_size_bytes=int(max_buffer_size_gb * (1024**3)),
        )
        self.lock = threading.Lock()
        self._tcp_server: Optional[socketserver.ThreadingTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the TCP listener in a daemon thread."""
        server = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self):
                sock = self.request
                try:
                    while True:
                        cmd_byte = _recvall(sock, 1)[0]

                        if cmd_byte == CMD_PUT_TOKENS:
                            meta = _recvall(sock, 12)
                            seq_len, page_start, page_count = struct.unpack(
                                "!III", meta
                            )

                            token_bytes = _recvall(sock, seq_len * 4)
                            token_ids = _unpack_tokens(token_bytes)

                            pages: List[bytes] = []
                            for _ in range(page_count):
                                dlen = struct.unpack(
                                    _ITEM_HDR_FMT,
                                    _recvall(sock, _ITEM_HDR_SIZE),
                                )[0]
                                pages.append(_recvall(sock, dlen))

                            results = server.insert_tokens(
                                token_ids, page_start, pages
                            )
                            if (
                                logger.isEnabledFor(logging.DEBUG)
                                and _looks_like_decode_page_start(page_start)
                                and page_count <= 128
                            ):
                                logger.debug(
                                    "TCP_RECV decode: page_start=%d, "
                                    "count=%d, tokens=%d, ok=%d/%d",
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

                        else:
                            break

                except (ConnectionError, struct.error):
                    pass

        self._tcp_server = socketserver.ThreadingTCPServer(
            ("0.0.0.0", self.port), _Handler
        )
        self._tcp_server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._tcp_server.serve_forever, daemon=True
        )
        self._server_thread.start()
        logger.info(
            "PeerCacheServer listening on port %d (buffer %.1f GB, page_size=%d)",
            self.port,
            self.buffer.max_size / (1024**3),
            self.page_size,
        )

    def stop(self):
        if self._tcp_server:
            self._tcp_server.shutdown()

    # -- token-based API (thread-safe via self.lock) ------------------------

    def insert_tokens(
        self,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
    ) -> List[bool]:
        with self.lock:
            return self.buffer.insert_pages(token_ids, page_start, pages)

    def match_prefix_from(
        self, token_ids: List[int], start_page: int = 0
    ) -> tuple:
        with self.lock:
            return self.buffer.match_prefix_from(token_ids, start_page)

    def get_pages_by_tokens(
        self,
        token_ids: List[int],
        start_page: int,
        count: int,
    ) -> List[Optional[bytes]]:
        with self.lock:
            return self.buffer.get_pages(token_ids, start_page, count)

    def clear(self):
        with self.lock:
            self.buffer.clear()


# ---------------------------------------------------------------------------
# PeerCacheClient – TCP sender
# ---------------------------------------------------------------------------


class PeerCacheClient:
    """TCP client that sends KV Cache pages to a remote PeerCacheServer."""

    def __init__(self, peer_url: str):
        parts = peer_url.split(":")
        self.host = parts[0]
        self.port = int(parts[1])
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)
        sock.connect((self.host, self.port))
        self._sock = sock
        return sock

    def _reset(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def put_pages_tokens(
        self,
        token_ids: List[int],
        page_start: int,
        pages: List[bytes],
    ) -> List[bool]:
        """Send pages with their token context to the remote PeerCacheServer."""
        with self._lock:
            try:
                sock = self._connect()
                seq_len = len(token_ids)
                page_count = len(pages)

                hdr = struct.pack(
                    "!BIII", CMD_PUT_TOKENS, seq_len, page_start, page_count
                )
                sock.sendall(hdr)
                sock.sendall(_pack_tokens(token_ids))

                for page in pages:
                    sock.sendall(struct.pack(_ITEM_HDR_FMT, len(page)))
                    sock.sendall(page)

                resp_hdr = _recvall(sock, 5)
                _, count = struct.unpack("!BI", resp_hdr)
                resp_data = _recvall(sock, count)
                return [bool(b) for b in resp_data]

            except Exception as exc:
                logger.warning("PeerCacheClient.put_pages_tokens failed: %s", exc)
                self._reset()
                return [False] * len(pages)

    def close(self):
        with self._lock:
            self._reset()
