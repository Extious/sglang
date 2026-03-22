# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
PeerCacheServer: TCP service that accepts KV Cache pages from a peer worker
and stores them in local DRAM with LRU eviction.

PeerCacheClient: TCP client that sends KV Cache pages to a remote
PeerCacheServer for replication.

Binary protocol over TCP:
  Request:  [cmd:1B][count:4B] then per item [key_len:4B][key][data_len:4B][data]
  Response: [status:1B][count:4B] then per item [ok:1B]

Commands: PUT=1, GET=2, EXISTS=3
"""

from __future__ import annotations

import logging
import socket
import socketserver
import struct
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CMD_PUT = 1
CMD_GET = 2
CMD_EXISTS = 3

_HEADER_FMT = "!BI"  # cmd (1B) + count (4B)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_ITEM_HDR_FMT = "!I"  # length prefix (4B)
_ITEM_HDR_SIZE = struct.calcsize(_ITEM_HDR_FMT)


def _recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def _recv_items(sock: socket.socket, count: int, with_data: bool):
    """Receive count items from socket. Each item is [key_len][key][data_len][data]."""
    keys: List[str] = []
    pages: List[bytes] = []
    for _ in range(count):
        key_len = struct.unpack(_ITEM_HDR_FMT, _recvall(sock, _ITEM_HDR_SIZE))[0]
        key = _recvall(sock, key_len).decode("utf-8")
        keys.append(key)
        if with_data:
            data_len = struct.unpack(_ITEM_HDR_FMT, _recvall(sock, _ITEM_HDR_SIZE))[0]
            data = _recvall(sock, data_len)
            pages.append(data)
    return keys, pages


def _send_items(sock: socket.socket, keys: List[str], pages: List[bytes]):
    """Send items with their data."""
    for key, page in zip(keys, pages):
        key_bytes = key.encode("utf-8")
        sock.sendall(struct.pack(_ITEM_HDR_FMT, len(key_bytes)))
        sock.sendall(key_bytes)
        sock.sendall(struct.pack(_ITEM_HDR_FMT, len(page)))
        sock.sendall(page)


class PeerCacheServer:
    """Accepts KV Cache pages from a peer worker and stores in local DRAM.

    The buffer is accessed in two ways:
    - Via TCP from a remote PeerCacheClient (put_pages during replication)
    - In-process from PeerCacheStorage (get_pages/exists during failover)
    """

    def __init__(self, port: int, max_buffer_size_gb: float = 8.0):
        self.port = port
        self.max_size = int(max_buffer_size_gb * (1024**3))
        self.current_size = 0
        self.buffer: Dict[str, bytes] = {}
        self.lru: OrderedDict = OrderedDict()
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
                        hdr = _recvall(sock, _HEADER_SIZE)
                        cmd, count = struct.unpack(_HEADER_FMT, hdr)

                        if cmd == CMD_PUT:
                            keys, pages = _recv_items(sock, count, with_data=True)
                            results = server.put_pages(keys, pages)
                            resp = struct.pack("!BI", 0, len(results))
                            resp += b"".join(
                                struct.pack("!?", ok) for ok in results
                            )
                            sock.sendall(resp)

                        elif cmd == CMD_GET:
                            keys, _ = _recv_items(sock, count, with_data=False)
                            pages = server.get_pages(keys)
                            found = [p is not None for p in pages]
                            resp = struct.pack("!BI", 0, len(found))
                            resp += b"".join(
                                struct.pack("!?", f) for f in found
                            )
                            sock.sendall(resp)
                            for page in pages:
                                if page is not None:
                                    sock.sendall(
                                        struct.pack(_ITEM_HDR_FMT, len(page))
                                    )
                                    sock.sendall(page)

                        elif cmd == CMD_EXISTS:
                            keys, _ = _recv_items(sock, count, with_data=False)
                            hit_count = server.exists(keys)
                            sock.sendall(struct.pack("!BI", 0, hit_count))

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
        logger.info("PeerCacheServer listening on port %d (buffer %.1f GB)",
                     self.port, self.max_size / (1024**3))

    def stop(self):
        if self._tcp_server:
            self._tcp_server.shutdown()

    def put_pages(self, keys: List[str], pages: List[bytes]) -> List[bool]:
        """Store pages in the local buffer with LRU eviction."""
        results = []
        with self.lock:
            for key, page in zip(keys, pages):
                page_size = len(page)
                if key in self.buffer:
                    old_size = len(self.buffer[key])
                    self.current_size -= old_size
                    self.lru.move_to_end(key)
                else:
                    self.lru[key] = True

                self.buffer[key] = page
                self.current_size += page_size

                self._evict_if_needed()
                results.append(True)
        return results

    def get_pages(self, keys: List[str]) -> List[Optional[bytes]]:
        """Retrieve pages from the local buffer, updating LRU order."""
        results = []
        with self.lock:
            for key in keys:
                page = self.buffer.get(key)
                if page is not None:
                    self.lru.move_to_end(key)
                results.append(page)
        return results

    def exists(self, keys: List[str]) -> int:
        """Return the number of consecutive existing keys from the start."""
        with self.lock:
            for i, key in enumerate(keys):
                if key not in self.buffer:
                    return i
            return len(keys)

    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.lru.clear()
            self.current_size = 0

    def _evict_if_needed(self):
        """Evict oldest entries until under max_size. Must hold self.lock."""
        while self.current_size > self.max_size and self.lru:
            oldest_key, _ = self.lru.popitem(last=False)
            if oldest_key in self.buffer:
                self.current_size -= len(self.buffer[oldest_key])
                del self.buffer[oldest_key]


class PeerCacheClient:
    """TCP client that sends KV Cache pages to a remote PeerCacheServer."""

    def __init__(self, peer_url: str):
        parts = peer_url.split(":")
        self.host = parts[0]
        self.port = int(parts[1])
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        """Get or create a TCP connection to the peer server."""
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

    def put_pages(self, keys: List[str], pages: List[bytes]) -> List[bool]:
        """Send pages to the remote PeerCacheServer."""
        with self._lock:
            try:
                sock = self._connect()
                sock.sendall(struct.pack(_HEADER_FMT, CMD_PUT, len(keys)))
                _send_items(sock, keys, pages)

                resp_hdr = _recvall(sock, 5)  # status(1) + count(4)
                _, count = struct.unpack("!BI", resp_hdr)
                resp_data = _recvall(sock, count)
                return [bool(b) for b in resp_data]
            except Exception as exc:
                logger.warning("PeerCacheClient.put_pages failed: %s", exc)
                self._reset()
                return [False] * len(keys)

    def close(self):
        with self._lock:
            self._reset()
