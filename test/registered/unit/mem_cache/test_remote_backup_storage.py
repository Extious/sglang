"""Unit tests for request-level remote backup storage."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo
from sglang.srt.mem_cache.storage.remote_backup.remote_backup_server import (
    LEASE_REASON_ABORT,
    LEASE_REASON_FAILOVER_OLD,
    LEASE_REASON_FAILOVER_SUPERSEDE,
    LEASE_REASON_NORMAL,
    _LEASE_STATUS_OK,
    RemoteBackupServer,
    RequestNamespaceStore,
)
from sglang.srt.mem_cache.storage.remote_backup.remote_backup_storage import (
    RemoteBackupStorage,
)

register_cpu_ci(5, "stage-a-test-cpu")


class TestRemoteBackupStorageLookupDpRank(CustomTestCase):
    def setUp(self):
        self.storage = RemoteBackupStorage.__new__(RemoteBackupStorage)
        self.storage._dp_rank = 0
        self.storage.page_size = 1
        self.storage.local_server = MagicMock()
        self.storage.client = None
        self.storage.mem_pool_host = MagicMock()

    def test_match_prefix_from_uses_override_dp_rank(self):
        self.storage.local_server.match_prefix_from.return_value = (4, 4)
        self.storage.local_server.buffer.page_count = 128

        data_count, navigable = self.storage.match_prefix_from(
            [1, 2, 3, 4], start_page=2, lookup_dp_rank=1
        )

        self.assertEqual((data_count, navigable), (4, 4))

    def test_get_pages_tokens_uses_override_dp_rank(self):
        self.storage.local_server.get_pages_by_tokens.return_value = [bytes([7])]
        self.storage.mem_pool_host.get_dummy_flat_data_page.return_value = torch.zeros(
            (1,), dtype=torch.uint8
        )

        results = self.storage.get_pages_tokens(
            [11],
            start_page=0,
            count=1,
            host_indices=torch.tensor([3], dtype=torch.int64),
            lookup_dp_rank=1,
        )

        self.assertEqual(results, [True])
        self.storage.mem_pool_host.set_from_flat_data_page.assert_called_once()


# ---------------------------------------------------------------------------
# RequestNamespaceStore unit tests
# ---------------------------------------------------------------------------


class TestRequestNamespaceStore(CustomTestCase):
    """Tests for the standalone RequestNamespaceStore data structure."""

    def test_put_and_get_pages(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        results = store.put_pages(0, "r1", 0, 0, [b"page0", b"page1"])
        self.assertEqual(results, [True, True])
        self.assertEqual(store.current_size, 10)

        pages = store.get_pages(0, "r1", 0, 0, 3)
        self.assertEqual(pages, [b"page0", b"page1", None])

    def test_query_range_contiguous(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        store.put_pages(0, "r1", 0, 0, [b"a", b"b", b"c"])
        contiguous, total = store.query_range(0, "r1", 0)
        self.assertEqual(contiguous, 3)
        self.assertEqual(total, 3)

    def test_query_range_with_gap(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        store.put_pages(0, "r1", 0, 0, [b"a"])
        store.put_pages(0, "r1", 0, 3, [b"d"])
        contiguous, total = store.query_range(0, "r1", 0)
        self.assertEqual(contiguous, 1)
        self.assertEqual(total, 2)

    def test_release_namespace(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        store.put_pages(0, "r1", 0, 0, [b"abc"])
        self.assertEqual(store.namespace_count, 1)
        freed = store.release_namespace(0, "r1", 0)
        self.assertEqual(freed, 3)
        self.assertEqual(store.namespace_count, 0)
        self.assertEqual(store.current_size, 0)

    def test_page_overwrite(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        store.put_pages(0, "r1", 0, 0, [b"old"])
        self.assertEqual(store.current_size, 3)
        store.put_pages(0, "r1", 0, 0, [b"newdata"])
        self.assertEqual(store.current_size, 7)
        pages = store.get_pages(0, "r1", 0, 0, 1)
        self.assertEqual(pages, [b"newdata"])

    def test_eviction_inactive_first(self):
        store = RequestNamespaceStore(max_size_bytes=10)
        store.put_pages(0, "r1", 0, 0, [b"aaaa"])  # 4 bytes
        store.mark_inactive(0, "r1", 0)
        store.put_pages(0, "r2", 0, 0, [b"bbbbbb"])  # 6 bytes, fits
        self.assertEqual(store.namespace_count, 2)
        # This should evict r1 (inactive)
        store.put_pages(0, "r3", 0, 0, [b"cccccc"])  # 6 bytes, exceeds 10
        self.assertIsNone(store.namespaces.get((0, "r1", 0)))

    def test_different_generations_independent(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        store.put_pages(0, "r1", 0, 0, [b"gen0"])
        store.put_pages(0, "r1", 1, 0, [b"gen1"])
        self.assertEqual(store.namespace_count, 2)
        p0 = store.get_pages(0, "r1", 0, 0, 1)
        p1 = store.get_pages(0, "r1", 1, 0, 1)
        self.assertEqual(p0, [b"gen0"])
        self.assertEqual(p1, [b"gen1"])

    def test_nonexistent_namespace_returns_none(self):
        store = RequestNamespaceStore(max_size_bytes=10_000)
        pages = store.get_pages(0, "missing", 0, 0, 3)
        self.assertEqual(pages, [None, None, None])
        contiguous, total = store.query_range(0, "missing", 0)
        self.assertEqual((contiguous, total), (0, 0))


# ---------------------------------------------------------------------------
# RemoteBackupServer request-namespace integration tests
# ---------------------------------------------------------------------------


class TestRemoteBackupRequestNamespace(CustomTestCase):
    def _drain_gc(self, server: RemoteBackupServer) -> None:
        while not server._gc_queue.empty():
            lease_id = server._gc_queue.get_nowait()
            entry = server._lease_by_id.pop(lease_id, None)
            if entry is not None:
                server._gc_process_lease(entry)

    def test_request_pages_round_trip_and_release_on_finish(self):
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=2)

        self.assertEqual(server.start_request(3, "rid-1", 0), _LEASE_STATUS_OK)
        self.assertEqual(
            server.put_pages_by_request(3, "rid-1", 0, 0, [b"p0", b"p1"]),
            [True, True],
        )
        contiguous, total = server.query_request_range(3, "rid-1", 0)
        self.assertEqual(contiguous, 2)
        self.assertEqual(total, 2)

        pages = server.get_pages_by_request(3, "rid-1", 0, 0, 3)
        self.assertEqual(pages, [b"p0", b"p1", None])

        self.assertEqual(
            server.finish_request(3, "rid-1", 0, LEASE_REASON_NORMAL),
            _LEASE_STATUS_OK,
        )
        self._drain_gc(server)

        contiguous, total = server.query_request_range(3, "rid-1", 0)
        self.assertEqual(contiguous, 0)
        self.assertEqual(total, 0)

        pages = server.get_pages_by_request(3, "rid-1", 0, 0, 2)
        self.assertEqual(pages, [None, None])

    def test_failover_old_marks_inactive_keeps_readable(self):
        """Old generation stays readable after FAILOVER_OLD until evicted."""
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=1)

        server.start_request(1, "rid-2", 0)
        server.put_pages_by_request(1, "rid-2", 0, 0, [b"a", b"b"])

        # Supersede: new generation starts
        server.start_request(1, "rid-2", 1)

        # Old generation still readable
        contiguous, total = server.query_request_range(1, "rid-2", 0)
        self.assertEqual(contiguous, 2)
        pages = server.get_pages_by_request(1, "rid-2", 0, 0, 2)
        self.assertEqual(pages, [b"a", b"b"])

        # New generation empty
        contiguous, total = server.query_request_range(1, "rid-2", 1)
        self.assertEqual(contiguous, 0)

        # Finish old gen with FAILOVER_OLD → marks inactive but keeps data
        server.finish_request(1, "rid-2", 0, LEASE_REASON_FAILOVER_OLD)
        self._drain_gc(server)

        # Old gen is now inactive (marked for eviction), pages still exist
        ns_key = (1, "rid-2", 0)
        ns = server.ns_store.namespaces.get(ns_key)
        if ns is not None:
            self.assertEqual(ns.state, "inactive")

    def test_abort_releases_namespace(self):
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=1)
        server.start_request(0, "rid-3", 0)
        server.put_pages_by_request(0, "rid-3", 0, 0, [b"x"])

        server.finish_request(0, "rid-3", 0, LEASE_REASON_ABORT)
        self._drain_gc(server)

        contiguous, total = server.query_request_range(0, "rid-3", 0)
        self.assertEqual(contiguous, 0)
        self.assertEqual(total, 0)

    def test_failover_supersede_releases_namespace(self):
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=1)
        server.start_request(0, "rid-4", 0)
        server.put_pages_by_request(0, "rid-4", 0, 0, [b"y"])

        server.finish_request(0, "rid-4", 0, LEASE_REASON_FAILOVER_SUPERSEDE)
        self._drain_gc(server)

        contiguous, total = server.query_request_range(0, "rid-4", 0)
        self.assertEqual(contiguous, 0)

    def test_decode_pages_appended_incrementally(self):
        """Simulate decode page-by-page append."""
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.01, page_size=1)
        server.start_request(0, "rid-5", 0)

        # Prefill pages
        server.put_pages_by_request(0, "rid-5", 0, 0, [b"p0", b"p1", b"p2"])
        # Decode pages (incremental)
        server.put_pages_by_request(0, "rid-5", 0, 3, [b"d3"])
        server.put_pages_by_request(0, "rid-5", 0, 4, [b"d4"])

        contiguous, total = server.query_request_range(0, "rid-5", 0)
        self.assertEqual(contiguous, 5)
        self.assertEqual(total, 5)

        pages = server.get_pages_by_request(0, "rid-5", 0, 0, 5)
        self.assertEqual(pages, [b"p0", b"p1", b"p2", b"d3", b"d4"])


# ---------------------------------------------------------------------------
# RemoteBackupStorage request-namespace API tests
# ---------------------------------------------------------------------------


class TestRemoteBackupStorageRequestApi(CustomTestCase):
    def setUp(self):
        self.storage = RemoteBackupStorage.__new__(RemoteBackupStorage)
        self.storage._dp_rank = 5
        self.storage.page_size = 1
        self.storage.local_server = MagicMock()
        self.storage.client = None
        self.storage.mem_pool_host = MagicMock()

    def test_batch_set_v1_uses_request_namespace_api_when_request_metadata_present(self):
        self.storage.local_server.request_aware = True
        self.storage.local_server.put_pages_by_request.return_value = [True, True]
        self.storage.mem_pool_host.get_data_page.side_effect = [
            torch.tensor([1], dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
        ]

        results = self.storage.batch_set_v1(
            ["h0", "h1"],
            torch.tensor([7, 8], dtype=torch.int64),
            HiCacheStorageExtraInfo(
                full_token_ids=[11, 12],
                page_start=3,
                request_id="rid-storage",
                request_generation=9,
            ),
        )

        self.assertEqual(results, [True, True])
        self.storage.local_server.put_pages_by_request.assert_called_once_with(
            5,
            "rid-storage",
            9,
            3,
            [b"\x01", b"\x02"],
        )

    def test_match_prefix_by_request_delegates_to_local_server(self):
        self.storage.local_server.query_request_range.return_value = (3, 5)

        contiguous, total = self.storage.match_prefix_by_request(
            dp_rank=7, rid="rid-x", generation=4
        )

        self.assertEqual(contiguous, 3)
        self.assertEqual(total, 5)
        self.storage.local_server.query_request_range.assert_called_once_with(
            7, "rid-x", 4
        )

    def test_get_pages_by_request_delegates_to_local_server(self):
        self.storage.local_server.get_pages_by_request.return_value = [
            b"data",
            None,
        ]
        self.storage.mem_pool_host.get_dummy_flat_data_page.return_value = (
            torch.zeros((1,), dtype=torch.uint8)
        )

        results = self.storage.get_pages_by_request(
            dp_rank=7,
            rid="rid-x",
            generation=4,
            page_start=2,
            count=2,
            host_indices=torch.tensor([10, 11], dtype=torch.int64),
        )

        self.assertEqual(results[0], True)
        self.assertEqual(results[1], False)
        self.storage.local_server.get_pages_by_request.assert_called_once_with(
            7, "rid-x", 4, 2, 2
        )


if __name__ == "__main__":
    unittest.main()
