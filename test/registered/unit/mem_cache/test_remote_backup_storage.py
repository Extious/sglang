"""Unit tests for request-level remote backup storage."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo
from sglang.srt.mem_cache.storage.remote_backup.remote_backup_server import (
    LEASE_REASON_FAILOVER_OLD,
    LEASE_REASON_NORMAL,
    _LEASE_STATUS_OK,
    RemoteBackupServer,
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
        self.storage.local_server.match_prefix_from.assert_called_once_with(
            1, [1, 2, 3, 4], 2
        )

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
        self.storage.local_server.get_pages_by_tokens.assert_called_once_with(
            1, [11], 0, 1
        )
        self.storage.mem_pool_host.set_from_flat_data_page.assert_called_once()


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
        self.assertEqual(
            server.match_request_pages(3, "rid-1", 0, start_page=0, count=4),
            (2, 2),
        )
        self.assertEqual(
            server.get_pages_by_request(3, "rid-1", 0, start_page=0, count=3),
            [b"p0", b"p1", None],
        )

        self.assertEqual(
            server.finish_request(3, "rid-1", 0, LEASE_REASON_NORMAL),
            _LEASE_STATUS_OK,
        )
        self._drain_gc(server)

        self.assertEqual(
            server.match_request_pages(3, "rid-1", 0, start_page=0, count=4),
            (0, 0),
        )
        self.assertEqual(
            server.get_pages_by_request(3, "rid-1", 0, start_page=0, count=2),
            [None, None],
        )

    def test_new_generation_does_not_remove_old_namespace_until_finished(self):
        server = RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=1)

        self.assertEqual(server.start_request(1, "rid-2", 0), _LEASE_STATUS_OK)
        self.assertEqual(
            server.put_pages_by_request(1, "rid-2", 0, 0, [b"a", b"b"]),
            [True, True],
        )

        self.assertEqual(server.start_request(1, "rid-2", 1), _LEASE_STATUS_OK)
        self.assertEqual(
            server.match_request_pages(1, "rid-2", 0, start_page=0, count=4),
            (2, 2),
        )
        self.assertEqual(
            server.get_pages_by_request(1, "rid-2", 0, start_page=0, count=2),
            [b"a", b"b"],
        )
        self.assertEqual(
            server.match_request_pages(1, "rid-2", 1, start_page=0, count=4),
            (0, 0),
        )

        self.assertEqual(
            server.finish_request(1, "rid-2", 0, LEASE_REASON_FAILOVER_OLD),
            _LEASE_STATUS_OK,
        )
        self._drain_gc(server)

        self.assertEqual(
            server.match_request_pages(1, "rid-2", 0, start_page=0, count=4),
            (0, 0),
        )
        self.assertEqual(
            server.match_request_pages(1, "rid-2", 1, start_page=0, count=4),
            (0, 0),
        )


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

    def test_match_request_pages_uses_exact_failover_dp_rank(self):
        self.storage.local_server.match_request_pages.return_value = (3, 3)

        result = self.storage.match_request_pages(
            "rid-storage", generation=4, start_page=2, count=5, lookup_dp_rank=7
        )

        self.assertEqual(result, (3, 3))
        self.storage.local_server.match_request_pages.assert_called_once_with(
            7, "rid-storage", 4, 2, 5
        )


if __name__ == "__main__":
    unittest.main()
