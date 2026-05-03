"""Unit tests for dp-rank override handling in RemoteBackupStorage."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

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


if __name__ == "__main__":
    unittest.main()
