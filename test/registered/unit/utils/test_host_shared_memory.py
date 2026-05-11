"""Unit tests for stale shared-memory recovery in HostSharedMemoryManager."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang.srt.utils.host_shared_memory import HostSharedMemoryManager, _Record

register_cpu_ci(5, "stage-a-test-cpu")


class TestHostSharedMemoryManager(CustomTestCase):
    def test_create_rank0_shm_reclaims_stale_segment(self):
        manager = HostSharedMemoryManager("sglang-test-shm")
        manager._closed = True  # Prevent atexit / __del__ cleanup noise in test.

        stale = MagicMock()
        fresh = MagicMock()

        def fake_shared_memory(*, name, create=False, size=None):
            self.assertEqual(name, "sglang-test-shm_op1")
            if create:
                if not hasattr(fake_shared_memory, "retried"):
                    fake_shared_memory.retried = True
                    raise FileExistsError()
                self.assertEqual(size, 128)
                return fresh
            return stale

        with patch(
            "sglang.srt.utils.host_shared_memory.shared_memory.SharedMemory",
            side_effect=fake_shared_memory,
        ):
            shm = manager._create_rank0_shm("sglang-test-shm_op1", 128)

        self.assertIs(shm, fresh)
        stale.close.assert_called_once_with()
        stale.unlink.assert_called_once_with()

    def test_close_unlinks_records_on_rank0(self):
        manager = HostSharedMemoryManager("sglang-test-shm")
        shm = MagicMock()
        tensor = MagicMock()
        tensor.data_ptr.return_value = 123
        manager._records = [_Record(shm=shm, np_array=MagicMock(), tensor=tensor)]

        with patch(
            "sglang.srt.utils.host_shared_memory.get_naive_distributed",
            return_value=SimpleNamespace(get_rank=lambda: 0),
        ), patch.object(manager, "_cuda_host_unregister") as unregister:
            manager.close()

        unregister.assert_called_once_with(123)
        shm.close.assert_called_once_with()
        shm.unlink.assert_called_once_with()
        self.assertEqual(manager._records, [])

    def test_close_does_not_unlink_on_non_owner_rank(self):
        manager = HostSharedMemoryManager("sglang-test-shm")
        shm = MagicMock()
        tensor = MagicMock()
        tensor.data_ptr.return_value = 456
        manager._records = [_Record(shm=shm, np_array=MagicMock(), tensor=tensor)]

        with patch(
            "sglang.srt.utils.host_shared_memory.get_naive_distributed",
            return_value=SimpleNamespace(get_rank=lambda: 1),
        ), patch.object(manager, "_cuda_host_unregister") as unregister:
            manager.close()

        unregister.assert_called_once_with(456)
        shm.close.assert_called_once_with()
        shm.unlink.assert_not_called()

    def test_open_non_owner_shm_retries_until_segment_is_visible(self):
        manager = HostSharedMemoryManager("sglang-test-shm")
        manager._closed = True  # Prevent atexit / __del__ cleanup noise in test.

        attached = MagicMock()
        call_count = {"value": 0}

        def fake_shared_memory(*, name, create=False, size=None):
            self.assertEqual(name, "sglang-test-shm_op1")
            self.assertFalse(create)
            self.assertIsNone(size)
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise FileNotFoundError()
            return attached

        with patch(
            "sglang.srt.utils.host_shared_memory.shared_memory.SharedMemory",
            side_effect=fake_shared_memory,
        ), patch("sglang.srt.utils.host_shared_memory.time.sleep") as sleep:
            shm = manager._open_non_owner_shm("sglang-test-shm_op1")

        self.assertIs(shm, attached)
        self.assertEqual(call_count["value"], 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
