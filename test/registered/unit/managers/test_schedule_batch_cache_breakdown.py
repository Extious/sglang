"""Unit tests for cache source breakdown accounting."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import apply_cache_source_breakdown

register_cpu_ci(5, "stage-a-test-cpu")


class TestScheduleBatchCacheBreakdown(CustomTestCase):
    def test_remote_prefetch_is_not_misattributed_as_device_reuse(self):
        req = SimpleNamespace(
            prefix_indices=torch.arange(20000, dtype=torch.int64),
            host_hit_length=5,
            storage_hit_length=19995,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            reused_tokens_device=0,
            reused_tokens_host=0,
            reused_tokens_storage=0,
        )

        apply_cache_source_breakdown(req)

        self.assertEqual(req.cached_tokens_device, 5)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 19995)
        self.assertEqual(req.reused_tokens_device, 5)
        self.assertEqual(req.reused_tokens_host, 0)
        self.assertEqual(req.reused_tokens_storage, 19995)

    def test_host_only_hits_still_count_as_host_reuse(self):
        req = SimpleNamespace(
            prefix_indices=torch.arange(512, dtype=torch.int64),
            host_hit_length=300,
            storage_hit_length=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            reused_tokens_device=0,
            reused_tokens_host=0,
            reused_tokens_storage=0,
        )

        apply_cache_source_breakdown(req)

        self.assertEqual(req.cached_tokens_device, 212)
        self.assertEqual(req.cached_tokens_host, 300)
        self.assertEqual(req.cached_tokens_storage, 0)
        self.assertEqual(req.reused_tokens_device, 212)
        self.assertEqual(req.reused_tokens_host, 300)
        self.assertEqual(req.reused_tokens_storage, 0)


if __name__ == "__main__":
    unittest.main()
