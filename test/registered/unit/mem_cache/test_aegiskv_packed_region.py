"""Unit tests for AegisKV's packed partial-layer HBM regions."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.aegiskv.kv_copy import (
    PackedRegion,
    arena_to_region,
    region_to_device,
)
from sglang.test.test_utils import CustomTestCase


def _region(*, token_capacity, resident_layers, num_layers=8, first_slot=100):
    m = PackedRegion.num_slots_needed(
        token_capacity=token_capacity,
        resident_layers=resident_layers,
        num_layers=num_layers,
    )
    return PackedRegion(
        slots=torch.arange(first_slot, first_slot + m),
        token_capacity=token_capacity,
        resident_layers=resident_layers,
        num_layers=num_layers,
    )


class TestPackedRegion(CustomTestCase):
    def test_cells_cover_region_exactly_once(self):
        """Every (layer, token) of the region maps to a distinct (pool layer, slot)
        inside the reserved slots; an overlap would let one layer overwrite another."""
        for capacity, layers in ((7, 3), (64, 4), (10, 8), (1, 1)):
            region = _region(token_capacity=capacity, resident_layers=layers)
            seen = set()
            for layer in range(layers):
                covered = 0
                for pool_layer, s_lo, s_hi, t_lo, t_hi in region.runs(
                    layer=layer, token_lo=0, token_hi=capacity
                ):
                    self.assertEqual(s_hi - s_lo, t_hi - t_lo)
                    self.assertLess(pool_layer, region.num_layers)
                    self.assertEqual(t_lo, covered)
                    covered = t_hi
                    for s in range(s_lo, s_hi):
                        self.assertNotIn((pool_layer, s), seen)
                        seen.add((pool_layer, s))
                self.assertEqual(covered, capacity)
            self.assertEqual(len(seen), layers * capacity)

    def test_partial_stages_round_trip(self):
        """Staging a region in several token ranges and expanding it back yields
        the original per-layer KV, as a takeover import relies on."""
        num_layers, heads, dim, capacity, layers = 8, 2, 4, 13, 3
        k = [torch.zeros(400, heads, dim) for _ in range(num_layers)]
        v = [torch.zeros(400, heads, dim) for _ in range(num_layers)]
        arena = torch.randn(num_layers, 2, 50, heads, dim)
        region = _region(token_capacity=capacity, resident_layers=layers)
        for lo, hi in ((0, 5), (5, 6), (6, 13)):
            arena_to_region(
                k_buffers=k,
                v_buffers=v,
                region=region,
                arena=arena,
                src_start=20,
                token_lo=lo,
                token_hi=hi,
            )
        dst = torch.arange(300, 300 + capacity)
        region_to_device(k_buffers=k, v_buffers=v, region=region, dst_slots=dst)
        for layer in range(layers):
            self.assertTrue(torch.equal(k[layer][dst], arena[layer, 0, 20:33]))
            self.assertTrue(torch.equal(v[layer][dst], arena[layer, 1, 20:33]))


if __name__ == "__main__":
    unittest.main()
