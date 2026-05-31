from __future__ import annotations

import unittest

from benchmark.sglsim.synthesis_requests.utils import response_to_detail_row


class ResponseToDetailRowCacheDetailsTest(unittest.TestCase):
    def test_load_back_tokens_from_cached_tokens_details_host(self):
        row = response_to_detail_row(
            "1-t2",
            {
                "usage": {
                    "prompt_tokens": 36081,
                    "completion_tokens": 512,
                    "prompt_tokens_details": {"cached_tokens": 15310},
                    "cached_tokens_details": {
                        "device": 12000,
                        "host": 3310,
                        "reused_device": 12000,
                        "reused_host": 3310,
                    },
                }
            },
            e2e_latency_s=45.0,
            prepared_prompt_len=36081,
            prepared_output_len=512,
        )

        self.assertEqual(row.cache_hit_tokens, 15310)
        self.assertEqual(row.load_back_tokens, 3310)

    def test_load_back_tokens_default_zero_without_details(self):
        row = response_to_detail_row(
            "1-t0",
            {
                "usage": {
                    "prompt_tokens": 14799,
                    "completion_tokens": 512,
                    "prompt_tokens_details": {"cached_tokens": 3},
                }
            },
            e2e_latency_s=55.0,
            prepared_prompt_len=14799,
            prepared_output_len=512,
        )

        self.assertEqual(row.cache_hit_tokens, 3)
        self.assertEqual(row.load_back_tokens, 0)


if __name__ == "__main__":
    unittest.main()
