import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.peer.peer_cache_storage import PeerCacheStorage


class TestPeerCacheStorageMapping(unittest.TestCase):
    @patch("sglang.srt.mem_cache.storage.peer.peer_cache_storage.PeerCacheClient")
    @patch("sglang.srt.mem_cache.storage.peer.peer_cache_storage.PeerCacheServer")
    def test_peer_target_map_uses_dp_pp_tp_stage_key(
        self, mock_server_cls, mock_client_cls
    ):
        mem_pool_host = SimpleNamespace(page_size=64)
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=1,
            pp_size=2,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="dummy",
            dp_rank=0,
            dp_size=2,
            extra_config={
                "peer_port": 29000,
                "peer_buffer_size_gb": 16.0,
                "peer_targets": {
                    "0:1:0": "node0:29000",
                    "1:1:0": "node1:29000",
                },
            },
        )

        PeerCacheStorage(storage_config, mem_pool_host)

        mock_server_cls.assert_called_once_with(29001, 16.0, page_size=64)
        mock_client_cls.assert_called_once_with("node1:29003")

    @patch("sglang.srt.mem_cache.storage.peer.peer_cache_storage.PeerCacheClient")
    @patch("sglang.srt.mem_cache.storage.peer.peer_cache_storage.PeerCacheServer")
    def test_peer_url_fallback_keeps_stage_port_alignment(
        self, mock_server_cls, mock_client_cls
    ):
        mem_pool_host = SimpleNamespace(page_size=32)
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=2,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="dummy",
            dp_rank=0,
            dp_size=2,
            extra_config={
                "peer_port": 29000,
                "peer_buffer_size_gb": 8.0,
                "peer_url": "backup-node:29000",
            },
        )

        PeerCacheStorage(storage_config, mem_pool_host)

        mock_server_cls.assert_called_once_with(29000, 8.0, page_size=32)
        mock_client_cls.assert_called_once_with("backup-node:29002")


if __name__ == "__main__":
    unittest.main()
