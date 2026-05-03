from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "benchmark" / "multi_agent" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_deploy_server_module():
    path = REPO_ROOT / "benchmark" / "multi_agent" / "src" / "server" / "deploy_server.py"
    spec = importlib.util.spec_from_file_location("deploy_server_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_remote_backup_server_module():
    path = (
        REPO_ROOT
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "storage"
        / "remote_backup"
        / "remote_backup_server.py"
    )
    spec = importlib.util.spec_from_file_location("remote_backup_server_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RemoteBackupLaunchPlanTest(unittest.TestCase):
    def test_remote_backup_config_waits_for_prefetch_completion(self):
        module = _load_deploy_server_module()

        cfg = module.load_json_config(
            REPO_ROOT / "benchmark" / "multi_agent" / "src" / "config" / "remote_backup" / "server.json"
        )

        self.assertEqual(cfg["load_balance_method"], "total_tokens")
        self.assertEqual(cfg["hicache_storage_prefetch_policy"], "wait_complete")

    def test_deploy_server_preserves_failover_events_env(self):
        deploy_server_path = (
            REPO_ROOT / "benchmark" / "multi_agent" / "src" / "server" / "deploy_server.py"
        )

        source = deploy_server_path.read_text(encoding="utf-8")

        self.assertNotIn('pop("SGLANG_FAILOVER_EVENTS_FILE"', source)

    def test_remote_backup_launch_uses_direct_process(self):
        module = _load_deploy_server_module()

        plan = module.build_remote_backup_launch_plan(
            repo_root=REPO_ROOT,
            venv_python=Path("/repo/.venv/bin/python"),
            backup_port=30000,
            backup_buffer_gb=120.0,
        )

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.client_host, "127.0.0.1")
        self.assertIsNone(plan.cleanup_cmd)
        self.assertEqual(
            plan.cmd[:3],
            [
                "/repo/.venv/bin/python",
                "-u",
                "-m",
            ],
        )
        self.assertNotIn("srun", plan.cmd)

    def test_remote_backup_launch_always_direct_regardless_of_node_count(self):
        module = _load_deploy_server_module()

        plan = module.build_remote_backup_launch_plan(
            repo_root=REPO_ROOT,
            venv_python=Path("/repo/.venv/bin/python"),
            backup_port=35000,
            backup_buffer_gb=32.0,
        )

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.client_host, "127.0.0.1")
        self.assertIsNone(plan.cleanup_cmd)
        self.assertNotIn("srun", plan.cmd)


class RemoteBackupServerLifecycleTest(unittest.TestCase):
    def test_remote_backup_server_stop_closes_tcp_server(self):
        module = _load_remote_backup_server_module()

        class FakeTCPServer:
            def __init__(self):
                self.shutdown_called = False
                self.server_close_called = False

            def shutdown(self):
                self.shutdown_called = True

            def server_close(self):
                self.server_close_called = True

        fake = FakeTCPServer()
        server = module.RemoteBackupServer(port=0, max_buffer_size_gb=0.001, page_size=1)
        server._tcp_server = fake

        server.stop()

        self.assertTrue(fake.shutdown_called)
        self.assertTrue(fake.server_close_called)
        self.assertIsNone(server._tcp_server)


if __name__ == "__main__":
    unittest.main()
