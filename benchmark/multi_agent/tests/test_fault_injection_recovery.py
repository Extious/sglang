from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "multi_agent" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_task_events(path: Path, count: int) -> None:
    path.write_text(
        "".join(json.dumps({"event": "task_completed"}) + "\n" for _ in range(count)),
        encoding="utf-8",
    )


class FaultInjectionRecoveryTest(unittest.TestCase):
    def test_fault_injection_config_timeline_only_has_no_job_trigger_or_extra_delay(self):
        from failure.config import FaultInjectionConfig

        cfg = FaultInjectionConfig.from_dict({"timeline_after_start_s": "20"})

        self.assertEqual(cfg.after_job, "")
        self.assertEqual(cfg.after_task, "")
        self.assertEqual(cfg.timeline_after_start_s, "20")
        self.assertEqual(cfg.delay, 0)

    def test_fault_injection_config_job_trigger_keeps_explicit_delay(self):
        from failure.config import FaultInjectionConfig

        cfg = FaultInjectionConfig.from_dict(
            {
                "inject_after_job": "10",
                "delay": 5,
                "recover_after_job": "12",
            }
        )

        self.assertEqual(cfg.after_job, "10")
        self.assertEqual(cfg.after_task, "")
        self.assertEqual(cfg.timeline_after_start_s, "")
        self.assertEqual(cfg.delay, 5)
        self.assertEqual(cfg.recover_after_job, "12")

    def test_monitor_until_dispatches_timeline_trigger_without_new_events(self):
        from failure.injector_core import FaultInjector, TimelineTrigger

        injector = FaultInjector.__new__(FaultInjector)
        injector.events_file = None
        injector._fault_threads = []
        injector._start_time = 100.0
        injector._triggers = [
            TimelineTrigger(
                fire_at_s=[1.0],
                stage_targets=[
                    {
                        "dp_rank": "0",
                        "pp_rank": "0",
                        "tp_rank": "0",
                        "node": "gpu0",
                        "node_url": "http://gpu0:28000",
                        "kill_pattern": "sglang::scheduler_DP0",
                    }
                ],
                inject_delay=0.0,
            )
        ]
        captured = []
        injector._schedule_injection = captured.append

        import failure.injector_core as injector_core

        original_time = injector_core.time.time
        original_sleep = injector_core.time.sleep
        current_time = {"value": 100.0}

        try:
            injector_core.time.time = lambda: current_time["value"]
            injector_core.time.sleep = lambda seconds: current_time.__setitem__(
                "value", current_time["value"] + max(float(seconds), 0.01)
            )
            injector.monitor_until(lambda: len(captured) == 1, timeout_s=2.0, poll_s=0.0)
        finally:
            injector_core.time.time = original_time
            injector_core.time.sleep = original_sleep

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].fields["trigger_kind"], "timeline")

    def test_recovery_task_target_advances_when_configured_target_already_passed(self):
        from failure.injector_core import FaultInjector

        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "events.jsonl"
            _write_task_events(events_file, 45)
            injector = FaultInjector.__new__(FaultInjector)
            injector.events_file = events_file

            self.assertEqual(injector._resolve_recovery_task_target(40), 46)

    def test_recovery_task_target_preserves_future_absolute_target(self):
        from failure.injector_core import FaultInjector

        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "events.jsonl"
            _write_task_events(events_file, 32)
            injector = FaultInjector.__new__(FaultInjector)
            injector.events_file = events_file

            self.assertEqual(injector._resolve_recovery_task_target(40), 40)


if __name__ == "__main__":
    unittest.main()
