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
