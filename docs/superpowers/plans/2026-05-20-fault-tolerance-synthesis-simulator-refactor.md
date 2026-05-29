# Fault Tolerance Synthesis Simulator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `benchmark/fault_tolerance` into a simulator-first synthesis workload benchmark with fast `offline` experiments and real-GPU-adjacent `online` blocking-server experiments, without depending on `benchmark/multi_agent`.

**Architecture:** Keep fault-tolerance mechanics in `tools/sglang-simulator` and keep benchmark orchestration under `benchmark/fault_tolerance/simulator/synthesis_requests`. The shared `common` package owns experiment config, synthetic request generation, simulator config generation, artifact collection, and reporting; `offline` runs in-process with logical time; `online` launches a simulator-hooked SGLang HTTP server in `BLOCKING` mode and drives it through a real client flow. The `real` directory is created as a boundary marker only, with no implementation in this plan.

**Tech Stack:** Python, pytest, SGLang simulator hooks, SGLang HTTP server, OpenAI-compatible request payloads, `aiconfigurator` with heuristic fallback, CSV/JSONL experiment artifacts.

---

## Scope

This plan includes:

- `benchmark/fault_tolerance/simulator/synthesis_requests/common`
- `benchmark/fault_tolerance/simulator/synthesis_requests/offline`
- `benchmark/fault_tolerance/simulator/synthesis_requests/online`
- `benchmark/fault_tolerance/real/README.md` as a placeholder boundary document
- simulator metadata/predictor improvements needed by both offline and online
- tests for config, dataset generation, payload generation, reporting, offline orchestration, and online command/client construction

This plan excludes:

- `crewai_agents`
- migrating old `benchmark/multi_agent` code
- changing production non-simulator failover logic under `python/sglang/srt`
- running real GPU integration tests in CI

## Target File Structure

Create:

- `benchmark/fault_tolerance/real/README.md`
  - states that non-simulator real-GPU experiments will live here in a separate phase.
- `benchmark/fault_tolerance/simulator/synthesis_requests/__init__.py`
  - package marker.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/__init__.py`
  - package marker and selected exports.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/schema.py`
  - dataclasses/enums for experiment config, strategy, execution mode, request rows, and artifact paths.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/config.py`
  - load/save experiment config from JSON files and generate strategy-specific simulator configs.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/dataset.py`
  - deterministic synthetic request generator, replacing the old fixed-agent wording.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/payload.py`
  - translate synthetic requests into SGLang generate kwargs and HTTP JSON payloads with simulation metadata.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/artifacts.py`
  - collect raw simulator outputs from single-DP and multi-DP output directories.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/reporting.py`
  - write `job_summary.csv`, `task_summary.csv`, `cache_hits.csv`, and `request_detail.csv`.
- `benchmark/fault_tolerance/simulator/synthesis_requests/common/suite.py`
  - resolve strategy suites and output directories.
- `benchmark/fault_tolerance/simulator/synthesis_requests/offline/__init__.py`
  - package marker.
- `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_one.py`
  - run one in-process offline simulator experiment.
- `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_suite.py`
  - run baseline, host backup, and remote backup offline experiments.
- `benchmark/fault_tolerance/simulator/synthesis_requests/online/__init__.py`
  - package marker.
- `benchmark/fault_tolerance/simulator/synthesis_requests/online/server.py`
  - build and launch simulator-hooked SGLang server command.
- `benchmark/fault_tolerance/simulator/synthesis_requests/online/client.py`
  - send synthetic requests to the running server and trigger profile output.
- `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_one.py`
  - run one online experiment by launching server, running client, collecting artifacts, and shutting down server.
- `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_suite.py`
  - run baseline, host backup, and remote backup online experiments.
- `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/base.json`
  - shared synthesis workload/server/predictor/failure defaults.
- `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/strategies.json`
  - strategy overrides for `baseline`, `host_backup`, `remote_backup`.
- `benchmark/fault_tolerance/simulator/synthesis_requests/README.md`
  - usage for offline and online experiments.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py`
  - config and simulator config generation tests.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py`
  - deterministic request generation tests.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py`
  - payload metadata tests.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py`
  - artifact collection and CSV reporting tests.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py`
  - offline orchestration tests with fakes.
- `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py`
  - online command/client orchestration tests with fakes.
- `tools/sglang-simulator/test/test_simulation_predictor_metadata.py`
  - predictor default/fallback metadata tests.

Modify:

- `tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py`
  - expose predictor metadata and keep `aiconfigurator` as the default.
- `tools/sglang-simulator/src/sglang_simulator/time_predictor/base.py`
  - add a stable predictor name property if missing.
- `tools/sglang-simulator/src/sglang_simulator/time_predictor/aiconfigurator.py`
  - set predictor name to `aiconfigurator`.
- `tools/sglang-simulator/src/sglang_simulator/time_predictor/heuristic.py`
  - set predictor name to `heuristic`.
- `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
  - include predictor/execution metadata in `metrics.json` or a sidecar `simulator_metadata.json`.
- `benchmark/fault_tolerance/simulator/src/simulator/*`
  - remove after replacement or leave thin compatibility wrappers that import from `synthesis_requests.common` for one transition.

---

## Task 1: Add Synthesis Request Schema And Config Loader

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/__init__.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/__init__.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/schema.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/config.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/base.json`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/strategies.json`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py`:

```python
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "synthesis_requests"
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    build_simulator_config,
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupStrategy,
    ExecutionMode,
)


def test_load_experiment_suite_from_json_files(tmp_path):
    cfg_dir = tmp_path / "synthesis-a100"
    cfg_dir.mkdir()
    (cfg_dir / "base.json").write_text(
        json.dumps(
            {
                "name": "synthesis-a100",
                "server": {
                    "model_path": "Qwen/Qwen3-8B",
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 1,
                    "accelerator_name": "a100_sxm",
                    "port": 30000
                },
                "workload": {
                    "input_len": 16,
                    "output_len": 4,
                    "num_requests": 8,
                    "app_workers": 2,
                    "request_rate": "inf",
                    "seed": 7
                },
                "failure": {
                    "enabled": True,
                    "inject_after_job": 2,
                    "recover_after_job": 6,
                    "delay": 0,
                    "dp_rank": 0
                },
                "predictor": {
                    "name": "aiconfigurator",
                    "database_mode": "SILICON"
                },
                "platform": {
                    "disk_read_bandwidth_gb": 8,
                    "disk_write_bandwidth_gb": 8,
                    "memory_read_bandwidth_gb": 64,
                    "memory_write_bandwidth_gb": 64,
                    "num_device_per_node": 8
                }
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "strategies.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "baseline": {"backup_policy": "none"},
                    "host_backup": {"backup_policy": "host"},
                    "remote_backup": {"backup_policy": "remote_backup"}
                }
            }
        ),
        encoding="utf-8",
    )

    suite = load_experiment_suite(cfg_dir)

    assert suite.name == "synthesis-a100"
    assert suite.server.model_path == "Qwen/Qwen3-8B"
    assert suite.workload.input_len == 16
    assert suite.failure.inject_after_job == 2
    assert suite.predictor.name == "aiconfigurator"
    assert [s.strategy for s in suite.strategies] == [
        BackupStrategy.BASELINE,
        BackupStrategy.HOST_BACKUP,
        BackupStrategy.REMOTE_BACKUP,
    ]


def test_build_simulator_config_sets_strategy_and_mode(tmp_path):
    suite = load_experiment_suite(
        Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100")
    )
    experiment = next(s for s in suite.strategies if s.strategy == BackupStrategy.HOST_BACKUP)

    payload = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.OFFLINE,
    )

    assert payload["predictor"]["name"] == "aiconfigurator"
    assert payload["scheduler"]["dp_size"] == suite.server.dp_size
    assert payload["scheduler"]["tp_size"] == suite.server.tp_size
    assert payload["failure"]["enabled"] is True
    assert payload["failure"]["backup_policy"] == "host"
    assert payload["failure"]["failed_dp_rank"] == suite.failure.dp_rank
    assert payload["benchmark"]["workload"] == "synthesis_requests"
    assert payload["benchmark"]["execution_mode"] == "offline"
    assert payload["benchmark"]["strategy"] == "host_backup"
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `benchmark.fault_tolerance.simulator.synthesis_requests`.

- [ ] **Step 3: Implement schema dataclasses and enums**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/schema.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BackupStrategy(str, Enum):
    BASELINE = "baseline"
    HOST_BACKUP = "host_backup"
    REMOTE_BACKUP = "remote_backup"


class BackupPolicy(str, Enum):
    NONE = "none"
    HOST = "host"
    REMOTE_BACKUP = "remote_backup"


class ExecutionMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"

    @property
    def simulator_env_value(self) -> str:
        return "OFFLINE" if self is ExecutionMode.OFFLINE else "BLOCKING"


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    dp_size: int
    tp_size: int
    pp_size: int
    accelerator_name: str
    port: int = 30000
    host: str = "127.0.0.1"


@dataclass(frozen=True)
class WorkloadConfig:
    input_len: int
    output_len: int
    num_requests: int
    app_workers: int
    request_rate: float
    seed: int


@dataclass(frozen=True)
class FailureConfig:
    enabled: bool
    inject_after_job: int
    recover_after_job: int
    delay: float
    dp_rank: int


@dataclass(frozen=True)
class PredictorConfig:
    name: str = "aiconfigurator"
    database_mode: str = "SILICON"
    database_path: str | None = None
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0


@dataclass(frozen=True)
class PlatformConfig:
    disk_read_bandwidth_gb: float
    disk_write_bandwidth_gb: float
    memory_read_bandwidth_gb: float
    memory_write_bandwidth_gb: float
    num_device_per_node: int


@dataclass(frozen=True)
class StrategyExperiment:
    strategy: BackupStrategy
    backup_policy: BackupPolicy


@dataclass(frozen=True)
class ExperimentSuite:
    name: str
    config_dir: Path
    server: ServerConfig
    workload: WorkloadConfig
    failure: FailureConfig
    predictor: PredictorConfig
    platform: PlatformConfig
    strategies: tuple[StrategyExperiment, ...]


@dataclass(frozen=True)
class SyntheticRequest:
    job_id: str
    worker_id: str
    worker_seq: int
    assigned_dp_rank: int
    token_ids: list[int]
    output_length: int
    created_time: float
    backup_policy: BackupPolicy
    attempt: int = 0

    @property
    def custom_params(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "worker_seq": self.worker_seq,
            "parallel_worker_client": True,
            "assigned_dp_rank": self.assigned_dp_rank,
            "backup_policy": self.backup_policy.value,
            "attempt": self.attempt,
            "created_time": self.created_time,
        }


@dataclass(frozen=True)
class ArtifactPaths:
    output_dir: Path
    simulator_raw_dir: Path
    simulator_config_path: Path
```

- [ ] **Step 4: Implement JSON config loader and simulator config builder**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/config.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    BackupPolicy,
    BackupStrategy,
    ExecutionMode,
    ExperimentSuite,
    FailureConfig,
    PlatformConfig,
    PredictorConfig,
    ServerConfig,
    StrategyExperiment,
    WorkloadConfig,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_request_rate(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    return float(value)


def _server_from_raw(raw: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        model_path=str(raw["model_path"]),
        dp_size=int(raw.get("dp_size", 1)),
        tp_size=int(raw.get("tp_size", 1)),
        pp_size=int(raw.get("pp_size", 1)),
        accelerator_name=str(raw.get("accelerator_name", "a100_sxm")),
        port=int(raw.get("port", 30000)),
        host=str(raw.get("host", "127.0.0.1")),
    )


def _workload_from_raw(raw: dict[str, Any]) -> WorkloadConfig:
    return WorkloadConfig(
        input_len=int(raw["input_len"]),
        output_len=int(raw["output_len"]),
        num_requests=int(raw["num_requests"]),
        app_workers=int(raw.get("app_workers", 1)),
        request_rate=_parse_request_rate(raw.get("request_rate", "inf")),
        seed=int(raw.get("seed", 1)),
    )


def _failure_from_raw(raw: dict[str, Any]) -> FailureConfig:
    return FailureConfig(
        enabled=bool(raw.get("enabled", True)),
        inject_after_job=int(raw.get("inject_after_job", 0) or 0),
        recover_after_job=int(raw.get("recover_after_job", 0) or 0),
        delay=float(raw.get("delay", 0) or 0),
        dp_rank=int(raw.get("dp_rank", raw.get("failed_dp_rank", 0)) or 0),
    )


def _predictor_from_raw(raw: dict[str, Any]) -> PredictorConfig:
    return PredictorConfig(
        name=str(raw.get("name", "aiconfigurator")),
        database_mode=str(raw.get("database_mode", "SILICON")),
        database_path=raw.get("database_path"),
        prefill_scale_factor=float(raw.get("prefill_scale_factor", 1.0)),
        decode_scale_factor=float(raw.get("decode_scale_factor", 1.0)),
    )


def _platform_from_raw(raw: dict[str, Any]) -> PlatformConfig:
    return PlatformConfig(
        disk_read_bandwidth_gb=float(raw.get("disk_read_bandwidth_gb", 8)),
        disk_write_bandwidth_gb=float(raw.get("disk_write_bandwidth_gb", 8)),
        memory_read_bandwidth_gb=float(raw.get("memory_read_bandwidth_gb", 64)),
        memory_write_bandwidth_gb=float(raw.get("memory_write_bandwidth_gb", 64)),
        num_device_per_node=int(raw.get("num_device_per_node", 8)),
    )


def load_experiment_suite(config_dir: Path) -> ExperimentSuite:
    base = _load_json(config_dir / "base.json")
    strategies_raw = _load_json(config_dir / "strategies.json")
    strategies = tuple(
        StrategyExperiment(
            strategy=BackupStrategy(name),
            backup_policy=BackupPolicy(raw["backup_policy"]),
        )
        for name, raw in strategies_raw["strategies"].items()
    )
    return ExperimentSuite(
        name=str(base["name"]),
        config_dir=config_dir,
        server=_server_from_raw(base["server"]),
        workload=_workload_from_raw(base["workload"]),
        failure=_failure_from_raw(base["failure"]),
        predictor=_predictor_from_raw(base.get("predictor", {})),
        platform=_platform_from_raw(base.get("platform", {})),
        strategies=strategies,
    )


def build_simulator_config(
    *,
    suite: ExperimentSuite,
    experiment: StrategyExperiment,
    mode: ExecutionMode,
) -> dict[str, Any]:
    predictor = {
        "name": suite.predictor.name,
        "database_mode": suite.predictor.database_mode,
        "prefill_scale_factor": suite.predictor.prefill_scale_factor,
        "decode_scale_factor": suite.predictor.decode_scale_factor,
    }
    if suite.predictor.database_path:
        predictor["database_path"] = suite.predictor.database_path

    return {
        "platform": {
            "accelerator": {
                "name": suite.server.accelerator_name,
                "hbm_capacity_gb": 80,
            },
            "disk_read_bandwidth_gb": suite.platform.disk_read_bandwidth_gb,
            "disk_write_bandwidth_gb": suite.platform.disk_write_bandwidth_gb,
            "memory_read_bandwidth_gb": suite.platform.memory_read_bandwidth_gb,
            "memory_write_bandwidth_gb": suite.platform.memory_write_bandwidth_gb,
            "num_device_per_node": suite.platform.num_device_per_node,
        },
        "predictor": predictor,
        "scheduler": {
            "tp_size": suite.server.tp_size,
            "pp_size": suite.server.pp_size,
            "dp_size": suite.server.dp_size,
            "backend_version": "0.5.9",
        },
        "failure": {
            "enabled": suite.failure.enabled,
            "backup_policy": experiment.backup_policy.value,
            "failed_dp_rank": suite.failure.dp_rank,
            "inject_after_job": suite.failure.inject_after_job,
            "inject_after_task": 0,
            "inject_delay_s": suite.failure.delay,
            "recover_after_job": suite.failure.recover_after_job,
            "recover_after_task": 0,
        },
        "benchmark": {
            "suite": suite.name,
            "workload": "synthesis_requests",
            "execution_mode": mode.value,
            "strategy": experiment.strategy.value,
            "request_rate": (
                "inf"
                if suite.workload.request_rate == float("inf")
                else suite.workload.request_rate
            ),
        },
    }
```

- [ ] **Step 5: Add default configs**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/base.json`:

```json
{
  "name": "synthesis-a100",
  "server": {
    "model_path": "Qwen/Qwen3-8B",
    "dp_size": 2,
    "tp_size": 1,
    "pp_size": 1,
    "accelerator_name": "a100_sxm",
    "host": "127.0.0.1",
    "port": 30000
  },
  "workload": {
    "input_len": 20000,
    "output_len": 1000,
    "num_requests": 32,
    "app_workers": 4,
    "request_rate": "inf",
    "seed": 1
  },
  "failure": {
    "enabled": true,
    "inject_after_job": 10,
    "recover_after_job": 20,
    "delay": 5,
    "dp_rank": 0
  },
  "predictor": {
    "name": "aiconfigurator",
    "database_mode": "SILICON",
    "prefill_scale_factor": 1.0,
    "decode_scale_factor": 1.0
  },
  "platform": {
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 8
  }
}
```

Create `benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100/strategies.json`:

```json
{
  "strategies": {
    "baseline": {
      "backup_policy": "none"
    },
    "host_backup": {
      "backup_policy": "host"
    },
    "remote_backup": {
      "backup_policy": "remote_backup"
    }
  }
}
```

- [ ] **Step 6: Run config tests and verify pass**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py -v
```

Expected: PASS for both tests.

- [ ] **Step 7: Commit config/schema work**

```bash
git add benchmark/fault_tolerance/simulator/synthesis_requests benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py
git commit -m "refactor: add synthesis simulator config schema"
```

---

## Task 2: Add Deterministic Synthetic Request Generation

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/dataset.py`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.dataset import (
    build_synthetic_requests,
    split_requests_by_worker,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    WorkloadConfig,
)


class FakeTokenizer:
    vocab_size = 1000


def test_build_synthetic_requests_sets_lengths_and_metadata():
    workload = WorkloadConfig(
        input_len=8,
        output_len=3,
        num_requests=4,
        app_workers=2,
        request_rate=float("inf"),
        seed=1,
    )

    requests = build_synthetic_requests(
        tokenizer=FakeTokenizer(),
        workload=workload,
        dp_size=2,
        backup_policy=BackupPolicy.REMOTE_BACKUP,
    )

    assert len(requests) == 4
    assert len(requests[0].token_ids) == 8
    assert requests[0].output_length == 3
    assert requests[0].job_id == "1"
    assert requests[0].worker_id == "1"
    assert requests[0].worker_seq == 0
    assert requests[2].worker_id == "1"
    assert requests[2].worker_seq == 1
    assert requests[0].assigned_dp_rank == 0
    assert requests[1].assigned_dp_rank == 1
    assert requests[0].backup_policy == BackupPolicy.REMOTE_BACKUP


def test_build_synthetic_requests_is_deterministic():
    workload = WorkloadConfig(
        input_len=6,
        output_len=2,
        num_requests=3,
        app_workers=1,
        request_rate=float("inf"),
        seed=9,
    )

    first = build_synthetic_requests(
        tokenizer=FakeTokenizer(),
        workload=workload,
        dp_size=2,
        backup_policy=BackupPolicy.NONE,
    )
    second = build_synthetic_requests(
        tokenizer=FakeTokenizer(),
        workload=workload,
        dp_size=2,
        backup_policy=BackupPolicy.NONE,
    )

    assert [r.token_ids for r in first] == [r.token_ids for r in second]


def test_split_requests_by_worker():
    workload = WorkloadConfig(
        input_len=4,
        output_len=2,
        num_requests=10,
        app_workers=3,
        request_rate=float("inf"),
        seed=1,
    )
    requests = build_synthetic_requests(
        tokenizer=FakeTokenizer(),
        workload=workload,
        dp_size=2,
        backup_policy=BackupPolicy.NONE,
    )

    queues = split_requests_by_worker(requests, app_workers=3)

    assert sum(len(v) for v in queues.values()) == 10
    assert len(queues["1"]) == 4
    assert len(queues["2"]) == 3
    assert len(queues["3"]) == 3
```

- [ ] **Step 2: Run dataset tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py -v
```

Expected: FAIL because `common.dataset` does not exist.

- [ ] **Step 3: Implement synthetic request generator**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/dataset.py`:

```python
from __future__ import annotations

import random

from .schema import BackupPolicy, SyntheticRequest, WorkloadConfig


def build_synthetic_requests(
    *,
    tokenizer,
    workload: WorkloadConfig,
    dp_size: int,
    backup_policy: BackupPolicy,
) -> list[SyntheticRequest]:
    rng = random.Random(workload.seed)
    vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    low = max(1, vocab // 4)
    high = max(low + 1, vocab * 3 // 4)
    worker_count = max(int(workload.app_workers), 1)
    dp_count = max(int(dp_size), 1)

    requests: list[SyntheticRequest] = []
    for idx in range(max(int(workload.num_requests), 0)):
        token_ids = [rng.randrange(low, high) for _ in range(max(workload.input_len, 1))]
        requests.append(
            SyntheticRequest(
                job_id=str(idx + 1),
                worker_id=str((idx % worker_count) + 1),
                worker_seq=idx // worker_count,
                assigned_dp_rank=idx % dp_count,
                token_ids=token_ids,
                output_length=max(workload.output_len, 1),
                created_time=0.0,
                backup_policy=backup_policy,
            )
        )
    return requests


def split_requests_by_worker(
    requests: list[SyntheticRequest],
    *,
    app_workers: int,
) -> dict[str, list[SyntheticRequest]]:
    worker_count = max(int(app_workers), 1)
    queues: dict[str, list[SyntheticRequest]] = {str(i + 1): [] for i in range(worker_count)}
    for request in requests:
        queues.setdefault(request.worker_id, []).append(request)
    return queues
```

- [ ] **Step 4: Run dataset tests and verify pass**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit dataset work**

```bash
git add benchmark/fault_tolerance/simulator/synthesis_requests/common/dataset.py benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py
git commit -m "refactor: add synthesis request generator"
```

---

## Task 3: Add Payload Translation Shared By Offline And Online

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/payload.py`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py`

- [ ] **Step 1: Write failing payload tests**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.payload import (
    build_generate_kwargs,
    build_http_generate_payload,
    with_total_request,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    SyntheticRequest,
)


def _request() -> SyntheticRequest:
    return SyntheticRequest(
        job_id="7",
        worker_id="2",
        worker_seq=3,
        assigned_dp_rank=1,
        token_ids=[11, 12, 13],
        output_length=5,
        created_time=4.5,
        backup_policy=BackupPolicy.HOST,
    )


def test_with_total_request_adds_total_request_without_mutating_request():
    params = with_total_request(_request(), total_request=9)

    assert params["job_id"] == "7"
    assert params["total_request"] == 9
    assert params["assigned_dp_rank"] == 1
    assert params["backup_policy"] == "host"


def test_build_generate_kwargs_routes_dp_rank_and_custom_params():
    kwargs = build_generate_kwargs(_request(), total_request=9)

    assert kwargs["input_ids"] == [11, 12, 13]
    assert kwargs["sampling_params"]["max_new_tokens"] == 5
    assert kwargs["sampling_params"]["ignore_eos"] is True
    assert kwargs["sampling_params"]["custom_params"]["simulation"]["total_request"] == 9
    assert kwargs["routed_dp_rank"] == 1


def test_build_http_generate_payload_uses_sampling_params_custom_params():
    payload = build_http_generate_payload(_request(), total_request=9)

    assert payload["input_ids"] == [11, 12, 13]
    assert payload["sampling_params"]["max_new_tokens"] == 5
    assert payload["sampling_params"]["custom_params"]["simulation"]["worker_id"] == "2"
    assert payload["sampling_params"]["custom_params"]["simulation"]["parallel_worker_client"] is True
```

- [ ] **Step 2: Run payload tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py -v
```

Expected: FAIL because `common.payload` does not exist.

- [ ] **Step 3: Implement payload helpers**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/payload.py`:

```python
from __future__ import annotations

from typing import Any

from .schema import SyntheticRequest


def with_total_request(request: SyntheticRequest, *, total_request: int) -> dict[str, Any]:
    simulation = dict(request.custom_params)
    simulation["total_request"] = int(total_request)
    return simulation


def build_generate_kwargs(request: SyntheticRequest, *, total_request: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": "",
        "input_ids": list(request.token_ids),
        "sampling_params": {
            "ignore_eos": True,
            "max_new_tokens": int(request.output_length),
            "custom_params": {
                "simulation": with_total_request(request, total_request=total_request),
            },
        },
    }
    kwargs["routed_dp_rank"] = int(request.assigned_dp_rank)
    return kwargs


def build_http_generate_payload(request: SyntheticRequest, *, total_request: int) -> dict[str, Any]:
    return build_generate_kwargs(request, total_request=total_request)
```

- [ ] **Step 4: Run payload tests and verify pass**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit payload work**

```bash
git add benchmark/fault_tolerance/simulator/synthesis_requests/common/payload.py benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py
git commit -m "refactor: add synthesis request payload helpers"
```

---

## Task 4: Move Artifact Collection And Reporting Into Common

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/artifacts.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/reporting.py`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py`
- Modify: `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`

- [ ] **Step 1: Write failing artifact/reporting tests**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py`:

```python
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.artifacts import (
    collect_jsonl_records,
    copy_simulator_artifacts,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.reporting import (
    write_outputs,
)


def test_collect_jsonl_records_reads_dp_directories(tmp_path):
    dp0 = tmp_path / "dp_0"
    dp1 = tmp_path / "dp_1"
    dp0.mkdir()
    dp1.mkdir()
    (dp0 / "request.jsonl").write_text('{"rid":"a"}\n', encoding="utf-8")
    (dp1 / "request.jsonl").write_text('{"rid":"b"}\n', encoding="utf-8")

    rows = collect_jsonl_records(tmp_path, "request.jsonl")

    assert [row["rid"] for row in rows] == ["a", "b"]


def test_copy_simulator_artifacts_writes_flat_files(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    (raw / "dp_0").mkdir(parents=True)
    (raw / "dp_0" / "request.jsonl").write_text('{"rid":"a"}\n', encoding="utf-8")
    (raw / "dp_0" / "iteration.jsonl").write_text('{"mode":"decode"}\n', encoding="utf-8")
    (raw / "dp_0" / "failure_events.jsonl").write_text(
        '{"event":"sim_gpu_failure"}\n',
        encoding="utf-8",
    )
    out.mkdir()

    copy_simulator_artifacts(raw, out)

    assert (out / "simulator_request.jsonl").read_text(encoding="utf-8").strip() == '{"rid":"a"}'
    assert (out / "simulator_iteration.jsonl").read_text(encoding="utf-8").strip() == '{"mode":"decode"}'
    assert (out / "sim_failure_events.jsonl").read_text(encoding="utf-8").strip() == '{"event":"sim_gpu_failure"}'


def test_write_outputs_creates_csvs(tmp_path):
    request_stats = [
        {
            "rid": "r1",
            "job_id": "1",
            "worker_id": "1",
            "assigned_dp_rank": 0,
            "backup_policy": "remote_backup",
            "status": "completed",
            "input_length": 10,
            "output_length": 4,
            "created_time": 0.0,
            "queue_start": 0.0,
            "queue_end": 0.5,
            "last_event_time": 1.7,
            "queue_time_s": 0.5,
            "prefetch_time_s": 0.2,
            "backup_time_s": 0.1,
            "inference_time_s": 1.0,
            "prefill_inference_time_s": 0.3,
            "decode_inference_time_s": 0.7,
            "is_failover_retried": True,
            "failure_impacted": True,
            "failure_time_s": 1.2,
            "recovery_time_s": -1,
            "pre_failover_output_tokens": 2,
            "pre_failover_backed_up_tokens": 2,
            "retry_prefill_tokens": 10,
            "failover_penalty_s": 0.2,
            "final_reused_tokens": 0,
            "prefetch_complete_tokens": 2,
        }
    ]

    write_outputs(tmp_path, request_stats)

    assert (tmp_path / "job_summary.csv").is_file()
    assert (tmp_path / "task_summary.csv").is_file()
    assert (tmp_path / "cache_hits.csv").is_file()
    assert (tmp_path / "request_detail.csv").is_file()
    detail_rows = list(csv.DictReader((tmp_path / "request_detail.csv").open()))
    assert detail_rows[0]["waiting_time_s"] == "0.5"
    assert detail_rows[0]["total_latency_s"] == "1.7"
    cache_rows = list(csv.DictReader((tmp_path / "cache_hits.csv").open()))
    assert cache_rows[0]["remote_prefetch"] == "2"
    assert cache_rows[0]["reused_storage"] == "2"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py -v
```

Expected: FAIL because `common.artifacts` and `common.reporting` do not exist.

- [ ] **Step 3: Implement artifact helpers**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def collect_jsonl_records(base_dir: Path, filename: str) -> list[dict]:
    records: list[dict] = []
    paths = sorted(base_dir.glob(f"dp_*/{filename}"))
    if not paths:
        paths = [base_dir / filename]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _copy_jsonl(raw_dir: Path, output_dir: Path, source_name: str, target_name: str) -> None:
    lines: list[str] = []
    paths = sorted(raw_dir.glob(f"dp_*/{source_name}"))
    if not paths:
        paths = [raw_dir / source_name]
    for path in paths:
        if path.is_file():
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    (output_dir / target_name).write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )


def copy_simulator_artifacts(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_jsonl(raw_dir, output_dir, "request.jsonl", "simulator_request.jsonl")
    _copy_jsonl(raw_dir, output_dir, "iteration.jsonl", "simulator_iteration.jsonl")
    _copy_jsonl(raw_dir, output_dir, "failure_events.jsonl", "sim_failure_events.jsonl")
```

- [ ] **Step 4: Move existing reporting logic into common**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/reporting.py` by moving the current implementation from `benchmark/fault_tolerance/simulator/src/simulator/reporting.py` and renaming the public writer to `write_outputs(output_dir: Path, request_stats: list[dict]) -> None`.

The compatibility wrapper in `benchmark/fault_tolerance/simulator/src/simulator/reporting.py` should contain:

```python
from __future__ import annotations

from pathlib import Path

from benchmark.fault_tolerance.simulator.synthesis_requests.common.reporting import (
    write_outputs,
)


def write_simulator_fixed_outputs(output_dir: Path, request_stats: list[dict]) -> None:
    write_outputs(output_dir, request_stats)
```

- [ ] **Step 5: Run artifact/reporting tests and existing reporting tests**

Run:

```bash
PYTHONPATH=. pytest \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py \
  benchmark/fault_tolerance/simulator/tests/test_simulator_reporting.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit artifact/reporting work**

```bash
git add \
  benchmark/fault_tolerance/simulator/synthesis_requests/common/artifacts.py \
  benchmark/fault_tolerance/simulator/synthesis_requests/common/reporting.py \
  benchmark/fault_tolerance/simulator/src/simulator/reporting.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py
git commit -m "refactor: share synthesis simulator artifact reporting"
```

---

## Task 5: Implement Offline Runner And Suite

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/common/suite.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/offline/__init__.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_one.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_suite.py`
- Modify: `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py`

- [ ] **Step 1: Write failing offline runner tests**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py`:

```python
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupStrategy,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one import (
    run_offline_experiment,
)


class FakeTokenizer:
    vocab_size = 1000


class FakeRunner:
    def __init__(self, server_args):
        self.server_args = server_args
        self.shutdown_called = False

    def benchmark(self, benchmark_config, dataset):
        self.dataset_size = len(dataset)
        return {"completed": len(dataset)}

    def get_request_stats(self):
        return [
            {
                "rid": "r1",
                "job_id": "1",
                "worker_id": "1",
                "backup_policy": "host",
                "status": "completed",
                "queue_time_s": 0.0,
                "prefetch_time_s": 0.0,
                "backup_time_s": 0.0,
                "inference_time_s": 0.1,
                "last_event_time": 0.1,
            }
        ]

    def shutdown(self):
        self.shutdown_called = True


def test_run_offline_experiment_writes_config_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda model_path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one.SGLangBenchmarkRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one.ServerArgs",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = run_offline_experiment(
        config_dir=Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"),
        strategy=BackupStrategy.HOST_BACKUP,
        output_dir=tmp_path,
    )

    config = json.loads((tmp_path / "simulator_config.json").read_text(encoding="utf-8"))
    assert result.metrics["completed"] == 32
    assert config["benchmark"]["execution_mode"] == "offline"
    assert config["failure"]["backup_policy"] == "host"
    assert (tmp_path / "request_detail.csv").is_file()
```

- [ ] **Step 2: Run offline runner tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py -v
```

Expected: FAIL because `offline.run_one` does not exist.

- [ ] **Step 3: Implement suite helpers**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/common/suite.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema import ArtifactPaths, BackupStrategy, ExperimentSuite, StrategyExperiment


@dataclass(frozen=True)
class ExperimentResult:
    strategy: BackupStrategy
    output_dir: Path
    metrics: dict


def get_strategy(suite: ExperimentSuite, strategy: BackupStrategy) -> StrategyExperiment:
    for experiment in suite.strategies:
        if experiment.strategy == strategy:
            return experiment
    raise ValueError(f"Strategy {strategy.value!r} is not configured in suite {suite.name!r}")


def default_output_dir(
    *,
    root: Path,
    suite_name: str,
    mode: str,
    strategy: BackupStrategy,
) -> Path:
    return root / suite_name / mode / strategy.value


def prepare_artifact_paths(output_dir: Path) -> ArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "simulator_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        output_dir=output_dir,
        simulator_raw_dir=raw_dir,
        simulator_config_path=output_dir / "simulator_config.json",
    )
```

- [ ] **Step 4: Implement offline single-run entry point**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_one.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from sglang.srt.server_args import ServerArgs
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner

from ..common.artifacts import copy_simulator_artifacts
from ..common.config import build_simulator_config, load_experiment_suite
from ..common.dataset import build_synthetic_requests
from ..common.reporting import write_outputs
from ..common.schema import BackupStrategy, ExecutionMode
from ..common.suite import ExperimentResult, get_strategy, prepare_artifact_paths


class _OfflineDataset:
    def __init__(self, requests):
        self.requests = list(requests)

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)


def run_offline_experiment(
    *,
    config_dir: Path,
    strategy: BackupStrategy,
    output_dir: Path,
) -> ExperimentResult:
    suite = load_experiment_suite(config_dir)
    experiment = get_strategy(suite, strategy)
    artifacts = prepare_artifact_paths(output_dir)
    sim_config = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.OFFLINE,
    )
    artifacts.simulator_config_path.write_text(
        json.dumps(sim_config, indent=2),
        encoding="utf-8",
    )

    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(artifacts.simulator_config_path)
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(artifacts.simulator_raw_dir)
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    tokenizer = AutoTokenizer.from_pretrained(suite.server.model_path)
    requests = build_synthetic_requests(
        tokenizer=tokenizer,
        workload=suite.workload,
        dp_size=suite.server.dp_size,
        backup_policy=experiment.backup_policy,
    )
    dataset = _OfflineDataset(requests)
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=suite.server.model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_storage_backend="file",
            tp_size=suite.server.tp_size,
            pp_size=suite.server.pp_size,
            dp_size=suite.server.dp_size,
            page_size=1,
        )
    )
    try:
        metrics = runner.benchmark(BenchmarkConfig(), dataset)
        request_stats = runner.get_request_stats()
        write_outputs(artifacts.output_dir, request_stats)
        copy_simulator_artifacts(artifacts.simulator_raw_dir, artifacts.output_dir)
        return ExperimentResult(strategy=strategy, output_dir=artifacts.output_dir, metrics=metrics)
    finally:
        runner.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=[item.value for item in BackupStrategy], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_offline_experiment(
        config_dir=args.config_dir,
        strategy=BackupStrategy(args.strategy),
        output_dir=args.output_dir,
    )
    print(json.dumps({"strategy": result.strategy.value, "output_dir": str(result.output_dir), "metrics": result.metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement offline suite entry point**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/offline/run_suite.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config import load_experiment_suite
from ..common.schema import BackupStrategy
from ..common.suite import default_output_dir
from .run_one import run_offline_experiment


def run_offline_suite(*, config_dir: Path, output_root: Path) -> list[dict]:
    suite = load_experiment_suite(config_dir)
    results = []
    for experiment in suite.strategies:
        output_dir = default_output_dir(
            root=output_root,
            suite_name=suite.name,
            mode="offline",
            strategy=experiment.strategy,
        )
        result = run_offline_experiment(
            config_dir=config_dir,
            strategy=experiment.strategy,
            output_dir=output_dir,
        )
        results.append(
            {
                "strategy": result.strategy.value,
                "output_dir": str(result.output_dir),
                "metrics": result.metrics,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark/fault_tolerance/results/synthesis_requests"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_offline_suite(config_dir=args.config_dir, output_root=args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Replace old fixed pipeline with compatibility wrapper**

Replace `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py` with:

```python
from __future__ import annotations

from pathlib import Path

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    build_simulator_config,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one import (
    main,
    run_offline_experiment,
)


def build_simulator_config_payload(*args, **kwargs):
    raise RuntimeError(
        "build_simulator_config_payload was replaced by "
        "benchmark.fault_tolerance.simulator.synthesis_requests.common.config.build_simulator_config"
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

If existing tests still import `build_simulator_config_payload`, update those tests in the same commit to import `build_simulator_config` from the new path.

- [ ] **Step 7: Run offline tests**

Run:

```bash
PYTHONPATH=. pytest \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_config.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_dataset.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_payload.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_artifacts_reporting.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit offline runner work**

```bash
git add \
  benchmark/fault_tolerance/simulator/synthesis_requests/common/suite.py \
  benchmark/fault_tolerance/simulator/synthesis_requests/offline \
  benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py \
  benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_offline_runner.py
git commit -m "refactor: add synthesis offline simulator runner"
```

---

## Task 6: Add Predictor Metadata And Default Verification

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/time_predictor/base.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/time_predictor/aiconfigurator.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/time_predictor/heuristic.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
- Test: `tools/sglang-simulator/test/test_simulation_predictor_metadata.py`

- [ ] **Step 1: Write failing predictor metadata tests**

Create `tools/sglang-simulator/test/test_simulation_predictor_metadata.py`:

```python
import json
from pathlib import Path
from types import SimpleNamespace

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.time_predictor.heuristic import HeuristicTimePredictor


def test_predictor_config_defaults_to_aiconfigurator(tmp_path, monkeypatch):
    config_path = tmp_path / "sim.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
                "scheduler": {"tp_size": 1, "dp_size": 1, "pp_size": 1, "backend_version": "0.5.9"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    assert ConfigManager.get_predictor_config()["name"] == "aiconfigurator"


def test_heuristic_predictor_has_stable_name():
    predictor = HeuristicTimePredictor(
        model=SimpleNamespace(num_hidden_layers=32, hidden_size=4096),
        hw=SimpleNamespace(name="a100_sxm"),
        config=SimpleNamespace(),
    )

    assert predictor.name == "heuristic"
```

- [ ] **Step 2: Run predictor metadata tests and verify failure**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_predictor_metadata.py -v
```

Expected: FAIL because `ConfigManager.get_predictor_config` or `predictor.name` is missing.

- [ ] **Step 3: Add predictor name to base and concrete predictors**

Modify `tools/sglang-simulator/src/sglang_simulator/time_predictor/base.py` so `InferTimePredictor` exposes:

```python
class InferTimePredictor:
    name = "base"

    def __init__(self, model, hw, config):
        self.model = model
        self.hw = hw
        self.config = config
```

Modify `tools/sglang-simulator/src/sglang_simulator/time_predictor/aiconfigurator.py`:

```python
class AIConfiguratorTimePredictor(InferTimePredictor):
    name = "aiconfigurator"
```

Modify `tools/sglang-simulator/src/sglang_simulator/time_predictor/heuristic.py`:

```python
class HeuristicTimePredictor(InferTimePredictor):
    name = "heuristic"
```

- [ ] **Step 4: Add predictor config accessor**

Modify `tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py`:

```python
@classmethod
def get_predictor_config(cls) -> dict:
    raw = cls._get_raw_config().get("predictor", {})
    predictor_config = dict(raw) if isinstance(raw, dict) else {}
    predictor_config.setdefault("name", "aiconfigurator")
    return predictor_config
```

Then update `get_inference_time_predictor` to call:

```python
predictor_config = cls.get_predictor_config()
predictor_name = predictor_config.get("name", "aiconfigurator")
```

- [ ] **Step 5: Emit predictor metadata from scheduler profile output**

Modify `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py` inside the profile dump block, next to `metrics.json`, to write:

```python
metadata = {
    "predictor": getattr(C_SchedulerHook.INFERENCE_PREDICTOR, "name", "unknown"),
    "simulation_mode": C_SchedulerHook.SIM_MODE.value,
    "failure_enabled": C_SchedulerHook.FAILURE_MANAGER is not None,
}
metadata_path = os.path.join(base_output_dir, "simulator_metadata.json")
with open(metadata_path, "w") as f:
    f.write(json.dumps(metadata) + "\n")
```

- [ ] **Step 6: Run predictor tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_predictor_metadata.py -v
```

Expected: PASS.

- [ ] **Step 7: Run simulator unit tests touched by predictor/config**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest \
  tools/sglang-simulator/test/test_simulation_failure_manager.py \
  tools/sglang-simulator/test/test_simulation_time_predictor.py \
  tools/sglang-simulator/test/test_simulation_predictor_metadata.py \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit predictor metadata work**

```bash
git add \
  tools/sglang-simulator/src/sglang_simulator/time_predictor/base.py \
  tools/sglang-simulator/src/sglang_simulator/time_predictor/aiconfigurator.py \
  tools/sglang-simulator/src/sglang_simulator/time_predictor/heuristic.py \
  tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py \
  tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py \
  tools/sglang-simulator/test/test_simulation_predictor_metadata.py
git commit -m "feat: record simulator predictor metadata"
```

---

## Task 7: Implement Online Server Command Builder

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/online/__init__.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/online/server.py`
- Test: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py`

- [ ] **Step 1: Write failing online server command test**

Create `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py` with this first test:

```python
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.server import (
    build_server_command,
    build_server_env,
)


def test_build_server_command_uses_simulator_launch_server(tmp_path):
    suite = load_experiment_suite(
        Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100")
    )
    sim_config = tmp_path / "simulator_config.json"
    raw_dir = tmp_path / "raw"

    cmd = build_server_command(suite=suite, sim_config_path=sim_config)
    env = build_server_env(sim_config_path=sim_config, simulator_raw_dir=raw_dir)

    assert cmd[:3] == [sys.executable, "-m", "sglang_simulator.simulation.sglang.launch_server"]
    assert "--model-path" in cmd
    assert suite.server.model_path in cmd
    assert "--port" in cmd
    assert str(suite.server.port) in cmd
    assert env["SGLANG_SIMULATOR_CONFIG_PATH"] == str(sim_config)
    assert env["SGLANG_SIMULATOR_OUTPUT_DIR"] == str(raw_dir)
    assert env["SGLANG_SIMULATOR_OUTPUT_MODE"] == "BLOCKING"
    assert "SGLANG_USE_CPU_ENGINE" not in env
```

- [ ] **Step 2: Run online test and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py::test_build_server_command_uses_simulator_launch_server -v
```

Expected: FAIL because `online.server` does not exist.

- [ ] **Step 3: Implement online server helpers**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/online/server.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..common.schema import ExperimentSuite


def build_server_command(*, suite: ExperimentSuite, sim_config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        suite.server.model_path,
        "--host",
        suite.server.host,
        "--port",
        str(suite.server.port),
        "--tp-size",
        str(suite.server.tp_size),
        "--dp-size",
        str(suite.server.dp_size),
        "--enable-hierarchical-cache",
        "--hicache-storage-backend",
        "file",
        "--sim-config-path",
        str(sim_config_path),
    ]


def build_server_env(*, sim_config_path: Path, simulator_raw_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config_path)
    env["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(simulator_raw_dir)
    env["SGLANG_SIMULATOR_OUTPUT_MODE"] = "BLOCKING"
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    return env


def start_server(
    *,
    suite: ExperimentSuite,
    sim_config_path: Path,
    simulator_raw_dir: Path,
) -> subprocess.Popen:
    return subprocess.Popen(
        build_server_command(suite=suite, sim_config_path=sim_config_path),
        env=build_server_env(
            sim_config_path=sim_config_path,
            simulator_raw_dir=simulator_raw_dir,
        ),
    )
```

- [ ] **Step 4: Run online server command test and verify pass**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py::test_build_server_command_uses_simulator_launch_server -v
```

Expected: PASS.

- [ ] **Step 5: Commit online server helper work**

```bash
git add benchmark/fault_tolerance/simulator/synthesis_requests/online benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py
git commit -m "feat: add synthesis online server harness"
```

---

## Task 8: Implement Online HTTP Client And Run-One Orchestration

**Files:**
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/online/client.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_one.py`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_suite.py`
- Modify: `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py`

- [ ] **Step 1: Extend online tests for client payload and run orchestration**

Append to `benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py`:

```python
import json
from types import SimpleNamespace

from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    BackupStrategy,
    SyntheticRequest,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.client import (
    build_generate_url,
    run_online_client,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one import (
    run_online_experiment,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


class FakeHttpClient:
    def __init__(self):
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return FakeResponse(json)


def test_build_generate_url():
    assert build_generate_url("127.0.0.1", 30000) == "http://127.0.0.1:30000/generate"


def test_run_online_client_sends_all_requests():
    client = FakeHttpClient()
    requests = [
        SyntheticRequest(
            job_id="1",
            worker_id="1",
            worker_seq=0,
            assigned_dp_rank=0,
            token_ids=[1, 2],
            output_length=3,
            created_time=0.0,
            backup_policy=BackupPolicy.NONE,
        )
    ]

    results = run_online_client(
        host="127.0.0.1",
        port=30000,
        requests=requests,
        total_request=1,
        http_client=client,
    )

    assert results == [{"ok": True}]
    assert client.posts[0][0] == "http://127.0.0.1:30000/generate"
    assert client.posts[0][1]["sampling_params"]["custom_params"]["simulation"]["job_id"] == "1"


def test_run_online_experiment_writes_config_and_stops_server(tmp_path, monkeypatch):
    class FakeTokenizer:
        vocab_size = 1000

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()

    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda model_path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one.start_server",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one.wait_for_server_ready",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one.run_online_client",
        lambda **kwargs: [{"ok": True}],
    )
    monkeypatch.setattr(
        "benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one.trigger_profile",
        lambda **kwargs: None,
    )
    (tmp_path / "simulator_raw").mkdir()
    (tmp_path / "simulator_raw" / "request.jsonl").write_text(
        json.dumps({"rid": "r1", "job_id": "1", "status": "completed"}) + "\n",
        encoding="utf-8",
    )

    result = run_online_experiment(
        config_dir=Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"),
        strategy=BackupStrategy.BASELINE,
        output_dir=tmp_path,
    )

    assert result.metrics["client_completed"] == 1
    assert json.loads((tmp_path / "simulator_config.json").read_text())["benchmark"]["execution_mode"] == "online"
    assert process.terminated is True
```

- [ ] **Step 2: Run online tests and verify failure**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py -v
```

Expected: FAIL because `online.client` and `online.run_one` do not exist.

- [ ] **Step 3: Implement online client**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/online/client.py`:

```python
from __future__ import annotations

import time
from typing import Any

import requests as http_requests

from ..common.payload import build_http_generate_payload
from ..common.schema import SyntheticRequest


def build_generate_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}/generate"


def build_profile_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}/start_profile"


def wait_for_server_ready(*, host: str, port: int, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{int(port)}/health"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = http_requests.get(url, timeout=5)
            if response.status_code < 500:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout_s}s: {last_error}")


def run_online_client(
    *,
    host: str,
    port: int,
    requests: list[SyntheticRequest],
    total_request: int,
    http_client: Any = http_requests,
) -> list[dict]:
    url = build_generate_url(host, port)
    results: list[dict] = []
    for request in requests:
        response = http_client.post(
            url,
            json=build_http_generate_payload(request, total_request=total_request),
            timeout=3600,
        )
        response.raise_for_status()
        results.append(response.json())
    return results


def trigger_profile(*, host: str, port: int, http_client: Any = http_requests) -> None:
    response = http_client.post(build_profile_url(host, port), json={}, timeout=300)
    response.raise_for_status()
```

- [ ] **Step 4: Implement online single-run orchestration**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_one.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from ..common.artifacts import collect_jsonl_records, copy_simulator_artifacts
from ..common.config import build_simulator_config, load_experiment_suite
from ..common.dataset import build_synthetic_requests
from ..common.reporting import write_outputs
from ..common.schema import BackupStrategy, ExecutionMode
from ..common.suite import ExperimentResult, get_strategy, prepare_artifact_paths
from .client import run_online_client, trigger_profile, wait_for_server_ready
from .server import start_server


def _stop_server(process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except Exception:
        process.kill()
        process.wait(timeout=30)


def run_online_experiment(
    *,
    config_dir: Path,
    strategy: BackupStrategy,
    output_dir: Path,
) -> ExperimentResult:
    suite = load_experiment_suite(config_dir)
    experiment = get_strategy(suite, strategy)
    artifacts = prepare_artifact_paths(output_dir)
    sim_config = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.ONLINE,
    )
    artifacts.simulator_config_path.write_text(
        json.dumps(sim_config, indent=2),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(suite.server.model_path)
    synthetic_requests = build_synthetic_requests(
        tokenizer=tokenizer,
        workload=suite.workload,
        dp_size=suite.server.dp_size,
        backup_policy=experiment.backup_policy,
    )
    process = start_server(
        suite=suite,
        sim_config_path=artifacts.simulator_config_path,
        simulator_raw_dir=artifacts.simulator_raw_dir,
    )
    try:
        wait_for_server_ready(host=suite.server.host, port=suite.server.port)
        client_results = run_online_client(
            host=suite.server.host,
            port=suite.server.port,
            requests=synthetic_requests,
            total_request=len(synthetic_requests),
        )
        trigger_profile(host=suite.server.host, port=suite.server.port)
        request_stats = collect_jsonl_records(artifacts.simulator_raw_dir, "request.jsonl")
        write_outputs(artifacts.output_dir, request_stats)
        copy_simulator_artifacts(artifacts.simulator_raw_dir, artifacts.output_dir)
        return ExperimentResult(
            strategy=strategy,
            output_dir=artifacts.output_dir,
            metrics={"client_completed": len(client_results)},
        )
    finally:
        _stop_server(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=[item.value for item in BackupStrategy], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_online_experiment(
        config_dir=args.config_dir,
        strategy=BackupStrategy(args.strategy),
        output_dir=args.output_dir,
    )
    print(json.dumps({"strategy": result.strategy.value, "output_dir": str(result.output_dir), "metrics": result.metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement online suite entry point**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/online/run_suite.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config import load_experiment_suite
from ..common.suite import default_output_dir
from .run_one import run_online_experiment


def run_online_suite(*, config_dir: Path, output_root: Path) -> list[dict]:
    suite = load_experiment_suite(config_dir)
    results = []
    for experiment in suite.strategies:
        output_dir = default_output_dir(
            root=output_root,
            suite_name=suite.name,
            mode="online",
            strategy=experiment.strategy,
        )
        result = run_online_experiment(
            config_dir=config_dir,
            strategy=experiment.strategy,
            output_dir=output_dir,
        )
        results.append(
            {
                "strategy": result.strategy.value,
                "output_dir": str(result.output_dir),
                "metrics": result.metrics,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark/fault_tolerance/results/synthesis_requests"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_online_suite(config_dir=args.config_dir, output_root=args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run online tests and verify pass**

Run:

```bash
PYTHONPATH=. pytest benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit online orchestration work**

```bash
git add benchmark/fault_tolerance/simulator/synthesis_requests/online benchmark/fault_tolerance/simulator/tests/synthesis_requests/test_online_runner.py
git commit -m "feat: add synthesis online experiment runner"
```

---

## Task 9: Add Boundary Docs And Update README

**Files:**
- Create: `benchmark/fault_tolerance/real/README.md`
- Create: `benchmark/fault_tolerance/simulator/synthesis_requests/README.md`
- Modify: `benchmark/fault_tolerance/simulator/docs/simulator_fixed_failure.md`

- [ ] **Step 1: Create real boundary README**

Create `benchmark/fault_tolerance/real/README.md`:

```markdown
# Real Fault Tolerance Experiments

This directory is reserved for non-simulator fault-tolerance experiments that run against real SGLang GPU inference.

The current implementation work is focused on `benchmark/fault_tolerance/simulator/synthesis_requests`.
```

- [ ] **Step 2: Create synthesis requests README**

Create `benchmark/fault_tolerance/simulator/synthesis_requests/README.md`:

```markdown
# Synthesis Request Fault Tolerance Simulator

This benchmark runs synthetic fixed-length request workloads against SGLang simulator fault-tolerance behavior.

## Modes

- `offline`: runs an in-process simulator engine with logical time. This is the fast research/debug path.
- `online`: launches `sglang_simulator.simulation.sglang.launch_server` in `BLOCKING` mode and drives it through HTTP requests. This is the real-GPU-adjacent path for validating server/client behavior.

## Strategies

- `baseline`: no KV backup, retry re-prefills all required tokens.
- `host_backup`: simulated GPU failure with host memory surviving.
- `remote_backup`: simulated GPU failure with remote KV backup available.

## Offline Suite

```bash
PYTHONPATH=.:tools/sglang-simulator/src \
python -m benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_suite \
  --config-dir benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100 \
  --output-root benchmark/fault_tolerance/results/synthesis_requests
```

## Online Suite

The online suite starts a simulator-hooked SGLang server and does not set `SGLANG_USE_CPU_ENGINE`.

```bash
PYTHONPATH=.:tools/sglang-simulator/src \
python -m benchmark.fault_tolerance.simulator.synthesis_requests.online.run_suite \
  --config-dir benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100 \
  --output-root benchmark/fault_tolerance/results/synthesis_requests
```

## Outputs

Each strategy directory contains:

- `simulator_config.json`
- `simulator_request.jsonl`
- `simulator_iteration.jsonl`
- `sim_failure_events.jsonl`
- `request_detail.csv`
- `job_summary.csv`
- `task_summary.csv`
- `cache_hits.csv`
```

- [ ] **Step 3: Update old simulator fixed doc to point to the new path**

Modify `benchmark/fault_tolerance/simulator/docs/simulator_fixed_failure.md` to start with:

```markdown
> This document describes the first fixed-workload prototype. New synthesis request experiments live under `benchmark/fault_tolerance/simulator/synthesis_requests`.
```

- [ ] **Step 4: Commit docs**

```bash
git add benchmark/fault_tolerance/real/README.md benchmark/fault_tolerance/simulator/synthesis_requests/README.md benchmark/fault_tolerance/simulator/docs/simulator_fixed_failure.md
git commit -m "docs: document synthesis fault tolerance simulator"
```

---

## Task 10: Verification And Cleanup

**Files:**
- Modify: `benchmark/fault_tolerance/simulator/src/simulator/*`
- Modify: `benchmark/fault_tolerance/simulator/tests/*`
- Modify: `tools/sglang-simulator/test/*`

- [ ] **Step 1: Run benchmark fault-tolerance unit tests**

Run:

```bash
PYTHONPATH=.:tools/sglang-simulator/src pytest benchmark/fault_tolerance/simulator/tests -v
```

Expected: PASS.

- [ ] **Step 2: Run simulator unit tests relevant to failure, runner, scheduler, and predictor**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest \
  tools/sglang-simulator/test/test_simulation_failure_manager.py \
  tools/sglang-simulator/test/test_simulation_scheduler_failure.py \
  tools/sglang-simulator/test/test_simulation_sglang_runner.py \
  tools/sglang-simulator/test/test_simulation_sglang_scheduler.py \
  tools/sglang-simulator/test/test_simulation_predictor_metadata.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run a small offline smoke experiment**

Run:

```bash
PYTHONPATH=.:tools/sglang-simulator/src \
python -m benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one \
  --config-dir benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100 \
  --strategy baseline \
  --output-dir /tmp/sglang-ft-synthesis-offline-baseline
```

Expected:

- command exits with code 0;
- `/tmp/sglang-ft-synthesis-offline-baseline/simulator_config.json` exists;
- `/tmp/sglang-ft-synthesis-offline-baseline/request_detail.csv` exists;
- `/tmp/sglang-ft-synthesis-offline-baseline/simulator_request.jsonl` exists.

- [ ] **Step 4: Inspect generated failure config**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/sglang-ft-synthesis-offline-baseline/simulator_config.json')
data = json.loads(p.read_text())
print(data['benchmark']['execution_mode'], data['benchmark']['strategy'], data['failure']['backup_policy'])
PY
```

Expected output:

```text
offline baseline none
```

- [ ] **Step 5: Run import checks for online modules without launching GPU server**

Run:

```bash
PYTHONPATH=.:tools/sglang-simulator/src python - <<'PY'
from pathlib import Path
from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import load_experiment_suite
from benchmark.fault_tolerance.simulator.synthesis_requests.online.server import build_server_command
suite = load_experiment_suite(Path('benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100'))
cmd = build_server_command(suite=suite, sim_config_path=Path('/tmp/sim.json'))
print(cmd[0], cmd[1], cmd[2])
PY
```

Expected output contains:

```text
-m sglang_simulator.simulation.sglang.launch_server
```

- [ ] **Step 6: Remove obsolete generated cache files from git status**

Run:

```bash
git status --short
```

Expected: source/test/doc changes only. If `__pycache__`, `.pytest_cache`, or experiment outputs appear, remove those untracked generated files with:

```bash
find benchmark/fault_tolerance tools/sglang-simulator -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf tools/sglang-simulator/.pytest_cache
```

- [ ] **Step 7: Final commit**

```bash
git add benchmark/fault_tolerance tools/sglang-simulator docs/superpowers/plans/2026-05-20-fault-tolerance-synthesis-simulator-refactor.md
git commit -m "refactor: organize fault tolerance synthesis simulator"
```

---

## Manual Online GPU Smoke Test

Run this only on a machine with the intended GPUs and model access:

```bash
PYTHONPATH=.:tools/sglang-simulator/src \
python -m benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one \
  --config-dir benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100 \
  --strategy baseline \
  --output-dir /tmp/sglang-ft-synthesis-online-baseline
```

Expected:

- server launches through `sglang_simulator.simulation.sglang.launch_server`;
- `SGLANG_SIMULATOR_OUTPUT_MODE` is `BLOCKING`;
- generated `simulator_config.json` has `"execution_mode": "online"`;
- `request_detail.csv` and `simulator_request.jsonl` are written;
- server process is terminated after profile/artifact collection.

---

## Self-Review

- Spec coverage: covers the requested `simulator` and `real` split, focuses only on `simulator/synthesis_requests`, includes both `offline` and `online`, excludes `crewai_agents`, and includes simulator predictor metadata.
- Placeholder scan: no open-ended implementation placeholders remain; each code-changing task includes exact file paths, expected test failures, implementation snippets, and verification commands.
- Type consistency: `BackupStrategy`, `BackupPolicy`, `ExecutionMode`, `ExperimentSuite`, `SyntheticRequest`, and `ExperimentResult` are introduced before they are consumed by later tasks.
