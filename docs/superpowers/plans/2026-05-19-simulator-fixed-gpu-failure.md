# Simulator Fixed GPU Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-workload, offline simulator path that runs baseline, host-backup, and remote-backup GPU-failure experiments with scheduler-internal retry and per-request cache/latency CSV outputs.

**Architecture:** Leave the existing `benchmark/multi_agent` GPU pipeline untouched and create a new standalone benchmark under `benchmark/fault_tolerance/simulator`. The simulator scheduler receives all requests in `OFFLINE` mode, injects logical GPU failure inside the scheduler event loop, captures affected in-flight requests, rebuilds retry requests through SGLang's existing `handle_generate_request` path, and attributes waiting, prefetch, backup, and inference time to each logical request.

**Tech Stack:** Python, pytest, SGLang scheduler hooks, `tools/sglang-simulator`, benchmark CSV writers in `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`.

---

## Scope And Semantics

This plan implements the "second version": GPU failure and retry happen inside the simulator scheduler hook, not as post-processing.

The first implementation covers fixed workload only:

- one fixed request is one benchmark job and one benchmark task;
- `baseline` means no backup, retry re-prefills all original input plus pre-failure output;
- `host_backup` means simulated GPU failure with host memory still alive, so host KV can be reused;
- `remote_backup` means simulated GPU failure with remote KV storage available, so remote storage can be reused across the failed logical DP rank;
- all requests are submitted before scheduling starts through simulator `OFFLINE` mode;
- the scheduler uses a logical `assigned_dp_rank` per request to decide which requests are affected by failure.

The first implementation does not modify the production SGLang failover code paths under `python/sglang/srt/`; it modifies simulator hooks and adds benchmark-side simulator orchestration under `benchmark/fault_tolerance/simulator`.

## File Structure

Create:

- `benchmark/fault_tolerance/simulator/src/simulator/__init__.py`
  - package marker for simulator experiment modules.
- `benchmark/fault_tolerance/simulator/src/simulator/fixed_dataset.py`
  - deterministic fixed-token dataset builder using `sglang_simulator.dataset.GenericRequest`.
- `benchmark/fault_tolerance/simulator/src/simulator/config.py`
  - local config loader for `benchmark/fault_tolerance/simulator/configs/<name>/{server,client,failure}.json`.
- `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py`
  - CLI entry point for running one fixed simulator experiment from the new fault-tolerance simulator config directories.
- `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`
  - converts simulator request/iteration/failure outputs into `job_summary.csv`, `task_summary.csv`, `cache_hits.csv`, and `request_detail.csv`.
- `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/server.json`
  - fixed baseline server/topology config.
- `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/client.json`
  - fixed workload config.
- `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/failure.json`
  - fixed GPU failure config.
- `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/server.json`
  - fixed host-backup server/topology config.
- `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/client.json`
  - fixed workload config.
- `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/failure.json`
  - fixed GPU failure config.
- `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/server.json`
  - fixed remote-backup server/topology config.
- `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/client.json`
  - fixed workload config.
- `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/failure.json`
  - fixed GPU failure config.
- `benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py`
  - dataset and config translation tests.
- `benchmark/fault_tolerance/simulator/tests/test_simulator_reporting.py`
  - CSV reporting tests using synthetic request stats.
- `tools/sglang-simulator/src/sglang_simulator/simulation/manager/failure.py`
  - simulator-only failure configuration and runtime state.
- `tools/sglang-simulator/test/test_simulation_failure_manager.py`
  - unit tests for failure trigger, affected request selection, backup policy calculations.
- `tools/sglang-simulator/test/test_simulation_scheduler_failure.py`
  - hook-level tests using small CPU/offline simulator runs.

Modify:

- `tools/sglang-simulator/src/sglang_simulator/simulation/types.py`
  - extend `RequestStats` with logical benchmark identity, DP rank, timing breakdowns, and failure fields.
- `tools/sglang-simulator/src/sglang_simulator/simulation/manager/__init__.py`
  - export `FailureManager`, `FailureConfig`, and `BackupPolicy`.
- `tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py`
  - parse optional `failure` and `benchmark` sections from simulator JSON config.
- `tools/sglang-simulator/src/sglang_simulator/simulation/manager/state.py`
  - expose current logical iteration time without changing existing clock semantics.
- `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/bench_runner.py`
  - preserve dataset `custom_params` instead of replacing them with only `created_time` and `total_request`.
- `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
  - inject failure events, capture affected requests, re-dispatch retries, and write richer request/iteration JSONL.
- `tools/sglang-simulator/src/sglang_simulator/simulation/utils.py`
  - compute metrics from completed logical requests and expose timing breakdown totals.
- `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`
  - optionally add a writer for `request_detail.csv` or keep it in simulator reporting if the format is simulator-only.

---

## Task 1: Define Simulator Failure Runtime Types

**Files:**
- Create: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/failure.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/__init__.py`
- Test: `tools/sglang-simulator/test/test_simulation_failure_manager.py`

- [ ] **Step 1: Write failing tests for trigger parsing and event scheduling**

Add tests that cover:

```python
from sglang_simulator.simulation.manager.failure import (
    BackupPolicy,
    FailureConfig,
    FailureManager,
)


def test_failure_config_defaults_disabled():
    cfg = FailureConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.backup_policy == BackupPolicy.NONE


def test_failure_after_request_schedules_at_completion_plus_delay():
    cfg = FailureConfig.from_dict({
        "enabled": True,
        "backup_policy": "remote_backup",
        "failed_dp_rank": 0,
        "inject_after_request": 2,
        "inject_delay_s": 3.0,
        "recover_after_request": 4,
    })
    mgr = FailureManager(cfg)
    assert mgr.on_request_completed("r1", completion_time_s=10.0) is None
    event = mgr.on_request_completed("r2", completion_time_s=12.0)
    assert event is not None
    assert event.fire_time_s == 15.0
    assert event.failed_dp_rank == 0
    assert event.backup_policy == BackupPolicy.REMOTE_BACKUP
```

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_failure_manager.py -v
```

Expected: fail because `failure.py` does not exist.

- [ ] **Step 2: Implement failure config and trigger state**

Implement:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BackupPolicy(str, Enum):
    NONE = "none"
    HOST = "host"
    REMOTE_BACKUP = "remote_backup"


@dataclass(frozen=True)
class FailureConfig:
    enabled: bool = False
    backup_policy: BackupPolicy = BackupPolicy.NONE
    failed_dp_rank: int = 0
    inject_after_request: int = 0
    inject_after_task: int = 0
    inject_after_job: int = 0
    timeline_after_start_s: float = 0.0
    inject_delay_s: float = 0.0
    recover_after_request: int = 0
    recover_after_task: int = 0
    recover_after_job: int = 0
    recovery_delay_s: float = 0.0
    host_backup_ratio: float = 1.0
    remote_backup_ratio: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict) -> "FailureConfig":
        if not raw:
            return cls()
        policy = BackupPolicy(str(raw.get("backup_policy", "none")))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            backup_policy=policy,
            failed_dp_rank=int(raw.get("failed_dp_rank", raw.get("dp_rank", 0))),
            inject_after_request=int(raw.get("inject_after_request", 0) or 0),
            inject_after_task=int(raw.get("inject_after_task", 0) or 0),
            inject_after_job=int(raw.get("inject_after_job", 0) or 0),
            timeline_after_start_s=float(raw.get("timeline_after_start_s", 0.0) or 0.0),
            inject_delay_s=float(raw.get("inject_delay_s", raw.get("delay", 0.0)) or 0.0),
            recover_after_request=int(raw.get("recover_after_request", 0) or 0),
            recover_after_task=int(raw.get("recover_after_task", 0) or 0),
            recover_after_job=int(raw.get("recover_after_job", 0) or 0),
            recovery_delay_s=float(raw.get("recovery_delay_s", raw.get("recovery_delay", 0.0)) or 0.0),
            host_backup_ratio=float(raw.get("host_backup_ratio", 1.0)),
            remote_backup_ratio=float(raw.get("remote_backup_ratio", 1.0)),
        )


@dataclass(frozen=True)
class FailureEvent:
    event_id: int
    fire_time_s: float
    failed_dp_rank: int
    backup_policy: BackupPolicy


class FailureManager:
    def __init__(self, config: FailureConfig):
        self.config = config
        self.completed_requests = 0
        self._scheduled: Optional[FailureEvent] = None
        self._fired = False
        self._recovered = False

    def on_request_completed(self, rid: str, completion_time_s: float) -> Optional[FailureEvent]:
        if not self.config.enabled or self._scheduled is not None:
            return None
        self.completed_requests += 1
        threshold = (
            self.config.inject_after_request
            or self.config.inject_after_task
            or self.config.inject_after_job
        )
        if threshold and self.completed_requests >= threshold:
            self._scheduled = FailureEvent(
                event_id=1,
                fire_time_s=completion_time_s + self.config.inject_delay_s,
                failed_dp_rank=self.config.failed_dp_rank,
                backup_policy=self.config.backup_policy,
            )
            return self._scheduled
        return None

    def maybe_schedule_timeline(self) -> Optional[FailureEvent]:
        if not self.config.enabled or self._scheduled is not None:
            return None
        if self.config.timeline_after_start_s > 0:
            self._scheduled = FailureEvent(
                event_id=1,
                fire_time_s=self.config.timeline_after_start_s + self.config.inject_delay_s,
                failed_dp_rank=self.config.failed_dp_rank,
                backup_policy=self.config.backup_policy,
            )
            return self._scheduled
        return None

    def should_fire(self, now_s: float) -> Optional[FailureEvent]:
        if self._fired or self._scheduled is None:
            return None
        if now_s >= self._scheduled.fire_time_s:
            self._fired = True
            return self._scheduled
        return None
```

- [ ] **Step 3: Export from manager package**

In `tools/sglang-simulator/src/sglang_simulator/simulation/manager/__init__.py`, add:

```python
from sglang_simulator.simulation.manager.failure import (
    BackupPolicy,
    FailureConfig,
    FailureEvent,
    FailureManager,
)
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_failure_manager.py -v
```

Expected: pass.

---

## Task 2: Parse Failure And Benchmark Config From Simulator JSON

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py`
- Test: `tools/sglang-simulator/test/test_simulation_failure_manager.py`

- [ ] **Step 1: Add failing tests for config parsing**

Append:

```python
import json
import os

from sglang_simulator.simulation.manager import ConfigManager


def test_config_manager_loads_failure_config(tmp_path, monkeypatch):
    config_path = tmp_path / "sim.json"
    config_path.write_text(json.dumps({
        "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
        "predictor": {"name": "aiconfigurator"},
        "scheduler": {"tp_size": 1, "dp_size": 2, "backend_version": "0.5.9"},
        "failure": {
            "enabled": True,
            "backup_policy": "host",
            "failed_dp_rank": 1,
            "inject_after_request": 3,
        },
        "benchmark": {"request_rate": 4.0},
    }))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()
    failure = ConfigManager.get_failure_config()
    assert failure.enabled is True
    assert failure.backup_policy.value == "host"
    assert failure.failed_dp_rank == 1
    assert ConfigManager.get_benchmark_config()["request_rate"] == 4.0
```

- [ ] **Step 2: Implement parser methods**

In `ConfigManager`, add cached fields and methods:

```python
from sglang_simulator.simulation.manager.failure import FailureConfig

_failure_config: Optional[FailureConfig] = None
_benchmark_config: Optional[dict] = None

@classmethod
def reset_config_cache(cls):
    cls._raw_config = None
    cls._model_info = None
    cls._platform_config = None
    cls._scheduler_config = None
    cls._failure_config = None
    cls._benchmark_config = None

@classmethod
def get_failure_config(cls) -> FailureConfig:
    if cls._failure_config is None:
        cls._failure_config = FailureConfig.from_dict(
            cls._get_raw_config().get("failure", {})
        )
    return cls._failure_config

@classmethod
def get_benchmark_config(cls) -> dict:
    if cls._benchmark_config is None:
        raw = cls._get_raw_config().get("benchmark", {})
        cls._benchmark_config = raw if isinstance(raw, dict) else {}
    return cls._benchmark_config
```

- [ ] **Step 3: Run tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_failure_manager.py -v
```

Expected: pass.

---

## Task 3: Preserve Request Metadata Through Offline Submission

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/bench_runner.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/types.py`
- Test: `tools/sglang-simulator/test/test_simulation_sglang_runner.py`

- [ ] **Step 1: Write failing test for request custom metadata**

Add a small direct test around `SGLangBenchmarkRunner.get_request` using a fake dataset:

```python
from sglang_simulator.dataset import GenericRequest, SimpleDataset


def test_runner_preserves_custom_request_metadata(monkeypatch):
    from sglang.srt.server_args import ServerArgs
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner

    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    dataset = SimpleDataset(reqs=[
        GenericRequest(
            token_ids=[1, 2, 3],
            output_length=4,
            custom_params={"job_id": "7", "worker_id": "2", "assigned_dp_rank": 1},
        )
    ])
    req, params = next(runner.get_request(dataset, ignore_timestamp=True, request_rate=1.0))
    assert req.custom_params["job_id"] == "7"
    assert params["job_id"] == "7"
    assert params["worker_id"] == "2"
    assert params["assigned_dp_rank"] == 1
    assert params["total_request"] == 1
    assert "created_time" in params
```

- [ ] **Step 2: Modify `get_request` to merge custom params**

Change the body to:

```python
simulation_params = dict(req.custom_params or {})
simulation_params.setdefault("total_request", len(dataset))
simulation_params["created_time"] = (
    simulation_params.get("created_time", created_time)
)
```

Keep existing `created_time` behavior for request-rate replay.

- [ ] **Step 3: Extend `RequestStats`**

In `tools/sglang-simulator/src/sglang_simulator/simulation/types.py`, add:

```python
job_id: str = ""
worker_id: str = ""
assigned_dp_rank: int = 0
attempt: int = 0
backup_policy: str = "none"
status: str = "running"
is_failover_retried: bool = False
failure_impacted: bool = False
failure_time_s: float = -1
recovery_time_s: float = -1
pre_failover_output_tokens: int = 0
pre_failover_backed_up_tokens: int = 0
retry_prefill_tokens: int = 0
queue_time_s: float = 0
prefetch_time_s: float = 0
backup_time_s: float = 0
inference_time_s: float = 0
prefill_inference_time_s: float = 0
decode_inference_time_s: float = 0
failover_penalty_s: float = 0
```

Change `is_complete` to:

```python
def is_complete(self) -> bool:
    return self.status == "completed" or bool(self.gen_token_latencies)
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_sglang_runner.py::test_runner_preserves_custom_request_metadata -v
```

Expected: pass.

---

## Task 4: Attribute Request And Iteration Timing In Scheduler

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
- Test: `tools/sglang-simulator/test/test_simulation_sglang_runner.py`

- [ ] **Step 1: Add failing test for request stats detail fields**

Extend existing `test_benchmark_sglang` assertions:

```python
request_stats = runner.get_request_stats()
first = request_stats[0]
assert "queue_time_s" in first
assert "prefetch_time_s" in first
assert "backup_time_s" in first
assert "inference_time_s" in first
assert first["inference_time_s"] >= 0
iteration_stats = runner.get_iteration_stats()
assert iteration_stats
assert "rids" in iteration_stats[0]
assert "mode" in iteration_stats[0]
```

- [ ] **Step 2: Populate identity fields in `wrapped_recv_requests`**

After `simulation_args` is extracted:

```python
req_stats.job_id = str(simulation_args.get("job_id", req.rid))
req_stats.worker_id = str(simulation_args.get("worker_id", ""))
req_stats.assigned_dp_rank = int(simulation_args.get("assigned_dp_rank", 0) or 0)
req_stats.backup_policy = str(simulation_args.get("backup_policy", "none") or "none")
req_stats.attempt = int(simulation_args.get("attempt", 0) or 0)
```

- [ ] **Step 3: Attribute prefill/decode inference time in `wrapped_process_batch_result`**

Use `C_SchedulerHook.SIMULATION_BATCH.is_prefill()` and `.is_decode()` after `current_inference_dur` is known:

```python
active_rids = [req.rid for req in batch.reqs]
per_req_forward = current_inference_dur / max(len(active_rids), 1)
per_req_load = hicache_l2_load_dur / max(len(active_rids), 1)
per_req_backup = hicache_l2_backup_dur / max(len(active_rids), 1)
for req in batch.reqs:
    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
    req_stats.prefetch_time_s += per_req_load
    req_stats.backup_time_s += per_req_backup
    req_stats.inference_time_s += per_req_forward
    if C_SchedulerHook.SIMULATION_BATCH.is_prefill():
        req_stats.prefill_inference_time_s += per_req_forward
    else:
        req_stats.decode_inference_time_s += per_req_forward
```

- [ ] **Step 4: Add iteration fields**

Change iteration record to include:

```python
"rids": active_rids,
"mode": (
    "prefill"
    if C_SchedulerHook.SIMULATION_BATCH.is_prefill()
    else "decode"
),
"global_clock_s": StateManager.get_global_clock(),
```

- [ ] **Step 5: Mark completed requests**

After `original_process_batch_result`, loop over `batch.reqs`:

```python
for req in batch.reqs:
    if req.finished():
        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
        req_stats.status = "completed"
        req_stats.queue_time_s = max(req_stats.queue_end - req_stats.queue_start, 0)
```

- [ ] **Step 6: Run targeted test**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_sglang_runner.py::test_benchmark_sglang -v
```

Expected: pass.

---

## Task 5: Implement Scheduler-Internal GPU Failure And Retry

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/failure.py`
- Test: `tools/sglang-simulator/test/test_simulation_scheduler_failure.py`

- [ ] **Step 1: Write failing integration test**

Create a small offline run with `dp_size=2`, `failed_dp_rank=0`, `inject_after_request=1`, fixed short requests, and assert at least one request is retried.

```python
def test_scheduler_internal_failure_retries_impacted_requests(tmp_path, monkeypatch):
    import json
    import os
    from transformers import AutoTokenizer
    from sglang.srt.server_args import ServerArgs
    from sglang_simulator.dataset import GenericRequest, SimpleDataset
    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner

    config_path = tmp_path / "sim.json"
    config_path.write_text(json.dumps({
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {"name": "aiconfigurator"},
        "scheduler": {"tp_size": 1, "dp_size": 2, "backend_version": "0.5.9"},
        "failure": {
            "enabled": True,
            "backup_policy": "none",
            "failed_dp_rank": 0,
            "inject_after_request": 1,
            "inject_delay_s": 0.0,
        },
    }))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    model_path = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = SimpleDataset(tokenizer=tokenizer, reqs=[
        GenericRequest(
            token_ids=[10, 11, 12, 13],
            output_length=4,
            custom_params={"job_id": "1", "assigned_dp_rank": 0},
        ),
        GenericRequest(
            token_ids=[20, 21, 22, 23],
            output_length=4,
            custom_params={"job_id": "2", "assigned_dp_rank": 0},
        ),
    ])
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            max_total_tokens=8192,
            page_size=1,
        )
    )
    metrics = runner.benchmark(BenchmarkConfig(request_rate=float("inf")), dataset)
    request_stats = runner.get_request_stats()
    runner.shutdown()

    assert metrics["completed"] == 2
    assert any(row["is_failover_retried"] for row in request_stats)
    assert any(row["failure_impacted"] for row in request_stats)
```

- [ ] **Step 2: Initialize a `FailureManager` on scheduler hook**

In `C_SchedulerHook`, add class fields:

```python
FAILURE_MANAGER = None
FAILED_DP_RANKS: set[int] = set()
FAILURE_EVENTS: list[dict] = []
RETRY_SALT: int = 0
```

In `wrapped_init` after `ConfigManager.set_scheduler_config`:

```python
from sglang_simulator.simulation.manager import FailureManager
C_SchedulerHook.FAILURE_MANAGER = FailureManager(ConfigManager.get_failure_config())
C_SchedulerHook.FAILURE_MANAGER.maybe_schedule_timeline()
```

- [ ] **Step 3: Add helper to create retry input**

In scheduler hook, add a nested helper or classmethod:

```python
def _build_retry_input(self, req, event):
    import copy
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

    old_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
    generated = list(getattr(req, "output_ids", []) or [])
    original_input = list(getattr(req, "origin_input_ids", []) or [])
    remaining = int(req.sampling_params.max_new_tokens) - len(generated)
    if remaining <= 0:
        return None

    sp = copy.deepcopy(req.sampling_params)
    sp.max_new_tokens = remaining
    C_SchedulerHook.RETRY_SALT += 1
    retry_rid = f"{req.rid}:sim_retry:{C_SchedulerHook.RETRY_SALT}"
    retry_input_ids = original_input + generated

    backed_up = C_SchedulerHook.FAILURE_MANAGER.estimate_backed_up_tokens(
        backup_policy=event.backup_policy,
        generated_tokens=len(generated),
    )

    retry_stats = C_SchedulerHook.REQUEST_STATS[retry_rid]
    retry_stats.rid = retry_rid
    retry_stats.job_id = old_stats.job_id
    retry_stats.worker_id = old_stats.worker_id
    retry_stats.assigned_dp_rank = C_SchedulerHook.FAILURE_MANAGER.pick_recovery_dp_rank(
        failed_dp_rank=event.failed_dp_rank,
        dp_size=ConfigManager.get_scheduler_config().dp_size,
    )
    retry_stats.input_length = len(retry_input_ids)
    retry_stats.output_length = remaining
    retry_stats.created_time = StateManager.get_global_clock()
    retry_stats.queue_start = StateManager.get_global_clock()
    retry_stats.backup_policy = event.backup_policy.value
    retry_stats.attempt = old_stats.attempt + 1
    retry_stats.is_failover_retried = True
    retry_stats.failure_impacted = True
    retry_stats.failure_time_s = event.fire_time_s
    retry_stats.pre_failover_output_tokens = len(generated)
    retry_stats.pre_failover_backed_up_tokens = backed_up
    retry_stats.retry_prefill_tokens = len(retry_input_ids) - backed_up

    old_stats.failure_impacted = True
    old_stats.status = "failed_over"

    return TokenizedGenerateReqInput(
        rid=retry_rid,
        input_text="",
        input_ids=retry_input_ids,
        sampling_params=sp,
        return_logprob=getattr(req, "return_logprob", False),
        logprob_start_len=getattr(req, "logprob_start_len", -1),
        top_logprobs_num=getattr(req, "top_logprobs_num", 0),
        token_ids_logprob=getattr(req, "token_ids_logprob", None),
        stream=getattr(req, "stream", False),
        return_hidden_states=getattr(req, "return_hidden_states", False),
        return_routed_experts=getattr(req, "return_routed_experts", False),
        input_embeds=getattr(req, "input_embeds", None),
        original_prompt_len=len(original_input),
        failover_prefix_ids=generated,
        pre_failover_output_tokens=len(generated),
        pre_failover_backed_up_tokens=backed_up,
        failed_dp_rank=event.failed_dp_rank,
        custom_params={
            "simulation": {
                "job_id": retry_stats.job_id,
                "worker_id": retry_stats.worker_id,
                "assigned_dp_rank": retry_stats.assigned_dp_rank,
                "backup_policy": event.backup_policy.value,
                "attempt": retry_stats.attempt,
                "created_time": StateManager.get_global_clock(),
                "total_request": len(C_SchedulerHook.REQUEST_STATS),
            }
        },
    )
```

- [ ] **Step 4: Add helper to fire failure and re-dispatch**

Implement:

```python
def _maybe_fire_failure(self):
    mgr = C_SchedulerHook.FAILURE_MANAGER
    if mgr is None:
        return
    event = mgr.should_fire(StateManager.get_global_clock())
    if event is None:
        return
    C_SchedulerHook.FAILED_DP_RANKS.add(event.failed_dp_rank)
    C_SchedulerHook.FAILURE_EVENTS.append({
        "event": "sim_gpu_failure",
        "event_id": event.event_id,
        "timestamp_s": event.fire_time_s,
        "failed_dp_rank": event.failed_dp_rank,
        "backup_policy": event.backup_policy.value,
    })
    active_reqs = list(getattr(self.running_batch, "reqs", []) or [])
    retry_inputs = []
    for req in active_reqs:
        stats = C_SchedulerHook.REQUEST_STATS[req.rid]
        if int(stats.assigned_dp_rank) != int(event.failed_dp_rank):
            continue
        retry_input = _build_retry_input(self, req, event)
        if retry_input is not None:
            retry_inputs.append(retry_input)
        req.set_finish_with_abort("simulated gpu failure")
    for retry_input in retry_inputs:
        self.handle_generate_request(retry_input)
```

Call `_maybe_fire_failure(self)`:

- after `StateManager.step_global_clock(...)` in `wrapped_process_batch_result`;
- after jumping to next future request time in `wrapped_get_new_batch_prefill`;
- before returning from `wrapped_recv_requests`.

- [ ] **Step 5: Estimate backed-up tokens and recovery DP rank**

In `FailureManager`, add:

```python
def estimate_backed_up_tokens(self, backup_policy, generated_tokens: int) -> int:
    if backup_policy == BackupPolicy.NONE:
        return 0
    if backup_policy == BackupPolicy.HOST:
        return int(generated_tokens * self.config.host_backup_ratio)
    if backup_policy == BackupPolicy.REMOTE_BACKUP:
        return int(generated_tokens * self.config.remote_backup_ratio)
    return 0

def pick_recovery_dp_rank(self, failed_dp_rank: int, dp_size: int) -> int:
    if dp_size <= 1:
        return failed_dp_rank
    return (failed_dp_rank + 1) % dp_size
```

- [ ] **Step 6: Run integration test**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_scheduler_failure.py -v
```

Expected: pass.

---

## Task 6: Add Backup Policy Timing Effects

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/manager/failure.py`
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
- Test: `tools/sglang-simulator/test/test_simulation_failure_manager.py`

- [ ] **Step 1: Add tests for restore penalty**

```python
def test_restore_penalty_uses_memory_for_host_and_disk_for_remote():
    cfg = FailureConfig.from_dict({"enabled": True, "backup_policy": "host"})
    mgr = FailureManager(cfg)
    assert mgr.restore_penalty_s(
        backup_policy=BackupPolicy.NONE,
        kv_bytes_per_token=100,
        restored_tokens=10,
        memory_read_bandwidth=1000,
        disk_read_bandwidth=100,
    ) == 0
    assert mgr.restore_penalty_s(
        backup_policy=BackupPolicy.HOST,
        kv_bytes_per_token=100,
        restored_tokens=10,
        memory_read_bandwidth=1000,
        disk_read_bandwidth=100,
    ) == 1.0
    assert mgr.restore_penalty_s(
        backup_policy=BackupPolicy.REMOTE_BACKUP,
        kv_bytes_per_token=100,
        restored_tokens=10,
        memory_read_bandwidth=1000,
        disk_read_bandwidth=100,
    ) == 10.0
```

- [ ] **Step 2: Implement restore penalty**

```python
def restore_penalty_s(
    self,
    *,
    backup_policy: BackupPolicy,
    kv_bytes_per_token: int,
    restored_tokens: int,
    memory_read_bandwidth: float | None,
    disk_read_bandwidth: float | None,
) -> float:
    if restored_tokens <= 0 or backup_policy == BackupPolicy.NONE:
        return 0.0
    bytes_to_restore = kv_bytes_per_token * restored_tokens
    if backup_policy == BackupPolicy.HOST:
        bw = memory_read_bandwidth or 1.0
    else:
        bw = disk_read_bandwidth or 1.0
    return float(bytes_to_restore / bw)
```

- [ ] **Step 3: Apply penalty to retry stats**

In `_build_retry_input`, after `backed_up` is calculated:

```python
kv_bytes = ConfigManager.get_kv_cache_bytes()
platform = ConfigManager.get_platform_config()
penalty = C_SchedulerHook.FAILURE_MANAGER.restore_penalty_s(
    backup_policy=event.backup_policy,
    kv_bytes_per_token=kv_bytes,
    restored_tokens=backed_up,
    memory_read_bandwidth=platform.memory_read_bandwidth,
    disk_read_bandwidth=platform.disk_read_bandwidth,
)
retry_stats.failover_penalty_s = penalty
retry_stats.prefetch_time_s += penalty
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_failure_manager.py tools/sglang-simulator/test/test_simulation_scheduler_failure.py -v
```

Expected: pass.

---

## Task 7: Build Fixed Simulator Dataset And Config Translator

**Files:**
- Create: `benchmark/fault_tolerance/simulator/src/simulator/__init__.py`
- Create: `benchmark/fault_tolerance/simulator/src/simulator/fixed_dataset.py`
- Create: `benchmark/fault_tolerance/simulator/src/simulator/config.py`
- Test: `benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py`

- [ ] **Step 1: Write dataset tests**

```python
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
TOOLS = Path(__file__).resolve().parents[4] / "tools" / "sglang-simulator" / "src"
sys.path[:0] = [str(SRC), str(TOOLS)]

from simulator.fixed_dataset import build_fixed_dataset


class FakeTokenizer:
    vocab_size = 1000


def test_fixed_dataset_sets_lengths_and_metadata():
    dataset = build_fixed_dataset(
        tokenizer=FakeTokenizer(),
        input_len=8,
        output_len=3,
        num_requests=4,
        app_workers=2,
        seed=1,
        dp_size=2,
        backup_policy="remote_backup",
        request_rate=2.0,
    )
    assert len(dataset) == 4
    req0 = dataset[0]
    assert len(req0.token_ids) == 8
    assert req0.output_length == 3
    assert req0.custom_params["job_id"] == "1"
    assert req0.custom_params["worker_id"] == "1"
    assert req0.custom_params["assigned_dp_rank"] == 0
    assert req0.custom_params["backup_policy"] == "remote_backup"
    assert req0.custom_params["created_time"] == 0.0
```

- [ ] **Step 2: Implement dataset builder**

```python
from __future__ import annotations

import random

from sglang_simulator.dataset import GenericRequest, SimpleDataset


def build_fixed_dataset(
    *,
    tokenizer,
    input_len: int,
    output_len: int,
    num_requests: int,
    app_workers: int,
    seed: int,
    dp_size: int,
    backup_policy: str,
    request_rate: float = float("inf"),
) -> SimpleDataset:
    rng = random.Random(seed)
    reqs = []
    created = 0.0
    vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    low = max(1, vocab // 4)
    high = max(low + 1, vocab * 3 // 4)
    for idx in range(max(num_requests, 0)):
        token_ids = [rng.randrange(low, high) for _ in range(max(input_len, 1))]
        if idx > 0 and request_rate != float("inf") and request_rate > 0:
            created += 1.0 / request_rate
        reqs.append(GenericRequest(
            token_ids=token_ids,
            input_length=len(token_ids),
            output_length=max(output_len, 1),
            custom_params={
                "job_id": str(idx + 1),
                "worker_id": str((idx % max(app_workers, 1)) + 1),
                "assigned_dp_rank": idx % max(dp_size, 1),
                "backup_policy": backup_policy,
                "attempt": 0,
                "created_time": created,
            },
        ))
    return SimpleDataset(tokenizer=tokenizer, reqs=reqs)
```

- [ ] **Step 3: Run tests**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src pytest benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py -v
```

Expected: pass.

- [ ] **Step 4: Write local config loader tests**

Append:

```python
from simulator.config import load_experiment_config


def test_load_experiment_config_from_fault_tolerance_tree(tmp_path):
    cfg_root = tmp_path / "configs" / "baseline-fixed-a100"
    cfg_root.mkdir(parents=True)
    (cfg_root / "server.json").write_text('{"kv_backup":"none","server":{"model_path":"Qwen/Qwen3-8B","dp_size":2,"tp_size":1,"pp_size":1}}')
    (cfg_root / "client.json").write_text('{"fixed":{"input_len":16,"output_len":4,"num_requests":8,"app_workers":2,"seed":1}}')
    (cfg_root / "failure.json").write_text('{"inject_after_job":"2","delay":0,"dp_rank":"0"}')
    cfg = load_experiment_config("baseline-fixed-a100", config_root=tmp_path / "configs")
    assert cfg.name == "baseline-fixed-a100"
    assert cfg.kv_backup == "none"
    assert cfg.fixed.input_len == 16
    assert cfg.server.dp_size == 2
```

- [ ] **Step 5: Implement local config loader**

Create `benchmark/fault_tolerance/simulator/src/simulator/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    model_path: str = "Qwen/Qwen3-8B"
    dp_size: int = 2
    tp_size: int = 1
    pp_size: int = 1
    accelerator_name: str = "a100_sxm"


@dataclass
class FixedConfig:
    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    seed: int = 1


@dataclass
class ExperimentConfig:
    name: str
    server: ServerConfig
    fixed: FixedConfig
    failure: dict
    kv_backup: str


def _default_config_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(
    name: str,
    *,
    config_root: Path | None = None,
) -> ExperimentConfig:
    root = config_root or _default_config_root()
    cfg_dir = Path(name)
    if not cfg_dir.is_absolute():
        cfg_dir = root / name
    server_raw = _load_json(cfg_dir / "server.json")
    client_raw = _load_json(cfg_dir / "client.json")
    failure_raw = _load_json(cfg_dir / "failure.json")
    server_section = server_raw.get("server", server_raw)
    fixed_section = client_raw.get("fixed", client_raw)
    return ExperimentConfig(
        name=cfg_dir.name,
        server=ServerConfig(
            model_path=str(server_section.get("model_path", "Qwen/Qwen3-8B")),
            dp_size=int(server_section.get("dp_size", 2)),
            tp_size=int(server_section.get("tp_size", 1)),
            pp_size=int(server_section.get("pp_size", 1)),
            accelerator_name=str(server_section.get("accelerator_name", server_raw.get("accelerator_name", "a100_sxm"))),
        ),
        fixed=FixedConfig(
            input_len=int(fixed_section.get("input_len", 1024)),
            output_len=int(fixed_section.get("output_len", 128)),
            num_requests=int(fixed_section.get("num_requests", 64)),
            app_workers=int(fixed_section.get("app_workers", 4)),
            seed=int(fixed_section.get("seed", 1)),
        ),
        failure=failure_raw,
        kv_backup=str(server_raw.get("kv_backup", "none")),
    )
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src pytest benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py -v
```

Expected: pass.

---

## Task 8: Implement Fixed Simulator Pipeline CLI

**Files:**
- Create: `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py`
- Create: `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/server.json`
- Create: `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/client.json`
- Create: `benchmark/fault_tolerance/simulator/configs/baseline-fixed-a100/failure.json`
- Create: `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/server.json`
- Create: `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/client.json`
- Create: `benchmark/fault_tolerance/simulator/configs/host_backup-fixed-a100/failure.json`
- Create: `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/server.json`
- Create: `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/client.json`
- Create: `benchmark/fault_tolerance/simulator/configs/remote_backup-fixed-a100/failure.json`
- Test: `benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py`

- [ ] **Step 1: Add config translation test**

```python
from simulator.fixed_pipeline import build_simulator_config_payload


def test_build_simulator_config_payload_contains_failure_section():
    payload = build_simulator_config_payload(
        accelerator_name="a100_sxm",
        dp_size=2,
        tp_size=1,
        pp_size=1,
        backup_policy="host",
        failure={
            "inject_after_job": "10",
            "recover_after_job": "20",
            "delay": 5,
            "dp_rank": "0",
        },
    )
    assert payload["scheduler"]["dp_size"] == 2
    assert payload["failure"]["enabled"] is True
    assert payload["failure"]["backup_policy"] == "host"
    assert payload["failure"]["inject_after_job"] == 10
```

- [ ] **Step 2: Implement config builder**

Implement:

```python
def _int_from_failure(raw: dict, key: str, default: int = 0) -> int:
    value = raw.get(key, default)
    if value in ("", None):
        return default
    return int(value)


def build_simulator_config_payload(
    *,
    accelerator_name: str,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    backup_policy: str,
    failure: dict,
) -> dict:
    return {
        "platform": {
            "accelerator": {"name": accelerator_name, "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {"name": "aiconfigurator"},
        "scheduler": {
            "tp_size": max(int(tp_size), 1),
            "pp_size": max(int(pp_size), 1),
            "dp_size": max(int(dp_size), 1),
            "backend_version": "0.5.9",
        },
        "failure": {
            "enabled": True,
            "backup_policy": backup_policy,
            "failed_dp_rank": _int_from_failure(failure, "dp_rank", 0),
            "inject_after_job": _int_from_failure(failure, "inject_after_job", 0),
            "inject_after_task": _int_from_failure(failure, "inject_after_task", 0),
            "inject_delay_s": float(failure.get("delay", 0) or 0),
            "recover_after_job": _int_from_failure(failure, "recover_after_job", 0),
            "recover_after_task": _int_from_failure(failure, "recover_after_task", 0),
        },
    }
```

- [ ] **Step 3: Add built-in experiment configs**

Create baseline server config:

```json
{
  "name": "baseline-fixed-a100",
  "kv_backup": "none",
  "server": {
    "model_path": "Qwen/Qwen3-8B",
    "accelerator_name": "a100_sxm",
    "dp_size": 2,
    "tp_size": 1,
    "pp_size": 1
  }
}
```

Create host server config:

```json
{
  "name": "host_backup-fixed-a100",
  "kv_backup": "host",
  "server": {
    "model_path": "Qwen/Qwen3-8B",
    "accelerator_name": "a100_sxm",
    "dp_size": 2,
    "tp_size": 1,
    "pp_size": 1
  }
}
```

Create remote server config:

```json
{
  "name": "remote_backup-fixed-a100",
  "kv_backup": "remote_backup",
  "server": {
    "model_path": "Qwen/Qwen3-8B",
    "accelerator_name": "a100_sxm",
    "dp_size": 2,
    "tp_size": 1,
    "pp_size": 1
  }
}
```

For all three `client.json` files, use:

```json
{
  "fixed": {
    "input_len": 20000,
    "output_len": 1000,
    "num_requests": 40,
    "app_workers": 10,
    "seed": 1
  }
}
```

For all three `failure.json` files, use:

```json
{
  "inject_after_job": "10",
  "recover_after_job": "20",
  "delay": 5,
  "dp_rank": "0"
}
```

- [ ] **Step 4: Implement CLI skeleton**

The CLI must:

- load config via `simulator.config.load_experiment_config`;
- validate required fixed workload fields from the new fault-tolerance config tree;
- write a generated simulator config under the experiment output directory;
- set `SGLANG_SIMULATOR_CONFIG_PATH`;
- create `SGLangBenchmarkRunner`;
- run `runner.benchmark(benchmark_config, dataset)`;
- call reporting.

Use this implementation shape:

```python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from sglang.srt.server_args import ServerArgs
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner
from transformers import AutoTokenizer

from simulator.config import load_experiment_config
from simulator.fixed_dataset import build_fixed_dataset
from simulator.reporting import write_simulator_fixed_outputs


def _parse_rate(raw: str) -> float:
    return float("inf") if raw == "inf" else float(raw)


def _default_backup_policy(kv_backup: str) -> str:
    if kv_backup in {"host", "remote_backup"}:
        return kv_backup
    return "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-config-output", type=Path, default=None)
    parser.add_argument("--accelerator-name", default="")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--backup-policy", choices=["none", "host", "remote_backup"], default="")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    backup_policy = args.backup_policy or _default_backup_policy(cfg.kv_backup)
    accelerator = args.accelerator_name or cfg.server.accelerator_name
    sim_payload = build_simulator_config_payload(
        accelerator_name=accelerator,
        dp_size=cfg.server.dp_size,
        tp_size=cfg.server.tp_size,
        pp_size=cfg.server.pp_size,
        backup_policy=backup_policy,
        failure=cfg.failure,
    )
    sim_config_path = args.sim_config_output or output_dir / "simulator_config.json"
    sim_config_path.parent.mkdir(parents=True, exist_ok=True)
    sim_config_path.write_text(json.dumps(sim_payload, indent=2), encoding="utf-8")
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config_path)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    tokenizer = AutoTokenizer.from_pretrained(cfg.server.model_path)
    dataset = build_fixed_dataset(
        tokenizer=tokenizer,
        input_len=cfg.fixed.input_len,
        output_len=cfg.fixed.output_len,
        num_requests=cfg.fixed.num_requests,
        app_workers=cfg.fixed.app_workers,
        seed=cfg.fixed.seed,
        dp_size=cfg.server.dp_size,
        backup_policy=backup_policy,
        request_rate=_parse_rate(args.request_rate),
    )
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=cfg.server.model_path,
            load_format="dummy",
            device="cpu",
            max_total_tokens=max(cfg.fixed.input_len + cfg.fixed.output_len, 8192),
            page_size=1,
        )
    )
    try:
        runner.benchmark(
            BenchmarkConfig(request_rate=_parse_rate(args.request_rate)),
            dataset=dataset,
        )
        write_simulator_fixed_outputs(output_dir, runner.get_request_stats())
        shutil.copyfile(
            Path("/tmp/sglang_simulator/output/request.jsonl"),
            output_dir / "simulator_request.jsonl",
        )
        shutil.copyfile(
            Path("/tmp/sglang_simulator/output/iteration.jsonl"),
            output_dir / "simulator_iteration.jsonl",
        )
        failure_src = Path("/tmp/sglang_simulator/output/failure_events.jsonl")
        if failure_src.is_file():
            shutil.copyfile(failure_src, output_dir / "sim_failure_events.jsonl")
    finally:
        runner.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src pytest benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py -v
```

Expected: pass.

---

## Task 9: Generate CSV Reporting Including Request Detail

**Files:**
- Create: `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`
- Test: `benchmark/fault_tolerance/simulator/tests/test_simulator_reporting.py`

- [ ] **Step 1: Write reporting tests**

```python
import csv
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from simulator.reporting import write_simulator_fixed_outputs


def test_write_simulator_outputs(tmp_path):
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
    write_simulator_fixed_outputs(tmp_path, request_stats)
    assert (tmp_path / "job_summary.csv").is_file()
    assert (tmp_path / "task_summary.csv").is_file()
    assert (tmp_path / "cache_hits.csv").is_file()
    assert (tmp_path / "request_detail.csv").is_file()
    rows = list(csv.DictReader((tmp_path / "request_detail.csv").open()))
    assert rows[0]["waiting_time_s"] == "0.5"
    cache = list(csv.DictReader((tmp_path / "cache_hits.csv").open()))
    assert cache[0]["remote_prefetch"] == "2"
```

- [ ] **Step 2: Implement reporting**

Implement local CSV writers inside `benchmark/fault_tolerance/simulator/src/simulator/reporting.py`. Do not import from `benchmark/multi_agent`; the new benchmark must remain self-contained.

For `request_detail.csv`, include exact columns:

```text
job_id,rid,worker_id,assigned_dp_rank,backup_policy,status,created_time_s,queue_start_s,queue_end_s,finish_time_s,waiting_time_s,prefetch_time_s,backup_time_s,inference_time_s,total_latency_s,prefill_inference_time_s,decode_inference_time_s,failure_impacted,is_failover_retried,failure_time_s,recovery_time_s,pre_failover_output_tokens,pre_failover_backed_up_tokens,retry_prefill_tokens,failover_penalty_s
```

For `cache_hits.csv`:

- `baseline`: all backup fields zero;
- `host`: `l2_match` and `reused_host` from `pre_failover_backed_up_tokens`;
- `remote_backup`: `remote_match`, `remote_prefetch`, and `reused_storage` from `pre_failover_backed_up_tokens` / `prefetch_complete_tokens`.

- [ ] **Step 3: Run reporting tests**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src pytest benchmark/fault_tolerance/simulator/tests/test_simulator_reporting.py -v
```

Expected: pass.

---

## Task 10: Persist Failure Events And Reset Hook State Cleanly

**Files:**
- Modify: `tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py`
- Modify: `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py`
- Test: `tools/sglang-simulator/test/test_simulation_scheduler_failure.py`

- [ ] **Step 1: Write assertion for failure event file**

Extend scheduler failure test:

```python
failure_events = [
    row for row in runner.get_failure_events()
    if row["event"] == "sim_gpu_failure"
]
assert failure_events
assert failure_events[0]["failed_dp_rank"] == 0
```

- [ ] **Step 2: Add runner helper**

In `bench_runner.py`, add:

```python
def get_failure_events(self) -> list[dict]:
    data = []
    file_path = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/failure_events.jsonl"
    if os.path.exists(file_path):
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data
```

- [ ] **Step 3: Dump failure events in scheduler profile**

In `wrapped_profile`, write:

```python
with open(f"{output_dir}/failure_events.jsonl", "w") as f:
    for item in C_SchedulerHook.FAILURE_EVENTS:
        f.write(json.dumps(item) + "\n")
```

Reset:

```python
C_SchedulerHook.FAILURE_EVENTS.clear()
C_SchedulerHook.FAILED_DP_RANKS.clear()
C_SchedulerHook.RETRY_SALT = 0
C_SchedulerHook.FAILURE_MANAGER = None
```

- [ ] **Step 4: Run scheduler failure tests**

Run:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest tools/sglang-simulator/test/test_simulation_scheduler_failure.py -v
```

Expected: pass.

---

## Task 11: End-To-End Fixed Simulator Runs

**Files:**
- Modify: `benchmark/fault_tolerance/simulator/src/simulator/fixed_pipeline.py`
- Test: manual smoke command

- [ ] **Step 1: Add CLI args**

CLI should support:

```text
--config
--output-dir
--sim-config-output
--accelerator-name
--request-rate
--backup-policy
```

Default `backup_policy`:

- `none` when config `kv_backup` is `none`;
- `remote_backup` when config `kv_backup` is `remote_backup`;
- explicit `--backup-policy host` for host-backup experiments.

- [ ] **Step 2: Run baseline smoke**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src \
CUDA_VISIBLE_DEVICES="" \
python -m simulator.fixed_pipeline \
  --config baseline-fixed-a100 \
  --output-dir /tmp/ft-sim-baseline \
  --backup-policy none \
  --request-rate inf
```

Expected files:

```text
/tmp/ft-sim-baseline/job_summary.csv
/tmp/ft-sim-baseline/task_summary.csv
/tmp/ft-sim-baseline/cache_hits.csv
/tmp/ft-sim-baseline/request_detail.csv
/tmp/ft-sim-baseline/simulator_request.jsonl
/tmp/ft-sim-baseline/simulator_iteration.jsonl
/tmp/ft-sim-baseline/sim_failure_events.jsonl
```

- [ ] **Step 3: Run host backup smoke**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src \
CUDA_VISIBLE_DEVICES="" \
python -m simulator.fixed_pipeline \
  --config host_backup-fixed-a100 \
  --output-dir /tmp/ft-sim-host \
  --backup-policy host \
  --request-rate inf
```

Expected:

- `cache_hits.csv` has non-zero `l2_match` / `reused_host` for impacted retries;
- `request_detail.csv` has `is_failover_retried=True` for impacted retries.

- [ ] **Step 4: Run remote backup smoke**

Run:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src \
CUDA_VISIBLE_DEVICES="" \
python -m simulator.fixed_pipeline \
  --config remote_backup-fixed-a100 \
  --output-dir /tmp/ft-sim-remote \
  --backup-policy remote_backup \
  --request-rate inf
```

Expected:

- `cache_hits.csv` has non-zero `remote_prefetch` / `reused_storage` for impacted retries;
- `request_detail.csv` has non-zero `prefetch_time_s` and `failover_penalty_s`.

---

## Task 12: Add Documentation And Comparison Notes

**Files:**
- Create: `benchmark/fault_tolerance/simulator/docs/simulator_fixed_failure.md`
- Modify: no production docs required unless requested

- [ ] **Step 1: Document usage**

Include:

```markdown
# Fixed Workload Simulator Failure Experiments

This simulator path runs fixed-length workload experiments in `OFFLINE` mode.
It simulates GPU failure inside the simulator scheduler and writes CSV outputs
compatible with the existing fault-tolerance simulator benchmark plotting scripts.

## Policies

- `none`: no KV backup; impacted requests retry from scratch.
- `host`: GPU failure with host memory surviving; impacted requests reuse host KV.
- `remote_backup`: GPU failure with remote KV backup; impacted requests reuse remote KV.

## Outputs

- `job_summary.csv`
- `task_summary.csv`
- `cache_hits.csv`
- `request_detail.csv`
- `sim_failure_events.jsonl`
- `simulator_request.jsonl`
- `simulator_iteration.jsonl`
```

- [ ] **Step 2: Add limitation note**

Mention:

- simulator uses logical DP rank assignment;
- output tokens are simulated, not semantically meaningful;
- failure is modeled as GPU failure, not host/node failure;
- host backup assumes CPU memory survives the GPU failure.

---

## Verification Matrix

Run unit tests:

```bash
PYTHONPATH=tools/sglang-simulator/src pytest \
  tools/sglang-simulator/test/test_simulation_failure_manager.py \
  tools/sglang-simulator/test/test_simulation_scheduler_failure.py \
  -v
```

Run benchmark simulator tests:

```bash
PYTHONPATH=benchmark/fault_tolerance/simulator/src:tools/sglang-simulator/src pytest \
  benchmark/fault_tolerance/simulator/tests/test_simulator_fixed_dataset.py \
  benchmark/fault_tolerance/simulator/tests/test_simulator_reporting.py \
  -v
```

Run a path isolation check:

```bash
grep -R "benchmark/multi_agent" benchmark/fault_tolerance/simulator || true
```

Expected: no output except documentation comments that explicitly describe old implementation as background. The implementation and tests must not import from `benchmark/multi_agent`.

Run smoke commands from Task 11 for `none`, `host`, and `remote_backup`.

## Commit Plan

Use one commit per major stage:

1. `feat(simulator): add failure manager config`
2. `feat(simulator): preserve fixed request metadata`
3. `feat(simulator): add scheduler-internal gpu failure retry`
4. `feat(fault-tolerance): add fixed simulator pipeline`
5. `feat(fault-tolerance): write simulator fixed csv outputs`
6. `docs: document fixed simulator failure experiments`

## Self-Review Notes

Coverage:

- Scheduler-internal failure: Tasks 1, 2, 5, 6, 10.
- Offline fixed workload: Tasks 3, 7, 8, 11.
- `cache_hits.csv`: Task 9.
- Per-request waiting/prefetch/inference detail: Tasks 4 and 9.
- Three policies baseline/host/remote: Tasks 6, 8, 9, 11.

Known risk:

- Constructing retry `TokenizedGenerateReqInput` from a live `Req` may need adjustment for fields that are required by the current SGLang version. The implementation should reuse the existing production failover construction in `tokenizer_manager.py` as the reference and keep tests short enough to expose constructor mismatches quickly.
