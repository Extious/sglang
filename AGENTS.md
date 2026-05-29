# Project Context For Coding Agents

## Research Context

The user is a PhD student working on fault tolerance for SGLang. This repository is used primarily for research experiments and fast iteration, not only production feature work.

The broader project goal is to study failure injection, request retry, and KV-cache backup strategies for SGLang. Existing non-simulator work has already implemented failure injection and request retry paths, with experiments under `benchmark/multi_agent` using fixed agents and CrewAI agents as clients. Those real experiments are slow because they require actual GPU inference and communication.

The current focus is to use `tools/sglang-simulator` for faster fault-tolerance research experiments.

## Fault Tolerance Experiment Layout

The intended fault-tolerance benchmark layout is:

- `benchmark/fault_tolerance/real`: reserved for non-simulator, real SGLang GPU inference experiments.
- `benchmark/fault_tolerance/simulator`: simulator-backed experiments.
- `benchmark/fault_tolerance/simulator/synthesis_requests`: the current primary simulator workload.
- `benchmark/fault_tolerance/simulator/crewai_agents`: planned for later; do not prioritize unless explicitly requested.

For now, focus on `benchmark/fault_tolerance/simulator/synthesis_requests` rather than the older `benchmark/multi_agent` experiment style.

## Synthesis Requests Workload

`synthesis_requests` replaces the older "fixed agent" naming because the workload is synthetic fixed-length requests rather than semantically meaningful agent behavior.

It has two modes:

- `offline`: fast in-process simulator mode using logical time. This is the main debug/research path.
- `online`: launches `sglang_simulator.simulation.sglang.launch_server` in `BLOCKING` mode and drives it over HTTP. This should stay close to a real GPU blocking server experiment.

Online mode should not set `SGLANG_USE_CPU_ENGINE`. It should remove that env var from the launched server environment. Online server commands should skip SGLang warmup with `--skip-server-warmup`, because warmup generate requests do not carry simulator metadata.

Online client behavior should respect workload concurrency semantics:

- use `app_workers` as parallel blocking clients;
- preserve per-request `worker_id`, `worker_seq`, `assigned_dp_rank`, and `created_time`;
- for finite `request_rate`, schedule arrivals according to request `created_time`.

## Strategies

The core strategies are:

- `baseline`: no KV backup; retry re-prefills required tokens.
- `host_backup`: simulated GPU failure with host memory surviving.
- `remote_backup`: simulated GPU failure with remote KV backup available.

Backup policy values used in configs/payloads include:

- `none`
- `host`
- `remote_backup`

## Simulator Implementation Notes

The current simulator work extends `tools/sglang-simulator` with:

- failure config and failure manager support;
- scheduler failure injection and recovery behavior;
- retry request construction and rerouting;
- request/iteration/failure event artifacts;
- predictor metadata sidecars;
- default time predictor config using `aiconfigurator` with heuristic fallback when optional dependencies or databases are unavailable.

`aiconfigurator` is an optional dependency in many local environments. Imports of `AIConfiguratorTimePredictor` should be lazy or guarded so tests and default config paths can fall back to `HeuristicTimePredictor`.

## Request Metadata

Synthetic request metadata is important for scheduler behavior. Preserve these fields when refactoring:

- `job_id`
- `worker_id`
- `worker_seq`
- `parallel_worker_client`
- `assigned_dp_rank`
- `backup_policy`
- `attempt`
- `created_time`
- `total_request`

For offline parallel-worker sequencing, keep each worker on one DP rank. The expected assignment is based on worker index modulo DP count, not global request index.

## Testing And Verification

Prefer TDD for feature or bug-fix work.

Use `PYTHONDONTWRITEBYTECODE=1` when running pytest to avoid generating new `__pycache__` files. Clean generated caches before finishing:

```bash
find benchmark/fault_tolerance tools/sglang-simulator -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf tools/sglang-simulator/.pytest_cache .pytest_cache
```

Useful verification commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/sglang-simulator/src pytest benchmark/fault_tolerance/simulator/tests -v
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/sglang-simulator/src pytest \
  tools/sglang-simulator/test/test_simulation_failure_manager.py \
  tools/sglang-simulator/test/test_simulation_scheduler_failure.py \
  tools/sglang-simulator/test/test_simulation_sglang_runner.py \
  tools/sglang-simulator/test/test_simulation_sglang_scheduler.py \
  tools/sglang-simulator/test/test_simulation_predictor_metadata.py \
  -v
```

Some local environments do not have optional runtime dependencies such as `transformers`, `sglang`, `sglang_simulator`, or `aiconfigurator` importable by default. Distinguish environment limitations from actual regressions, and make optional dependency imports lazy where possible.

## Documentation

The synthesis simulator docs live at:

- `benchmark/fault_tolerance/simulator/synthesis_requests/README.md`
- `benchmark/fault_tolerance/real/README.md`

The legacy fixed-workload simulator prototype under `benchmark/fault_tolerance/simulator/src/simulator` has been removed. Prefer extending `synthesis_requests` instead of recreating the old fixed-agent path.
