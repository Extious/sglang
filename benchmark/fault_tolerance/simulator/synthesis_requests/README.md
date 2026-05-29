# Synthesis Request Fault Tolerance Simulator

This benchmark runs synthetic fixed-length request workloads against SGLang simulator fault-tolerance behavior.

Synthetic requests use Request ID as the experiment identity. They do not carry
multi-agent fields such as `task_label` or `agent_role`.

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
- `figures/backup_cache_hit_profile.png`
- `figures/trace_log_profile.png`

## Workload Cache Locality

Set `workload.shared_prefix_len` in `base.json` to make synthetic requests reuse
the same deterministic prefix while keeping `input_len` unchanged. This creates
a small amount of realistic prefix overlap for checking whether retried requests
can reuse backed-up KV after a failure.
