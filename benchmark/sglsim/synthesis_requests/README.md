# SGLSim-aligned synthesis_requests (real GPU)

GSP (`generated-shared-prefix`) workload aligned with `sglsim/benchmark/synthesis_requests`, executed on real `sglang.launch_server` GPU inference.

## Layout

```
benchmark/sglsim/synthesis_requests/
├── run.py
├── compare.py
├── client.py / server.py / utils.py
├── cache_store.py / progress.py
├── config/
│   ├── synthesis-a100/
│   └── synthesis-a100-smoke/
├── cache/
├── logs/
└── results/<suite>/online/YYYYMMDD_HHMMSS/
```

## Run (real GPU)

From sglang repo root:

```bash
pip install transformers numpy tqdm
python -m benchmark.sglsim.synthesis_requests.run \
  --config-dir benchmark/sglsim/synthesis_requests/config/synthesis-a100-smoke
```

If the server is already up:

```bash
python -m benchmark.sglsim.synthesis_requests.run \
  --config-dir benchmark/sglsim/synthesis_requests/config/synthesis-a100-smoke \
  --skip-server
```

Outputs under `results/<suite>/online/<timestamp>/`:

- `server_config.json`
- `metrics.json`
- `request_details.csv`
- `server.log`

Client progress and cache hits: `logs/client.log`. Shared workload cache: `cache/`.

## client.json (GSP)

Same fields as sglsim `benchmark/synthesis_requests/config/client.json`:

| Field | Default (smoke) |
|-------|-----------------|
| `request_rate` | `"inf"` (burst; use `max_concurrency`) |
| `max_concurrency` | `4` |
| `gsp_num_groups` | `2` |
| `gsp_prompts_per_group` | `4` |
| `gsp_system_prompt_len` | `512` |
| `gsp_question_len` | `1024` |
| `gsp_output_len` | `128` |

`synthesis-a100`: 64 groups x 1 prompt, ~20k question / 2k output tokens.

## Compare simulator vs GPU

```bash
python -m benchmark.sglsim.synthesis_requests.compare \
  --sim-request-details /path/to/sglsim/benchmark/synthesis_requests/results/.../request_details.csv \
  --real-request-details benchmark/sglsim/synthesis_requests/results/synthesis-a100-smoke/online/.../request_details.csv \
  --real-metrics benchmark/sglsim/synthesis_requests/results/synthesis-a100-smoke/online/.../metrics.json
```

Ratio `sim/real` < 1 means the simulator is faster than GPU (typical for offline logical clock).
