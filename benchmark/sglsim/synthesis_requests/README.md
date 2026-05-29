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
│   ├── client.json
│   └── server.json
├── cache/
├── logs/
└── results/YYYYMMDD_HHMMSS/
```

## Run

From sglang repo root:

```bash
pip install transformers numpy tqdm
python -m benchmark.sglsim.synthesis_requests.run
```

Custom config directory:

```bash
python -m benchmark.sglsim.synthesis_requests.run \
  --config-dir benchmark/sglsim/synthesis_requests/config
```

If the server is already up:

```bash
python -m benchmark.sglsim.synthesis_requests.run --skip-server
```

Outputs under `results/<timestamp>/`:

- `server_config.json`
- `metrics.json`
- `request_details.csv`
- `server.log`

## client.json

Same fields as sglsim `benchmark/synthesis_requests/config/client.json`.

| Field | SGLang CLI | Default |
|-------|------------|---------|
| `request_rate` | `--request-rate` | `"inf"` |
| `max_concurrency` | `--max-concurrency` | `32` |
| `gsp_num_groups` | `--gsp-num-groups` | `8` |
| `gsp_prompts_per_group` | `--gsp-prompts-per-group` | `16` |
| `gsp_system_prompt_len` | `--gsp-system-prompt-len` | `4096` |
| `gsp_question_len` | `--gsp-question-len` | `20000` |
| `gsp_output_len` | `--gsp-output-len` | `1024` |
| `gsp_range_ratio` | `--gsp-range-ratio` | `0.8` |
| `gsp_fast_prepare` | `--gsp-fast-prepare` | `false` |
| `gsp_send_routing_key` | `--gsp-send-routing-key` | `false` |
| `gsp_num_turns` | `--gsp-num-turns` | `1` |
| `gsp_ordered` | `--gsp-ordered` | `false` |
| `seed` | `--seed` | `1` |

Total requests: `gsp_num_groups * gsp_prompts_per_group` (128 with defaults).

## server.json

GPU deploy fields only (no simulator `platform` / `predictor` / `cache`):

| Field | Description |
|-------|-------------|
| `experiment_name` | Run label in `server_config.json` |
| `model_path` | HuggingFace model for `sglang.launch_server` |
| `host` / `port` | HTTP bind address |
| `scheduler.tp_size` / `pp_size` / `dp_size` | Parallelism for launch |
| `extra_args` | Extra CLI flags appended to `launch_server` |

Match `scheduler` sizes with sglsim when comparing runs. sglsim may use `dp_size > 1`; set the same here if your GPU setup supports it.

## Compare simulator vs GPU

```bash
python -m benchmark.sglsim.synthesis_requests.compare \
  --sim-request-details /path/to/sglsim/benchmark/synthesis_requests/results/.../request_details.csv \
  --real-request-details benchmark/sglsim/synthesis_requests/results/.../request_details.csv \
  --real-metrics benchmark/sglsim/synthesis_requests/results/.../metrics.json
```

Ratio `sim/real` < 1 means the simulator is faster than GPU.
