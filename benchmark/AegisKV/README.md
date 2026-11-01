# AegisKV on SGLang (real inference)

AegisKV protects agentic Workflows against fail-stop Worker failures by keeping
KV backups on peer Workers, chosen by each Agent's slack. This directory holds
the experiment harness; the serving-side data plane is
`python/sglang/srt/aegiskv/`. The branch is based on SGLang **v0.5.21**.

## Layout

| Path | Content |
| --- | --- |
| `python/sglang/srt/aegiskv/` | Worker data plane behind `POST /aegiskv/control` (`--enable-aegiskv`) |
| `harness/` | Master/router, workloads, failure injection, policies, profiling, analysis |
| `configs/` | Experiment configs (JSON, `harness/config.py` schema) |
| `scripts/` | Cluster launch scripts |
| `paper/overleaf` | The manuscript (git submodule of the Overleaf project) |
| `results/` | Measured results; start with [`results/a100_main`](results/a100_main/README.md) |

Branches: `aegiskv` (this one: data plane, harness, AegisKV policy, paper),
`baseline-lumen` (LUMEN-CP\*), `baseline-hbm-replica` (HBM-Replica). Baseline
branches fork from the shared-infrastructure commit and add one policy module,
so every method runs on the same data plane, router and workload.

## System

Each Worker is an unmodified `sglang.launch_server` (TP=1, one GPU) started with
`--enable-aegiskv --page-size 1`. The data plane runs inside the scheduler loop
and does four things on command:

* **Checkpoint** (`protect`): every 64 committed tokens, copy a running request's
  new KV (all layers) into a peer-owned DRAM arena, a `/dev/shm` file pinned
  with `cudaHostRegister`. Copies run on a side stream ordered after the
  forward stream and finish before the next scheduling step.
* **Peer-HBM region** (`reserve_region`/`stage_region`): the holder reserves
  slots in its own KV pool for the first `L` layers of a peer's request, packed
  so `L` layers of `n` tokens cost `ceil(L*n/32)` slots. Bytes are staged
  through the arena because the A100 PCIe GPUs have no peer-to-peer access.
* **Import** (`import`): on takeover, rebuild the committed prefix from the
  region (D2D) and/or the arena (H2D) into fresh slots, insert it into the radix
  cache and lock it until the resubmitted request has matched it. A full-layer
  replica is adopted without copying.
* **Poll**: commit frontiers, finished stages/imports and KV-pool usage.

The Master (`harness/master.py`) runs the closed-loop Workflows, picks for each
Agent call an execution Worker and a distinct retry Worker (the holder of its
backups), and calls the policy every poll tick. A failure is a SIGKILL of the
Worker's process group, which also loses its arena; interrupted calls resume on
their retry Worker with `prompt + accepted tokens` after importing the latest
committed backup.

Policies (`harness/*_policy.py`):

* `rebuild`: no backups.
* `aegiskv`: Tracker (GBDT duration + critical-path slack) -> Agent Selection
  (zero slack: full history in DRAM plus the first `L` layers in HBM; positive
  slack: smallest prefix `d*` with `T^D(n, d*) <= s`) -> Adapter (HBM quota from
  forecast foreground demand, Reclaim of the earliest regions).
* `lumen` / `hbm_replica`: on their baseline branches.

MiniCPM4.1-8B's released checkpoints use dense attention, so Region Selection
keeps every block of the first `L` layers (Top-K = 100%); sparse Top-K needs
InfLLM-v2 and is not exercised here. Restores are bulk imports before the
resubmission, not layer-streamed into the replay.

## Results

[`results/a100_main`](results/a100_main/README.md): 4x A100-40GB, 32 Workflows at
concurrency 8, one Worker failure per run (each Worker once). Against LUMEN-CP\*,
HBM-Replica and Rebuild, AegisKV gives critical-path victims the earliest first
resumed token (2.69 s vs 3.78 / 4.76 / 6.86 s), the lowest exposed delay and added
WCT, and the highest critical-path KV coverage (81%), with 3.7 GiB of DRAM and no
HBM.

## Running

```bash
cd benchmark/AegisKV
# Data-plane check on 2 GPUs: checkpoint, region staging and import round trips.
python -m harness.smoke --config configs/smoke_local.json
# One set of Workers for a whole suite: warm-up (also the Tracker's training data),
# recovery profile, Tracker, No-Failure reference, then runs with one failure each.
python -m harness.run --config configs/<cfg>.json --warmup 12 --out-dir runs/<x> \
    --suite profile,tracker,nofail,rebuild@0,aegiskv@0,rebuild@1,aegiskv@1
# Baselines run from their own branches with the same failure time.
python -m harness.run --config configs/<cfg>.json --warmup 12 --out-dir runs/<x>-lumen \
    --suite lumen@0,lumen@1 --fail-at "$(cat runs/<x>/fail_at_s.txt)"
python -m harness.analyze --nofail runs/<x>/nofail --out results/<x> \
    --runs rebuild=runs/<x>/rebuild_w0,runs/<x>/rebuild_w1 aegiskv=... lumen=...
```

`scripts/blsc/` holds the BLSC A100-40GB (aarch64) environment notes and Slurm
scripts used for `results/a100_main`.
