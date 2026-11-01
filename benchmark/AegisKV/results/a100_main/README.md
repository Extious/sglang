# A100-40GB comparison: one concurrency, one failure per run

All numbers come from real MiniCPM4.1-8B inference on SGLang v0.5.21 (this
branch), run on 2026-11-01/02 on one BLSC node: 4x A100-PCIE-40GB without GPU
peer-to-peer, Kunpeng-920 aarch64, driver 535 with the CUDA 13 forward-compat
runtime. `raw/` holds every run's `result.json`, `events.jsonl` and
`config.json`; `analysis/` holds the pooled summary and the figure.

![comparison](analysis/failure_comparison.png)

## Setup

| Item | Value |
| --- | --- |
| Model | MiniCPM4.1-8B AutoAWQ (W4A16, Marlin), dense attention, BF16 KV |
| Workers | 4, TP=1, one per GPU; KV pool 921,608 tokens per worker |
| Server flags | `--page-size 1 --chunked-prefill-size 4096 --enable-priority-scheduling --disable-priority-preemption --cuda-graph-backend-prefill disabled` |
| Workload | 32 Workflows, closed loop at concurrency 8; A -> {B, C, D, E} -> F; every Agent owns 16,384 random history tokens plus its predecessors' outputs; outputs 256 tokens, C 768 (greedy, `ignore_eos`) |
| Failure | SIGKILL of one Worker at 57.1 s (35% of the 163.2 s No-Failure makespan); it rejoins after 60 s. Each policy runs four times, failing Worker 0, 1, 2 and 3 |
| Backup capacity per holder | DRAM checkpoints 65,536 tokens (2 GiB, about 0.07x the KV pool); peer-HBM quota 460,000 tokens (0.5x pool); 200,000-token host staging for HBM copies |
| Checkpoint cadence | every 64 committed tokens, all policies |

Policies (same data plane, router, workload and cadence; only placement differs):

* **Rebuild**: no backups; the retry Worker re-prefills the accepted history.
* **LUMEN-CP\***: full-history peer-DRAM checkpoints for every call, first come
  first served until the holder's DRAM budget is full (`baseline-lumen`).
* **HBM-Replica**: full KV replica of every call in the holder's KV pool under
  the static quota; takeover adopts it without a copy (`baseline-hbm-replica`).
* **AegisKV**: GBDT Tracker + critical-path slack; zero-slack calls get a full
  peer-DRAM checkpoint (peer HBM only if DRAM is full) and resume at higher
  priority; positive-slack calls keep the smallest prefix whose modeled recovery
  fits their slack, which here is none.

Offline inputs, produced inside the same job on the same GPUs: the recovery
profile (`raw/aegiskv/profile.json`: import 1.78 us/token, about 18 GB/s, so a
16K prefix imports in 32 ms; prefill of 16K tokens about 2.2 s) and the Tracker
(trained on 72 calls of a warm-up run with a different seed, training MAE
2.9 s on a mean duration of 11.0 s).

## Results (pooled over the four failed Workers)

| Metric | Rebuild | LUMEN-CP\* | HBM-Replica | AegisKV |
| --- | ---: | ---: | ---: | ---: |
| Victims / on the critical path | 16 / 8 | 16 / 9 | 17 / 8 | 16 / 8 |
| First resumed token, critical-path victims (s) | 6.86 | 3.78 | 4.76 | **2.69** |
| Exposed delay, all victims (s) | 3.42 | 2.12 | 2.25 | **1.34** |
| Added WCT of affected Workflows, mean / P95 (s) | 10.87 / 23.32 | 7.88 / 17.97 | 8.26 / 16.56 | **6.34 / 15.36** |
| KV restored, all / critical-path victims | 0% / 0% | 53% / 61% | 39% / 58% | 40% / **81%** |
| WCT of pre-failure Workflows vs No-Failure | 1.00 | 1.01 | 1.00 | 1.00 |
| Mean backup footprint | none | 4.7 GiB DRAM | 6.9 GiB HBM (peak 11.4) | 3.7 GiB DRAM |

"Critical path" means zero slack in the No-Failure run (slack <= 1 s), computed
from actual call durations. Exposed delay is the first resumed token minus that
reference slack, floored at 0. Twelve Workflows were affected per policy.

## Reading the results

* **Where AegisKV wins.** The DRAM budget fits the critical calls but not every
  call. LUMEN-CP\* fills it with whichever calls arrived first, so a critical
  call that starts later is often unprotected. AegisKV leaves positive-slack
  calls (B, D, E with about 9-11 s of slack against a 2-5 s rebuild) without
  backups, so critical calls always find room. It restores more critical-path
  KV (81% vs 61%) with less memory (3.7 vs 4.7 GiB), and it resumes them ahead
  of the slack-absorbing rebuilds on the retry Worker. The Tracker's online slack
  matched the reference slack: 0 for C/A/F, 8.7-11.3 s predicted vs 10.5-15.4 s
  actual for B/D/E.
* **HBM-Replica.** The quota was large enough for every call, but without P2P a
  peer-HBM replica is two copies (source GPU to host staging, then host to the
  holder's GPU) plus a control round trip. It lags the DRAM checkpoint by 1-3 s,
  so calls that started shortly before the failure had no replica yet. It also
  occupied 6.9 GiB of the holders' KV pools on average. In this setup the KV pool
  was not under pressure, so that footprint cost no measurable throughput.
* **Resident depth L = 0.** On this hardware a layer of 16K-token KV copies
  (about 0.9 ms) faster than a layer of replay compute (about 2.2 ms), so the
  zero-wait depth is 0. AegisKV therefore placed no HBM regions here.
* **Remaining recovery time.** Even a restored critical call waits 0.3-4.8 s for
  its first token: the import is handled at the retry Worker's next scheduler
  iteration (up to one 4,096-token prefill chunk), and the resumed request then
  waits for any in-flight chunked prefill on that Worker.
* **Variance.** Added WCT varies widely per Workflow (panel b), so with 12
  affected Workflows per policy the WCT ordering is indicative, not significant.
  Recovery latency and coverage of critical-path victims are the clearer signals.

## Deviations from the paper's setup

* Real inference on 4 Workers instead of trace-driven emulation on 8.
* Dense attention: Region Selection keeps every block (Top-K = 100%), and
  restores are bulk imports before resubmission rather than layer-streamed
  into the replay. With L = 0 this costs nothing here.
* The checkpoint DRAM budget is 0.07x the KV pool rather than 0.5x. The node
  grants about 55 GB of host memory per GPU, and the budget has to bind for the
  policies to differ; at 0.5x every call fits and LUMEN-CP\* equals AegisKV.
* One call per Agent (no multi-turn reuse).

## Reproducing

From `benchmark/AegisKV` on the BLSC login node (see `scripts/blsc/SETUP.md`):

```bash
sbatch -J main -t 01:20:00 --export=ALL,BRANCH_DIR=$HOME/aegis/src/aegiskv,CONFIG=configs/a100_main.json,\
SUITE=profile:tracker:nofail:rebuild@0:aegiskv@0:rebuild@1:aegiskv@1:rebuild@2:aegiskv@2:rebuild@3:aegiskv@3,\
WARMUP=12,OUT=$HOME/aegis/runs/main/aegiskv scripts/blsc/run_suite.sbatch
# then, from the baseline-lumen / baseline-hbm-replica checkouts:
sbatch -J lumen -t 00:40:00 --export=ALL,BRANCH_DIR=$HOME/aegis/src/baseline-lumen,\
CONFIG=$HOME/aegis/src/aegiskv/benchmark/AegisKV/configs/a100_main.json,SUITE=lumen@0:lumen@1:lumen@2:lumen@3,\
WARMUP=12,OUT=$HOME/aegis/runs/main/lumen,FAIL_AT_FILE=$HOME/aegis/runs/main/aegiskv/fail_at_s.txt \
  $HOME/aegis/src/baseline-lumen/benchmark/AegisKV/scripts/blsc/run_suite.sbatch
python -m harness.analyze --nofail raw/aegiskv/nofail --runs rebuild=... lumen=... hbm_replica=... aegiskv=... --out analysis
```

GPU time for this comparison: about 33 min (main) + 2 x 15 min (baselines) on
4 GPUs, about 4.2 GPU-hours.
