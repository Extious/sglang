# AegisKV workspace notes

- Scope (2026-11-01): AegisKV runs as real inference on SGLang v0.5.21; the
  simulator-era runtime and results were removed from the branch. The full
  pre-restructure tree is the local-only branch `aegiskv-legacy-20261101`; old
  docs, experiments and the figure5 dataset are under `~/aegiskv-archive-20261101/`.
- Old results are not in the workspace: `experiments/`, `profiling/` and
  `workflows/` moved to `~/aegiskv-archive-20261101/benchmark-AegisKV/` on
  2026-11-02. Paper figure scripts under `paper/figure_sources/` still name
  `experiments/`; restore it from the archive before regenerating an old figure.
  `docs/` (related-work PDFs) and `paper/*` except the `paper/overleaf`
  submodule stay local and git-excluded.
- The manuscript is `paper/overleaf` (Overleaf git remote). Never commit its
  remote URL with credentials.
- Baselines live on their own branches; keep the data plane and Master identical
  across branches and change only `harness/<policy>_policy.py`.
- GPU time on the BLSC A100 cluster is billed: smoke-test small first, prebuild
  JIT kernels on the login node, and make every job tear its Workers down.
