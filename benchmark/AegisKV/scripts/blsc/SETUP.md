# BLSC A100-40GB environment

Nodes: 4x A100-PCIE-40GB (no GPU P2P), Kunpeng-920 aarch64, 254 GB RAM, driver
535.104 (CUDA 12.2). The login node is aarch64 and has no GPU. Long downloads
from the login node break at about 160 MB, and GitHub access is intermittent,
so every download below resumes and retries.

1. Python: Miniforge (aarch64) from the TUNA mirror, `conda create -p ~/aegis/env python=3.12`.
2. PyTorch 2.13.0+cu130 aarch64: fetch the wheel from `mirrors.nju.edu.cn/pytorch/whl/cu130`
   with `curl -C -`, then `pipget.sh ./torch-...whl`.
3. CUDA 13 on the R535 driver: extract `cuda-compat-13-0-580.196.07-1.el8.aarch64.rpm`
   (NVIDIA rhel8/sbsa repo) to `~/aegis/compat`; `env.sh` puts its `libcuda.so` first.
4. CUDA 13.0 toolkit for JIT kernels: `conda create -p ~/aegis/cuda13 -c conda-forge
   cuda-version=13.0 cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev cuda-profiler-api
   cuda-nvtx-dev libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev cuda-driver-dev`,
   then `ln -s lib ~/aegis/cuda13/lib64` (the JIT linker looks in `lib64`).
5. SGLang dependencies: the `dependencies` of `python/pyproject.toml` minus `torchcodec`,
   `llguidance`, `tilelang` and `cuda-tile` (no aarch64 wheels; unused here), installed
   with `pipget.sh -r deps.txt --only-binary :all:`; then `pip install --no-deps -e python`.
6. Master-only packages: `pip install --no-deps --target ~/aegis/master_site scikit-learn`
   (not in the shared env; see env.sh).
7. Host compiler: JIT kernels need C++20; `env.sh` uses the site gcc 13.2.
8. Model: `hfget.sh openbmb/MiniCPM4.1-8B-AutoAWQ ~/aegis/models/MiniCPM4.1-8B-AutoAWQ`.
9. Prebuild the Marlin GEMM for sm80 on the login node: `python scripts/blsc/prebuild_jit.py`.

`envcheck.sbatch` checks one GPU; `run_suite.sbatch` runs an experiment suite.
