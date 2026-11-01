"""Prebuild the slow JIT kernels for sm80 on a GPU-less login node.

The AWQ (Marlin) GEMM takes minutes to compile; building it here keeps that
off billed GPU time. The cache key depends on nvcc and flags, so run it with
the same env.sh as the jobs.
"""

import time

import torch

from sglang.kernels.jit.utils.arch import override_jit_cuda_arch

with override_jit_cuda_arch(8, 0):
    from sglang.kernels.ops.gemm.gptq_marlin import _jit_gptq_marlin_module
    from sglang.kernels.ops.quantization.awq_marlin_repack import (
        _jit_awq_marlin_repack_module,
    )

    for build in (
        lambda: _jit_gptq_marlin_module(torch.bfloat16),
        _jit_awq_marlin_repack_module,
    ):
        t = time.time()
        build()
        print(f"built in {time.time() - t:.0f}s", flush=True)
