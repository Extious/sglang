# Environment for the BLSC A100-PCIE-40GB nodes (aarch64 Kunpeng-920, driver 535).
# SGLang v0.5.21 needs CUDA 13; the A100 runs it through NVIDIA's forward-compat
# user-space driver (cuda-compat-13-0) on the R535 kernel driver.
AEGIS=${AEGIS:-$HOME/aegis}
# JIT kernels need C++20 host headers; the system gcc is 7.3, so use the site gcc 13.
export PATH=$AEGIS/env/bin:$AEGIS/cuda13/bin:/home/bingxing2/apps/compilers/gcc/13.2.0/bin:$PATH
export CUDA_HOME=$AEGIS/cuda13
export LD_LIBRARY_PATH=$AEGIS/compat/usr/local/cuda-13.0/compat:$AEGIS/cuda13/lib:${LD_LIBRARY_PATH:-}
export SGLANG_JIT_CACHE_DIR=$AEGIS/jitcache
export TRITON_CACHE_DIR=$AEGIS/triton_cache
export TORCHINDUCTOR_CACHE_DIR=$AEGIS/inductor_cache
export TMPDIR=$AEGIS/tmp
export HF_HUB_OFFLINE=1
# scikit-learn (the master's Tracker GBDT) lives outside the shared env: on
# aarch64 its bundled libgomp cannot be dlopen'ed after torch ("cannot allocate
# memory in static TLS block"), and workers would import it through transformers.
export AEGIS_MASTER_SITE=$AEGIS/master_site
mkdir -p $TMPDIR $SGLANG_JIT_CACHE_DIR $TRITON_CACHE_DIR
