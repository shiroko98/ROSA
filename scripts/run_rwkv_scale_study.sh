#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rwkv-lm-v7/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export CC="${CC:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-g++}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.codex-build/rwkv_native_cuda_h800}"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4
export TORCH_WARM_POOL=0 TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RWKV_MY_TESTING=x070 RWKV_KERNEL=@rwkv3 RWKV_JIT_ON=0 RWKV_HEAD_SIZE=64
export RWKV_HEAD_L2WRAP_CE_CHUNK=0 RWKV_SFT_MASKED_FUSED_CE_CHUNK=1024 RWKV_FLOAT_MODE=bf16
source "$ROOT/third_party/rwkv_lm_v7_native/scripts/setup_rwkv_cuda_env.sh"
export PYTHONPATH="$ROOT/third_party/rwkv_lm_v7_native:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/third_party/rwkv_lm_v7_native"
exec "$PYTHON_BIN" -u -m rwkv_rosa.native.scale_study "$@"
