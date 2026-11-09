#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ROSA_RWKV_ENV:-rosa-rwkv}"
if [[ "$ENV_NAME" == "base" ]]; then
  echo "Refusing to install into base; choose an independent ROSA_RWKV_ENV." >&2
  exit 2
fi
if ! conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" python=3.11 pip
fi
# Native SFT kernels need a CUDA compiler compatible with this Torch CUDA build.
conda install -y -n "$ENV_NAME" -c nvidia 'cuda-nvcc=12.6' 'cuda-cudart-dev=12.6' \
  'gcc_linux-64=12' 'gxx_linux-64=12'
export DS_BUILD_OPS=0
# Independent environment only; never alter base or an existing RWKV environment.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  -r "$ROOT/requirements-rwkv.txt" 'numpy==1.26.4' 'jinja2==3.1.6' \
  'coverage>=7.6,<8' 'pytest>=8,<10'
conda run --no-capture-output -n "$ENV_NAME" python -c \
  'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())'
