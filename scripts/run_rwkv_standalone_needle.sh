#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=''
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://dian.sea-group.org:9098}"
export WANDB_ENTITY="${WANDB_ENTITY:-mobius}"
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_RUN_ID WANDB_RESUME
cd "$ROOT"
exec "${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}" -u \
  -m rwkv_rosa.native.standalone_needle "$@"
