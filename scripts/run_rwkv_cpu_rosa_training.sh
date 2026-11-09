#!/usr/bin/env bash
# Frozen RWKV-7 on GPU, learned improved ROSA + optimizers on CPU.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}"
CONFIG="${1:-$ROOT/configs/rwkv_rosa/rwkv7_2_9b_cpu_rosa_sft.json}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
"$PYTHON" -u train_rwkv_rosa_compare.py --config "$CONFIG" "$@"
train_rc=$?
# This wrapper intentionally preserves offline records even when remote auth or
# network is unavailable. Upload errors are NOT reported as training failures,
# nor is an offline cache described as successfully uploaded.
if [[ $train_rc -ne 0 ]]; then
    echo "TRAINING_FAILED exit=$train_rc at $(date --iso-8601=seconds)"
    exit "$train_rc"
fi
echo "TRAINING_COMPLETE at $(date --iso-8601=seconds)"
