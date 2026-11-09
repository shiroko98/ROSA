#!/usr/bin/env bash
# Identical backbone, SAM, table size, data order and hyperparameters; output-only ROSA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROSA_POSITION=post_norm
CONTEXT="${ROSA_TRAIN_CONTEXT:-131072}"
export ROSA_OUTPUT_DIR="${ROSA_OUTPUT_DIR:-/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx${CONTEXT}_20261109}"
# Do not block the first real optimizer update on a long CPU baseline monitor.
# Keep regular SFT eval, but NIAH runs only in a separate checkpoint worker.
exec bash "$ROOT/scripts/run_rwkv_3b_lastlayer_training.sh" \
  --rosa_eval_initial 0 --rosa_needle_every_eval 0 --rosa_needle_timeout 14400 "$@"
