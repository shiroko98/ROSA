#!/usr/bin/env bash
# Full original 192513 stored tokens -> 192512 model inputs, one window per row.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROSA_TRAIN_CONTEXT=192512
export ROSA_OUTPUT_DIR="${ROSA_OUTPUT_DIR:-/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109}"
exec bash "$ROOT/scripts/run_rwkv_3b_postnorm_training.sh" "$@"
