#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rwkv_rosa/toy.json}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec conda run --no-capture-output -n "${ROSA_RWKV_ENV:-rosa-rwkv}" \
  python -u train_rwkv_rosa_compare.py --config "$CONFIG" "$@"
