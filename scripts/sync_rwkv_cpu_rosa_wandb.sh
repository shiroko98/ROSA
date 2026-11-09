#!/usr/bin/env bash
# Invoke after training finishes, or after W&B connectivity/credentials recover.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/outputs/rwkv7_3b_cpu_rosa_frozen_sft_20261109}"
# Respect an explicit WANDB_BASE_URL or the user's W&B settings file.
# Do not override a configured self-hosted W&B server with the public service.
shopt -s nullglob
RUNS=("$OUT"/improved/wandb/offline-run-*)
if [[ ${#RUNS[@]} -eq 0 ]]; then echo 'No offline W&B runs found.' >&2; exit 1; fi
for run in "${RUNS[@]}"; do
    timeout 120 "${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}" -m wandb sync "$run"
done
