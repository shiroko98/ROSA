#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

export RUN_MODELS="baseline"
export OUT_DIR="${BASELINE_OUT_DIR:-${OUT_DIR%/}_capacity_baseline}"
export WANDB_RUN_NAME="${BASELINE_WANDB_RUN_NAME:-${WANDB_RUN_NAME:-online_v1_capacity_baseline}}"
export WANDB_GROUP="${BASELINE_WANDB_GROUP:-${WANDB_GROUP:-online_v1_capacity_compare}}"
if [[ -n "${WANDB_TAGS:-}" ]]; then
  export WANDB_TAGS="${WANDB_TAGS},baseline,capacity_matched"
else
  export WANDB_TAGS="baseline,capacity_matched"
fi

exec bash "${SCRIPT_DIR}/launch_8gpu_ddp_online_v1.sh" --baseline_capacity_match_rosa "$@"
