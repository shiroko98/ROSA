#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROSA_RECIPE="${ROSA_RECIPE:-online_v2}"
export MASTER_PORT="${MASTER_PORT:-29512}"

exec bash "${SCRIPT_DIR}/launch_8gpu_ddp_capacity_baseline.sh" "$@"
