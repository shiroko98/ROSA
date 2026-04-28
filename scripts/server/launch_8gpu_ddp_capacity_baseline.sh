#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_rosa_server.sh"

rosa_apply_capacity_baseline_defaults

exec bash "${SCRIPT_DIR}/launch_8gpu_ddp_online_v1.sh" --baseline_capacity_match_rosa "$@"
