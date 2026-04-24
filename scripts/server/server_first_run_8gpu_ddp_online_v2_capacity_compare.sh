#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[ROSA] repo: ${ROOT_DIR}"
echo "[ROSA] step 1/3: install environment"
bash "${SCRIPT_DIR}/install_model_env_server.sh"

echo "[ROSA] step 2/3: prepare memmap dataset if needed"
bash "${SCRIPT_DIR}/prepare_memmap_dataset.sh"

echo "[ROSA] step 3/3: launch 8-gpu ddp online_v2 capacity compare"
bash "${SCRIPT_DIR}/launch_8gpu_ddp_online_v2_capacity_compare.sh" "$@"
