#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_rosa_server.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-model}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"
MEMMAP_OUT_DIR="${MEMMAP_OUT_DIR:-${ROSA_ROOT}/rosa_runs/memmap_dataset}"
DATA_FORMAT="${DATA_FORMAT:-jsonl}"
SPLIT_MODE="${SPLIT_MODE:-paragraph}"
JSON_TEXT_KEYS="${JSON_TEXT_KEYS:-text,content,body,message}"

ROSA_REQUIRE_SPLITS="${ROSA_REQUIRE_SPLITS:-1}"

if [[ "${ROSA_REQUIRE_SPLITS}" == "1" ]]; then
  rosa_require_nonempty TRAIN_DATA_PATH
  rosa_require_nonempty VAL_DATA_PATH
  rosa_require_nonempty TEST_DATA_PATH
fi

rosa_prepare_memmap_manifest_if_needed

echo "[ROSA] memmap dataset is ready."
rosa_print_kv "manifest" "${PRETOKENIZED_MANIFEST}"
