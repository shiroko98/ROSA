#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

ROSA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

rosa_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    "${PYTHON_BIN}" "$@"
  elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}

rosa_require_nonempty() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -z "${value}" ]]; then
    echo "[ROSA] missing required variable: ${var_name}" >&2
    exit 1
  fi
}

rosa_require_file() {
  local path="$1"
  local label="${2:-file}"
  if [[ ! -f "${path}" ]]; then
    echo "[ROSA] ${label} not found: ${path}" >&2
    exit 1
  fi
}

rosa_print_kv() {
  printf "  %-32s %s\n" "$1" "$2"
}

rosa_recipe_tag() {
  local raw="${1:-${ROSA_RECIPE:-custom}}"
  raw="${raw// /_}"
  raw="${raw//[^[:alnum:]_.-]/_}"
  printf "%s\n" "${raw}"
}

rosa_append_csv_tag() {
  local current="${1:-}"
  local extra="${2:-}"
  if [[ -z "${current}" ]]; then
    printf "%s\n" "${extra}"
  elif [[ -z "${extra}" ]]; then
    printf "%s\n" "${current}"
  else
    printf "%s,%s\n" "${current}" "${extra}"
  fi
}

rosa_apply_capacity_compare_defaults() {
  local recipe_tag
  recipe_tag="$(rosa_recipe_tag "${ROSA_RECIPE:-custom}")"

  export RUN_MODELS="${RUN_MODELS:-both}"
  export OUT_DIR="${COMPARE_OUT_DIR:-${OUT_DIR%/}_${recipe_tag}_capacity_compare}"
  export WANDB_RUN_NAME="${COMPARE_WANDB_RUN_NAME:-${WANDB_RUN_NAME:-${recipe_tag}_capacity_compare}}"
  export WANDB_GROUP="${COMPARE_WANDB_GROUP:-${WANDB_GROUP:-${recipe_tag}_capacity_compare}}"
  export WANDB_TAGS="$(rosa_append_csv_tag "${WANDB_TAGS:-}" "capacity_compare,baseline_capacity_matched,${recipe_tag}")"
}

rosa_maybe_build_compiled_cpu_extension() {
  local requested_impl="${1:-compiled_cpu}"
  if [[ "${requested_impl}" != "compiled_cpu" ]]; then
    printf "%s\n" "${requested_impl}"
    return 0
  fi

  echo "[ROSA] trying to build compiled CPU address extension..." >&2
  if (
    cd "${ROSA_ROOT}" && \
    rosa_python -c "from rosa_cpp_extension import build_rosa_sam_cpu_extension; build_rosa_sam_cpu_extension(verbose=True, force_rebuild=False)"
  ); then
    echo "[ROSA] compiled CPU address extension is ready." >&2
    printf "%s\n" "compiled_cpu"
    return 0
  fi

  echo "[ROSA] compiled CPU extension build failed, falling back to fast implementation." >&2
  printf "%s\n" "fast"
}

rosa_prepare_memmap_manifest_if_needed() {
  if [[ -n "${PRETOKENIZED_MANIFEST:-}" ]]; then
    rosa_require_file "${PRETOKENIZED_MANIFEST}" "PRETOKENIZED_MANIFEST"
    return 0
  fi

  rosa_require_nonempty TOKENIZER_NAME_OR_PATH
  rosa_require_nonempty MEMMAP_OUT_DIR
  PRETOKENIZED_MANIFEST="${MEMMAP_OUT_DIR%/}/dataset_manifest.json"

  if [[ -f "${PRETOKENIZED_MANIFEST}" ]]; then
    echo "[ROSA] reuse existing memmap manifest: ${PRETOKENIZED_MANIFEST}" >&2
    return 0
  fi

  mkdir -p "${MEMMAP_OUT_DIR}"
  local args=(
    build_rosa_memmap_dataset.py
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
    --data_format "${DATA_FORMAT:-jsonl}"
    --split_mode "${SPLIT_MODE:-paragraph}"
    --json_text_keys "${JSON_TEXT_KEYS:-text,content,body,message}"
    --tokenize_workers "${TOKENIZE_WORKERS:-1}"
    --tokenize_batch_docs "${TOKENIZE_BATCH_DOCS:-64}"
    --progress_docs "${PROGRESS_DOCS:-5000}"
    --out_dir "${MEMMAP_OUT_DIR}"
  )

  if [[ -n "${TRAIN_DATA_PATH:-}" && -n "${VAL_DATA_PATH:-}" && -n "${TEST_DATA_PATH:-}" ]]; then
    args+=(
      --train_data_path "${TRAIN_DATA_PATH}"
      --val_data_path "${VAL_DATA_PATH}"
      --test_data_path "${TEST_DATA_PATH}"
    )
    if [[ -n "${MAX_TRAIN_DOCS:-}" ]]; then
      args+=(--max_train_docs "${MAX_TRAIN_DOCS}")
    fi
    if [[ -n "${MAX_VAL_DOCS:-}" ]]; then
      args+=(--max_val_docs "${MAX_VAL_DOCS}")
    fi
    if [[ -n "${MAX_TEST_DOCS:-}" ]]; then
      args+=(--max_test_docs "${MAX_TEST_DOCS}")
    fi
  elif [[ -n "${DATA_PATH:-}" ]]; then
    args+=(--data_path "${DATA_PATH}")
    if [[ -n "${MAX_DOCS:-}" ]]; then
      args+=(--max_docs "${MAX_DOCS}")
    fi
  else
    echo "[ROSA] set PRETOKENIZED_MANIFEST, or provide TRAIN_DATA_PATH/VAL_DATA_PATH/TEST_DATA_PATH, or provide DATA_PATH." >&2
    exit 1
  fi

  echo "[ROSA] building memmap dataset..." >&2
  (
    cd "${ROSA_ROOT}" && \
    rosa_python "${args[@]}"
  )
  rosa_require_file "${PRETOKENIZED_MANIFEST}" "generated PRETOKENIZED_MANIFEST"
}

rosa_print_launch_header() {
  local title="$1"
  echo "[ROSA] ${title}"
  rosa_print_kv "repo" "${ROSA_ROOT}"
  rosa_print_kv "python bin" "${PYTHON_BIN:-python/conda}"
  rosa_print_kv "conda env" "${CONDA_ENV_NAME:-<unset>}"
  rosa_print_kv "tokenizer" "${TOKENIZER_NAME_OR_PATH:-<unset>}"
  rosa_print_kv "manifest" "${PRETOKENIZED_MANIFEST:-<auto>}"
  rosa_print_kv "output dir" "${OUT_DIR:-<unset>}"
  rosa_print_kv "recipe" "${ROSA_RECIPE:-<unset>}"
  rosa_print_kv "run models" "${RUN_MODELS:-<unset>}"
  rosa_print_kv "online sam impl" "${ROSA_ONLINE_SAM_IMPL:-<unset>}"
  rosa_print_kv "wandb enabled" "${ENABLE_WANDB:-0}"
  rosa_print_kv "wandb mode" "${WANDB_MODE:-disabled}"
  rosa_print_kv "wandb dir" "${WANDB_DIR:-${OUT_DIR:-<unset>}}"
  rosa_print_kv "distributed strategy" "${DISTRIBUTED_STRATEGY:-<unset>}"
  rosa_print_kv "backend" "${DISTRIBUTED_BACKEND:-<unset>}"
  rosa_print_kv "world size" "${NPROC_PER_NODE:-<unset>}"
  rosa_print_kv "batch size per gpu" "${BATCH_SIZE:-<unset>}"
  rosa_print_kv "grad accum steps" "${GRAD_ACCUM_STEPS:-<unset>}"
  rosa_print_kv "seq len" "${SEQ_LEN:-<unset>}"
  rosa_print_kv "epochs" "${EPOCHS:-<unset>}"
}
