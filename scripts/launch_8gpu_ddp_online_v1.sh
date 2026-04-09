#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_rosa_server.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-model}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"
PRETOKENIZED_MANIFEST="${PRETOKENIZED_MANIFEST:-}"
MEMMAP_OUT_DIR="${MEMMAP_OUT_DIR:-${ROSA_ROOT}/rosa_runs/memmap_dataset}"
OUT_DIR="${OUT_DIR:-${ROSA_ROOT}/rosa_runs/8gpu_ddp_online_v1}"

DATA_FORMAT="${DATA_FORMAT:-jsonl}"
SPLIT_MODE="${SPLIT_MODE:-paragraph}"
JSON_TEXT_KEYS="${JSON_TEXT_KEYS:-text,content,body,message}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

ARCH_STYLE="${ARCH_STYLE:-qwen}"
SEQ_LEN="${SEQ_LEN:-1024}"
STRIDE="${STRIDE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEED="${SEED:-42}"

DIM="${DIM:-768}"
N_LAYERS="${N_LAYERS:-16}"
N_HEADS="${N_HEADS:-12}"
N_KV_HEADS="${N_KV_HEADS:-12}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-3072}"

ROSA_RECIPE="${ROSA_RECIPE:-online_v1}"
ROSA_ONLINE_SAM_IMPL="${ROSA_ONLINE_SAM_IMPL:-compiled_cpu}"
DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-ddp}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
RUN_MODELS="${RUN_MODELS:-rosa_fused}"
TRAIN_TIMING="${TRAIN_TIMING:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

EXTRA_ARGS=("$@")

rosa_require_nonempty TOKENIZER_NAME_OR_PATH
rosa_prepare_memmap_manifest_if_needed
ROSA_ONLINE_SAM_IMPL="$(rosa_maybe_build_compiled_cpu_extension "${ROSA_ONLINE_SAM_IMPL}")"

mkdir -p "${OUT_DIR}"
rosa_print_launch_header "8 GPU DDP online_v1 launch"

TRAIN_ARGS=(
  train_qwen_llama_vs_rosa_v2.py
  --pretokenized_manifest "${PRETOKENIZED_MANIFEST}"
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
  --arch_style "${ARCH_STYLE}"
  --seq_len "${SEQ_LEN}"
  --stride "${STRIDE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --dim "${DIM}"
  --n_layers "${N_LAYERS}"
  --n_heads "${N_HEADS}"
  --n_kv_heads "${N_KV_HEADS}"
  --intermediate_size "${INTERMEDIATE_SIZE}"
  --rosa_recipe "${ROSA_RECIPE}"
  --rosa_online_sam_impl "${ROSA_ONLINE_SAM_IMPL}"
  --distributed_strategy "${DISTRIBUTED_STRATEGY}"
  --distributed_backend "${DISTRIBUTED_BACKEND}"
  --run_models "${RUN_MODELS}"
  --save_every_epochs "${SAVE_EVERY_EPOCHS}"
  --out_dir "${OUT_DIR}"
)

if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  TRAIN_ARGS+=(--activation_checkpointing)
fi

if [[ "${TRAIN_TIMING}" == "1" ]]; then
  TRAIN_ARGS+=(--train_timing)
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
fi

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

(
  cd "${ROSA_ROOT}" && \
  rosa_python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}"
)
