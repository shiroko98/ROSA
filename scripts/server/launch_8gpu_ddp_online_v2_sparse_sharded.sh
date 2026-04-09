#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_rosa_server.sh"

ROSA_RECIPE="${ROSA_RECIPE:-online_v2_sparse_sharded}"
MASTER_PORT="${MASTER_PORT:-29511}"

EXTRA_ARGS=("$@")

rosa_require_nonempty TOKENIZER_NAME_OR_PATH
rosa_prepare_memmap_manifest_if_needed
ROSA_ONLINE_SAM_IMPL="$(rosa_maybe_build_compiled_cpu_extension "${ROSA_ONLINE_SAM_IMPL}")"

mkdir -p "${OUT_DIR}"
rosa_print_launch_header "8 GPU DDP online_v2_sparse_sharded launch"

TRAIN_ARGS=(
  train_qwen_llama_vs_rosa_v2.py
  --pretokenized_manifest "${PRETOKENIZED_MANIFEST}"
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
  --arch_style "${ARCH_STYLE}"
  --seq_len "${SEQ_LEN}"
  --stride "${STRIDE}"
  --batch_size "${BATCH_SIZE}"
  --train_num_workers "${TRAIN_NUM_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --train_log_every_steps "${TRAIN_LOG_EVERY_STEPS}"
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

if [[ "${ENABLE_BF16}" == "1" ]]; then
  TRAIN_ARGS+=(--bf16)
fi

if [[ "${DATALOADER_PIN_MEMORY}" == "1" ]]; then
  TRAIN_ARGS+=(--dataloader_pin_memory)
fi

if [[ "${DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
  TRAIN_ARGS+=(--dataloader_persistent_workers)
fi

if [[ "${TRAIN_TIMING}" == "1" ]]; then
  TRAIN_ARGS+=(--train_timing)
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
fi

if [[ "${ENABLE_WANDB}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_dir "${WANDB_DIR}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    TRAIN_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    TRAIN_ARGS+=(--wandb_group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    TRAIN_ARGS+=(--wandb_tags "${WANDB_TAGS}")
  fi
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
