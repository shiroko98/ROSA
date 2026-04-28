#!/usr/bin/env bash

# =========================
# 0. 代理配置
# =========================
export SOCKS_PROXY="${SOCKS_PROXY:-socks5://192.168.9.1:7893}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://192.168.9.1:7893}"
export HTTP_PROXY="${HTTP_PROXY:-http://192.168.9.1:7893}"
export ALL_PROXY="${ALL_PROXY:-${SOCKS_PROXY}}"

export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export all_proxy="${all_proxy:-${ALL_PROXY}}"

# =========================
# 1. 基础环境
# =========================
export ENV_NAME="${ENV_NAME:-ROSA}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${ENV_NAME}}"
export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
export BUILD_ROSA_CPU_EXTENSION="${BUILD_ROSA_CPU_EXTENSION:-1}"
export FORCE_ENV_UPDATE="${FORCE_ENV_UPDATE:-0}"

# =========================
# 2. 数据与模型路径
# =========================
export TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-/mnt/lab/Models/qwen/Qwen3.5-9B}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/mnt/data/Datas/minipile/jsonl/train-*.jsonl}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-/mnt/data/Datas/minipile/jsonl/validation-00000-of-00001-a2192e61a091cecb.jsonl}"
export TEST_DATA_PATH="${TEST_DATA_PATH:-/mnt/data/Datas/minipile/jsonl/test-00000-of-00001-010a6231c4b54d31.jsonl}"
export MEMMAP_OUT_DIR="${MEMMAP_OUT_DIR:-/mnt/data/Codes/RWKV/ROSA/memmap_out}"
export OUT_DIR="${OUT_DIR:-/mnt/data/Codes/RWKV/ROSA/rosa_runs}"
export PRETOKENIZED_MANIFEST="${PRETOKENIZED_MANIFEST:-}"

export DATA_FORMAT="${DATA_FORMAT:-jsonl}"
export SPLIT_MODE="${SPLIT_MODE:-paragraph}"
export JSON_TEXT_KEYS="${JSON_TEXT_KEYS:-text,content,body,message}"
export TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-8}"
export TOKENIZE_BATCH_DOCS="${TOKENIZE_BATCH_DOCS:-64}"
export PROGRESS_DOCS="${PROGRESS_DOCS:-5000}"

# 小规模验证时可打开这些限制；正式全量训练时留空即可。
export MAX_TRAIN_DOCS="${MAX_TRAIN_DOCS:-}"
export MAX_VAL_DOCS="${MAX_VAL_DOCS:-}"
export MAX_TEST_DOCS="${MAX_TEST_DOCS:-}"

# =========================
# 3. 分布式训练配置
# =========================
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# =========================
# 4. 模型与训练超参
# =========================
export ARCH_STYLE="${ARCH_STYLE:-qwen}"
export SEQ_LEN="${SEQ_LEN:-2048}"
export STRIDE="${STRIDE:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EPOCHS="${EPOCHS:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export SEED="${SEED:-42}"

export DIM="${DIM:-1536}"
export N_LAYERS="${N_LAYERS:-24}"
export N_HEADS="${N_HEADS:-16}"
export N_KV_HEADS="${N_KV_HEADS:-8}"
export INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-6144}"

# =========================
# 4.1 DataLoader 配置
# =========================
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"

# =========================
# 5. ROSA 主线配置
# =========================
export ROSA_RECIPE="${ROSA_RECIPE:-online_v1}"
export ROSA_ONLINE_SAM_IMPL="${ROSA_ONLINE_SAM_IMPL:-compiled_cpu}"
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-ddp}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export RUN_MODELS="${RUN_MODELS:-rosa_fused}"
export SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
export TRAIN_TIMING="${TRAIN_TIMING:-0}"
export TRAIN_LOG_EVERY_STEPS="${TRAIN_LOG_EVERY_STEPS:-10}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
export ENABLE_BF16="${ENABLE_BF16:-1}"

# =========================
# 6. wandb 监控
# =========================
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-ROSA}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_GROUP="${WANDB_GROUP:-}"
export WANDB_TAGS="${WANDB_TAGS:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_SAVE_DIR:-${OUT_DIR%/}/wandb}}"
