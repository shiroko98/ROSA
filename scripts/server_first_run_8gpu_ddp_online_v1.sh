#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =========================
# 1. 基础环境
# =========================
export ENV_NAME="${ENV_NAME:-model}"
export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
export BUILD_ROSA_CPU_EXTENSION="${BUILD_ROSA_CPU_EXTENSION:-1}"

# =========================
# 2. 数据与模型路径
# 请按服务器实际路径修改
# =========================
export TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-/data/models/Qwen3.5-0.8B}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/minipile/train.jsonl}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-/data/minipile/val.jsonl}"
export TEST_DATA_PATH="${TEST_DATA_PATH:-/data/minipile/test.jsonl}"
export MEMMAP_OUT_DIR="${MEMMAP_OUT_DIR:-/data/rosa_runs/minipile_memmap}"
export OUT_DIR="${OUT_DIR:-/data/rosa_runs/online_v1_ddp}"
export TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-8}"
export TOKENIZE_BATCH_DOCS="${TOKENIZE_BATCH_DOCS:-64}"
export PROGRESS_DOCS="${PROGRESS_DOCS:-5000}"

# 如果已经提前构好 manifest，可直接指定这一项，
# 脚本会跳过预分词构建阶段。
export PRETOKENIZED_MANIFEST="${PRETOKENIZED_MANIFEST:-}"

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
# 这里给的是“先跑通”的稳妥默认值
# =========================
export ARCH_STYLE="${ARCH_STYLE:-qwen}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export STRIDE="${STRIDE:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export SEED="${SEED:-42}"

export DIM="${DIM:-768}"
export N_LAYERS="${N_LAYERS:-16}"
export N_HEADS="${N_HEADS:-12}"
export N_KV_HEADS="${N_KV_HEADS:-12}"
export INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-3072}"

# =========================
# 5. ROSA 主线配置
# =========================
export ROSA_RECIPE="${ROSA_RECIPE:-online_v1}"
export ROSA_ONLINE_SAM_IMPL="${ROSA_ONLINE_SAM_IMPL:-compiled_cpu}"
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-ddp}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export RUN_MODELS="${RUN_MODELS:-rosa_fused}"
export SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
export TRAIN_TIMING="${TRAIN_TIMING:-1}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

echo "[ROSA] repo: ${ROOT_DIR}"
echo "[ROSA] step 1/3: install environment"
bash "${ROOT_DIR}/scripts/install_model_env_server.sh"

echo "[ROSA] step 2/3: prepare memmap dataset if needed"
bash "${ROOT_DIR}/scripts/prepare_memmap_dataset.sh"

echo "[ROSA] step 3/3: launch 8-gpu ddp training"
bash "${ROOT_DIR}/scripts/launch_8gpu_ddp_online_v1.sh"
