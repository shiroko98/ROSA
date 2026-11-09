#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
# CUDA 12.6 supports GCC <=13; use this environment's GCC 12 rather than host GCC 15.
export CC="${CC:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-g++}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.codex-build/rwkv_native_cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_WARM_POOL=0
export TORCHINDUCTOR_COMPILE_THREADS=1
# Respect an explicit WANDB_BASE_URL or the user's W&B settings file.
# Do not override a configured self-hosted W&B server with the public service.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_CONSOLE=off
source "$ROOT/third_party/rwkv_lm_v7_native/scripts/setup_rwkv_cuda_env.sh"
COMMAND=("$PYTHON_BIN" -u)
if [[ "${ROSA_COVERAGE:-0}" == "1" ]]; then
  export COVERAGE_FILE="$ROOT/.coverage"
  COMMAND+=(-m coverage run --parallel-mode --rcfile="$ROOT/.coveragerc")
fi
exec "${COMMAND[@]}" "$ROOT/train_rwkv_native_rosa.py" \
  --load_model /hikscale/models/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --data_type sft_binidx --vocab_size 65536 \
  --n_layer 12 --n_embd 768 --dim_att 768 --dim_ffn 3072 --head_size 64 \
  --d_decay_lora 64 --d_aaa_lora 64 --d_mv_lora 32 --d_gate_lora 128 \
  --my_testing x070 --kernel @rwkv3 --precision bf16 \
  --accelerator gpu --devices 1 --num_nodes 1 --strategy auto \
  --micro_bsz 1 --accumulate_grad_batches 1 --grad_cp 1 \
  --head_chunk 0 --sft_masked_ce_chunk 32 --sft_masked_fused_ce_chunk 0 \
  --lr_init 1e-4 --lr_final 1e-4 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
  --warmup_steps 0 --weight_decay 0.01 --grad_clip 1.0 \
  --train_stage 0 --load_partial 0 --random_seed 42 \
  --rosa 1 --rosa_layers 0 --rosa_scale 0.05 \
  --epoch_steps 1000 --epoch_count 1 --epoch_save 1 \
  --sft_train_shuffle 1 --sft_train_shuffle_seed 42 \
  --wandb ROSA-RWKV7 "$@"
