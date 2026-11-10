#!/usr/bin/env bash
# Fully replicated data parallelism. No ZeRO or parameter/optimizer offload.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rwkv-lm-v7/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export CC="${CC:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$(dirname "$PYTHON_BIN")/x86_64-conda-linux-gnu-g++}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.codex-build/rwkv_native_cuda_h800}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_WARM_POOL=0 TORCHINDUCTOR_COMPILE_THREADS=1
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://dian.sea-group.org:9098}"
export WANDB_ENTITY="${WANDB_ENTITY:-mobius}" WANDB_MODE=online WANDB_CONSOLE=off
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rwkv7_3b_dp8_192512_dynamic}"
export WANDB_TAGS="${WANDB_TAGS:-rwkv7,2.9b,rosa,dp,ctx192512,cpu-needle}"
OUT="${ROSA_OUTPUT_DIR:-/hikscale/models/ROSA/rwkv7_3b_rosa_dp8_ctx192512_dynamic_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC_PER_NODE:-8}" BATCH="${MICRO_BSZ:-2}"
DATA_PREFIX="${ROSA_DATA_PREFIX:-/mnt/data/Datasets/SFT_RWKV7_13B_rwkv_g1_ctx192512/results/SFT_RWKV7_13B}"
PAD_ID="${ROSA_SFT_PAD_ID:-0}" EVAL_DOCS="${SFT_EVAL_TAIL_DOCS:-2117}"
SCALE_MODE="${ROSA_SCALE_MODE:-logit_gap}" FIXED_SCALE="${ROSA_FIXED_SCALE:-0.05}"
MIN_MATCH="${ROSA_DYNAMIC_MIN_MATCH:-6}" DYNAMIC_MAX="${ROSA_DYNAMIC_MAX:-0.2}"
DOCUMENTS="$("$PYTHON_BIN" -c 'import struct,sys; f=open(sys.argv[1]+".idx","rb"); f.seek(18); print(struct.unpack("<Q",f.read(8))[0])' "$DATA_PREFIX")"
if (( DOCUMENTS <= EVAL_DOCS )); then
  echo "Dataset must have more than $EVAL_DOCS rows; found $DOCUMENTS" >&2
  exit 2
fi
mkdir -p "$OUT"
if [[ -f "$OUT/wandb_run_id.txt" ]]; then
  export WANDB_RUN_ID="$(cat "$OUT/wandb_run_id.txt")"
else
  export WANDB_RUN_ID="$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex[:8])')"
  printf '%s\n' "$WANDB_RUN_ID" > "$OUT/wandb_run_id.txt"
fi
export WANDB_RESUME=allow
source "$ROOT/third_party/rwkv_lm_v7_native/scripts/setup_rwkv_cuda_env.sh"
DP=1 STRATEGY=ddp
if [[ "$NPROC" == 1 ]]; then DP=0 STRATEGY=auto; fi
STEPS=$(( (DOCUMENTS - EVAL_DOCS + NPROC * BATCH - 1) / (NPROC * BATCH) ))
exec "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$ROOT/train_rwkv_native_rosa.py" \
  --load_model /mnt/data/Models/RWKV-7/rwkv7-g1k-2.9b-20260930-ctx25600.pth \
  --data_file "$DATA_PREFIX" --sft_pad_token_id "$PAD_ID" \
  --data_type sft_binidx --vocab_size 65536 --ctx_len 192512 \
  --n_layer 32 --n_embd 2560 --dim_att 2560 --dim_ffn 10240 --head_size 64 \
  --d_decay_lora 96 --d_aaa_lora 96 --d_mv_lora 64 --d_gate_lora 320 \
  --my_testing x070 --kernel @rwkv3 --precision bf16 \
  --accelerator gpu --devices "$NPROC" --num_nodes 1 --strategy "$STRATEGY" --rosa_dp "$DP" \
  --micro_bsz "$BATCH" --accumulate_grad_batches 1 --grad_cp 1 \
  --head_chunk 0 --sft_masked_ce_chunk 0 --sft_masked_fused_ce_chunk 1024 \
  --lr_init 3e-5 --lr_final 3e-6 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
  --warmup_steps 200 --weight_decay 0.01 --grad_clip 1.0 \
  --lr_wsd_decay_style cosine --lr_wsd_decay_iters "$((STEPS / 10))" \
  --train_stage 0 --load_partial 0 --random_seed 42 \
  --rosa 1 --rosa_device cuda --rosa_position post_norm --rosa_layers '' --rosa_output_norm rms \
  --rosa_address_impl compiled_cpu --rosa_scale_mode "$SCALE_MODE" --rosa_scale "$FIXED_SCALE" \
  --rosa_dynamic_min_match "$MIN_MATCH" --rosa_dynamic_max "$DYNAMIC_MAX" --rosa_dynamic_margin 0.5 --rosa_dynamic_chunk 128 \
  --sft_one_pass 1 --epoch_steps "$STEPS" --epoch_count 1 --epoch_save 0 \
  --sft_train_shuffle 1 --sft_train_shuffle_seed 42 \
  --sft_eval_tail_docs "$EVAL_DOCS" --sft_eval_include_in_train 0 --sft_eval_every_n_steps 200 --sft_eval_steps 1 \
  --rosa_eval_initial 0 --rosa_needle_every_eval 0 \
  --rosa_checkpoint_seconds 7200 --rosa_checkpoint_initial 1 --rosa_checkpoint_adapter_only 1 \
  --rosa_save 1 --save_every_n_steps 0 --enable_progress_bar False --logger False \
  --proj_dir "$OUT" --wandb ROSA-RWKV7 "$@"
