#!/usr/bin/env bash
# Longest safety-qualified native context measured on this host; frozen 2.9B.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}"
CONTEXT="${ROSA_TRAIN_CONTEXT:-131072}"
POSITION="${ROSA_POSITION:-block}"
LAYERS=31
if [[ "$POSITION" == post_norm ]]; then LAYERS=""; fi
OUT="${ROSA_OUTPUT_DIR:-/hikscale/models/ROSA/rwkv7_3b_rosa_last_ctx${CONTEXT}_20261109}"
mkdir -p "$OUT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://dian.sea-group.org:9098}" WANDB_ENTITY="${WANDB_ENTITY:-mobius}"
export WANDB_MODE=online WANDB_RESUME=allow
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rwkv7_3b_position_20261109}"
export WANDB_TAGS="rwkv7,2.9b,improved-rosa,$POSITION,frozen-backbone,cpu-needle-eval"
if [[ -f "$OUT/wandb_run_id.txt" ]]; then
  export WANDB_RUN_ID="$(cat "$OUT/wandb_run_id.txt")"
else
  export WANDB_RUN_ID="$("$PYTHON_BIN" -c 'import uuid;print(uuid.uuid4().hex[:8])')"
  printf '%s\n' "$WANDB_RUN_ID" > "$OUT/wandb_run_id.txt"
fi
# Exactly one pass over ALL virtual windows; native physical train/eval split
# remains seed-42, 2117 held-out rows. No truncation/repacking of source files.
WINDOWS_PER_ROW=$(( (192512 + CONTEXT - 1) / CONTEXT ))
TOTAL_STEPS=$(( 209516 * WINDOWS_PER_ROW ))
DECAY_STEPS=$(( TOTAL_STEPS / 10 ))
exec bash "$ROOT/scripts/run_rwkv_3b_native_rosa.sh" \
  --data_file /hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B \
  --ctx_len "$CONTEXT" --rosa_windowed_data 1 --micro_bsz 1 --accumulate_grad_batches 1 \
  --rosa_position "$POSITION" --rosa_layers "$LAYERS" --rosa_device cuda --rosa_address_impl compiled_cpu --rosa_scale 0.05 --grad_cp 1 \
  --lr_init 3e-5 --lr_final 3e-6 --warmup_steps 200 \
  --lr_wsd_decay_style cosine --lr_wsd_decay_iters "$DECAY_STEPS" \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --weight_decay 0.01 --grad_clip 1.0 \
  --epoch_steps "$TOTAL_STEPS" --epoch_count 1 --max_steps "$TOTAL_STEPS" --sft_one_pass 0 \
  --sft_train_shuffle 1 --sft_train_shuffle_seed 42 \
  --sft_eval_tail_docs 2117 --sft_eval_include_in_train 0 --sft_eval_every_n_steps 200 --sft_eval_steps 4 \
  --rosa_eval_initial 1 --rosa_needle_every_eval 1 --rosa_needle_length 8192 \
  --rosa_needle_depths 0.1,0.5,0.9 --rosa_needle_max_new_tokens 24 --rosa_needle_cpu_threads 4 \
  --rosa_checkpoint_seconds 3600 --rosa_checkpoint_initial 1 --rosa_checkpoint_adapter_only 1 \
  --epoch_save 0 --save_every_n_steps 0 --rosa_save 1 --enable_progress_bar False \
  --proj_dir "$OUT" "$@"
