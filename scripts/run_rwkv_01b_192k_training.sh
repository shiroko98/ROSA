#!/usr/bin/env bash
# Measured 192,512-token recipe; frozen native RWKV, learned improved ROSA only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROSA_OUTPUT_DIR:-/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109}"
mkdir -p "$OUT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# This is the user's configured self-hosted W&B service. Credentials are resolved
# from the matching netrc host (including port) by the native Python entrypoint.
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://dian.sea-group.org:9098}"
export WANDB_ENTITY="${WANDB_ENTITY:-mobius}"
export WANDB_MODE=online
export WANDB_RUN_GROUP=rwkv7_01b_rosa_192512_20261109
export WANDB_TAGS=rwkv7,0.1b,improved-rosa,frozen-backbone,192512,gpu-rosa
export WANDB_RESUME=allow
if [[ -f "$OUT/wandb_run_id.txt" ]]; then
  export WANDB_RUN_ID="$(cat "$OUT/wandb_run_id.txt")"
else
  export WANDB_RUN_ID="$("${ROSA_RWKV_PYTHON:-/mnt/data/anaconda3/envs/rosa-rwkv/bin/python}" -c 'import uuid;print(uuid.uuid4().hex[:8])')"
  printf '%s\n' "$WANDB_RUN_ID" > "$OUT/wandb_run_id.txt"
fi
exec bash "$ROOT/scripts/run_rwkv_native_rosa.sh" \
  --data_file /hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B \
  --ctx_len 192512 --micro_bsz 2 --accumulate_grad_batches 1 \
  --rosa_device cuda --rosa_layers 0 --rosa_scale 0.05 \
  --grad_cp 1 --rosa_no_checkpoint_layers 11 \
  --sft_masked_ce_chunk 0 --sft_masked_fused_ce_chunk 2048 \
  --lr_init 1e-4 --lr_final 1e-5 --warmup_steps 200 \
  --lr_wsd_decay_style cosine --lr_wsd_decay_iters 10476 \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --weight_decay 0.01 --grad_clip 1.0 \
  --sft_train_shuffle 1 --sft_train_shuffle_seed 42 \
  --sft_eval_tail_docs 2117 --sft_eval_include_in_train 0 \
  --sft_eval_every_n_steps 200 --sft_eval_steps 4 \
  --sft_one_pass 1 --max_steps 104758 \
  --epoch_save 0 --save_every_n_steps 0 \
  --rosa_save 1 --rosa_checkpoint_initial 1 --rosa_checkpoint_seconds 3600 \
  --enable_progress_bar False --log_every_n_steps 1 \
  --proj_dir "$OUT" "$@"
