#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_rwkv_native_rosa.sh" \
  --load_model /hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth \
  --n_layer 32 --n_embd 2560 --dim_att 2560 --dim_ffn 10240 \
  --d_decay_lora 96 --d_aaa_lora 96 --d_mv_lora 64 --d_gate_lora 320 \
  --rosa_layers 31 --rosa_device cuda --sft_masked_ce_chunk 0 --sft_masked_fused_ce_chunk 1024 "$@"
