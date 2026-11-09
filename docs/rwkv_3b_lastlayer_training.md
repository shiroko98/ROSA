# 3B + last-layer improved ROSA, 128K training (2026-11-09)

## Configuration / scope

- Native upstream `new_sft` train.py, Lightning, SFT loss and x070 CUDA kernels,
  wrapped by `/mnt/data/Codes/ROSA/train_rwkv_native_rosa.py`.
- Base: `/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`.
  All **2,948,065,280** body parameters frozen BF16 on CUDA; only **180,884,482**
  FP32 improved-ROSA parameters train, on CUDA. SAM addressing remains CPU.
- Inject immediately before final block **31** (32 blocks, zero-based).
  Frozen prefix does not need reverse-mode checkpoints; final trainable path does.
- Context **131,072** inputs; batch 1, accumulation 1. Capacity exploration stopped
  at user's request. Earlier 3-update check: peak allocated 42.796 GiB, reserved
  44.533 GiB. 139,264 and 192,512 OOM. Not an infinite-run stability guarantee.
- Extra first/last-layer quality comparison was NOT completed. Last-layer gains
  must be determined by subsequent eval, not assumed.
- Previous 0.1B training stopped at logged update 916; last complete checkpoint is
  update 724. That output and W&B run remain intact.

## Data / optimization

Source: `/hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B`.
211,633 physical 192,513-token rows. Split physical rows first, seed 42:
2,117 held out, 209,516 train. Each physical row maps to two virtual windows at
128K; every original target/mask is retained exactly once. Input windows overlap
at one boundary token. Second windows pad with native pad ID 65532; padded targets
have loss mask zero. RWKV state resets between windows; this is NOT 192K-context
training. No repacking/114GB corpus copy. Total 419,032 optimizer updates/pass.

LR 3e-5 -> 3e-6; 200 warmup updates; final 10% cosine decay. Dense AdamW plus
SparseAdam value tables; betas .9/.99, eps 1e-8, dense weight decay .01, clip 1.
Fused masked CE chunks 1024, FP32 ROSA/no TF32, native BF16 body/no outer AMP.

## SAM fix

The old Python and C++ implementations eagerly updated endpos along the entire
suffix-link chain for every token. Repeated padding makes this quadratic.
The compiled CPU implementation now uses a rooted link-cut tree with lazy
root-to-last endpos assignment, point queries and correct clone reparenting.
These tree operations have amortized logarithmic cost; no padding/special token
is skipped, and no approximate/truncated search is substituted.

- `--rosa_address_impl compiled_cpu`, algorithm `suffix_link_lct_v1`.
- Pure Python `fast` remains the original reference for regression comparison.
- Equality checks cover addr_ids, raw_match_lens, fired_match_lens, valid_mask,
  special_mask, memory, special filtering, min-match thresholds, clone-heavy
  random sequences, causal prefixes, and a closed-form 131K repeated-token case.
- Full addressing latency is logged as **rosa_address_seconds** for each update,
  separately from forward/backward/eval/checkpoint time.
- Results/protocol:
  `/mnt/data/Codes/ROSA/outputs/native_rwkv_3b_lastlayer_20261109/sam_optimization.json`
  and `sam_scaling.py` in the same directory.
- Binary rebuild checks source/setup timestamps. Restart an already-running
  Python process to use a rebuilt extension; it cannot hot-replace loaded code.

## Eval / persistence

Launcher: `/mnt/data/Codes/ROSA/scripts/run_rwkv_3b_lastlayer_training.sh`.
Output: `/hikscale/models/ROSA/rwkv7_3b_rosa_last_ctx131072_20261109`.

SFT eval at initialization and every 200 updates, 4 held-out windows. Each eval
also runs a synchronous CPU FP32 NIAH monitor: fixed 8K documents, depths .1/.5/.9,
unrelated/similar-archive distractors, QA/exact-prefix tasks, greedy max 24 tokens,
4 CPU threads. Base outputs are signature-keyed/cached. This is an 8K diagnostic,
not 128K NIAH or native-BF16 parity. CPU eval pauses training and can be slow.
External base fingerprints include the native parameter-name order, since .pth
mapping order may differ from nn.Module registration order; bytes/dtypes/shapes
are still checked. Old unordered metadata uses its historical hash behavior.

W&B: self-hosted `http://dian.sea-group.org:9098`, mobius/ROSA-RWKV7, online.
`train.log`, `native_steps.jsonl`, `native_audit.json`, `train_log.txt`,
`needle/step*/{request.json,worker.log,report.json}`, `needle/history.jsonl`.

Save first completed update, then approximately hourly at optimizer boundaries.
`checkpoints/native-*.ckpt`, `native-last.ckpt` symlink; no hourly overwrite.
Compact checkpoints preserve adapter, both optimizer states and Lightning loop
state, and reference original base path + verified hash + parameter-name order.
They require the immutable base to resume; they are NOT standalone full models.
The ROSA-aware conversion utility can reconstruct a full inference checkpoint;
ROSA is not a static matrix merge. Resume is unit-tested; no claim of a completed
3B crash/restart experiment until one is actually performed.

For a graceful future stop, create `OUTPUT/STOP`: save at the next optimizer
boundary and exit. Do not use SIGKILL. Source/config hashes are snapshotted at
production launch; no credentials are included.
