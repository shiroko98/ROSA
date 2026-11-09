# RWKV7 3B frozen backbone + CPU improved ROSA

## Implemented placement and update contract (2026-11-09)

- Source base checkpoint: `/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`.
- `--variants improved --freeze-backbone --rosa-device cpu` selects `rosa_only` training.
- ALL 2,948,065,280 backbone parameters are frozen, including embedding, output norms and head.
- All 180,884,482 trainable improved-ROSA parameters live on CPU: per-layer values, projections, norms, gate bias and length scaling. SAM address computation is also CPU.
- Backbone execution remains CUDA. Hidden states cross to CPU for ROSA; masked residuals cross back to CUDA. These transfers are differentiable, with no `.detach()` on the learned path.
- The frozen backbone must NOT run under `no_grad` during training: input gradients through its layers are required to train the upstream ROSA residual.
- SparseAdam value updates and dense AdamW module updates and their states are CPU-only. GPU trainable parameter count is zero.
- `freeze_audit.json` records exact full backbone hashes before training, after the first optimizer update and on completion. It rejects backbone gradients or parameter mutations and records CPU optimizer placement and actual changes to learned values.
- Delta checkpoints contain only trainable ROSA parameters and optimizer/RNG/cursor, not a second full 3B model. The base checkpoint remains required.

## Actual checks performed

Real 2.9B model + supplied SFT dataset, two update steps:

`/mnt/data/Codes/ROSA/outputs/rwkv7_3b_cpu_rosa_frozen_preflight_20261109/`

- Inputs: 256 tokens per step, targets: 128 supervised tokens per step.
- CUDA trainable params: 0. CPU trainable params: 180,884,482.
- Full backbone hash unchanged after both updates. No backbone gradients.
- ROSA value-table hash changed after the first update. Optimizer tensor states all CPU.
- Peak allocated CUDA memory ~11.47 GiB; steady second step ~9.17 s on this reference path.
- Initial 64-token full/streaming FP32 verification: 0 next-token top-1 changes, max logit discrepancy ~9.16e-5. Not an 84K parity test.
- Full regression suite: 180 tests passed; Python coverage 95.60%. Includes actual CUDA-backbone/CPU-ROSA gradient flow and checkpoint restore tests.

## Dataset handling and the 84K distinction

Dataset source: `/hikscale/H800backup/Datasets/SFT_RWKV7_13B_2026_07_30/`.

- Merged token binidx: 442,795 physical packed rows, ALL 86,017 tokens long.
- Next-token shift gives 86,016 input tokens (84 * 1024). Physical rows contain EOT-separated examples and masked EOT padding, not necessarily one 84K conversation.
- Stored total: 38,087,897,515 tokens including packing/padding.
- Token `.bin` ~70.94 GiB (uint16), loss-mask `.bin` ~35.47 GiB (uint8).
- Read-only NumPy memory maps use host virtual memory / OS page cache. The full corpus is NOT copied into GPU VRAM or eagerly loaded into a Python RAM list.
- The new reader selects seeded random physical rows, unpacks EOT-delimited examples, removes exact token duplicates, and then splits examples 80/10/10. Related/near-duplicate conversations are NOT certified absent.
- Fixed CPU history+target windows are constructed from supplied masks; only the current batch enters CUDA. This is NOT full 84K-context training and does NOT carry RWKV state between windows.

### Current reference backend cannot do full 84K BPTT on this GPU

The exact recurrence saves `T+1` distinct float32 head-state matrices for autograd. Instrumented actual saved-tensor storages at T=16,32,64,128: 17,33,65,129 matrices.

For 40 heads * 64 * 64 float32 values per state, one layer at T=86,016 requires ~52.5006 GiB in these saved state matrices alone, exceeding the ~48 GiB GPU even before ~10.98 GiB frozen FP32 weights and other activations. Block activation checkpointing still recomputes the same differentiable op, so it does not resolve its per-block memory requirement.

Additional actual-checkpoint probes used real masked token spans (forward + backward only, no optimizer update):

| Input length | CUDA peak allocated | Forward/backward time |
|---|---:|---:|
| 256 | 11.47 GiB | 9.63 s |
| 1024 | 12.49 GiB | 35.95 s |

Both confirmed absent backbone gradients and CPU placement of all observed ROSA gradients. Raw probe: `/mnt/data/Codes/ROSA/outputs/rwkv7_84k_feasibility_20261109/bounded_real_backprop.json`.

This is an extrapolation grounded in exact operator instrumentation, **not a completed 84K full-model training/OOM test**. Report:

`/mnt/data/Codes/ROSA/outputs/rwkv7_84k_feasibility_20261109/report.json`

Options requiring separate implementation/validation: optimized chunkwise WKV forward/backward with matching precision semantics; activation offload or finer recurrence checkpointing; state-carrying truncated BPTT (not full 84K gradients). The checkpoint filename only indicates a 25,600-token context configuration, not validated 84K quality.

## Training configuration and tracking status

Ready config:
`/mnt/data/Codes/ROSA/configs/rwkv_rosa/rwkv7_2_9b_cpu_rosa_sft.json`

- FP32; improved only; frozen backbone; CPU ROSA.
- Bounded first experiment: seed 42, 2,048 random packed source rows, max 1,000 optimizer steps, 128 target + up to 128 history tokens, batch 1 / accumulation 4, LR 1e-4, injection layer 0, residual scale .05.
- Fixed 32 seed-random validation windows and 32 test windows; validation/checkpoint every 100 steps. No full-corpus or full-context quality claims.
- As of the length-feasibility investigation, only short preflights have been executed; no 84K full-context long training has been launched.

W&B `Api`/GraphQL credential checks return HTTP 403. SDK 0.19.11 offline logging was tested. A CLI sync probe printed a success-like message with an empty entity, but remote visibility has NOT been independently verified; do not describe it as confirmed upload. Private source text/token arrays and model checkpoints are not attached to W&B. Offline metrics can be explicitly synced after connectivity/permissions are verified.

Launch and sync scripts:

```bash
/mnt/data/Codes/ROSA/scripts/run_rwkv_cpu_rosa_training.sh
/mnt/data/Codes/ROSA/scripts/sync_rwkv_cpu_rosa_wandb.sh
```

Do not run the second command on an incomplete offline run. After successful SDK sync, confirm remote run visibility under the intended authenticated entity.

## Actual maximum-length search

A separate isolated-process search was launched on 2026-11-09 after the user requested an empirical limit:

```bash
cd /mnt/data/Codes/ROSA
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/mnt/data/anaconda3/envs/rosa-rwkv/bin/python -u -m rwkv_rosa.length_probe \
  --out-dir /mnt/data/Codes/ROSA/outputs/rwkv7_cpu_rosa_length_search_20261109
```

- FP32, batch 1, frozen 3B CUDA backbone, all learned ROSA/optimizer CPU, layer-0 injection, per-block activation checkpointing.
- Real contiguous suffix of physical packed row 0 and its original supervision mask. Each case starts from identical base initialization and executes a forward, full backward, and actual ROSA optimizer update. It checks frozen full parameter hashes and confirms learned table changes.
- Starts at 2,048 tokens and doubles toward 86,016; after an actual CUDA OOM, bisects to a 1,024-token resolution. This finds an empirical bracket, not an exact token-by-token maximum.
- Each length uses a fresh subprocess, keeping OOM recovery independent and avoiding stale allocator fragmentation between lengths.
- A one-hour **per-case** time limit is recorded separately from CUDA OOM. A timed-out/failed process does not establish an OOM bound. A single successful step is not a guarantee of long-run stability or safe lengths for all corpus examples.
- Current state and completed-case results: `/mnt/data/Codes/ROSA/outputs/rwkv7_cpu_rosa_length_search_20261109/search.json`.
- Per-case stage JSON/log files expose initialization, forward, backward and optimizer progress. Parent log: `/mnt/data/Codes/ROSA/outputs/launch_logs/rwkv7_cpu_rosa_length_search_20261109.log`.
- W&B remains offline; numeric length/memory/timing metrics are preserved for later verified upload.

The two earlier short probes are not reused as successes in this stricter search, because they did not include an optimizer step at the longer length.

## Superseded as the preferred training path

The preceding reference-runner length/memory conclusions apply **only to the
Python reference backend**, not to the native SFT CUDA training backend. The
reference 8K search and direct 50K test were interrupted when the user required
actual upstream native training. Neither interrupted test establishes an OOM
bound. New native SFT integration and 0.1B placement comparisons are documented
in `/mnt/data/Codes/ROSA/docs/rwkv_native_sft.md`.
