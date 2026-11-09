# Native RWKV `new_sft` training + improved ROSA

## Corrected training foundation

The preferred training entry is now **`train_rwkv_native_rosa.py`**, which executes
vendored upstream `train.py`, not the previous custom reference training loop.

Upstream: TimeMobius/RWKV-LM-V7, branch `new_sft`, pinned commit
`1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1`.

Reused without replacing the training pipeline:

- Native `train.py` argument processing, strict pretrained load, Lightning Trainer and `trainer.fit`.
- `src.model.RWKV` and fused x070 CUDA forward/backward, not the Python recurrence.
- `src.dataset.MyDataset` / binidx token and supervision-mask handling.
- Native SFT masked CE, supervised-token-weighted gradient accumulation, LR scheduler,
  held-out evaluation, logging/W&B and checkpoint callbacks.

Complete upstream modifications and original/local hashes are retained in
`/mnt/data/Codes/ROSA/third_party/rwkv_lm_v7_native/UPSTREAM_PATCH.diff` and
`UPSTREAM_FILES.json`. License and attribution retained. Changes are limited to
extension installation/injection, sparse gradient norms, single-device checkpoint
integration, optional diagnostic checkpoint disabling and Ada-compatible scalar
atomics for kernels whose vector atomics require SM90+.

## Current base model and freezing contract

`/hikscale/models/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.pth`

The filename labels it 0.1B; actual strict-loaded shape is 12 layers, hidden 768,
12 heads × 64, vocab 65,536; **191,084,544 backbone parameters**. All stay frozen,
including embedding, output norm and head. Improved ROSA has **51,512,834 trainable
parameters** at injection layer 0. Only ROSA parameters update.

`--rosa_device cpu` pins learned values/projections/norms/gates and gradient/moment
states to CPU; `--rosa_device cuda` puts these on GPU. SAM address preprocessing
is CPU in BOTH cases; this is a learned-fusion placement comparison, not GPU SAM.
The historical module identifier `model.rosa_cpu` is shared by both placements;
the explicit device argument and audit, not this identifier, determine placement.

Differentiable transfers preserve loss -> frozen native RWKV input gradients ->
ROSA gradients. Injection is outside block checkpointing. The pinned adapter
ignores recursive backbone dtype/device conversions so Lightning cannot silently
move/cast it away from the configured device. The native Trainer uses a single
combined optimizer containing AdamW + SparseAdam groups on that device, retaining
native LR scheduling and supervised-token weighting. Scalar Adam step counters
may remain CPU in the CUDA case, but moment tensors remain CUDA.

Precision: BF16 frozen backbone/native kernels, FP32 learned ROSA in both arms,
TF32 disabled for FP32 ROSA matmuls. Outer AMP is disabled for this single-device
extension because it promotes LayerNorm outputs to FP32, incompatible with the
upstream BF16-only custom operators. Consequently the internal Trainer precision
field is 32 while the actual frozen weights/operator inputs are **BF16**, not
FP32. No bitwise or streamed-generation equivalence is claimed.

Currently **single GPU only**; CPU/GPU mixed placement is explicitly rejected for
DeepSpeed/DDP multi-GPU, rather than silently using an unsupported mixed strategy.

## Completed actual placement benchmark — 2026-11-09

Hardware: RTX 4090 D (~48 GiB), EPYC 9654 CPU; CPU branch used 8 OMP/MKL threads.

Data: real token/mask suffixes from
`/hikscale/H800backup/Datasets/SFT_RWKV7_13B_2026_07_30/`. Each derived diagnostic
binidx has 16 rows of exactly `ctx_len+1`, preserving original token IDs/masks.
The upstream loader intentionally rejects oversized rows; therefore the 84K
source is not silently passed to a shorter-context loader. These bounded suffixes
are for controlled profiling, **not full-corpus or quality evaluation**.

Each of 6 runs completed 10 real optimizer updates. Exclude steps 1–2 as warmup;
compare the median over steps 3–10. No coverage/profiler instrumentation in timed
training workers; kernel compilation, model loading, hash audits and checkpoint
serialization are excluded from these iteration timings. CUDA/CPU order is
alternated across contexts. All paired input/mask batch hashes matched exactly.

| Input context | ROSA CUDA median step | ROSA CPU median step | CUDA speedup | CUDA ROSA peak allocated | CPU ROSA peak allocated |
|---|---:|---:|---:|---:|---:|
| 1,024 | 0.06756 s | 0.08956 s | 1.33× | 1.069 GiB | 0.468 GiB |
| 8,192 | 0.37872 s | 0.57337 s | 1.51× | 1.764 GiB | 1.025 GiB |
| 51,200 | 2.30732 s | 4.37595 s | 1.90× | 5.973 GiB | 4.387 GiB |

All six runs confirmed:

- Frozen full backbone parameter hash before/after identical:
  `415018aacc1917d2559fb5800b7267660ad95d36cd8df8e32041fe295dce9df3`.
- Backbone has no gradients. ROSA value table changed after actual optimizer update.
- CPU case: 51,512,834 CPU trainables, zero CUDA trainables; CUDA case vice versa.
- Optimizer moment tensor devices match selected placement.

Initial CPU/CUDA loss differences were recorded, not hidden: approximately
0.00109 (1K), 0.00144 (8K), 0.00304 (50K). Both use native BF16 backbone and FP32
ROSA, but device arithmetic is not bitwise identical. This benchmark does not
establish equal optimization trajectories or compare finetuning quality.
50K passing on the 8K-named checkpoint establishes execution feasibility only,
not that the pretrained model has validated 50K-context quality.

**Recommendation from these measured settings:** use GPU learned ROSA when its
additional ~0.60–1.59 GiB peak allocation fits. CPU ROSA saves GPU memory but is
slower here. This is not a universal CPU/GPU result across hardware/thread counts.

Validation: 189 tests passed in the CUDA-capable environment. A separate
forced-CPU-only run executed 189 tests with 3 GPU tests skipped and 95.12% Python
line coverage (7,150/7,517), without relying on training-run coverage to pass the
95% threshold. Full-epoch native CPU checkpoint restore was also exercised from
step 10 through two real updates to step 12, preserving the frozen backbone hash.
A Lightning 1.9 completed-epoch counter normalization was added for this restore;
partial-epoch cursor equivalence has NOT been validated. Timing source snapshots
(45 files matching recorded hashes) remain under the raw report's `source_snapshot/`.

Raw report:
`/mnt/data/Codes/ROSA/outputs/native_rwkv_01b_rosa_placement_20261109/comparison.json`

Per-run `native_audit.json` and `native_steps.jsonl` provide full checks/timings.
The two 8K cases also saved both native `.pth` model states and full Lightning
`native-last.ckpt` (including optimizer/loop state). Other diagnostic lengths
intentionally skipped checkpoint output.

## Reproduction

Independent environment: `rosa-rwkv`, Python 3.11, Torch 2.7.1+cu126,
Lightning 1.9.5, DeepSpeed 0.16.9, TorchMetrics 1.6.3, NumPy 1.26.4,
Setuptools 80.9.0, W&B 0.19.11; CUDA nvcc 12.6.85 + conda GCC 12.4.
The launcher selects GCC 12 rather than incompatible host GCC 15 and reuses
upstream CUDA include/library discovery.

```bash
cd /mnt/data/Codes/ROSA
# Native training on full original 84K binidx; explicit length, split and limits.
bash scripts/run_rwkv_native_rosa.sh \
  --data_file /hikscale/H800backup/Datasets/SFT_RWKV7_13B_2026_07_30/results/SFT_RWKV7_13B \
  --ctx_len 86016 --rosa_device cuda \
  --epoch_steps 1000 --max_steps 1000 \
  --sft_eval_tail_ratio 0.01 --sft_eval_every_n_steps 100 --sft_eval_steps 8 \
  --proj_dir /mnt/data/Codes/ROSA/outputs/native_rwkv_01b_full_data_NEW
```

The full 84K command above is an example **not executed as part of the completed
1K/8K/50K comparison**. Use a fresh output directory and do not confuse it with a
completed full-corpus experiment. Native checkpoints include full frozen weights,
not only ROSA deltas. Resume with original `--load_model` and
`--rosa_resume /absolute/path/native-last.ckpt`, preserving dataset/context/LR
semantics; extend epoch_count/max_steps if the saved epoch has already ended.
Full-epoch restore is verified; do not assume mid-epoch exact-cursor equivalence.

Repeat the paired timing benchmark against already prepared diagnostic data:

```bash
# On another checkout, first prepare fresh bounded profiling data:
for ctx in 1024 8192 51200; do
  python -m rwkv_rosa.native.prepare_probe_data \
    --source /hikscale/H800backup/Datasets/SFT_RWKV7_13B_2026_07_30/ \
    --output /mnt/data/Codes/ROSA/outputs/native_probe_NEW/ctx${ctx} \
    --context "$ctx" --rows 16
done
python -m rwkv_rosa.native.benchmark \
  --data-root /mnt/data/Codes/ROSA/outputs/native_probe_NEW \
  --out-dir /mnt/data/Codes/ROSA/outputs/native_placement_NEW \
  --contexts 1024,8192,51200 --steps 10 --warmup 2
```

The benchmark defaults to the user's 0.1B checkpoint and saves 8K checkpoints.
`prepare_probe_data` can create a fresh bounded profiling binidx for other
lengths, without changing the original dataset or the native loader.

## W&B status

Each run used upstream W&B logging and retains an offline run under its output
`wandb/` directory. Only configuration/metrics are logged; no raw source text,
token arrays or model artifacts are uploaded. Online API verification has returned
HTTP 403; do not claim offline logs are confirmed uploads. See the benchmark's
`wandb_status.json`. After account/connectivity is repaired, explicitly sync the
completed offline runs and verify they appear under the intended entity.

## 192,512-token production run — 2026-11-09

Requested dataset: `/hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B`.
Index inspection found **211,633 physical rows**, each 192,513 stored tokens,
therefore **192,512 shifted input tokens**. Original token IDs and loss masks
are read directly by the native memmap loader: no suffix extraction, synthetic
repetition, or sequence shortening in this run.

Measured recipe: `/mnt/data/Codes/ROSA/scripts/run_rwkv_01b_192k_training.sh`.
Backbone remains the user's 0.1B-named checkpoint (191,084,544 frozen parameters),
BF16 CUDA; only 51,512,834 improved ROSA parameters update, FP32 CUDA, layer 0,
residual scale .05. CPU SAM addressing is unchanged. Outer AMP and ROSA TF32
remain disabled. Native full-sequence forward/backward is used, not truncated BPTT.

### Short tuning probes (not convergence or long-term stability evidence)

All successful timing cases start from the same base/ROSA initialization and
use the same first shuffled training batches. Exclude first update as warmup.
Different CE implementations have small numerical differences; no equal-trajectory
or bitwise-equivalence claim is made.

| Configuration, batch 2 | Median update seconds | Peak allocated GiB |
|---|---:|---:|
| Standard checkpointed masked CE, chunk 256 | 16.950 | 38.631 |
| Standard checkpointed masked CE, chunk 2048 | 15.302 | 38.632 |
| Native fused masked CE, chunk 256 | 14.514 | 38.632 |
| Native fused masked CE, chunk 2048 | 14.450 | 38.632 |
| Fused chunk 2048; block 11 without recomputation (**selected**) | 14.359 | 38.081 |

Selected setting completed 4 optimizer updates, unchanged frozen-backbone hash,
updated ROSA values; peak reserved memory 39.914 GiB. Approximately **1.18×**
throughput versus the batch-2 chunk-256 standard-CE baseline in these short probes.
Batch 3 and batch 2 with both final blocks 10–11 uncheckpointed caused actual CUDA
OOM (wrapped in TorchScript RuntimeError). The selected setting leaves allocator,
CUDA-context, and data-dependent sparse-gradient headroom instead of filling
unused VRAM with nonproductive allocations. It is not a globally optimal recipe.

Raw tuning report:
`/mnt/data/Codes/ROSA/outputs/native_rwkv_01b_192k_tuning_20261109/tuning_report.json`.

### Production hyperparameters and outputs

- Micro batch 2, gradient accumulation 1; 385,024 input positions per update.
- Hold out 2,117 complete physical rows (~1%); train on 209,516 rows for **one
  pass / 104,758 optimizer updates**. Fixed seed 42 and native train shuffling.
- Peak LR 1e-4; 200-update warmup from 1% of peak; final 10,476 updates cosine
  decay to 1e-5. AdamW dense + SparseAdam values; betas (.9,.99), eps 1e-8,
  dense decay .01, sparse decay zero, gradient norm clip 1.0. These are initial
  conservative settings, not a quality-tuned optimum.
- Held-out evaluation every 200 updates, 4 micro batches / 8 rows per event.
- Activation checkpoint blocks 0–10, block 11 direct. Native fused masked head
  CE chunk 2048. Single GPU, 8 OMP/MKL threads.
- Output root:
  `/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/`.
- `train.log`: full stdout/stderr; `train_log.txt`: upstream epoch log;
  `native_steps.jsonl`: every-update loss, LR, gradient norm, times, memory,
  input throughput and addressing statistics. Evaluation metrics use native W&B.
- `source_snapshot/`, `source_hashes.json`, `run_manifest.json`, `train.pid` and
  `tuning_report.json` preserve implementation/launch provenance. No API key is
  written to these files or placed on a command line.

### Hourly full-state checkpoints

`--rosa_checkpoint_seconds 3600 --rosa_checkpoint_initial 1` saves one full
checkpoint after the first actual update, then at the first completed optimizer
update after each elapsed hour. Saving is synchronous and therefore includes a
small I/O pause; not an exact wall-clock alarm in the middle of a CUDA operation.
Full model, both child optimizer states and Lightning loop state are retained.
Files under `checkpoints/native-hourly-step*.ckpt` are written to a temporary name
and atomically renamed; `native-last.ckpt` is an atomically replaced relative
symlink to the latest complete file. Old hourly files are **not deleted**.
Final fit also saves a full final checkpoint. A full checkpoint is about
1,000,470,227 bytes (~955 MiB) in the measured environment.

The timer was accelerated to one second in a real 192,512-token batch-2 probe,
producing initial, periodic and final full checkpoints. A step-2 periodic
checkpoint was restored to perform actual update 3, with matching next input/mask
hash and loss versus the uninterrupted diagnostic run and frozen backbone
unchanged. This is one verified mid-epoch restore, not proof of universal exact
cursor/RNG equivalence under all Lightning/data-loader conditions.

Resume (same data/context/batch/schedule; original base is still required):

```bash
ROSA_OUTPUT_DIR=/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109 \
  bash /mnt/data/Codes/ROSA/scripts/run_rwkv_01b_192k_training.sh \
  --rosa_resume /hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/native-last.ckpt
```

Do not start this while its `train.pid` is still running.

### W&B correction and verification

The user's configured service is `http://dian.sea-group.org:9098`, account
`mobius`, project `ROSA-RWKV7`. The old launcher incorrectly overrode the configured
server with public `api.wandb.ai`, explaining prior 403 checks. The native launcher
now respects explicit configuration; the entrypoint resolves matching netrc
credentials **including the port** for this older SDK. Credentials are used only
for that selected server and never logged. An online preflight uploaded a numeric
metric and read it back through the remote API successfully. Production uses
online mode, with persistent run ID in `wandb_run_id.txt`; after launch its actual
remote training metrics must also be verified separately. No corpus text/token
arrays or model artifacts are uploaded. Earlier placement runs remain historical
offline runs unless explicitly synced; the preflight does not retroactively make
them uploaded.

Validation after these changes: **192 tests passed (3 CUDA-only tests skipped in
forced-CPU validation), 95.13% Python line coverage (7,225/7,595)**; real CUDA tuning,
full-state checkpoint save and step-2-to-3 restore separately passed.

Production online run **93b9f1f2** was separately verified after launch: remote API
history contained actual native/train loss, learning rate, timing, memory and
gradient norm for updates 1–4. `wandb_status.json` retains that verification.
The first full checkpoint was atomically saved on the requested NFS destination
(1,000,470,227 bytes); writing/closing it took **132.1 seconds**, unlike the ~1.3
seconds measured on local SSD. GPU compute updates then resumed at ~14.4–14.6
seconds. Hourly saves will introduce shared-storage I/O pauses; training-step
timing does not hide this cost, which appears in native upstream end-to-end perf
metrics. At current short-run throughput, the planned ~40.3 billion input-token
pass is a roughly 18-day workload, not a several-hour experiment. This is an
estimate, not a completion-date promise.

CPU checkpoint export and base-vs-ROSA diagnostic protocol: see
`/mnt/data/Codes/ROSA/docs/rwkv_cpu_checkpoint_comparison.md`.

3B / last-block work and every-eval CPU passkey monitoring are documented in
`/mnt/data/Codes/ROSA/docs/rwkv_3b_lastlayer_training.md`. This supersedes the
0.1B production launch; its source/checkpoints and results remain preserved.

SAM 长序列退化与精确寻址加速：见 `/mnt/data/Codes/ROSA/docs/rwkv_sam_performance.md`。

显式 post_norm 输出分支及3B试训：见 `/mnt/data/Codes/ROSA/docs/rwkv_post_norm_training.md`。
