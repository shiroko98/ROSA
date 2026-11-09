# Independent native GPU NIAH evaluation

This job never creates a Trainer/optimizer, never resumes training, and evaluates
one immutable ROSA snapshot against its exact external frozen base. The loader
validates both tensor-state SHA256 fingerprints and explicit `post_norm` metadata.
There is no static merge, hard-coded copy bypass, or reference CUDA recurrence.

## Runtime

- Native vendored TimeMobius `new_sft` RWKV CUDA kernels (`@rwkv3`), BF16 body/head.
- Native ROSA formula and trained FP32 values/projections on the same GPU.
- Compiled **CPU** suffix automaton performs integer token addressing. This is
  intentional and does not mean the neural model/ROSA projections run on CPU.
- Inference mode, frozen parameters, TF32 off, no optimizer/gradients.
- Tail-pad to 16 tokens only for the native kernel; select real-token positions.
- Context lengths in the matrix refer to document tokens. Query, generated
  answer, and kernel padding are additional. The runtime assertion is extended
  to `max(document lengths) + 1024`; model weights/architecture do not change.
- Post-norm does not alter body state. Identical token histories therefore share
  one frozen forward between base/trained. Divergent generations get separate
  forwards. Native training kernels do not expose a reusable inference state;
  each new token replays the full history. This is slower than an optimized
  streaming decoder, but avoids silently changing the evaluated recurrence.

## Protocol

Default: 8192/32768/65536/131072/192512 document tokens; five insertion depths
(10/30/50/70/90%); unrelated background and confusable archive entries; ordinary
QA and literal-prefix completion. One reproducible passkey per depth/difficulty,
shared across lengths. This gives **100 matched case/mode pairs = 200 generated
answers** at one seed, not 100 independent random passkeys. Expand `--seeds` for
statistical replication. Greedy decode, 24 new tokens maximum, both arms matched.

First, finish the complete teacher-forced probability/addressing sweep; second,
generate actual answers. Generation always starts from a **fresh prompt-only**
forward and then only its own predicted tokens; teacher-forced answer IDs and
logits are never fed into free generation. Generated-answer success uses the
existing six-digit answer parser, and strict digits-only accuracy is separate.

A suffix hit is **not** a neural-generation success. `probes.jsonl` contains
teacher-forced candidate IDs, answer NLL and first-token ranks; `results.jsonl`
adds actual greedy output and scoring. BF16 target-logit differences are measured
after the real BF16 residual addition/head, including quantization; they are not
an FP32 hypothetical residual. `post_norm_residual_zero_fraction` exposes values
rounded away by that addition. Short plain-RWKV/ChatML/English-continuation sanity
outputs are written to `sanity.json` so pathological base decoding is not hidden.

## Launch (absolute paths recommended)

```bash
WANDB_BASE_URL=http://dian.sea-group.org:9098 \
  bash /mnt/data/Codes/ROSA/scripts/run_rwkv_gpu_needle.sh \
  --train-dir /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109 \
  --checkpoint /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/checkpoints/rosa-adapter-step000000200.pth \
  --output /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/gpu_needle_step200_20261109 \
  --wandb
```

Choose a fresh output directory for each invocation; concurrent/mixed reruns are
rejected. The launcher normalizes file arguments before entering the upstream
working directory. W&B has a separate evaluation run grouped by the training run;
source step remains explicitly 200 rather than an invented training update.

Artifacts: `request.json`, `status.json`, `sanity.json`, `probes.jsonl`,
`results.jsonl`, atomically refreshed `report.json`, `worker.log`,
`wandb_run_id.txt`. Reports are partial unless status is `complete`. The per-case
JSONL is fsynced after every finished pair. A terminated job preserves completed
results; it does not restart itself or the training job.

## Training stop, 2026-11-09

The prior auto-switch controller and CPU watcher were cancelled **before** stopping
the trainer and synchronous CPU evaluator. GPU memory was verified released.
`run_manifest.json` and controller status record the explicit cancellation; STOP
markers remain. Step 200 adapter weights are preserved, but **no optimizer/loop
checkpoint exists at step 200**. Step 1 remains the latest exact-resume snapshot.
The partial CPU evaluation was cancelled and is not conflated with GPU results.

## Tests

`tests/test_rwkv_gpu_needle.py` covers tail-only padding, address slicing,
shared/divergent free-generation histories, no teacher-answer seeding, stop/length
scoring, summary separation, and rejection of wrong-position snapshots before
native CUDA loading. Real 3B GPU validation additionally ran 1K/8K matched cases.

## Three-way prompt-format comparison (user update, 2026-11-09)

The official comparisons are now:

| Variant | Template | Scored answers |
|---|---|---:|
| Frozen 3B base | ordinary `User: ...\n\nAssistant:` RWKV | 100 |
| Step 200 post-norm ROSA | ordinary RWKV | 100 |
| Step 200 post-norm ROSA | ChatML with existing vocabulary special tokens | 100 |

The GPU CLI defaults to `--prompt-template rwkv --arms base,trained`. The
ChatML follow-up explicitly uses `--prompt-template chatml --arms trained`;
base is **not** generated/scored again under ChatML. Its head may be computed
on identical trained hidden vectors only for post-norm logit-delta diagnostics.
All short sanity and pilot runs in that follow-up are trained-only as well.

- Current RWKV run:
  `/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/gpu_needle_step200_rwkv_20261109`
- Trained-only ChatML follow-up:
  `/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/gpu_needle_step200_chatml_trained_20261109`

`rwkv_rosa.native.gpu_needle_followup` waits for the specific existing GPU worker
PID **and creation time** to exit, then starts only the independent ChatML
inference command. It inherits lengths, depths, seeds, immutable checkpoint,
24-token greedy cap, and W&B setting from the RWKV request. It cannot launch
training. This avoids concurrent 192K CUDA forwards overcommitting the GPU.
`queue_status.json` distinguishes queued from actually running/completed. Touch
`CANCEL_QUEUE` in the follow-up folder to cancel **before** its worker launches.
If the RWKV worker exits incomplete, the follow-up still runs but records that
failure; no complete three-way conclusion is fabricated.

Both runs have separate W&B run IDs under the same training-run group. After
ChatML completes, `three_way_report.json` in its output combines exactly the
three requested variants, after checking matching matrix/checkpoint/decoding
settings. Model improvement is measured **only** by ROSA-RWKV vs base-RWKV;
ROSA-ChatML vs ROSA-RWKV measures prompt-format sensitivity, not isolated
training improvement. Prompt-token counts can differ between formats; each
case still has the requested total document-token budget and fixed seed.

The earlier ChatML paired diagnostic is retained separately in
`gpu_needle_step200_20261109` with status `cancelled_template_diagnostic`; it is
not counted as an official completed benchmark. It showed normal short base
responses in RWKV format but repetitive symbols in ChatML, motivating this
explicit format ablation rather than silently mixing incompatible templates.

The 24-token QA cap is deliberately inherited for matching; outputs that enter
`<think>` and never emit the passkey before this cap are marked truncated, not
proof that retrieval failed. The 1K short-context pilot is separate from the
100-case matrix and is not included in its accuracy denominator.
