# GPU stateful NIAH: implementation and completed ten-case result

## What changed

Training remains stopped. No weights or optimizers were updated.

The old `gpu_needle.py` is retained as a full-replay reference, not used for the
completed quick benchmark. `gpu_state.py` adds an inference-only stateful forward:

- Reuses the native SFT model's BF16 mix6, value residual, gates, key normalization,
  GroupNorm/residual/gating, channel-mix and frozen weights.
- Adds `native/cuda/stateful_wkv.cu`, a forward-only counterpart of `@rwkv3` with
  the same decay and recurrence/reduction order. It accepts/returns FP32 WKV
  state, instead of resetting to zero at every generation step. No backpropagation
  buffers or optimizer are allocated. State tensors are immutable inputs.
- Per layer, caches WKV state plus the previous normalized attention/FFN inputs.
  For this 3B model the total cache is **20.3125 MiB per independent history**.
- Prefill retains native tail padding/GEMM dimensions but snapshots state **at
  the last real token**, never after the synthetic pads. Subsequent decoding
  consumes just one new token. Identical base/ROSA histories share the frozen
  state/forward; diverged histories use independent state.
- Adds a persistent `IncrementalSAM` binding to the existing compiled CPU LCT
  suffix automaton. The training batch predictor is not replaced. Prefill builds
  once and generation calls `update_one`; special-token filtering matches the
  training address policy. SAM remains CPU integer addressing; ROSA tensors and
  projections remain FP32 CUDA.

## Numerical verification

Results: `/mnt/data/Codes/ROSA/outputs/gpu_state_audit_20261109/report.json` and
`numerical_audit.json` in the completed evaluation folder.

- Synthetic WKV forward: tested native output vs new kernel, arbitrary split
  points, final state: maximum absolute differences **0** in this test.
- Real checkpoint prefill on a short prompt and an 8233-token prompt: hidden and
  final-token logits versus native full forward: maximum differences **0**.
- Real checkpoint split continuation is **not bitwise equal**: observed hidden
  maximum difference up to 1.5 and logits up to 0.5 for the audited last-token
  split. Top-1 was unchanged in these two audited cases. Splitting changes BF16
  GEMM shapes; these are measurements, not a guarantee that all generation will
  match the replay backend. Do not silently compare the two runtimes as exact
  equivalents.
- Real CUDA tests verify arbitrary splitting, exclusion of tail padding from
  saved state, input-state immutability and independent branch outputs.
- CPU tests verify incremental SAM against static addressing on random/repeated/
  special-token histories, clones, and the exact balanced ten-case design.

Test logs:
`/tmp/rosa-all-tests-stateful.log` (247 tests, OK, 3 skipped) and
`/tmp/rosa-gpu-state-cuda-tests.log` (3 explicit CUDA tests, OK).

## Completed quick comparison, November 9, 2026 (Asia/Shanghai)

Output:
`/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/gpu_needle_step200_stateful10_20261109`

Base: `rwkv7-g1k-2.9b-20260930-ctx25600.pth`; immutable post-norm ROSA snapshot:
`checkpoints/rosa-adapter-step000000200.pth`. Loader validates both tensor hashes.

- 10 cases per variant, **30 generated answers total**.
- 8192 document tokens, depths 10/30/50/70/90%, QA and literal-prefix completion.
- Five unrelated-background cases and five similar-archive cases, balanced by
  alternating difficulty across depth/mode, selected before seeing any output.
- Same keys, secrets, seeds and difficulty identifiers across templates. Template
  tokens/header lengths differ, so identical document-token budgets are not
  identical full token sequences across RWKV and ChatML.
- Greedy, **128-token cap uniformly across all variants**. This is larger than
  the cancelled replay run's 24-token cap, to allow some thinking output. No early
  stop based on seeing the answer. Generated answers never use teacher-forced
  tokens; target-probability diagnostics run later on independent state branches.

| Variant | First six-digit passkey hit | QA | Prefix completion | Strict digits only |
|---|---:|---:|---:|---:|
| Base, RWKV format | 4/10 | 2/5 | 2/5 | 0/10 |
| Step 200 ROSA, RWKV format | 4/10 | 2/5 | 2/5 | 0/10 |
| Step 200 ROSA, ChatML | 0/10 | 0/5 | 0/5 | 0/10 |

All ten base-vs-ROSA RWKV generated token sequences are **identical**. Prefix
SAM diagnostic correctly retrieves all three answer tokens in 5/5 cases, but
neural prefix generation succeeds only in 2/5. Thus a suffix hit must not be
reported as neural answer accuracy.

On RWKV prompts, on average **96.49%** of hidden components have zero effective
change after the real BF16 residual addition. The target first-token head logit
change was zero on 9/10 cases and +0.0001220703125 on one. These support a very
weak effective output correction at this checkpoint; they do **not** establish
that precision alone, rather than residual magnitude/training/representation
alignment, is the sole cause.

**Scoring caveat:** the passkey-hit metric counts the first standalone six-digit
number anywhere in the generated text, including thinking text. It is not a
final-answer-only metric. All three variants fail strict digits-only formatting.
8/10 outputs of each RWKV variant and all 10 ChatML outputs reach the 128-token
limit, so this is a bounded quick test, not a definitive unlimited-generation
quality result. ChatML outputs are predominantly repetitive symbols/garbled text.
A ten-case sample is not a broad statistical benchmark or a completed 192K
free-generation evaluation.

### Measured speed and artifacts

- Complete job (including loading/audits): **98.61 seconds**; 30 answers finished
  at **19:53:47, November 9, 2026**. Actual case evaluation about 76.26 seconds.
- Median cached forward: ~17.15 ms on RWKV paired histories, ~16.85 ms on ChatML;
  median 8K prompt prefill ~0.58 seconds. Compared with the earlier ~0.46 s full
  replay **per token**, this is ~27x faster in the 8K decode phase, not a 27x
  end-to-end or 192K speed claim. Same-history base/ROSA forward sharing means
  per-arm throughput is not additive.
- Peak allocated memory: **7.315 GiB**.
- W&B run **`m231r44o`**, project `ROSA-RWKV7`, group `needle-of-dee9ca1a`.
  Worker log confirms file synchronization. `results.jsonl`, `report.json`,
  `request.json`, and `numerical_audit.json` were uploaded.
- Original broad replay probes and cancelled queued test remain in their own
  folders and are not combined into this accuracy denominator.

Launch a fresh run:

```bash
WANDB_BASE_URL=http://dian.sea-group.org:9098 \
  bash /mnt/data/Codes/ROSA/scripts/run_rwkv_fast_gpu_needle.sh \
  --train-dir /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109 \
  --checkpoint /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109/checkpoints/rosa-adapter-step000000200.pth \
  --output /absolute/path/to/a/fresh/output \
  --length 8192 --max-new-tokens 128 --wandb
```

No training is started/resumed by this launcher. No automatic evaluation worker
is left running after completion.
