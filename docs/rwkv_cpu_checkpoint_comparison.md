# Native ROSA checkpoint export and CPU comparison

The evaluation implementation is
`/mnt/data/Codes/ROSA/rwkv_rosa/native/cpu_compare.py`.
It does **not** change the running native CUDA training job or its environment.
All evaluation parameters, recurrent state, addressing and head logits execute
on CPU. Launch with `CUDA_VISIBLE_DEVICES=''`; the script fails if CUDA is visible.

## Conversion is not a static merge

The native Lightning `.ckpt` includes frozen BF16 RWKV weights, FP32 learned ROSA
weights, optimizer moments and training loop state. The exporter snapshots one
resolved source path and writes:

- `export/rwkv7-rosa.pth`: flat tensor-only native weights including `rosa_cpu.*`.
- `export/rosa-adapter.pth`: learned ROSA tensors and layer/scale/base metadata.
- `export/manifest.json`: checkpoint step, model dimensions, exact parameter-byte
  hashes and loading requirements.

The original full-state checkpoint remains intact. The exporter compares every
backbone tensor and dtype to the requested base, checks finite tensors, and
reloads the serialized `.pth` to verify its complete weight hash. These artifacts
are **not** a vanilla RWKV merged model. ROSA depends nonlinearly on current hidden
states and causal token-history retrieval and cannot be absorbed into a fixed
RWKV matrix. Vanilla RWKV loading may reject or ignore its extra parameters.
Use the supplied ROSA-aware CPU inference class (or the native training/inference
integration), not a loader which silently discards `rosa_cpu.*`.

```python
from rwkv_rosa.native.cpu_compare import load_cpu
model = load_cpu('/absolute/path/to/comparison/export', 'trained')
# CPU FP32 paired reference inference, not native BF16 CUDA equivalence.
```

## Matched CPU diagnostic protocol

Three arms use identical backbone weights, data tokens, masks and CPU FP32
arithmetic:

1. **base:** original RWKV without ROSA.
2. **initial:** original RWKV with initialized, untrained improved ROSA (value
   table copied from base embeddings, identity projections, original norms/gates).
3. **trained:** original RWKV plus all strictly loaded native trained ROSA tensors.

The initial arm separates the structural module effect from learned-weight gains.
CPU inference reuses the existing pinned upstream reference RWKV runtime, including
FP32 recurrent state, and calls the **actual native ROSA module** before exactly
the configured blocks. Streaming generation retains per-block WKV/time-shift
state and full causal ROSA address history. Regression tests check CPU base/runtime
identity, ROSA full-vs-streamed agreement, causal prefix agreement, strict export
reload, mask-weighted scoring and exact native held-out split reproduction.
CPU FP32 avoids comparing different arithmetic between arms, but does not claim
bitwise equivalence with native fused BF16 CUDA training or its validation losses.

### Held-out records and masks

**Important:** because production uses `sft_train_shuffle=1`, native SFT first
permutes all row indices with seed 42, then reserves the final 2,117 indices of
that permutation for evaluation. These are **not the last 2,117 physical rows**.
This comparison directly reuses `compute_sft_shuffled_split_indices`; it asserts
no selected row belongs to the native training partition. The first eight rows
of the native evaluation list are excluded because online monitoring repeatedly
uses those rows. Existing production training split is unchanged.

The initial comparison samples 16 different remaining held-out physical rows,
seed 20261109. Each produces a **2,048-input-token window** centred near the median
supervised target, preserving exact tokens and shifted masks. Initial recurrence
and address history reset at window start in **all** arms. Manifest records row
indices, offsets, counts and token/mask hashes. This is a bounded, supervision-
centred diagnostic; it is not a random-token estimate of the whole corpus, not
an evaluation of entire 192,512-token records, and not a long-distance retrieval
benchmark. Source packing may contain related/duplicate conversations across
rows; row-disjoint evaluation does not establish semantic deduplication.

Metrics: supervised-token-weighted mean cross entropy, perplexity, top-1 next-
token accuracy, paired per-window loss wins and a paired window-level bootstrap
95% interval. Bootstrap intervals describe only this sampled diagnostic and do
not remove context truncation, supervision selection or corpus-duplication bias.

### Generation samples

Eight fixed Chinese prompts cover arithmetic, translation, simple knowledge,
copying, short-context recall, sorting and explanation. Base/trained arms use
identical upstream chat template, prompt formatting and greedy decoding, max 64
new tokens, same stop IDs. Full texts are saved locally in `generations.json`;
they are not automatically uploaded. These are illustrative samples, not a
validated comprehensive task benchmark. Do not judge success by cherry-picking
one answer or assuming lower token loss guarantees better free generation.

## Run

```bash
cd /mnt/data/Codes/ROSA
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
WANDB_BASE_URL=http://dian.sea-group.org:9098 WANDB_ENTITY=mobius WANDB_MODE=online \
/mnt/data/anaconda3/envs/rosa-rwkv/bin/python -u -m rwkv_rosa.native.cpu_compare \
  --checkpoint /absolute/path/to/native-hourly-step000000479-TIMESTAMP.ckpt \
  --base /hikscale/models/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --data /hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B \
  --output /absolute/path/to/FRESH_comparison_output \
  --heldout 2117 --split-seed 42 --rows 16 --length 2048 \
  --threads 4 --seed 20261109 --max-new-tokens 64 --wandb
```

A separate W&B `cpu-evaluation` run receives numeric aggregate results only;
remote summary is read back before marking upload verified. The training W&B
run and its metric history are left unchanged. API keys are read from existing
matching-host credentials, not serialized in reports.

The initial launch on 2026-11-09 was stopped during conversion before any data
scoring while the native shuffled split policy was checked; only the `_v2`
comparison directory is the accepted evaluation. Incomplete initial export files
are not evaluation results.

## Completed step-479 results — 2026-11-09

Accepted output root:
`/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/cpu_compare_step479_20261109_v2`.

Completed: first 8 of the same 16 held-out rows, fixed without selection by loss improvement, expanded to 8,192 inputs. 54,352 supervised tokens. These overlap the 2K sample rows; the two context tests are not independent studies and should not be pooled as such.

| Arm | Mask-weighted loss | PPL | Next-token top-1 accuracy |
|---|---:|---:|---:|
| base | 1.523601 | 4.588719 | 63.4015% |
| trained | 1.502108 | 4.491146 | 63.6352% |

Trained-vs-base loss reduction: **1.411%**. PPL reduction **2.126%**. Accuracy change **+0.2337 percentage points**. Loss improves on **5/8** windows, not every row. Paired-window bootstrap 95% loss-gain interval: [0.004195163389653402, 0.042166665535008405]; this is only a small, supervision-centred diagnostic.

8K numeric W&B upload remotely verified: http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/cc73x9vl.

## Overall conclusion

The evaluated step-479 checkpoint has a **small positive supervised next-token likelihood benefit**, larger at 8K than 2K in this sample. Its 8K token accuracy also improves slightly, while 2K accuracy does not. This is **not yet consistent free-generation or instruction-following improvement**: the retained generation samples include gains, unchanged failures and repetition regressions. It does not establish full-192K or long-distance retrieval quality. CPU comparison left the running native GPU training job untouched.

Validation: 196 regression tests passed in CPU-only mode (3 CUDA-only tests skipped), 95.08% Python line coverage. The real exported CPU model also passed full-vs-chunked inference on a 48-token check; not a universal native-BF16 equivalence claim.


Full 2K metrics and all generation examples are in `RESULTS.md` and
`GENERATION_COMPARISON.md` under that output root. 2K loss base 1.551159,
untrained ROSA 1.551486, trained ROSA 1.544855; PPL 4.716934 -> 4.687291;
next-token accuracy 63.1624% -> 63.1278%. Do not hide this slight accuracy decline.
