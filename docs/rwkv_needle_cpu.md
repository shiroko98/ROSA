# CPU needle-in-a-haystack diagnostic for native improved ROSA

Implementation: `/mnt/data/Codes/ROSA/rwkv_rosa/native/needle_eval.py`.
The test does not alter or stop the ongoing native GPU training job.

## Why two query modes?

The repository's native ROSA uses literal-token online SAM addressing. It retrieves
a predicted **next token following a repeated suffix**. Its CPU/GPU learned
fusion is a gated projected value-vector residual before a RWKV block, not a
hard assignment of the retrieved token to the output distribution. Therefore
there are two separate empirical questions:

1. Does a repeated key retrieve the correct continuation token from a distant
   needle (addressing mechanism)?
2. Does the actual learned model output the correct passkey, rather than merely
   improve its probability (end-to-end task)?

The experiment does not assume either result and does not silently replace the
model's output with a retrieved token.

## Fixed protocol

- Freeze one completed checkpoint: **step 724**, the latest completed hourly
  checkpoint at experiment initiation on 2026-11-09. Convert and verify all
  backbone tensors exactly match the requested original 0.1B-named base.
- All inference is CPU FP32, 4 Torch/OMP/MKL threads, CUDA hidden. This is a
  matched reference comparison, not bitwise equivalence with fused BF16 CUDA.
- Seven deterministic fresh random eight-letter archive keys and six-digit
  secrets (seed 20261109, distinct seed per condition). Each secret occurs exactly
  once in its document and never occurs in its query. Filler is synthetic Chinese
  activity records with varying locations/actions/counts, not repeated token IDs.
- Common document prefills have exactly **1,024 / 8,192 / 32,768 tokens**.
  Query suffix tokens are additional and their exact full prompt lengths are
  recorded. These labels are document-prefill sizes, not rounded total lengths.
- At 1K and 8K, insert the needle at approximately 10%, 50%, 90% depth. At 32K,
  test the middle position only. Actual needle position/full prompt fraction is
  recorded. This yields **7 documents, 14 mode cases per full model arm**.
- Each document is tested with the **same two query modes**:
  - `qa`: ordinary question asking for that archive's passkey, digits only.
  - `exact_prefix`: assistant continuation ends in the same exact archive prefix
    as the needle, but contains **none of the secret**. This intentionally tests
    the best-matched literal retrieval condition, not general semantic retrieval.
- Full arms: **base** and **trained ROSA** on all 14 cases. A separate explicitly
  limited **initialized ROSA** control runs the two central-8K cases. It is not
  reported as if tested on the full matrix.
- Greedy decoding, max 24 new tokens, stop IDs 0 / 65531 / 65532. No target tokens
  are supplied during free generation. Content success requires the first
  standalone six-digit answer to equal the secret; digits-only exact match is
  recorded separately. An early wrong number cannot pass just by mentioning the
  right number later.
- Teacher-forced answer NLL and first-answer-token rank/probability are separate
  probability diagnostics, not substitutes for generation accuracy.
- Each condition's common-document recurrent cache is reused for its two query
  modes. It contains **only the document**, not a future query or answer. Tests
  check that the cached reference state is not mutated by decoding/probability
  probes, and cached teacher-forced NLL matches recomputing the full input.

## Address-only diagnostic is not model accuracy

Run the same native SAM addressing policy on each prompt plus correct prior
answer tokens. If every causally predicted address equals its corresponding
answer token, deterministic suffix-only free continuation of the known answer
length would be exact (by induction). This tests the addressing mechanism without
using future answer tokens at any predicted position. It is clearly labelled
`suffix_only_exact_match`, separate from the neural model's greedy output.
Normal QA is not expected to present the same exact suffix as the needle; the
measurement checks rather than assumes whether it hits.

For ROSA arms, record actual query-tail address/match length and the final query
position's injected-residual-to-embedding norm ratio. A correct address alone is
not counted as an actual neural passkey success.

## Limits and reproducibility

This is a **small diagnostic**, not an official or statistically comprehensive
NIAH benchmark. It tests one synthetic needle per document, only seven secrets,
no full-192K sequence, and no robust semantic paraphrase/multi-needle suite. The
1K/8K position cases differ in secret and filler as well as depth, so a position
trend cannot be isolated from a single example. Both modes share each document;
they are not independent trials. 32K exceeds the base filename's 8K context and
establishes only this CPU execution/test outcome, not general pretrained quality.

Accepted output:
`/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/needle_step724_20261109/`.

- `cases.json`: all synthetic inputs/keys/secrets, seeds, positions and hashes.
- `report.json`, `results.jsonl`: all generated answers, exact-match flags,
  teacher-forced diagnostics and per-length/mode summaries.
- `export/`: strictly verified ROSA-aware inference weights. Not a static merge.
- `source_snapshot/`, `source_hashes.json`: implementation and tokenizer provenance.
- W&B receives aggregate numeric metrics only; synthetic full inputs and answers
  remain local. Remote API readback must verify the upload before claiming it.

Run using the `rosa-rwkv` environment, `CUDA_VISIBLE_DEVICES=''`, and
`python -m rwkv_rosa.native.needle_eval --checkpoint ... --base ... --output FRESH_DIR --wandb`.

## Additional similar-record distractor variant

After early ordinary-background base-model successes suggested a ceiling effect,
a **separate** `similar_archives` variant was added; the original experiment is
retained in full, not replaced or filtered. Background lines now contain many
other random archive-key / six-digit-passkey pairs in the **same format** as the
needle. Keys and secrets are unique within each document. The target remains a
single specified key and its secret occurs exactly once; neither query contains
the secret. This stresses key-value selection rather than remembering the only
salient six-digit number in otherwise unrelated prose.

This variant has 8K at 10/50/90% depth and 32K at 50%: **4 documents, 8 mode cases
per full arm**. Initialized ROSA remains only a central-8K control. It reuses the
exact verified step-724 export from the ordinary-background experiment via an
`export` symlink; it does not reload a newer training checkpoint or change fusion
scale/weights. The two difficulty variants use different random-condition
assignments, so do not attribute every across-variant difference solely to
interference based on this small diagnostic.

Output:
`/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/needle_hard_step724_20261109/`.

Source snapshots record the original ordinary-background script and the later
extended script separately. Baseline and trained arms within each case use
identical prompt tokens, needle, depth and decoding rules.

## Completed results — 2026-11-09

# CPU needle-in-a-haystack results — step 724

## Conclusion

The tested checkpoint does not demonstrate a large end-to-end passkey retrieval improvement over base. Ordinary 1K/8K cases are already solved by base; both models fail the ordinary 32K case and the harder similar-archive distractor cases. Crucially, the exact-prefix SAM candidate mechanism can independently retrieve every tested secret, but the neural fusion does not reliably turn that candidate into the correct answer. This is a limitation of this evaluated integration/checkpoint, not a general proof about all ROSA designs.

All inference is CPU FP32. Native GPU training was left running and unchanged. Same model weights, prompts and decoding are used per paired case. These are small synthetic diagnostics, not a full-192K benchmark.

## End-to-end content retrieval

Correct means the first standalone six-digit number in the generated answer matches the secret. This is NOT strict digits-only instruction compliance. Each cell is successes / paired cases in that mode.

| Background | Document-prefill tokens | QA base | QA trained ROSA | Exact-prefix base | Exact-prefix trained ROSA |
|---|---:|---:|---:|---:|---:|
| 普通背景 | 1,024 | 3/3 | 3/3 | 3/3 | 3/3 |
| 普通背景 | 8,192 | 3/3 | 3/3 | 3/3 | 3/3 |
| 普通背景 | 32,768 | 0/1 | 0/1 | 0/1 | 0/1 |
| 相似档案干扰 | 8,192 | 0/3 | 0/3 | 0/3 | 0/3 |
| 相似档案干扰 | 32,768 | 0/1 | 0/1 | 0/1 | 0/1 |

1K/8K ordinary background: needle depths 10%, 50%, 90%. 32K: middle only. Hard background: 8K at 10/50/90%, 32K middle only. Query suffix tokens are additional to document-prefill length. Each experiment uses distinct small random condition sets; do not treat all modes as independent trials.

Neither base nor trained ROSA obeyed the strict digits-only output format in these tested cases; successful content answers often continued with copied/repeated document text.

## What the address engine can do

- Teacher-forced causal address continuation check: 11/11 exact-prefix secrets correct.
- Independent **free-running suffix-only candidate copy**, with no ground-truth tokens supplied to the engine: **11/11** correct. This is an address-only diagnostic with NO RWKV neural decoder, not the model’s task accuracy.
- Normal QA does not end in the needle’s exact suffix; its direct suffix-only continuation checks all failed (0/11). Exact literal retrieval is not automatically semantic question answering.

For the trained model, the query-tail layer-0 residual/embedding norm ratio in exact-prefix cases ranged from 5.06% to 5.79%, median 5.52%. The candidate is a small learned vector residual before 12 frozen RWKV blocks, not a hard override or explicit copy distribution at the output. The observations locate the gap between correct literal addressing and successful neural readout; they do not prove that scale alone is the cause.

## Representative harder central-8K case

Target archive AUCBHXDF, correct passkey **470012**. With the exact-prefix query:

- Base generated first passkey: **657511** (wrong).
- Initialized ROSA generated first passkey: **657511** (wrong).
- Trained ROSA generated first passkey: **657556** (wrong).
- Native SAM query address pointed to token 667, the correct initial `47` token, with match length 15. Free-running SAM-only continuation returned **470012**.
- Yet the trained neural model ranked that correct first answer token only 48th. Thus a correct retrieval address cannot be counted as a correct model answer.

## Further negative/neutral findings

- The limited initialized ROSA control (central 8K only) also failed the hard task; do not report it as a full-matrix result.
- In the ordinary 32K exact-prefix case, answer NLL got worse: base 0.988085 -> trained 1.649600. QA answer NLL sometimes improved without changing retrieval success.
- No training hyperparameters, fusion scale, placement or weights were changed to obtain these results.
- A separate hard variant was added after early base successes revealed an ordinary-background ceiling; both variants and every sample are retained.
- 199 CPU-only regression tests passed (3 CUDA-only tests skipped), 95.03% Python line coverage. Tests cover token-boundary/no-answer-leak construction, exact suffix retrieval, unmutated shared cache and agreement with full recomputation of target NLL.

## Artifacts / W&B

### 普通背景
- Directory: `/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/needle_step724_20261109`
- `cases.json`: all synthetic inputs/targets and positions.
- `report.json`, `results.jsonl`: all neural answers and metrics.
- `suffix_free_running.json`: separate free-running address-only check.
- `evaluation.log`, `source_snapshot/`, `source_hashes.json`: raw logs and provenance.
- Numeric W&B metrics remotely verified: http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/tqg8stgv

### 相似档案干扰
- Directory: `/hikscale/models/ROSA/rwkv7_01b_rosa_192512_20261109/needle_hard_step724_20261109`
- `cases.json`: all synthetic inputs/targets and positions.
- `report.json`, `results.jsonl`: all neural answers and metrics.
- `suffix_free_running.json`: separate free-running address-only check.
- `evaluation.log`, `source_snapshot/`, `source_hashes.json`: raw logs and provenance.
- Numeric W&B metrics remotely verified: http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/fh0ip9rc

