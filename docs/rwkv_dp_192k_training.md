# RWKV-7 ROSA replicated DP at 192512 tokens

`scripts/run_rwkv_3b_dp_192k_training.sh` launches eight independent CUDA
processes with `torchrun`. Each GPU keeps the entire frozen BF16 2.9B backbone,
FP32 ROSA adapter, AdamW and SparseAdam states. There is no ZeRO, FSDP, model
partitioning, or optimizer offload. The four embedding pieces are local storage
pieces replicated on **every** GPU, not distributed shards.

The pretrained model is
`/mnt/data/Models/RWKV-7/rwkv7-g1k-2.9b-20260930-ctx25600.pth`. Data comes directly
from `/mnt/data/Datasets/SFT_RWKV7_13B_rwkv_g1_ctx192512/results/SFT_RWKV7_13B`, using
the rebuilt ordinary `System:`, `User:`, `Assistant:` format, EOS/pad 0,
last-assistant supervision, and all 192512 input/target positions in each row.
The existing seed-42 physical split reserves 2117 held-out rows and uses the
remaining 224631 training rows. Each rank reads different rows. The final global
batch wraps a small number of rows to keep all ranks participating equally.

The adapter uses `post_norm`, `rms`, and **`logit_gap`** dynamic scaling.
The short CE study initially selected `min_match=2`, `dynamic_max=0.2`;
the final retrieval run used `min_match=6`, `dynamic_max=0.3`, margin 0.5.
The backbone remains frozen. These are new adapter and optimizer states,
not a resume of the historical fixed-scale experiments. The launcher defaults
remain `6` / `0.2`; pass both settings explicitly to reproduce the final run.
Training stopped after saving step 1800. See
[current status and evaluation results](rosa_relay_status.md) for the final
checkpoint and the distinction between the historical runs below.

Dense adapter gradients use PyTorch DDP. Sparse embedding weights are explicitly
excluded from DDP's dense reducer. Before token normalization and clipping,
`SparseGradientSync` computes the union of touched rows and averages gradients in
bounded chunks. Missing local gradients contribute zeros, and touched zero rows
remain represented for SparseAdam. Entirely untouched tables keep `grad=None`.
The existing native global supervised-token normalization also applies to
gradient accumulation. No duplicate `DistributedSampler` is installed.

After the first real update, audit records verify identical adapter hashes on
all ranks, distinct input batches, and unchanged frozen-backbone bytes.
Only rank zero publishes logs and checkpoint files; every rank participates in
the save/barrier. Checkpoint time and stop decisions are broadcast, preventing
rank-specific filesystem timing from changing collective order.

Checkpoints contain FP32 adapter weights, both optimizer states, Lightning loop
counters, context/data/fusion metadata and a verified reference to the immutable
base model. Resume checks global world size, per-GPU batch and accumulation.
`native-last.ckpt` atomically points to an immutable checkpoint. One initial
checkpoint after the first update seeds independent evaluation; subsequent saves
occur every **7200 seconds**, at optimizer boundaries, with a final save on fit
completion. Historical checkpoints are retained.

To start detached training plus CPU evaluation:

```bash
/mnt/data/anaconda3/envs/rwkv-lm-v7/bin/python scripts/start_rwkv_3b_dp_cpu_eval.py \
  --output /hikscale/models/ROSA/NEW_EXPERIMENT --batch 2 --cpu-threads 4
```

This command starts a new experiment. To select the final run's settings, add
`--min-match 6 --dynamic-max 0.3 --scale-mode logit_gap`.

`train.log`, `train.pid`, `run_manifest.json`, `native_audit.json`, and
`native_steps.jsonl` describe the running experiment. A source snapshot retains
the code including local DP edits. W&B uses the configured self-hosted server
and independent run IDs for training and evaluation.

The CPU workers watch only published immutable checkpoints. They never signal,
pause, or wait inside the trainer. Separate 8192- and 192512-token document
protocols each use three depths, two distractor types, QA and exact-prefix
prompts, paired base/trained arms, and greedy free generation capped at 24
tokens (24 answers per checkpoint per protocol). They use the ordinary RWKV
prompt template for both arms, matching the rebuilt dataset. Full CPU reports contain actual generated answers and
scores; teacher-forced likelihood diagnostics are reported separately. CPU FP32
quality measurements are not claims of numerical equality with native CUDA
BF16. The long CPU protocol can take much longer than checkpoint intervals.

Validation includes a real two-process Lightning test against a single global
weighted-batch oracle, disjoint sparse rows/missing local gradients, explicit
zero gradients, clipping, accumulation, and bitwise checkpoint continuation.
Full-context CUDA batch capacity is measured separately before launch; the
runtime manifest records the selected batch. The first multi-GPU update provides
an additional real-model cross-rank parameter and data audit.

## Historical launch (stopped; do not resume)

Run directory:
`/hikscale/models/ROSA/rwkv7_3b_rosa_dp8_ctx192512_dynamic_m6_cap02_20261109`.
W&B: `http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/7cf2ca2c`.
Per-GPU batch is 2, global batch 16, with 3,080,192 input tokens per update.
The one-pass schedule contains 13095 optimizer updates (four repeated tail rows).

Full-context single-GPU probes completed two updates at batch 2. Batch 3 failed
on its second update both with the default allocator and with expandable
segments, after optimizer states became resident. The latter needed another
5.51 GiB with 73.89 GiB already allocated on a 79.17 GiB device. Batch 2 is the
largest tested stable batch; capacity evidence is retained in
`validation/capacity_report.json` under the run directory.

The real eight-GPU run completed its first optimizer updates with identical
adapter hashes across all eight ranks and eight distinct input-batch hashes.
The backbone hash remained unchanged, while ROSA weights updated. Initial
checkpoint publication took 2.37 seconds and produced a 2.17 GB resumable file.
Subsequent updates initially took about 20.4 seconds, around 151K aggregate
input tokens/second. GPU memory reported by `nvidia-smi` was approximately
78,020 MiB per card, including allocator reservation and CUDA/NCCL overhead.
These observations establish startup and early update correctness, not completed
long-run training or NIAH quality improvement. Both CPU NIAH workers started
independently from the initial checkpoint while the eight-GPU trainer continued.

## Rebuilt-data launch after matched scale study

The historical run above was stopped at step 43 and must not be resumed for
this experiment. The rebuilt ordinary G1 dataset has 226748 physical rows,
224631 train rows and 2117 held-out rows. EOS/pad is 0. The one-pass schedule
contains 14040 optimizer updates, global batch 16 (microbatch 2 on eight GPUs).

A 36-configuration initial screen and eight matched 28-update training pilots,
followed by eight separate full-context confirmation rows, selected
`logit_gap`, `min_match=2`, `dynamic_max=0.2`, margin 0.5. Exact calibration CE
was 0.744443092; sampled population-weighted confirmation CE was 0.710058290.
Fixed scale 0.05 was close (confirmation CE 0.710125079), and its top-1 accuracy
was slightly higher. The difference between the leading two configurations is
within paired-row uncertainty; this is the best tested short-run CE setting,
not evidence of a universal optimum. The protocol and complete report are in
`/hikscale/models/ROSA/rosa_plain_scale_study_20261109/REPORT.md`.

Fresh long-run output:
`/hikscale/models/ROSA/rwkv7_3b_rosa_dp8_plain_ctx192512_best_m2_cap02_20261109`.
W&B: `http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/5ff82386`.
This loads the original pretrained base and initializes a new adapter and
optimizer; it resumes neither an old run nor a pilot. Initial observed losses
were 0.72036 and 0.72496, approximately 20.9 seconds per update and 147K aggregate
input tokens/second. The first-update audit confirmed all eight adapters were
identical, eight distinct input batches, unchanged frozen base, and updated
ROSA parameters. GPU reservation was approximately 78025 MiB per card.
Dynamic gain averaged approximately 0.03 and reached the configured 0.2 cap.
An initial 2.17 GB checkpoint seeded the independent 8K/192K CPU NIAH workers;
subsequent periodic saves are scheduled every two hours. Startup evidence is
retained in `startup_verification.json` in the run directory.
