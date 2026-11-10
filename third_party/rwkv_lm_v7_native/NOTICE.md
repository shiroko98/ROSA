# Native SFT training source provenance

Source repository: https://github.com/TimeMobius/RWKV-LM-V7, branch `new_sft`.
Pinned commit: `1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1`.
License: Apache-2.0; original LICENSE retained. This is separate from the older
`third_party/rwkv_lm_v7/` demo/reference runtime.

Reused native components: `train.py`, `src/model.py`, `src/dataset.py`, SFT loss,
token-weighted accumulation, LR scheduler, evaluation/checkpoint/W&B callbacks,
all required fused CUDA sources, and CUDA environment setup helper.

ROSA-specific changes (complete diff in UPSTREAM_PATCH.diff; source hashes in
UPSTREAM_FILES.json):

- `train.py`: add extension arguments and install the ROSA callback after strict
  checkpoint loading; select replicated DDP and sparse-gradient synchronization
  when explicitly enabled; retain native Trainer creation and `trainer.fit`.
- `src/model.py`: call ROSA once per forward and inject its differentiable residual
  before configured blocks; use PyTorch reentrant block checkpointing for the
  single-GPU ROSA path instead of requiring a distributed DeepSpeed engine;
  allow explicitly selected tail blocks to skip recomputation for measured memory/speed tuning.
- `src/trainer.py`: sparse-aware gradient norms on the configured ROSA device and
  explicit diagnostic-only checkpoint disabling. Original token-weighting,
  scheduler, training/evaluation/logging logic stays in use.
- CUDA sources containing `atomicAdd(float2*)`: retain vector atomics on SM90+;
  use two scalar atomic adds on Ada/pre-SM90 so original native kernels compile
  on the actual RTX 4090 D. No Python/reference WKV fallback.

All trainable ROSA integration/optimizer/verification code lives in
`rwkv_rosa/native/`, outside this vendored tree. The wrapper
`train_rwkv_native_rosa.py` executes this actual train.py via runpy with its
expected source root/current directory.

Precision: frozen backbone and native kernels BF16; learned ROSA FP32. The
single-device extension disables outer Lightning AMP to prevent it promoting
LayerNorm outputs to FP32 before BF16-only CUDA operators. A Trainer precision
value of 32 in this extension therefore does NOT mean FP32 backbone math; the
actual parameter/operator dtypes are BF16 and are recorded separately.

Device-placement comparisons keep SAM address preprocessing on CPU in both arms.
ROSA values/projections/norms/gates, gradients and moment tensors execute on the
chosen CPU or CUDA device. Adam may keep scalar step counters on CPU even with
CUDA parameters; it does not offload CUDA moment tensors.

2026-11-09 3B/last-layer extension: skip redundant block checkpointing before
the first trainable injection (frozen no-grad prefix); optional explicit virtual
windows on native physical-row splits; invoke the CPU passkey monitor after each
native SFT eval. Original native kernels/loss/scheduler/Trainer remain in use.
All windowing and monitoring implementation stays in rwkv_rosa/native/.

2026-11-10 DP extension: native Trainer accepts an opt-in ROSA DDP strategy and
sparse-gradient callback. Sparse-row reduction, rank coordination, checkpoint
publication and independent monitoring remain in rwkv_rosa/native/; no ZeRO
dependency or model/optimizer partitioning is introduced.
