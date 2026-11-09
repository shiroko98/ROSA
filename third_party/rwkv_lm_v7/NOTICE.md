# Upstream provenance

Source: https://github.com/TimeMobius/RWKV-LM-V7, branch `new_sft`.
Pinned commit: `1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1` (retrieved 2026-11-09).
License: Apache-2.0; original LICENSE and source attribution retained.

- `runtime.py`: verbatim `src/rwkv_v7_demo_runtime.py` (differentiable, stateful
  PyTorch RWKV-7 reference used by upstream checkpoint equivalence tests).
- `sft_loss.py`, `binidx.py`: verbatim files from `src/`.
- `tokenizer.py`: verbatim `data/tokenizer/rwkv_tokenizer.py`.
- `chat_template.jinja`: verbatim `data/SFT/sample/chat_template.jinja`.
- `sft_binidx.py`: `src/sft_binidx.py`; only two imports changed to relative
  imports (`.tokenizer`, `.binidx`) to avoid top-level `src` / `data` collisions.

The integration uses the actual upstream reference RWKV backbone and SFT mask
builder, not a Transformer relabeled RWKV. It does not import upstream's
Lightning/DeepSpeed trainer or its unconditional fused CUDA builds. The default
WKV backend is the slower, auditable upstream PyTorch recurrence; this is not a
claim of parity with upstream fused-kernel performance. ROSA integration,
comparison protocol, training runner and tests live outside this directory.

## Integration precision boundary

The vendored runtime remains unchanged. `rwkv_rosa.backbone.load_runtime()` wraps
its WKV operation with `rwkv_rosa.precision.fence_wkv_fp32`. This disables the
outer autocast policy for the recurrence itself: casting state to float32 alone
does not stop autocast from executing its matmuls in BF16. Unfenced execution is
retained only for counterfactual numerical auditing, not as a training default.
Other BF16 projections can still change streamed predictions; FP32 is the
accuracy-comparison default. See `docs/rwkv_bf16_audit.md`.
