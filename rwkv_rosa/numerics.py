"""Reproducible precision audit: same tokens, several streaming partitions, real cache checks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .backbone import infer_config, load_base_state
from .data import ROOT, build_tokenizer, load_documents
from .experiment import autocast_context, file_hash, seed_all, write_json
from .model import RWKVRosaLM, configure_train_scope
from .precision import WKV_PRECISION_POLICY, wkv_policy


class MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.bmm.default:
            self.calls.append({"inputs": [str(a.dtype) for a in args if isinstance(a, torch.Tensor)],
                               "output": str(result.dtype)})
        return result


def tensor_error(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError("Cannot compare different tensor shapes.")
    reference, actual = reference.float(), actual.float()
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise FloatingPointError("Non-finite values in precision audit.")
    delta = reference - actual
    return {"max_abs_diff": float(delta.abs().max()), "mean_abs_diff": float(delta.abs().mean()),
            "relative_rms_error": float(delta.square().mean().sqrt() /
                                        reference.square().mean().sqrt().clamp_min(1e-12))}


def logits_error(reference, actual, targets):
    result = tensor_error(reference, actual)
    reference, actual = reference.float(), actual.float()
    a, b = reference.argmax(-1), actual.argmax(-1)
    changed = a.ne(b)
    ref_logp, logp = reference.log_softmax(-1), actual.log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)
    margins = reference.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
    rows = []
    for batch, time in changed.nonzero().tolist()[:12]:
        rows.append({"batch": batch, "position": time, "reference_token": int(a[batch, time]),
                     "actual_token": int(b[batch, time]), "reference_top2_margin": float(margins[batch, time])})
    result.update(top1_changed=int(changed.sum()), predictions=changed.numel(),
                  top1_agreement=float((~changed).float().mean()),
                  mean_kl_reference_to_actual=float(kl.mean()),
                  nll_reference=float(-ref_logp.gather(-1, targets.unsqueeze(-1)).mean()),
                  nll_actual=float(-logp.gather(-1, targets.unsqueeze(-1)).mean()),
                  changed_examples=rows)
    result["nll_delta"] = result["nll_actual"] - result["nll_reference"]
    return result


def partition(length, mode):
    if length < 2:
        raise ValueError("Audit needs >=2 input tokens.")
    if mode == "halves":
        return [length // 2, length - length // 2]
    if mode == "uneven":
        if length == 2:
            return [1, 1]
        a, b = max(1, length // 5), max(1, length // 3)
        return [a, b, length - a - b]
    if mode == "decode":
        tail = min(16, length - 1)
        return [length - tail] + [1] * tail
    raise ValueError(f"Unknown partition: {mode}")


@contextmanager
def float32_model(model):
    storage = {name: p.dtype for name, p in model.named_parameters()}
    runtime = model.backbone._reference_runtime
    previous = runtime.DTYPE
    try:
        model.float()
        runtime.configure_runtime(dtype=torch.float32)
        yield
    finally:
        for name, p in model.named_parameters():
            p.data = p.data.to(storage[name])
        runtime.configure_runtime(dtype=previous)


def stream_forward(model, ids, sizes):
    if sum(sizes) != ids.shape[1] or any(s <= 0 for s in sizes):
        raise ValueError("Streaming partition must cover the sequence exactly.")
    pieces, state, start = [], None, 0
    for size in sizes:
        out = model(ids[:, start:start + size], state=state, return_state=True)
        pieces.append(out["logits"].float().cpu())
        state = out["state"]
        start += size
    return torch.cat(pieces, dim=1), state


def cache_error(reference, actual):
    wkv = [tensor_error(a["att"]["wkv"], b["att"]["wkv"])
           for a, b in zip(reference.rwkv, actual.rwkv)]
    shifts = [float((a["att"]["x"].float() - b["att"]["x"].float()).abs().max())
              for a, b in zip(reference.rwkv, actual.rwkv)]
    shifts += [float((a["ffn"].float() - b["ffn"].float()).abs().max())
               for a, b in zip(reference.rwkv, actual.rwkv)]
    return {"max_wkv_abs_diff": max(x["max_abs_diff"] for x in wkv),
            "max_wkv_relative_rms_error": max(x["relative_rms_error"] for x in wkv),
            "max_shift_abs_diff": max(shifts), "tokens_seen_equal": reference.tokens_seen == actual.tokens_seen,
            "wkv_storage_all_fp32": all(s["att"]["wkv"].dtype == torch.float32 for s in actual.rwkv)}


def address_agreement(model, ids):
    if not model.has_rosa:
        return None
    engine = model.address_engine
    full = engine.forward_seq(ids.cpu())
    state = engine.init_state(ids.shape[0])
    chunks, start = [], 0
    for size in partition(ids.shape[1], "uneven"):
        chunks.append(engine.forward_step(ids[:, start:start + size].cpu(), state))
        start += size
    fields = ("addr_ids", "raw_match_lens", "fired_match_lens", "valid_mask", "special_mask")
    matches = {key: torch.equal(getattr(full, key), torch.cat([getattr(c, key) for c in chunks], dim=1))
               for key in fields}
    if not all(matches.values()):
        raise AssertionError(f"ROSA address/state discrepancy, not numerical rounding: {matches}")
    return matches


def operator_probe(model, device):
    runtime = model.backbone._reference_runtime
    head = model.cfg.head_size
    generator = torch.Generator(device=device).manual_seed(97)
    xs = [torch.randn(1, 12, head, device=device, generator=generator) * .1 for _ in range(6)]
    xs[1] -= 2
    expected, expected_state = runtime._unfenced_wkv_op(*xs, return_state=True)
    result = {}
    for legacy in (True, False):
        tracer = MatmulDtypes()
        with wkv_policy(runtime, legacy=legacy), tracer, torch.autocast(device.type, dtype=torch.bfloat16):
            out, state = runtime.RWKV7_OP(*xs, return_state=True)
        calls = tracer.calls
        result["legacy_amp" if legacy else "fp32_wkv"] = {
            "matmul_calls": len(calls), "observed_dtypes": list({json.dumps(c, sort_keys=True) for c in calls}),
            "state_dtype": str(state.dtype), "output_error": tensor_error(expected, out),
            "state_error": tensor_error(expected_state, state)}
    return result


def trace_layers(model, ids, dtype):
    selected = {f"blocks.{i}" for i in range(model.cfg.n_layer)}
    selected |= {"blocks.0.att.receptance", "blocks.0.att.key", "blocks.0.att.value"}
    captured, handles = {}, []
    try:
        for name, module in model.backbone.named_modules():
            if name not in selected:
                continue
            def hook(module, inputs, output, name=name):
                value = output[0] if isinstance(output, tuple) else output
                captured.setdefault(name, []).append(value.detach())
            handles.append(module.register_forward_hook(hook))
        with autocast_context(ids.device, dtype):
            model(ids)
            full = {k: v[0] for k, v in captured.items()}
            captured.clear()
            stream_forward(model, ids, partition(ids.shape[1], "halves"))
        return {name: tensor_error(value, torch.cat(captured[name], dim=1)) for name, value in full.items()}
    finally:
        for handle in handles:
            handle.remove()


def make_cases(tokenizer, long_tokens):
    docs = load_documents(ROOT / "examples/rwkv_rosa_toy.json", tokenizer)
    first = docs[0].input_ids
    length = min(64, len(first) - 1)
    english = tokenizer.encode(("Record A17=603. Record B29=815. Repeat A17=603. Lookup A17=603.\n") * (long_tokens + 1))
    chinese = tokenizer.encode(("记录：编号甲17，数值603。编号乙29，数值815。请复述甲17的数值603。\n") * 40)
    chinese2 = tokenizer.encode(("记录：编号丙31，数值927。编号丁43，数值104。请复述丙31的数值927。\n") * 40)
    rows = [("sft64", [first[:length + 1]]), (f"repeat{long_tokens}", [english[:long_tokens + 1]]),
            ("chinese_batch2_64", [chinese[:65], chinese2[:65]])]
    return [(name, torch.tensor([r[:-1] for r in batch]), torch.tensor([r[1:] for r in batch]))
            for name, batch in rows]


@torch.no_grad()
def audit_model(model, cases, device):
    model.eval()
    result, full_by_policy = {}, {}
    result["operator_probe"] = operator_probe(model, device)
    for policy in ("legacy_amp", "fp32_wkv", "float32"):
        runtime = model.backbone._reference_runtime
        context = float32_model(model) if policy == "float32" else nullcontext()
        with context, wkv_policy(runtime, legacy=policy == "legacy_amp"):
            dtype = torch.float32 if policy == "float32" else torch.bfloat16
            measurements, full_by_policy[policy] = {}, {}
            for name, cpu_ids, targets in cases:
                ids = cpu_ids.to(device)
                with autocast_context(device, dtype):
                    full_out = model(ids, return_state=True)
                    full = full_out["logits"].float().cpu()
                    repeat = model(ids)["logits"].float().cpu()
                record = {"batch_size": ids.shape[0], "tokens": ids.shape[1],
                          "repeat_or_stateless_max_abs_diff": tensor_error(full, repeat)["max_abs_diff"],
                          "address_agreement": address_agreement(model, ids), "partitions": {}}
                for mode in ("halves", "uneven", "decode"):
                    with autocast_context(device, dtype):
                        out, state = stream_forward(model, ids, partition(ids.shape[1], mode))
                    stats = logits_error(full, out, targets)
                    stats["cache"] = cache_error(full_out["state"], state)
                    if policy == "float32" and not torch.allclose(full, out, atol=2e-3, rtol=2e-4):
                        raise AssertionError(f"FP32 streaming/state error: {name}/{mode}: {stats}")
                    record["partitions"][mode] = stats
                measurements[name] = record
                full_by_policy[policy][name] = full
                print(f"{model.variant}/{policy}/{name}: " + ", ".join(
                    f"{mode} flips={s['top1_changed']}/{s['predictions']} rms={s['relative_rms_error']:.4g}"
                    for mode, s in record["partitions"].items()), flush=True)
            if model.variant == "baseline":
                measurements["layer_trace_sft64_halves"] = trace_layers(model, cases[0][1].to(device), dtype)
            result[policy] = measurements
    result["against_fp32_full"] = {policy: {name: logits_error(full_by_policy["float32"][name], full_by_policy[policy][name], targets)
                                            for name, _, targets in cases}
                                   for policy in ("legacy_amp", "fp32_wkv")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--long-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.long_tokens < 2:
        parser.error("long-tokens must be >=2")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new precision audit output directory.")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This real BF16 checkpoint audit requires CUDA.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    state = load_base_state(args.checkpoint)
    cfg = infer_config(state)
    cases = make_cases(build_tokenizer("rwkv"), args.long_tokens)
    report = {"checkpoint": args.checkpoint, "checkpoint_sha256": file_hash(args.checkpoint),
              "config": asdict(cfg), "wkv_precision_policy": WKV_PRECISION_POLICY,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "reduced_precision_bf16_reduction": False, "tf32": False,
              "weights": "same raw pretrained checkpoint in every arm; no optimization updates",
              "cases": {name: {"input_ids": ids.tolist(), "targets": targets.tolist()} for name, ids, targets in cases},
              "runs": {}}
    for variant in ("baseline", "original", "improved"):
        seed_all(42)
        print(f"Building {variant}", flush=True)
        model = RWKVRosaLM(cfg, variant=variant, base_state=state, dtype=torch.bfloat16,
                           scale=.05, sparse_values=True, value_shards=4).to(device)
        configure_train_scope(model, "last_block")
        report["runs"][variant] = audit_model(model, cases, device)
        write_json(out / "precision_report.json", report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    main()
