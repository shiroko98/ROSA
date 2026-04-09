import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为 ROSA online/reference 路径建立性能与正确性基线。"
    )
    parser.add_argument("--data_path", type=str, required=True, help="用于 profiling 的数据路径。")
    parser.add_argument("--rosa_recipe", type=str, default="custom", choices=rosa_mod.available_rosa_recipe_names())
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["paragraph", "line", "stream"])
    parser.add_argument("--data_format", type=str, default="auto", choices=["auto", "text", "jsonl", "json"])
    parser.add_argument("--json_text_keys", type=str, default="text,content,body,message")
    parser.add_argument("--max_docs", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--prefill_tokens", type=int, default=64)
    parser.add_argument("--decode_steps", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=3)
    parser.add_argument("--train_consistency_samples", type=int, default=8)
    parser.add_argument("--train_consistency_stride", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda；为空时自动选择。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch_style", type=str, default="qwen", choices=["llama", "qwen"])
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_tie_word_embeddings", action="store_true")
    parser.add_argument("--rosa_min_match_len", type=int, default=2)
    parser.add_argument("--rosa_inject_layers", type=int, default=1)
    parser.add_argument("--rosa_inject_layer_ids", type=str, default="")
    parser.add_argument("--rosa_scale", type=float, default=0.15)
    parser.add_argument("--rosa_value_mode", type=str, default="shared", choices=["shared", "per_layer"])
    parser.add_argument("--rosa_seq_address_mode", type=str, default="reference_backend",
                        choices=["reference_backend", "online_exact", "online_sam"])
    parser.add_argument("--rosa_online_sam_impl", type=str, default="fast", choices=["fast", "compiled_cpu", "stateful"])
    parser.add_argument("--enable_rosa_train_address_cache", action="store_true")
    parser.add_argument("--disable_rosa_train_address_cache", action="store_true")
    parser.add_argument("--rosa_context_gate", action="store_true")
    parser.add_argument("--rosa_hot_cache_size", type=int, default=0)
    parser.add_argument("--rosa_prefetch", action="store_true")
    parser.add_argument("--rosa_prefetch_pinned", action="store_true")
    parser.add_argument("--rosa_backend", type=str, default="sam", choices=["sam", "naive"])
    parser.add_argument("--rosa_allow_special_target", action="store_true")
    parser.add_argument("--rosa_disable_match_len_gate", action="store_true")
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--rosa_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/profile_rosa_online_baseline")
    return parser


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_ms(values_s: Sequence[float], q: float) -> float:
    if not values_s:
        return 0.0
    ordered = sorted(values_s)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[idx] * 1000.0


def summarize_timings(values_s: Sequence[float], token_counts: Sequence[int]) -> Dict[str, float]:
    if not values_s:
        return {
            "runs": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "tokens_per_s": 0.0,
        }
    total_s = sum(values_s)
    total_tokens = sum(token_counts)
    return {
        "runs": len(values_s),
        "avg_ms": total_s * 1000.0 / len(values_s),
        "p50_ms": percentile_ms(values_s, 0.50),
        "p95_ms": percentile_ms(values_s, 0.95),
        "tokens_per_s": total_tokens / total_s if total_s > 0 else 0.0,
    }


def average_metric(rows: Sequence[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def compare_tensor_dicts(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in left.keys():
        if key not in right:
            continue
        if left[key].numel() == 0:
            out[f"{key}_eq_ratio"] = 1.0
            continue
        out[f"{key}_eq_ratio"] = left[key].eq(right[key]).float().mean().item()
    out["all_equal"] = 1.0 if all(v == 1.0 for v in out.values()) else 0.0
    return out


def summarize_address_tensors(addressed: Dict[str, torch.Tensor]) -> Dict[str, float]:
    valid_mask = addressed["valid_mask"]
    raw_match_lens = addressed["raw_match_lens"]
    fired_match_lens = addressed["fired_match_lens"]
    raw_has_match = raw_match_lens.gt(0)
    return {
        "fire_coverage": valid_mask.float().mean().item() if valid_mask.numel() > 0 else 0.0,
        "raw_match_coverage": raw_has_match.float().mean().item() if raw_has_match.numel() > 0 else 0.0,
        "avg_fired_match_len": fired_match_lens[valid_mask].float().mean().item() if valid_mask.any() else 0.0,
        "avg_raw_match_len": raw_match_lens[raw_has_match].float().mean().item() if raw_has_match.any() else 0.0,
    }


def extract_scalar_rosa_stats(output: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in output.items():
        if not key.startswith("rosa_"):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def iter_batches(rows: Sequence[Dict[str, List[int]]], batch_size: int) -> List[List[Dict[str, List[int]]]]:
    return [list(rows[i:i + batch_size]) for i in range(0, len(rows), batch_size)]


def select_profile_samples(
    docs_tokens: Sequence[Sequence[int]],
    *,
    prefill_tokens: int,
    decode_steps: int,
    num_samples: int,
    stride: int,
) -> List[Dict[str, List[int]]]:
    need = prefill_tokens + decode_steps
    samples: List[Dict[str, List[int]]] = []
    step = max(1, stride)
    for ids in docs_tokens:
        if len(ids) < need:
            continue
        max_start = len(ids) - need
        for start in range(0, max_start + 1, step):
            window = list(ids[start:start + need])
            samples.append(
                {
                    "prefill": window[:prefill_tokens],
                    "decode": window[prefill_tokens:prefill_tokens + decode_steps],
                }
            )
            if len(samples) >= num_samples:
                return samples
    return samples


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        return payload
    raise ValueError(f"无法从 checkpoint 解析 state_dict: {path}")


def maybe_load_checkpoint(model: torch.nn.Module, path: Optional[str]) -> None:
    if not path:
        return
    state_dict = load_state_dict(path)
    model.load_state_dict(state_dict)


def build_rosa_model(args, tokenizer, device: torch.device, *, seq_address_mode: Optional[str] = None):
    cfg = rosa_mod.build_model_config(args, tokenizer)
    rosa_mod.set_seed(args.seed)
    rosa_model = rosa_mod.RosaFusedLM(
        cfg,
        pad_id=tokenizer.pad_token_id,
        rosa_backend=args.rosa_backend,
        min_match_len=args.rosa_min_match_len,
        inject_layers=args.rosa_inject_layers,
        inject_layer_ids=rosa_mod.parse_int_csv_arg(args.rosa_inject_layer_ids),
        rosa_scale=args.rosa_scale,
        rosa_value_mode=args.rosa_value_mode,
        rosa_seq_address_mode=seq_address_mode or args.rosa_seq_address_mode,
        rosa_online_sam_impl=args.rosa_online_sam_impl,
        use_context_gate=args.rosa_context_gate,
        rosa_hot_cache_size=args.rosa_hot_cache_size,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        use_match_len_gate=not args.rosa_disable_match_len_gate,
    )
    maybe_load_checkpoint(rosa_model, args.rosa_ckpt)
    return rosa_model.to(device).eval()


def build_models(args, tokenizer, device: torch.device):
    cfg = rosa_mod.build_model_config(args, tokenizer)

    rosa_mod.set_seed(args.seed)
    baseline = rosa_mod.BaseLM(cfg)
    rosa_model = build_rosa_model(args, tokenizer, device)

    maybe_load_checkpoint(baseline, args.baseline_ckpt)

    baseline.to(device).eval()
    return baseline, rosa_model


def address_batch_to_dict(address_batch) -> Dict[str, torch.Tensor]:
    if isinstance(address_batch, dict):
        return {
            "addr_ids": address_batch["addr_ids"],
            "raw_match_lens": address_batch["raw_match_lens"],
            "fired_match_lens": address_batch["fired_match_lens"],
            "valid_mask": address_batch["valid_mask"],
            "special_mask": address_batch["special_mask"],
        }
    return {
        "addr_ids": address_batch.addr_ids,
        "raw_match_lens": address_batch.raw_match_lens,
        "fired_match_lens": address_batch.fired_match_lens,
        "valid_mask": address_batch.valid_mask,
        "special_mask": address_batch.special_mask,
    }


def run_train_path_consistency_profile(
    args,
    tokenizer,
    docs_tokens: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    if args.rosa_backend != "sam":
        return {
            "enabled": False,
            "reason": "train_path_consistency 仅在 sam backend 下支持 reference_precompute 对照。",
        }

    stride = args.train_consistency_stride or args.seq_len
    sample_cap = max(1, args.train_consistency_samples)
    online_ds, _, _, online_meta = rosa_mod.build_chunk_datasets(
        docs_tokens,
        docs_tokens,
        docs_tokens,
        seq_len=args.seq_len,
        pad_id=tokenizer.pad_token_id,
        stride=stride,
        rosa_memory_tokens=args.prefill_tokens,
        rosa_memory_mode="doc_local",
        rosa_global_memory_tokens=0,
        rosa_backend=args.rosa_backend,
        rosa_train_mode="online_seq",
        rosa_seq_address_mode=args.rosa_seq_address_mode,
        rosa_online_sam_impl=args.rosa_online_sam_impl,
        rosa_min_match_len=args.rosa_min_match_len,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        enable_train_address_cache=(
            getattr(args, "enable_rosa_train_address_cache", False)
            and not getattr(args, "disable_rosa_train_address_cache", False)
        ),
    )
    reference_ds, _, _, reference_meta = rosa_mod.build_chunk_datasets(
        docs_tokens,
        docs_tokens,
        docs_tokens,
        seq_len=args.seq_len,
        pad_id=tokenizer.pad_token_id,
        stride=stride,
        rosa_memory_tokens=args.prefill_tokens,
        rosa_memory_mode="doc_local",
        rosa_global_memory_tokens=0,
        rosa_backend=args.rosa_backend,
        rosa_train_mode="reference_precompute",
        rosa_seq_address_mode=args.rosa_seq_address_mode,
        rosa_online_sam_impl=args.rosa_online_sam_impl,
        rosa_min_match_len=args.rosa_min_match_len,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        enable_train_address_cache=(
            getattr(args, "enable_rosa_train_address_cache", False)
            and not getattr(args, "disable_rosa_train_address_cache", False)
        ),
    )
    if len(online_ds) != len(reference_ds):
        raise ValueError(
            f"在线训练数据集与 reference_precompute 数据集样本数不一致: {len(online_ds)} vs {len(reference_ds)}"
        )

    collate = rosa_mod.make_collate_fn(tokenizer.pad_token_id)
    online_model = build_rosa_model(args, tokenizer, device, seq_address_mode="online_exact")
    reference_model = build_rosa_model(args, tokenizer, device, seq_address_mode="online_exact")

    address_cmp_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    online_stats_rows: List[Dict[str, float]] = []
    reference_stats_rows: List[Dict[str, float]] = []
    logit_diffs: List[float] = []
    loss_diffs: List[float] = []

    for start in range(0, min(len(online_ds), sample_cap), max(1, args.batch_size)):
        stop = min(start + max(1, args.batch_size), min(len(online_ds), sample_cap))
        online_batch_cpu = collate([online_ds[idx] for idx in range(start, stop)])
        reference_batch_cpu = collate([reference_ds[idx] for idx in range(start, stop)])

        if not torch.equal(online_batch_cpu["input_ids"], reference_batch_cpu["input_ids"]):
            raise ValueError("online_seq 与 reference_precompute 的 input_ids 不一致。")
        if not torch.equal(online_batch_cpu["labels"], reference_batch_cpu["labels"]):
            raise ValueError("online_seq 与 reference_precompute 的 labels 不一致。")

        input_ids = online_batch_cpu["input_ids"].to(device)
        labels = online_batch_cpu["labels"].to(device)
        online_mem = online_batch_cpu["rosa_memory_ids"].to(device)
        reference_mem = reference_batch_cpu["rosa_memory_ids"].to(device)
        online_pre_ids = online_batch_cpu.get("rosa_precomputed_ids")
        online_pre_match = online_batch_cpu.get("rosa_precomputed_match_lens")
        online_pre_raw = online_batch_cpu.get("rosa_precomputed_raw_best_lens")
        online_pre_source = online_batch_cpu.get("rosa_precomputed_source")
        reference_pre_ids = reference_batch_cpu["rosa_precomputed_ids"].to(device)
        reference_pre_match = reference_batch_cpu["rosa_precomputed_match_lens"].to(device)
        reference_pre_raw = reference_batch_cpu["rosa_precomputed_raw_best_lens"].to(device)

        with torch.no_grad():
            online_address = online_model.compute_rosa_address_batch(
                input_ids,
                rosa_memory_ids=online_mem,
                rosa_precomputed_ids=online_pre_ids.to(device) if online_pre_ids is not None else None,
                rosa_precomputed_match_lens=online_pre_match.to(device) if online_pre_match is not None else None,
                rosa_precomputed_raw_best_lens=online_pre_raw.to(device) if online_pre_raw is not None else None,
                rosa_precomputed_source=online_pre_source,
            )
            reference_address = reference_model.compute_rosa_address_batch(
                input_ids,
                rosa_memory_ids=reference_mem,
                rosa_precomputed_ids=reference_pre_ids,
                rosa_precomputed_match_lens=reference_pre_match,
                rosa_precomputed_raw_best_lens=reference_pre_raw,
            )
            online_out = online_model(
                input_ids=input_ids,
                labels=labels,
                rosa_memory_ids=online_mem,
                rosa_precomputed_ids=online_pre_ids.to(device) if online_pre_ids is not None else None,
                rosa_precomputed_match_lens=online_pre_match.to(device) if online_pre_match is not None else None,
                rosa_precomputed_raw_best_lens=online_pre_raw.to(device) if online_pre_raw is not None else None,
                rosa_precomputed_source=online_pre_source,
            )
            reference_out = reference_model(
                input_ids=input_ids,
                labels=labels,
                rosa_memory_ids=reference_mem,
                rosa_precomputed_ids=reference_pre_ids,
                rosa_precomputed_match_lens=reference_pre_match,
                rosa_precomputed_raw_best_lens=reference_pre_raw,
            )

        reference_dict = address_batch_to_dict(reference_address)
        online_dict = address_batch_to_dict(online_address)
        address_cmp_rows.append(compare_tensor_dicts(reference_dict, online_dict))
        coverage_rows.append(summarize_address_tensors(reference_dict))
        online_stats_rows.append(extract_scalar_rosa_stats(online_out))
        reference_stats_rows.append(extract_scalar_rosa_stats(reference_out))
        logit_diffs.append(torch.max(torch.abs(reference_out["logits"] - online_out["logits"])).item())
        loss_diffs.append(abs(float(reference_out["loss"].item()) - float(online_out["loss"].item())))

    return {
        "enabled": True,
        "samples_compared": min(len(online_ds), sample_cap),
        "online_memory_meta": online_meta,
        "reference_memory_meta": reference_meta,
        "correctness": {
            "address_agreement": {
                key: average_metric(address_cmp_rows, key)
                for key in (address_cmp_rows[0].keys() if address_cmp_rows else [])
            },
            "logit_max_abs_diff": max(logit_diffs) if logit_diffs else 0.0,
            "loss_max_abs_diff": max(loss_diffs) if loss_diffs else 0.0,
        },
        "coverage": {
            key: average_metric(coverage_rows, key)
            for key in (coverage_rows[0].keys() if coverage_rows else [])
        },
        "online_model_stats": {
            key: average_metric(online_stats_rows, key)
            for key in (online_stats_rows[0].keys() if online_stats_rows else [])
        },
        "reference_model_stats": {
            key: average_metric(reference_stats_rows, key)
            for key in (reference_stats_rows[0].keys() if reference_stats_rows else [])
        },
    }


def bench_callable(fn, *, warmup_iters: int, measure_iters: int, device: torch.device):
    last = None
    for _ in range(max(0, warmup_iters)):
        with torch.no_grad():
            last = fn()
        sync_device(device)

    timings: List[float] = []
    for _ in range(max(1, measure_iters)):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            last = fn()
        sync_device(device)
        timings.append(time.perf_counter() - t0)
    return last, timings


def run_prefill_profile(
    baseline: torch.nn.Module,
    rosa_model: rosa_mod.RosaFusedLM,
    batches: Sequence[Sequence[Dict[str, List[int]]]],
    *,
    pad_id: int,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
) -> Dict[str, Any]:
    baseline_times: List[float] = []
    reference_times: List[float] = []
    online_times: List[float] = []
    baseline_tokens: List[int] = []
    reference_tokens: List[int] = []
    online_tokens: List[int] = []
    logit_max_abs_diff: List[float] = []
    address_cmp_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    model_stats_rows: List[Dict[str, float]] = []

    for batch in batches:
        input_ids = torch.tensor([row["prefill"] for row in batch], dtype=torch.long, device=device)
        empty_mem = torch.empty((input_ids.size(0), 0), dtype=torch.long, device=device)
        token_count = int(input_ids.numel())

        def run_online_prefill():
            session = rosa_model.init_online_session(input_ids.size(0))
            try:
                return session.prefill_seq(input_ids)
            finally:
                session.close()

        _, base_times = bench_callable(
            lambda: baseline(input_ids=input_ids),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        reference_out, ref_times = bench_callable(
            lambda: rosa_model(input_ids=input_ids, rosa_memory_ids=empty_mem),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        online_out, on_times = bench_callable(
            run_online_prefill,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )

        reference_address = rosa_mod.rosa_addressing_with_memory(
            input_ids=input_ids,
            memory_ids=empty_mem,
            min_match_len=rosa_model.min_match_len,
            pad_id=pad_id,
            special_ids=rosa_model.special_ids,
            forbid_special_target=rosa_model.forbid_special_target,
            backend=rosa_model.rosa_backend,
        )
        online_session = rosa_model.init_online_session(input_ids.size(0))
        online_session.prefill_seq(input_ids)
        online_address = address_batch_to_dict(online_session.last_payload.address)

        baseline_times.extend(base_times)
        reference_times.extend(ref_times)
        online_times.extend(on_times)
        baseline_tokens.extend([token_count] * len(base_times))
        reference_tokens.extend([token_count] * len(ref_times))
        online_tokens.extend([token_count] * len(on_times))
        logit_max_abs_diff.append(torch.max(torch.abs(reference_out["logits"] - online_out["logits"])).item())
        address_cmp_rows.append(compare_tensor_dicts(reference_address, online_address))
        coverage_rows.append(summarize_address_tensors(reference_address))
        model_stats_rows.append(extract_scalar_rosa_stats(reference_out))

    return {
        "timings": {
            "baseline": summarize_timings(baseline_times, baseline_tokens),
            "rosa_reference": summarize_timings(reference_times, reference_tokens),
            "rosa_online": summarize_timings(online_times, online_tokens),
        },
        "correctness": {
            "logit_max_abs_diff": max(logit_max_abs_diff) if logit_max_abs_diff else 0.0,
            "address_agreement": {
                key: average_metric(address_cmp_rows, key)
                for key in (address_cmp_rows[0].keys() if address_cmp_rows else [])
            },
        },
        "coverage": {
            key: average_metric(coverage_rows, key)
            for key in (coverage_rows[0].keys() if coverage_rows else [])
        },
        "model_stats": {
            key: average_metric(model_stats_rows, key)
            for key in (model_stats_rows[0].keys() if model_stats_rows else [])
        },
    }


def run_decode_micro_profile(
    baseline: torch.nn.Module,
    rosa_model: rosa_mod.RosaFusedLM,
    batches: Sequence[Sequence[Dict[str, List[int]]]],
    *,
    pad_id: int,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
    use_prefetch: bool = False,
    use_pinned_prefetch: bool = False,
) -> Dict[str, Any]:
    baseline_times: List[float] = []
    reference_times: List[float] = []
    online_times: List[float] = []
    baseline_tokens: List[int] = []
    reference_tokens: List[int] = []
    online_tokens: List[int] = []
    prefetch_times: List[float] = []
    prefetch_tokens: List[int] = []
    logit_max_abs_diff: List[float] = []
    address_cmp_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    prefetch_stats_rows: List[Dict[str, float]] = []
    model_stats_rows: List[Dict[str, float]] = []

    def run_baseline_decode(prefill_rows, decode_rows):
        last = None
        for step_idx in range(len(decode_rows[0])):
            step_ids = torch.tensor([[row[step_idx]] for row in decode_rows], dtype=torch.long, device=device)
            last = baseline(input_ids=step_ids)
        return last

    def run_reference_decode(prefill_rows, decode_rows):
        histories = [list(row) for row in prefill_rows]
        last = None
        all_address = []
        for step_idx in range(len(decode_rows[0])):
            step_values = [[row[step_idx]] for row in decode_rows]
            step_ids = torch.tensor(step_values, dtype=torch.long, device=device)
            mem = torch.tensor(histories, dtype=torch.long, device=device)
            last = rosa_model(input_ids=step_ids, rosa_memory_ids=mem)
            addressed = rosa_mod.rosa_addressing_with_memory(
                input_ids=step_ids,
                memory_ids=mem,
                min_match_len=rosa_model.min_match_len,
                pad_id=pad_id,
                special_ids=rosa_model.special_ids,
                forbid_special_target=rosa_model.forbid_special_target,
                backend=rosa_model.rosa_backend,
            )
            all_address.append(addressed)
            for hist, row in zip(histories, decode_rows):
                hist.append(row[step_idx])
        return last, all_address

    def run_online_decode(prefill_rows, decode_rows):
        session = rosa_model.init_online_session(len(prefill_rows))
        prefill_tensor = torch.tensor(prefill_rows, dtype=torch.long, device=device)
        session.prefill_seq(prefill_tensor)
        last = None
        all_address = []
        for step_idx in range(len(decode_rows[0])):
            step_values = [[row[step_idx]] for row in decode_rows]
            step_ids = torch.tensor(step_values, dtype=torch.long, device=device)
            last = session.decode_step(step_ids)
            all_address.append(session.last_payload.address)
        return last, all_address

    def run_prefetched_decode(prefill_rows, decode_rows):
        session = rosa_model.init_online_session(
            len(prefill_rows),
            use_prefetch=True,
            use_pinned_prefetch=use_pinned_prefetch,
            prefetch_workers=1,
        )
        prefill_tensor = torch.tensor(prefill_rows, dtype=torch.long, device=device)
        session.prefill_seq(prefill_tensor)
        address_batches = []
        last = None
        all_address = []
        for step_idx in range(len(decode_rows[0])):
            step_values = [[row[step_idx]] for row in decode_rows]
            step_ids = torch.tensor(step_values, dtype=torch.long, device=device)
            request_key = session.schedule_decode_step(
                step_ids,
                request_key=f"step-{step_idx}",
            )
            address_batches.append((step_ids, request_key))

        for step_ids, request_key in address_batches:
            last = session.decode_step(step_ids, request_key=request_key)
            all_address.append(session.last_payload.address)
        stats = session.prefetch_stats()
        session.close()
        return last, all_address, stats

    for batch in batches:
        prefill_rows = [row["prefill"] for row in batch]
        decode_rows = [row["decode"] for row in batch]
        token_count = len(batch) * len(decode_rows[0])

        _, base_times = bench_callable(
            lambda: run_baseline_decode(prefill_rows, decode_rows),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        (reference_out, reference_addresses), ref_times = bench_callable(
            lambda: run_reference_decode(prefill_rows, decode_rows),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        (online_out, online_addresses), on_times = bench_callable(
            lambda: run_online_decode(prefill_rows, decode_rows),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        if use_prefetch:
            (prefetch_out, prefetch_addresses, prefetch_stats), pf_times = bench_callable(
                lambda: run_prefetched_decode(prefill_rows, decode_rows),
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
                device=device,
            )
        else:
            prefetch_out, prefetch_addresses, prefetch_stats, pf_times = None, None, None, []

        baseline_times.extend(base_times)
        reference_times.extend(ref_times)
        online_times.extend(on_times)
        prefetch_times.extend(pf_times)
        baseline_tokens.extend([token_count] * len(base_times))
        reference_tokens.extend([token_count] * len(ref_times))
        online_tokens.extend([token_count] * len(on_times))
        prefetch_tokens.extend([token_count] * len(pf_times))
        logit_max_abs_diff.append(torch.max(torch.abs(reference_out["logits"] - online_out["logits"])).item())
        if use_prefetch and prefetch_out is not None:
            logit_max_abs_diff.append(torch.max(torch.abs(reference_out["logits"] - prefetch_out["logits"])).item())
        model_stats_rows.append(extract_scalar_rosa_stats(reference_out))

        for ref_addr, on_addr in zip(reference_addresses, online_addresses):
            ref_dict = {
                "addr_ids": ref_addr["addr_ids"] if isinstance(ref_addr, dict) else ref_addr.addr_ids,
                "raw_match_lens": ref_addr["raw_match_lens"] if isinstance(ref_addr, dict) else ref_addr.raw_match_lens,
                "fired_match_lens": ref_addr["fired_match_lens"] if isinstance(ref_addr, dict) else ref_addr.fired_match_lens,
                "valid_mask": ref_addr["valid_mask"] if isinstance(ref_addr, dict) else ref_addr.valid_mask,
                "special_mask": ref_addr["special_mask"] if isinstance(ref_addr, dict) else ref_addr.special_mask,
            }
            on_dict = {
                "addr_ids": on_addr["addr_ids"] if isinstance(on_addr, dict) else on_addr.addr_ids,
                "raw_match_lens": on_addr["raw_match_lens"] if isinstance(on_addr, dict) else on_addr.raw_match_lens,
                "fired_match_lens": on_addr["fired_match_lens"] if isinstance(on_addr, dict) else on_addr.fired_match_lens,
                "valid_mask": on_addr["valid_mask"] if isinstance(on_addr, dict) else on_addr.valid_mask,
                "special_mask": on_addr["special_mask"] if isinstance(on_addr, dict) else on_addr.special_mask,
            }
            address_cmp_rows.append(compare_tensor_dicts(ref_dict, on_dict))
            coverage_rows.append(summarize_address_tensors(ref_addr))
        if use_prefetch and prefetch_addresses is not None:
            prefetch_stats_rows.append(prefetch_stats)

    timings = {
        "baseline": summarize_timings(baseline_times, baseline_tokens),
        "rosa_reference": summarize_timings(reference_times, reference_tokens),
        "rosa_online": summarize_timings(online_times, online_tokens),
    }
    if use_prefetch:
        timings["rosa_online_prefetch"] = summarize_timings(prefetch_times, prefetch_tokens)
    for mode in timings.values():
        mode["avg_ms_per_token"] = 1000.0 / mode["tokens_per_s"] if mode["tokens_per_s"] > 0 else 0.0

    report = {
        "timings": timings,
        "correctness": {
            "logit_max_abs_diff": max(logit_max_abs_diff) if logit_max_abs_diff else 0.0,
            "address_agreement": {
                key: average_metric(address_cmp_rows, key)
                for key in (address_cmp_rows[0].keys() if address_cmp_rows else [])
            },
        },
        "coverage": {
            key: average_metric(coverage_rows, key)
            for key in (coverage_rows[0].keys() if coverage_rows else [])
        },
        "model_stats": {
            key: average_metric(model_stats_rows, key)
            for key in (model_stats_rows[0].keys() if model_stats_rows else [])
        },
        "notes": "该 decode 指标是无 KV cache 的单步 microbenchmark，主要用于比较 ROSA 分支开销与 online/reference 一致性。",
    }
    if use_prefetch and prefetch_stats_rows:
        report["prefetch"] = {
            key: average_metric(prefetch_stats_rows, key)
            for key in prefetch_stats_rows[0].keys()
        }
    return report


def build_report(args) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    args.seq_len = max(args.seq_len, args.prefill_tokens)

    rosa_mod.set_seed(args.seed)
    tokenizer = rosa_mod.build_tokenizer(args.tokenizer_name_or_path)
    docs, source_meta = rosa_mod.load_docs_from_path(
        args.data_path,
        data_format=args.data_format,
        split_mode=args.split_mode,
        json_text_keys=rosa_mod.parse_csv_arg(args.json_text_keys),
        max_docs=args.max_docs,
    )
    docs_tokens = rosa_mod.tokenize_docs(docs, tokenizer, add_bos=True, add_eos=True)
    samples = select_profile_samples(
        docs_tokens,
        prefill_tokens=args.prefill_tokens,
        decode_steps=args.decode_steps,
        num_samples=args.num_samples,
        stride=args.sample_stride,
    )
    if not samples:
        raise ValueError(
            f"没有足够长的样本可用于 profiling。至少需要 {args.prefill_tokens + args.decode_steps} 个 token。"
        )

    batches = iter_batches(samples, max(1, args.batch_size))
    baseline, rosa_model = build_models(args, tokenizer, device)

    rosa_model.reset_hot_cache(clear_cache=True)
    prefill_report = run_prefill_profile(
        baseline,
        rosa_model,
        batches,
        pad_id=tokenizer.pad_token_id,
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )
    prefill_hot_cache_report = rosa_model.get_hot_cache_stats(top_k=5)
    rosa_model.reset_hot_cache(clear_cache=True)
    decode_report = run_decode_micro_profile(
        baseline,
        rosa_model,
        batches,
        pad_id=tokenizer.pad_token_id,
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        use_prefetch=args.rosa_prefetch,
        use_pinned_prefetch=args.rosa_prefetch_pinned,
    )
    decode_hot_cache_report = rosa_model.get_hot_cache_stats(top_k=5)
    train_path_report = run_train_path_consistency_profile(
        args,
        tokenizer,
        docs_tokens,
        device=device,
    )

    report = {
        "meta": {
            "device": str(device),
            "data_path": args.data_path,
            "data_source": source_meta,
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "num_docs_loaded": len(docs),
            "num_samples_profiled": len(samples),
            "batch_size": args.batch_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_steps": args.decode_steps,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "train_consistency_samples": args.train_consistency_samples,
            "train_consistency_stride": args.train_consistency_stride or args.seq_len,
            "baseline_ckpt": args.baseline_ckpt,
            "rosa_ckpt": args.rosa_ckpt,
            "rosa_backend": args.rosa_backend,
            "rosa_min_match_len": args.rosa_min_match_len,
            "rosa_inject_layers": args.rosa_inject_layers,
            "rosa_inject_layer_ids": rosa_mod.parse_int_csv_arg(args.rosa_inject_layer_ids),
            "rosa_scale": args.rosa_scale,
            "rosa_value_mode": args.rosa_value_mode,
            "rosa_seq_address_mode": args.rosa_seq_address_mode,
            "rosa_online_sam_impl": args.rosa_online_sam_impl,
            "rosa_context_gate": args.rosa_context_gate,
            "rosa_hot_cache_size": args.rosa_hot_cache_size,
            "rosa_prefetch": args.rosa_prefetch,
            "rosa_prefetch_pinned": args.rosa_prefetch_pinned,
        },
        "prefill": prefill_report,
        "decode_micro": decode_report,
        "train_path_consistency": train_path_report,
        "hot_cache": {
            "enabled": bool(prefill_hot_cache_report.get("enabled") or decode_hot_cache_report.get("enabled")),
            "prefill": prefill_hot_cache_report,
            "decode_micro": decode_hot_cache_report,
        },
    }

    ref_prefill = prefill_report["timings"]["rosa_reference"]["avg_ms"]
    online_prefill = prefill_report["timings"]["rosa_online"]["avg_ms"]
    ref_decode = decode_report["timings"]["rosa_reference"]["avg_ms"]
    online_decode = decode_report["timings"]["rosa_online"]["avg_ms"]
    report["summary"] = {
        "prefill_online_vs_reference_speedup": ref_prefill / online_prefill if online_prefill > 0 else 0.0,
        "decode_online_vs_reference_speedup": ref_decode / online_decode if online_decode > 0 else 0.0,
        "prefill_online_logit_max_abs_diff": prefill_report["correctness"]["logit_max_abs_diff"],
        "decode_online_logit_max_abs_diff": decode_report["correctness"]["logit_max_abs_diff"],
        "prefill_hot_cache_token_hit_rate": float(prefill_hot_cache_report.get("token_hit_rate", 0.0)),
        "prefill_hot_cache_unique_hit_rate": float(prefill_hot_cache_report.get("unique_hit_rate", 0.0)),
        "decode_hot_cache_token_hit_rate": float(decode_hot_cache_report.get("token_hit_rate", 0.0)),
        "decode_hot_cache_unique_hit_rate": float(decode_hot_cache_report.get("unique_hit_rate", 0.0)),
        "train_path_address_agreement": float(
            train_path_report.get("correctness", {}).get("address_agreement", {}).get("all_equal", 0.0)
        ),
        "train_path_logit_max_abs_diff": float(
            train_path_report.get("correctness", {}).get("logit_max_abs_diff", 0.0)
        ),
    }

    save_json(report, out_dir / "profile_report.json")
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("=== ROSA Online Baseline Report ===")
    print(f"device: {report['meta']['device']}")
    print(f"samples: {report['meta']['num_samples_profiled']} | batch_size: {report['meta']['batch_size']}")
    print(
        "prefill avg ms | baseline {0:.3f} | ref {1:.3f} | online {2:.3f}".format(
            report["prefill"]["timings"]["baseline"]["avg_ms"],
            report["prefill"]["timings"]["rosa_reference"]["avg_ms"],
            report["prefill"]["timings"]["rosa_online"]["avg_ms"],
        )
    )
    print(
        "decode avg ms | baseline {0:.3f} | ref {1:.3f} | online {2:.3f}".format(
            report["decode_micro"]["timings"]["baseline"]["avg_ms"],
            report["decode_micro"]["timings"]["rosa_reference"]["avg_ms"],
            report["decode_micro"]["timings"]["rosa_online"]["avg_ms"],
        )
    )
    train_path = report.get("train_path_consistency", {})
    if train_path.get("enabled"):
        print(
            "train path | address agreement {0:.4f} | logit diff {1:.6f}".format(
                train_path["correctness"]["address_agreement"].get("all_equal", 0.0),
                train_path["correctness"]["logit_max_abs_diff"],
            )
        )
    print(
        "address agreement | prefill {0:.4f} | decode {1:.4f}".format(
            report["prefill"]["correctness"]["address_agreement"].get("all_equal", 0.0),
            report["decode_micro"]["correctness"]["address_agreement"].get("all_equal", 0.0),
        )
    )
    print(
        "logit diff | prefill {0:.6f} | decode {1:.6f}".format(
            report["summary"]["prefill_online_logit_max_abs_diff"],
            report["summary"]["decode_online_logit_max_abs_diff"],
        )
    )
    if report.get("hot_cache", {}).get("enabled"):
        print(
            "hot cache | prefill token_hit {0:.4f} | decode token_hit {1:.4f}".format(
                report["hot_cache"]["prefill"].get("token_hit_rate", 0.0),
                report["hot_cache"]["decode_micro"].get("token_hit_rate", 0.0),
            )
        )
    print(f"report: {Path(report['meta'].get('out_dir', 'outputs')).resolve()}")


def run_profile(args) -> Dict[str, Any]:
    recipe_meta = rosa_mod.apply_rosa_recipe(args)
    report = build_report(args)
    report["meta"]["recipe"] = recipe_meta
    report["meta"]["out_dir"] = str(Path(args.out_dir).resolve())
    save_json(report, Path(args.out_dir) / "profile_report.json")
    print_summary(report)
    return report


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
