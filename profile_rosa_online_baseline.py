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
    parser.add_argument("--rosa_context_gate", action="store_true")
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


def build_models(args, tokenizer, device: torch.device):
    cfg = rosa_mod.build_model_config(args, tokenizer)

    rosa_mod.set_seed(args.seed)
    baseline = rosa_mod.BaseLM(cfg)
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
        use_context_gate=args.rosa_context_gate,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        use_match_len_gate=not args.rosa_disable_match_len_gate,
    )

    maybe_load_checkpoint(baseline, args.baseline_ckpt)
    maybe_load_checkpoint(rosa_model, args.rosa_ckpt)

    baseline.to(device).eval()
    rosa_model.to(device).eval()
    return baseline, rosa_model


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
            lambda: rosa_model.forward_online(
                input_ids=input_ids,
                rosa_online_state=rosa_model.init_online_state(input_ids.size(0)),
            ),
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
        online_state = rosa_model.init_online_state(input_ids.size(0))
        online_address = online_state.address_tokens(input_ids, pad_id=pad_id, device=device)

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
        state = rosa_model.init_online_state(len(prefill_rows))
        address_state = rosa_model.init_online_state(len(prefill_rows))
        prefill_tensor = torch.tensor(prefill_rows, dtype=torch.long)
        state.prefill(prefill_tensor, pad_id=pad_id)
        address_state.prefill(prefill_tensor, pad_id=pad_id)
        last = None
        all_address = []
        for step_idx in range(len(decode_rows[0])):
            step_values = [[row[step_idx]] for row in decode_rows]
            step_ids = torch.tensor(step_values, dtype=torch.long, device=device)
            addressed = address_state.address_tokens(step_ids, pad_id=pad_id, device=device)
            last = rosa_model.forward_online(input_ids=step_ids, rosa_online_state=state)
            all_address.append(addressed)
        return last, all_address

    def run_prefetched_decode(prefill_rows, decode_rows):
        schedule_state = rosa_model.init_online_state(len(prefill_rows))
        prefill_tensor = torch.tensor(prefill_rows, dtype=torch.long)
        schedule_state.prefill(prefill_tensor, pad_id=pad_id)
        prefetcher = rosa_model.init_prefetcher(
            use_async=True,
            use_pinned_memory=use_pinned_prefetch,
            max_workers=1,
        )
        address_batches = []
        step_tensors = []
        for step_idx in range(len(decode_rows[0])):
            step_values = [[row[step_idx]] for row in decode_rows]
            step_ids = torch.tensor(step_values, dtype=torch.long, device=device)
            step_tensors.append(step_ids)
            address_batch = rosa_model.schedule_rosa_prefetch(
                prefetcher,
                f"step-{step_idx}",
                step_ids,
                rosa_online_state=schedule_state,
            )
            address_batches.append(address_batch)

        last = None
        all_address = []
        for step_idx, step_ids in enumerate(step_tensors):
            payload = rosa_model.consume_rosa_prefetch(
                prefetcher,
                f"step-{step_idx}",
                device=device,
                fallback_address_batch=address_batches[step_idx],
            )
            last = rosa_model.forward_prefetched(step_ids, payload)
            all_address.append(payload.address)
        stats = prefetcher.stats()
        prefetcher.shutdown()
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

    prefill_report = run_prefill_profile(
        baseline,
        rosa_model,
        batches,
        pad_id=tokenizer.pad_token_id,
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )
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
            "baseline_ckpt": args.baseline_ckpt,
            "rosa_ckpt": args.rosa_ckpt,
            "rosa_backend": args.rosa_backend,
            "rosa_min_match_len": args.rosa_min_match_len,
            "rosa_inject_layers": args.rosa_inject_layers,
            "rosa_inject_layer_ids": rosa_mod.parse_int_csv_arg(args.rosa_inject_layer_ids),
            "rosa_scale": args.rosa_scale,
            "rosa_value_mode": args.rosa_value_mode,
            "rosa_context_gate": args.rosa_context_gate,
            "rosa_prefetch": args.rosa_prefetch,
            "rosa_prefetch_pinned": args.rosa_prefetch_pinned,
        },
        "prefill": prefill_report,
        "decode_micro": decode_report,
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
    print(f"report: {Path(report['meta'].get('out_dir', 'outputs')).resolve()}")


def run_profile(args) -> Dict[str, Any]:
    report = build_report(args)
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
