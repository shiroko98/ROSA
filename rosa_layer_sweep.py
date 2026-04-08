import argparse
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import profile_rosa_online_baseline as profile_mod
import train_qwen_llama_vs_rosa_v2 as rosa_mod
from rosa_recipes import apply_rosa_recipe


@dataclass
class SweepTrainingContext:
    tokenizer: Any
    train_ds: Any
    val_ds: Any
    test_ds: Any
    memory_meta: Dict[str, Any]
    dataset_source: Dict[str, Any]
    total_docs: int


def parse_custom_layer_sets(text: str) -> List[Tuple[int, ...]]:
    if not text.strip():
        return []
    out: List[Tuple[int, ...]] = []
    for chunk in text.split(";"):
        ids = tuple(rosa_mod.parse_int_csv_arg(chunk))
        if ids:
            out.append(ids)
    return out


def build_layer_sets(
    *,
    scan_mode: str,
    layer_candidates: Sequence[int],
    custom_layer_sets: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    if scan_mode == "custom":
        return list(custom_layer_sets)
    if scan_mode == "single":
        return [(idx,) for idx in layer_candidates]
    if scan_mode == "pair":
        return list(itertools.combinations(layer_candidates, 2))
    if scan_mode == "single_and_pair":
        return [(idx,) for idx in layer_candidates] + list(itertools.combinations(layer_candidates, 2))
    raise ValueError(f"未知 scan_mode: {scan_mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = rosa_mod.build_arg_parser()
    parser.description = "统一扫描 ROSA 注入层位配置，可同时汇总训练小样本与 profiling 结果。"
    parser.add_argument("--experiment_mode", type=str, default="profile", choices=["profile", "train", "both"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--prefill_tokens", type=int, default=64)
    parser.add_argument("--decode_steps", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=8)
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("--measure_iters", type=int, default=3)
    parser.add_argument("--train_consistency_samples", type=int, default=8)
    parser.add_argument("--train_consistency_stride", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda；为空时自动选择。")
    parser.add_argument("--rosa_prefetch", action="store_true")
    parser.add_argument("--rosa_prefetch_pinned", action="store_true")
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--rosa_ckpt", type=str, default=None)
    parser.add_argument("--scan_mode", type=str, default="single", choices=["single", "pair", "single_and_pair", "custom"])
    parser.add_argument("--layer_candidates", type=str, default="")
    parser.add_argument("--custom_layer_sets", type=str, default="")
    parser.add_argument("--max_combinations", type=int, default=0, help="0 表示不限制。")
    parser.add_argument("--scan_out_file", type=str, default="layer_scan_report.json")
    return parser


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clone_args_with_layers(args, layer_ids: Sequence[int], out_dir: Path):
    run_args = copy.deepcopy(args)
    run_args.rosa_inject_layers = len(layer_ids)
    run_args.rosa_inject_layer_ids = ",".join(str(x) for x in layer_ids)
    run_args.out_dir = str(out_dir)
    return run_args


def load_split_docs(args, json_text_keys: Sequence[str]):
    explicit_split_mode = any([args.train_data_path, args.val_data_path, args.test_data_path])
    if explicit_split_mode:
        if not all([args.train_data_path, args.val_data_path, args.test_data_path]):
            raise ValueError("使用显式数据集切分时，--train_data_path/--val_data_path/--test_data_path 必须同时提供。")
        train_docs, train_source = rosa_mod.load_docs_from_path(
            args.train_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_train_docs,
        )
        val_docs, val_source = rosa_mod.load_docs_from_path(
            args.val_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_val_docs,
        )
        test_docs, test_source = rosa_mod.load_docs_from_path(
            args.test_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_test_docs,
        )
        dataset_source = {
            "mode": "explicit_splits",
            "train": train_source,
            "val": val_source,
            "test": test_source,
        }
        total_docs = len(train_docs) + len(val_docs) + len(test_docs)
        return train_docs, val_docs, test_docs, dataset_source, total_docs

    if not args.data_path:
        raise ValueError("未提供数据路径。请传 --data_path，或同时传 --train_data_path/--val_data_path/--test_data_path。")

    docs, source_meta = rosa_mod.load_docs_from_path(
        args.data_path,
        data_format=args.data_format,
        split_mode=args.split_mode,
        json_text_keys=json_text_keys,
        max_docs=args.max_docs,
    )
    train_docs, val_docs, test_docs = rosa_mod.train_val_test_split(docs, args.train_ratio, args.val_ratio, args.seed)
    dataset_source = {
        "mode": "single_source",
        "source": source_meta,
    }
    return train_docs, val_docs, test_docs, dataset_source, len(docs)


def build_training_context(args) -> SweepTrainingContext:
    tokenizer = rosa_mod.build_tokenizer(args.tokenizer_name_or_path)
    json_text_keys = rosa_mod.parse_csv_arg(args.json_text_keys)
    if not json_text_keys:
        raise ValueError("--json_text_keys 不能为空。")

    train_docs, val_docs, test_docs, dataset_source, total_docs = load_split_docs(args, json_text_keys)
    train_tok = rosa_mod.tokenize_docs(train_docs, tokenizer, add_bos=True, add_eos=True)
    val_tok = rosa_mod.tokenize_docs(val_docs, tokenizer, add_bos=True, add_eos=True)
    test_tok = rosa_mod.tokenize_docs(test_docs, tokenizer, add_bos=True, add_eos=True)
    train_ds, val_ds, test_ds, memory_meta = rosa_mod.build_chunk_datasets(
        train_tok,
        val_tok,
        test_tok,
        seq_len=args.seq_len,
        pad_id=tokenizer.pad_token_id,
        stride=args.stride,
        rosa_memory_tokens=args.rosa_memory_tokens,
        rosa_memory_mode=args.rosa_memory_mode,
        rosa_global_memory_tokens=args.rosa_global_memory_tokens,
        rosa_backend=args.rosa_backend,
        rosa_train_mode=args.rosa_train_mode,
        rosa_seq_address_mode=args.rosa_seq_address_mode,
        rosa_min_match_len=args.rosa_min_match_len,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        enable_train_address_cache=not getattr(args, "disable_rosa_train_address_cache", False),
    )
    return SweepTrainingContext(
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        memory_meta=memory_meta,
        dataset_source=dataset_source,
        total_docs=total_docs,
    )


def summarize_profile_entry(report: Dict[str, Any], layer_ids: Sequence[int]) -> Dict[str, Any]:
    decode = report["decode_micro"]
    prefill = report["prefill"]
    entry = {
        "layer_ids": list(layer_ids),
        "prefill_ref_ms": prefill["timings"]["rosa_reference"]["avg_ms"],
        "prefill_online_ms": prefill["timings"]["rosa_online"]["avg_ms"],
        "decode_ref_ms": decode["timings"]["rosa_reference"]["avg_ms"],
        "decode_online_ms": decode["timings"]["rosa_online"]["avg_ms"],
        "decode_online_vs_reference_speedup": (
            decode["timings"]["rosa_reference"]["avg_ms"] / decode["timings"]["rosa_online"]["avg_ms"]
            if decode["timings"]["rosa_online"]["avg_ms"] > 0
            else 0.0
        ),
        "fire_coverage": decode["coverage"].get("fire_coverage", 0.0),
        "raw_match_coverage": decode["coverage"].get("raw_match_coverage", 0.0),
        "fired_avg_match_len": decode["coverage"].get("avg_fired_match_len", 0.0),
        "avg_gate": decode.get("model_stats", {}).get("rosa_avg_gate", 0.0),
        "gate_hit": decode.get("model_stats", {}).get("rosa_gate_hit", 0.0),
        "hot_cache_token_hit_rate": decode.get("model_stats", {}).get("rosa_hot_cache_token_hit_rate", 0.0),
        "hot_cache_unique_hit_rate": decode.get("model_stats", {}).get("rosa_hot_cache_unique_hit_rate", 0.0),
        "address_agreement": decode["correctness"]["address_agreement"].get("all_equal", 0.0),
        "logit_max_abs_diff": decode["correctness"]["logit_max_abs_diff"],
    }
    if "prefetch" in decode:
        entry["prefetch_hit_rate"] = decode["prefetch"].get("hit_rate", 0.0)
        entry["prefetch_avg_wait_ms"] = decode["prefetch"].get("avg_wait_ms", 0.0)
    return entry


def build_rosa_model(args, tokenizer, layer_ids: Sequence[int]) -> rosa_mod.RosaFusedLM:
    cfg = rosa_mod.build_model_config(args, tokenizer)
    rosa_mod.set_seed(args.seed)
    return rosa_mod.RosaFusedLM(
        cfg,
        pad_id=tokenizer.pad_token_id,
        rosa_backend=args.rosa_backend,
        min_match_len=args.rosa_min_match_len,
        inject_layers=len(layer_ids),
        inject_layer_ids=layer_ids,
        rosa_scale=args.rosa_scale,
        rosa_value_mode=args.rosa_value_mode,
        rosa_seq_address_mode=args.rosa_seq_address_mode,
        use_context_gate=args.rosa_context_gate,
        rosa_hot_cache_size=args.rosa_hot_cache_size,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        use_match_len_gate=not args.rosa_disable_match_len_gate,
    )


def summarize_training_entry(
    history: Dict[str, List[Dict[str, float]]],
    test_metrics: Dict[str, float],
    *,
    layer_ids: Sequence[int],
    param_count: int,
) -> Dict[str, Any]:
    train_last = history["train"][-1]
    val_last = history["val"][-1]
    return {
        "layer_ids": list(layer_ids),
        "param_count": int(param_count),
        "train_loss": train_last["loss"],
        "train_token_acc": train_last["token_acc"],
        "train_fire_coverage": train_last.get("rosa_fire_coverage", 0.0),
        "train_avg_gate": train_last.get("rosa_avg_gate", 0.0),
        "val_loss": val_last["loss"],
        "val_token_acc": val_last["token_acc"],
        "val_fire_coverage": val_last.get("rosa_fire_coverage", 0.0),
        "val_avg_gate": val_last.get("rosa_avg_gate", 0.0),
        "test_loss": test_metrics["loss"],
        "test_token_acc": test_metrics["token_acc"],
        "test_fire_coverage": test_metrics.get("rosa_fire_coverage", 0.0),
        "test_avg_gate": test_metrics.get("rosa_avg_gate", 0.0),
        "test_gate_hit": test_metrics.get("rosa_gate_hit", 0.0),
        "test_address_source_online_seq": test_metrics.get("rosa_address_source_online_seq", 0.0),
    }


def run_training_entry(args, context: SweepTrainingContext, layer_ids: Sequence[int], run_dir: Path) -> Dict[str, Any]:
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    train_loader, val_loader, test_loader = rosa_mod.build_dataloaders(
        context.train_ds,
        context.val_ds,
        context.test_ds,
        batch_size=args.batch_size,
        pad_id=context.tokenizer.pad_token_id,
        train_seed=args.seed,
    )
    model = build_rosa_model(args, context.tokenizer, layer_ids)
    param_count = rosa_mod.count_params(model)
    history = rosa_mod.train_one_model(
        model,
        train_loader,
        val_loader,
        device,
        pad_id=context.tokenizer.pad_token_id,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_bf16=args.bf16,
    )
    test_metrics = rosa_mod.evaluate(model.to(device), test_loader, device, context.tokenizer.pad_token_id)
    summary = summarize_training_entry(history, test_metrics, layer_ids=layer_ids, param_count=param_count)
    save_json(
        {
            "layer_ids": list(layer_ids),
            "history": history,
            "test": test_metrics,
            "summary": summary,
            "memory_meta": context.memory_meta,
            "dataset_source": context.dataset_source,
        },
        run_dir / "train_summary.json",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def compose_result_entry(
    *,
    layer_ids: Sequence[int],
    profile_entry: Dict[str, Any] | None,
    train_entry: Dict[str, Any] | None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "layer_ids": list(layer_ids),
        "profile": profile_entry,
        "train": train_entry,
    }
    if profile_entry is not None:
        row["address_agreement"] = profile_entry.get("address_agreement", 0.0)
        row["decode_online_ms"] = profile_entry.get("decode_online_ms", 0.0)
        row["profile_fire_coverage"] = profile_entry.get("fire_coverage", 0.0)
    if train_entry is not None:
        row["val_loss"] = train_entry.get("val_loss", 0.0)
        row["test_token_acc"] = train_entry.get("test_token_acc", 0.0)
        row["train_fire_coverage"] = train_entry.get("test_fire_coverage", 0.0)
    return row


def rank_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        entries,
        key=lambda row: (
            -row.get("address_agreement", 1.0 if row.get("profile") is None else 0.0),
            -row.get("test_token_acc", 0.0),
            row.get("val_loss", float("inf")),
            row.get("decode_online_ms", float("inf")),
            -row.get("train_fire_coverage", 0.0),
            -row.get("profile_fire_coverage", 0.0),
        ),
    )


def run_sweep(args) -> Dict[str, Any]:
    recipe_meta = apply_rosa_recipe(args)
    custom_layer_sets = parse_custom_layer_sets(args.custom_layer_sets)
    layer_candidates = rosa_mod.parse_int_csv_arg(args.layer_candidates) or list(range(args.n_layers))
    layer_sets = build_layer_sets(
        scan_mode=args.scan_mode,
        layer_candidates=layer_candidates,
        custom_layer_sets=custom_layer_sets,
    )
    if args.max_combinations > 0:
        layer_sets = layer_sets[:args.max_combinations]
    if not layer_sets:
        raise ValueError("没有生成任何 layer set，请检查扫描参数。")

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    training_context = build_training_context(args) if args.experiment_mode in {"train", "both"} else None

    results = []
    for layer_ids in layer_sets:
        layer_dir = base_out_dir / f"layers_{'_'.join(str(x) for x in layer_ids)}"
        profile_entry = None
        train_entry = None
        if args.experiment_mode in {"profile", "both"}:
            profile_args = clone_args_with_layers(args, layer_ids, layer_dir / "profile")
            profile_report = profile_mod.run_profile(profile_args)
            profile_entry = summarize_profile_entry(profile_report, layer_ids)
        if args.experiment_mode in {"train", "both"}:
            train_entry = run_training_entry(args, training_context, layer_ids, layer_dir / "train")
        results.append(compose_result_entry(layer_ids=layer_ids, profile_entry=profile_entry, train_entry=train_entry))

    ranked = rank_entries(results)
    summary = {
        "meta": {
            "experiment_mode": args.experiment_mode,
            "recipe": recipe_meta,
            "scan_mode": args.scan_mode,
            "layer_candidates": layer_candidates,
            "custom_layer_sets": [list(x) for x in custom_layer_sets],
            "max_combinations": args.max_combinations,
            "num_runs": len(results),
        },
        "ranked_results": ranked,
        "best": ranked[0] if ranked else None,
    }
    save_json(summary, base_out_dir / args.scan_out_file)
    print("=== ROSA Injection Layer Scan ===")
    print(f"runs: {len(results)}")
    print(f"best: {summary['best']}")
    print(f"report: {(base_out_dir / args.scan_out_file).resolve()}")
    return summary
