import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import profile_rosa_online_baseline as profile_mod
import train_qwen_llama_vs_rosa_v2 as rosa_mod


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
    parser = profile_mod.build_arg_parser()
    parser.description = "扫描 ROSA 注入层位配置，并汇总 profiling 结果。"
    parser.add_argument("--scan_mode", type=str, default="single", choices=["single", "pair", "single_and_pair", "custom"])
    parser.add_argument("--layer_candidates", type=str, default="")
    parser.add_argument("--custom_layer_sets", type=str, default="")
    parser.add_argument("--max_combinations", type=int, default=0, help="0 表示不限制。")
    parser.add_argument("--scan_out_file", type=str, default="layer_scan_report.json")
    return parser


def summarize_entry(report: dict, layer_ids: Sequence[int]) -> dict:
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


def rank_entries(entries: Sequence[dict]) -> List[dict]:
    return sorted(
        entries,
        key=lambda row: (
            -row.get("address_agreement", 0.0),
            -row.get("gate_hit", 0.0),
            row.get("decode_online_ms", float("inf")),
            -row.get("fire_coverage", 0.0),
        ),
    )


def run_scan(args) -> dict:
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

    reports = []
    for layer_ids in layer_sets:
        run_args = copy.deepcopy(args)
        run_args.rosa_inject_layers = len(layer_ids)
        run_args.rosa_inject_layer_ids = ",".join(str(x) for x in layer_ids)
        run_args.out_dir = str(base_out_dir / f"layers_{'_'.join(str(x) for x in layer_ids)}")
        report = profile_mod.run_profile(run_args)
        reports.append(summarize_entry(report, layer_ids))

    ranked = rank_entries(reports)
    summary = {
        "meta": {
            "scan_mode": args.scan_mode,
            "layer_candidates": layer_candidates,
            "custom_layer_sets": [list(x) for x in custom_layer_sets],
            "max_combinations": args.max_combinations,
            "num_runs": len(reports),
        },
        "ranked_results": ranked,
        "best": ranked[0] if ranked else None,
    }
    out_path = base_out_dir / args.scan_out_file
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== ROSA Injection Layer Scan ===")
    print(f"runs: {len(reports)}")
    print(f"best: {summary['best']}")
    print(f"report: {out_path.resolve()}")
    return summary


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
