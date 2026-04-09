from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def parse_int_csv(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("至少需要一个 snapshot interval。")
    return values


def parse_mode_csv(text: str) -> List[str]:
    modes: List[str] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in {"sync", "async"}:
            raise ValueError(f"未知模式: {part}")
        if part not in modes:
            modes.append(part)
    if not modes:
        raise ValueError("至少需要一个运行模式（sync 或 async）。")
    return modes


def build_train_command(
    args,
    *,
    mode: str,
    snapshot_interval: Optional[int],
    out_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "train_qwen_llama_vs_rosa_v2.py",
        "--train_data_path",
        args.train_data_path,
        "--val_data_path",
        args.val_data_path,
        "--test_data_path",
        args.test_data_path,
        "--data_format",
        args.data_format,
        "--json_text_keys",
        args.json_text_keys,
        "--max_train_docs",
        str(args.max_train_docs),
        "--max_val_docs",
        str(args.max_val_docs),
        "--max_test_docs",
        str(args.max_test_docs),
        "--tokenizer_name_or_path",
        args.tokenizer_name_or_path,
        "--arch_style",
        args.arch_style,
        "--seed",
        str(args.seed),
        "--seq_len",
        str(args.seq_len),
        "--stride",
        str(args.stride),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--dim",
        str(args.dim),
        "--n_layers",
        str(args.n_layers),
        "--n_heads",
        str(args.n_heads),
        "--n_kv_heads",
        str(args.n_kv_heads),
        "--intermediate_size",
        str(args.intermediate_size),
        "--dropout",
        str(args.dropout),
        "--rosa_recipe",
        args.rosa_recipe,
        "--rosa_online_sam_impl",
        args.rosa_online_sam_impl,
        "--disable_rosa_train_address_cache",
        "--train_timing",
        "--out_dir",
        str(out_dir),
    ]
    if mode == "sync":
        cmd.append("--disable_rosa_train_address_async")
    if snapshot_interval is None:
        cmd.append("--disable_rosa_train_state_snapshot")
    else:
        cmd.extend(
            [
                "--enable_rosa_train_state_snapshot",
                "--rosa_train_state_snapshot_interval",
                str(snapshot_interval),
            ]
        )
    return cmd


def load_comparison(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_run_metrics(
    comparison: Dict,
    *,
    mode: str,
    snapshot_interval: Optional[int],
    out_dir: Path,
) -> Dict[str, float]:
    baseline_train = comparison["baseline"]["history"]["train"][-1]
    rosa_train = comparison["rosa_fused"]["history"]["train"][-1]
    rosa_test = comparison["rosa_fused"]["test"]
    meta = comparison["dataset"]["memory"]
    return {
        "mode": mode,
        "snapshot_interval": snapshot_interval if snapshot_interval is not None else -1,
        "snapshot_enabled": 0.0 if snapshot_interval is None else 1.0,
        "rosa_step_ms": float(rosa_train.get("timing_step_ms", 0.0)),
        "rosa_forward_ms": float(rosa_train.get("timing_forward_ms", 0.0)),
        "rosa_addr_ms": float(rosa_train.get("timing_model_rosa_address_ms", 0.0)),
        "baseline_step_ms": float(baseline_train.get("timing_step_ms", 0.0)),
        "test_loss": float(rosa_test["loss"]),
        "test_token_acc": float(rosa_test["token_acc"]),
        "fire_coverage": float(rosa_test.get("rosa_fire_coverage", 0.0)),
        "effective_history": meta["effective_history"],
        "address_build_policy": meta["address_build_policy"],
        "out_dir": str(out_dir),
    }


def summarize_runs(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict]:
    best_by_mode: Dict[str, Dict] = {}
    for mode in {str(row["mode"]) for row in runs}:
        mode_rows = [row for row in runs if row["mode"] == mode]
        mode_rows.sort(key=lambda row: (row["rosa_step_ms"], row["rosa_addr_ms"]))
        best_by_mode[mode] = mode_rows[0]
    return best_by_mode


def print_run_summary(run: Dict[str, float]) -> None:
    label = "no_snapshot" if run["snapshot_interval"] < 0 else f"snapshot_{int(run['snapshot_interval'])}"
    print(
        f"[{run['mode']}] {label} | "
        f"step {run['rosa_step_ms']:.2f}ms | "
        f"rosa_addr {run['rosa_addr_ms']:.2f}ms | "
        f"loss {run['test_loss']:.4f} | "
        f"acc {run['test_token_acc']:.5f}"
    )


def run_sweep(args) -> Dict:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, float]] = []
    for mode in parse_mode_csv(args.modes):
        mode_root = root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        candidates: List[Optional[int]] = [None] + parse_int_csv(args.snapshot_intervals)
        for snapshot_interval in candidates:
            label = "no_snapshot" if snapshot_interval is None else f"snapshot_{snapshot_interval}"
            out_dir = mode_root / label
            cmd = build_train_command(
                args,
                mode=mode,
                snapshot_interval=snapshot_interval,
                out_dir=out_dir,
            )
            print(f"\n=== running {mode}/{label} ===")
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent)
            comparison = load_comparison(out_dir / "comparison.json")
            metrics = extract_run_metrics(
                comparison,
                mode=mode,
                snapshot_interval=snapshot_interval,
                out_dir=out_dir,
            )
            runs.append(metrics)
            print_run_summary(metrics)

    report = {
        "meta": {
            "modes": parse_mode_csv(args.modes),
            "snapshot_intervals": parse_int_csv(args.snapshot_intervals),
            "recipe": args.rosa_recipe,
            "online_sam_impl": args.rosa_online_sam_impl,
            "train_docs": args.max_train_docs,
            "val_docs": args.max_val_docs,
            "test_docs": args.max_test_docs,
        },
        "runs": runs,
        "best_by_mode": summarize_runs(runs),
    }
    report_path = root / "snapshot_sweep_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nsummary: {report_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量比较 ROSA 训练状态快照 interval 在 sync/async 下的开销与效果。")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--data_format", type=str, default="jsonl")
    parser.add_argument("--json_text_keys", type=str, default="text")
    parser.add_argument("--max_train_docs", type=int, default=16)
    parser.add_argument("--max_val_docs", type=int, default=4)
    parser.add_argument("--max_test_docs", type=int, default=4)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--arch_style", type=str, default="qwen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rosa_recipe", type=str, default="online_v1")
    parser.add_argument("--rosa_online_sam_impl", type=str, default="fast")
    parser.add_argument("--snapshot_intervals", type=str, default="64,128,256,512")
    parser.add_argument("--modes", type=str, default="sync,async")
    parser.add_argument("--out_dir", type=str, default="outputs/snapshot_interval_sweep_smoke")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
