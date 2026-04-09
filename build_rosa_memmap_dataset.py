import argparse
import os

from rosa_memmap_builder import (
    build_pretokenized_memmap_dataset_from_single_source,
    build_streaming_pretokenized_memmap_dataset_from_explicit_splits,
)
from train_qwen_llama_vs_rosa_v2 import parse_csv_arg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 ROSA 原始语料预分词并写成 memmap/二进制数据集。")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["paragraph", "line", "stream"])
    parser.add_argument("--data_format", type=str, default="auto", choices=["auto", "text", "jsonl", "json"])
    parser.add_argument("--json_text_keys", type=str, default="text,content,body,message")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--max_train_docs", type=int, default=None)
    parser.add_argument("--max_val_docs", type=int, default=None)
    parser.add_argument("--max_test_docs", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_bos", action="store_true")
    parser.add_argument("--disable_eos", action="store_true")
    parser.add_argument("--manifest_name", type=str, default="dataset_manifest.json")
    parser.add_argument("--tokenize_workers", type=int, default=1,
                        help="预分词阶段的进程数。explicit split 模式下会按 batch 多进程分词。")
    parser.add_argument("--tokenize_batch_docs", type=int, default=64,
                        help="每次提交给分词器/worker 的文档 batch 大小。")
    parser.add_argument("--progress_docs", type=int, default=5000,
                        help="每处理多少篇文档打印一次进度。设为 0 可关闭中间进度。")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    json_text_keys = parse_csv_arg(args.json_text_keys)
    if not json_text_keys:
        raise ValueError("--json_text_keys 不能为空。")

    explicit_split_mode = any([args.train_data_path, args.val_data_path, args.test_data_path])
    if explicit_split_mode:
        if not all([args.train_data_path, args.val_data_path, args.test_data_path]):
            raise ValueError("使用显式数据集切分时，--train_data_path/--val_data_path/--test_data_path 必须同时提供。")
        print("[memmap] 使用 explicit split 流式构建模式。")
    else:
        if not args.data_path:
            raise ValueError("未提供数据路径。请传 --data_path，或同时传 --train_data_path/--val_data_path/--test_data_path。")
        print("[memmap] 使用 single source 兼容模式。该模式当前仍会先读入全部文档再切分。")

    add_bos = not args.disable_bos
    add_eos = not args.disable_eos

    os.makedirs(args.out_dir, exist_ok=True)
    if explicit_split_mode:
        manifest_path, manifest = build_streaming_pretokenized_memmap_dataset_from_explicit_splits(
            args.out_dir,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            test_data_path=args.test_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_train_docs=args.max_train_docs,
            max_val_docs=args.max_val_docs,
            max_test_docs=args.max_test_docs,
            add_bos=add_bos,
            add_eos=add_eos,
            manifest_name=args.manifest_name,
            tokenize_workers=args.tokenize_workers,
            tokenize_batch_docs=args.tokenize_batch_docs,
            progress_docs=args.progress_docs,
        )
    else:
        manifest_path, manifest = build_pretokenized_memmap_dataset_from_single_source(
            args.out_dir,
            data_path=args.data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_docs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            add_bos=add_bos,
            add_eos=add_eos,
            manifest_name=args.manifest_name,
        )

    print(f"manifest: {manifest_path}")
    print(
        f"train/val/test docs: {manifest['splits']['train']['doc_count']} / "
        f"{manifest['splits']['val']['doc_count']} / {manifest['splits']['test']['doc_count']}"
    )
    print(
        f"train/val/test tokens: {manifest['splits']['train']['token_count']} / "
        f"{manifest['splits']['val']['token_count']} / {manifest['splits']['test']['token_count']}"
    )


if __name__ == "__main__":
    main()
