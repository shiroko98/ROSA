import argparse
import os

from rosa_memmap_dataset import write_pretokenized_memmap_dataset
from train_qwen_llama_vs_rosa_v2 import (
    build_tokenizer,
    load_docs_from_path,
    parse_csv_arg,
    preview_doc,
    tokenize_docs,
    train_val_test_split,
)


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
    parser.add_argument("--out_dir", type=str, required=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer_name_or_path)
    json_text_keys = parse_csv_arg(args.json_text_keys)
    if not json_text_keys:
        raise ValueError("--json_text_keys 不能为空。")

    explicit_split_mode = any([args.train_data_path, args.val_data_path, args.test_data_path])
    if explicit_split_mode:
        if not all([args.train_data_path, args.val_data_path, args.test_data_path]):
            raise ValueError("使用显式数据集切分时，--train_data_path/--val_data_path/--test_data_path 必须同时提供。")
        train_docs, train_source = load_docs_from_path(
            args.train_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_train_docs,
        )
        val_docs, val_source = load_docs_from_path(
            args.val_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_val_docs,
        )
        test_docs, test_source = load_docs_from_path(
            args.test_data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_test_docs,
        )
        source_meta = {
            "mode": "explicit_splits",
            "train": train_source,
            "val": val_source,
            "test": test_source,
        }
    else:
        if not args.data_path:
            raise ValueError("未提供数据路径。请传 --data_path，或同时传 --train_data_path/--val_data_path/--test_data_path。")
        docs, source_meta_raw = load_docs_from_path(
            args.data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_docs,
        )
        train_docs, val_docs, test_docs = train_val_test_split(docs, args.train_ratio, args.val_ratio, args.seed)
        source_meta = {
            "mode": "single_source",
            "source": source_meta_raw,
        }

    add_bos = not args.disable_bos
    add_eos = not args.disable_eos
    train_tok = tokenize_docs(train_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)
    val_tok = tokenize_docs(val_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)
    test_tok = tokenize_docs(test_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path, manifest = write_pretokenized_memmap_dataset(
        args.out_dir,
        split_docs_tokens={
            "train": train_tok,
            "val": val_tok,
            "test": test_tok,
        },
        tokenizer_meta={
            "name_or_path": args.tokenizer_name_or_path or "byte-fallback",
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        source_meta=source_meta,
        preview_texts={
            "train": preview_doc(train_docs[0]) if train_docs else "",
            "val": preview_doc(val_docs[0]) if val_docs else "",
            "test": preview_doc(test_docs[0]) if test_docs else "",
        },
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
