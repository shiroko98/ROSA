from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from rosa_memmap_dataset import PRETOKENIZED_MEMMAP_FORMAT, PRETOKENIZED_MEMMAP_VERSION
from train_qwen_llama_vs_rosa_v2 import (
    build_tokenizer,
    extract_text_from_record,
    infer_data_format,
    preview_doc,
    read_text,
    resolve_data_files,
    split_docs,
    train_val_test_split,
)


@dataclass
class DocStreamStats:
    path_spec: str
    resolved_files: List[str]
    skipped_records: int = 0
    format_counts: Dict[str, int] = field(default_factory=dict)
    example_keys: Optional[List[str]] = None
    doc_count: int = 0

    def to_source_meta(self) -> Dict[str, Any]:
        return {
            "path_spec": self.path_spec,
            "resolved_files": self.resolved_files,
            "file_count": len(self.resolved_files),
            "doc_count": int(self.doc_count),
            "skipped_records": int(self.skipped_records),
            "format_counts": dict(self.format_counts),
            "example_keys": list(self.example_keys) if self.example_keys else None,
        }


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def iter_docs_from_path(
    path_spec: str,
    *,
    data_format: str,
    split_mode: str,
    json_text_keys: Sequence[str],
    max_docs: Optional[int] = None,
) -> Tuple[Iterator[str], DocStreamStats]:
    files = resolve_data_files(path_spec)
    stats = DocStreamStats(path_spec=path_spec, resolved_files=files)

    def _iterator() -> Iterator[str]:
        yielded = 0
        for path in files:
            cur_format = infer_data_format(path, data_format)
            stats.format_counts[cur_format] = stats.format_counts.get(cur_format, 0) + 1

            if cur_format == "text":
                for doc in split_docs(read_text(path), split_mode):
                    text = doc.strip()
                    if not text:
                        continue
                    yielded += 1
                    stats.doc_count = yielded
                    yield text
                    if max_docs is not None and yielded >= max_docs:
                        return
            elif cur_format == "jsonl":
                opener = open
                if path.lower().endswith(".gz"):
                    import gzip
                    opener = gzip.open
                with opener(path, "rt", encoding="utf-8") as f:
                    for line_no, raw_line in enumerate(f, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"JSONL 解析失败: {path}:{line_no} -> {exc}") from exc
                        if stats.example_keys is None and isinstance(record, dict):
                            stats.example_keys = sorted(record.keys())
                        doc = extract_text_from_record(record, json_text_keys)
                        if doc is None or not doc.strip():
                            stats.skipped_records += 1
                            continue
                        yielded += 1
                        stats.doc_count = yielded
                        yield doc.strip()
                        if max_docs is not None and yielded >= max_docs:
                            return
            elif cur_format == "json":
                opener = open
                if path.lower().endswith(".gz"):
                    import gzip
                    opener = gzip.open
                with opener(path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                records = payload if isinstance(payload, list) else [payload]
                for idx, record in enumerate(records, start=1):
                    if stats.example_keys is None and isinstance(record, dict):
                        stats.example_keys = sorted(record.keys())
                    doc = extract_text_from_record(record, json_text_keys)
                    if doc is None or not doc.strip():
                        stats.skipped_records += 1
                        continue
                    yielded += 1
                    stats.doc_count = yielded
                    yield doc.strip()
                    if max_docs is not None and yielded >= max_docs:
                        return
            else:
                raise ValueError(f"不支持的数据格式: {cur_format}")

    return _iterator(), stats


_WORKER_TOKENIZER = None
_WORKER_ADD_BOS = False
_WORKER_ADD_EOS = False


def _init_tokenize_worker(tokenizer_name_or_path: Optional[str], add_bos: bool, add_eos: bool) -> None:
    global _WORKER_TOKENIZER, _WORKER_ADD_BOS, _WORKER_ADD_EOS
    _WORKER_TOKENIZER = build_tokenizer(tokenizer_name_or_path)
    _WORKER_ADD_BOS = add_bos
    _WORKER_ADD_EOS = add_eos


def _tokenize_one_doc(doc: str, tokenizer, *, add_bos: bool, add_eos: bool) -> List[int]:
    ids: List[int] = []
    if add_bos and tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    ids.extend(tokenizer.encode(doc))
    if add_eos and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return ids


def _tokenize_docs_batch_worker(docs: Sequence[str]) -> List[List[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenize worker 尚未初始化。")
    return [
        _tokenize_one_doc(doc, _WORKER_TOKENIZER, add_bos=_WORKER_ADD_BOS, add_eos=_WORKER_ADD_EOS)
        for doc in docs
    ]


def _batched_docs(docs_iter: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for doc in docs_iter:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_tokenized_batch(
    f,
    tokenized_docs: Sequence[Sequence[int]],
    *,
    offsets: List[int],
    lengths: List[int],
    token_count: int,
) -> int:
    for ids in tokenized_docs:
        offsets.append(token_count)
        lengths.append(len(ids))
        arr = np.asarray(list(ids), dtype=np.int32)
        if arr.size > 0:
            arr.tofile(f)
            token_count += int(arr.size)
    return token_count


def write_streaming_pretokenized_memmap_split(
    out_dir: str,
    split_name: str,
    *,
    docs_iter: Iterable[str],
    tokenizer_name_or_path: Optional[str],
    add_bos: bool,
    add_eos: bool,
    preview_text: str = "",
    tokenize_workers: int = 1,
    tokenize_batch_docs: int = 64,
    progress_docs: int = 5000,
) -> Dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    tokens_path = root / f"{split_name}.tokens.bin"
    offsets_path = root / f"{split_name}.offsets.npy"
    lengths_path = root / f"{split_name}.lengths.npy"

    offsets: List[int] = []
    lengths: List[int] = []
    token_count = 0
    doc_count = 0
    split_preview = preview_text
    start_time = time.time()

    def _log_progress(force: bool = False) -> None:
        if doc_count == 0:
            return
        if not force and progress_docs > 0 and doc_count % progress_docs != 0:
            return
        elapsed = max(1e-6, time.time() - start_time)
        print(
            f"[memmap:{split_name}] docs={doc_count:,} tokens={token_count:,} "
            f"docs_per_s={doc_count / elapsed:.2f} tokens_per_s={token_count / elapsed:.2f}"
        )

    with open(tokens_path, "wb") as f:
        if tokenize_workers <= 1:
            tokenizer = build_tokenizer(tokenizer_name_or_path)
            for doc_batch in _batched_docs(docs_iter, tokenize_batch_docs):
                if not split_preview and doc_batch:
                    split_preview = preview_doc(doc_batch[0])
                tokenized_docs = [
                    _tokenize_one_doc(doc, tokenizer, add_bos=add_bos, add_eos=add_eos)
                    for doc in doc_batch
                ]
                token_count = _write_tokenized_batch(
                    f,
                    tokenized_docs,
                    offsets=offsets,
                    lengths=lengths,
                    token_count=token_count,
                )
                doc_count += len(tokenized_docs)
                f.flush()
                _log_progress()
        else:
            pending: Deque[Tuple[int, Future[List[List[int]]], List[str]]] = deque()
            max_pending_batches = max(2, tokenize_workers * 2)
            with ProcessPoolExecutor(
                max_workers=tokenize_workers,
                initializer=_init_tokenize_worker,
                initargs=(tokenizer_name_or_path, add_bos, add_eos),
            ) as executor:
                batch_index = 0
                for doc_batch in _batched_docs(docs_iter, tokenize_batch_docs):
                    if not split_preview and doc_batch:
                        split_preview = preview_doc(doc_batch[0])
                    pending.append((batch_index, executor.submit(_tokenize_docs_batch_worker, tuple(doc_batch)), doc_batch))
                    batch_index += 1
                    while len(pending) >= max_pending_batches:
                        _, future, _ = pending.popleft()
                        tokenized_docs = future.result()
                        token_count = _write_tokenized_batch(
                            f,
                            tokenized_docs,
                            offsets=offsets,
                            lengths=lengths,
                            token_count=token_count,
                        )
                        doc_count += len(tokenized_docs)
                        f.flush()
                        _log_progress()
                while pending:
                    _, future, _ = pending.popleft()
                    tokenized_docs = future.result()
                    token_count = _write_tokenized_batch(
                        f,
                        tokenized_docs,
                        offsets=offsets,
                        lengths=lengths,
                        token_count=token_count,
                    )
                    doc_count += len(tokenized_docs)
                    f.flush()
                    _log_progress()

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    np.save(offsets_path, offsets_arr)
    np.save(lengths_path, lengths_arr)
    _log_progress(force=True)

    return {
        "name": split_name,
        "doc_count": int(doc_count),
        "token_count": int(token_count),
        "preview_text": split_preview,
        "tokens_bin": _relative_path(tokens_path, root),
        "offsets_npy": _relative_path(offsets_path, root),
        "lengths_npy": _relative_path(lengths_path, root),
    }


def build_streaming_pretokenized_memmap_dataset_from_explicit_splits(
    out_dir: str,
    *,
    tokenizer_name_or_path: Optional[str],
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    data_format: str,
    split_mode: str,
    json_text_keys: Sequence[str],
    max_train_docs: Optional[int] = None,
    max_val_docs: Optional[int] = None,
    max_test_docs: Optional[int] = None,
    add_bos: bool = True,
    add_eos: bool = True,
    manifest_name: str = "dataset_manifest.json",
    tokenize_workers: int = 1,
    tokenize_batch_docs: int = 64,
    progress_docs: int = 5000,
) -> Tuple[str, Dict[str, Any]]:
    tokenizer = build_tokenizer(tokenizer_name_or_path)
    tokenizer_meta = {
        "name_or_path": tokenizer_name_or_path or "byte-fallback",
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": (train_data_path, max_train_docs),
        "val": (val_data_path, max_val_docs),
        "test": (test_data_path, max_test_docs),
    }

    splits_meta: Dict[str, Dict[str, Any]] = {}
    source_meta: Dict[str, Any] = {"mode": "explicit_splits"}
    for split_name, (path_spec, max_docs) in split_specs.items():
        docs_iter, stats = iter_docs_from_path(
            path_spec,
            data_format=data_format,
            split_mode=split_mode,
            json_text_keys=json_text_keys,
            max_docs=max_docs,
        )
        print(f"[memmap:{split_name}] start streaming from {path_spec}")
        splits_meta[split_name] = write_streaming_pretokenized_memmap_split(
            str(root),
            split_name,
            docs_iter=docs_iter,
            tokenizer_name_or_path=tokenizer_name_or_path,
            add_bos=add_bos,
            add_eos=add_eos,
            tokenize_workers=tokenize_workers,
            tokenize_batch_docs=tokenize_batch_docs,
            progress_docs=progress_docs,
        )
        source_meta[split_name] = stats.to_source_meta()

    manifest = {
        "format": PRETOKENIZED_MEMMAP_FORMAT,
        "version": PRETOKENIZED_MEMMAP_VERSION,
        "tokenizer": tokenizer_meta,
        "tokenization": {
            "add_bos": bool(add_bos),
            "add_eos": bool(add_eos),
        },
        "source": source_meta,
        "splits": splits_meta,
    }
    manifest_path = root / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path), manifest


def build_pretokenized_memmap_dataset_from_single_source(
    out_dir: str,
    *,
    data_path: str,
    data_format: str,
    split_mode: str,
    json_text_keys: Sequence[str],
    max_docs: Optional[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    tokenizer_name_or_path: Optional[str],
    add_bos: bool = True,
    add_eos: bool = True,
    manifest_name: str = "dataset_manifest.json",
) -> Tuple[str, Dict[str, Any]]:
    from rosa_memmap_dataset import write_pretokenized_memmap_dataset
    from train_qwen_llama_vs_rosa_v2 import load_docs_from_path, tokenize_docs

    docs, source_meta_raw = load_docs_from_path(
        data_path,
        data_format=data_format,
        split_mode=split_mode,
        json_text_keys=json_text_keys,
        max_docs=max_docs,
    )
    train_docs, val_docs, test_docs = train_val_test_split(docs, train_ratio, val_ratio, seed)
    source_meta = {
        "mode": "single_source",
        "source": source_meta_raw,
    }
    tokenizer = build_tokenizer(tokenizer_name_or_path)
    train_tok = tokenize_docs(train_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)
    val_tok = tokenize_docs(val_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)
    test_tok = tokenize_docs(test_docs, tokenizer, add_bos=add_bos, add_eos=add_eos)
    return write_pretokenized_memmap_dataset(
        out_dir,
        split_docs_tokens={
            "train": train_tok,
            "val": val_tok,
            "test": test_tok,
        },
        tokenizer_meta={
            "name_or_path": tokenizer_name_or_path or "byte-fallback",
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
        manifest_name=manifest_name,
    )
