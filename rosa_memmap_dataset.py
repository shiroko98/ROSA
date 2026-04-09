import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PRETOKENIZED_MEMMAP_FORMAT = "rosa_pretokenized_memmap_v1"
PRETOKENIZED_MEMMAP_VERSION = 1


def count_doc_samples(doc_len: int, seq_len: int, stride: Optional[int]) -> int:
    if doc_len < 2:
        return 0
    step = stride or seq_len
    max_start = max(1, doc_len - 1)
    count = 0
    for start in range(0, max_start, step):
        count += 1
        if start + seq_len + 1 >= doc_len:
            break
    return count


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_tokens_bin(path: Path, docs_tokens: Sequence[Sequence[int]]) -> int:
    total_tokens = 0
    with open(path, "wb") as f:
        for ids in docs_tokens:
            arr = np.asarray(list(ids), dtype=np.int32)
            total_tokens += int(arr.size)
            if arr.size > 0:
                arr.tofile(f)
    return total_tokens


def write_pretokenized_memmap_split(
    out_dir: str,
    split_name: str,
    docs_tokens: Sequence[Sequence[int]],
    *,
    preview_text: str = "",
) -> Dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    lengths = np.asarray([len(ids) for ids in docs_tokens], dtype=np.int64)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    tokens_path = root / f"{split_name}.tokens.bin"
    offsets_path = root / f"{split_name}.offsets.npy"
    lengths_path = root / f"{split_name}.lengths.npy"

    token_count = _write_tokens_bin(tokens_path, docs_tokens)
    np.save(offsets_path, offsets)
    np.save(lengths_path, lengths)

    return {
        "name": split_name,
        "doc_count": int(len(docs_tokens)),
        "token_count": int(token_count),
        "preview_text": preview_text,
        "tokens_bin": _relative_path(tokens_path, root),
        "offsets_npy": _relative_path(offsets_path, root),
        "lengths_npy": _relative_path(lengths_path, root),
    }


def write_pretokenized_memmap_dataset(
    out_dir: str,
    *,
    split_docs_tokens: Dict[str, Sequence[Sequence[int]]],
    tokenizer_meta: Dict[str, Any],
    source_meta: Optional[Dict[str, Any]] = None,
    preview_texts: Optional[Dict[str, str]] = None,
    add_bos: bool = True,
    add_eos: bool = True,
    manifest_name: str = "dataset_manifest.json",
) -> Tuple[str, Dict[str, Any]]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    preview_texts = preview_texts or {}

    splits_meta: Dict[str, Dict[str, Any]] = {}
    for split_name, docs_tokens in split_docs_tokens.items():
        splits_meta[split_name] = write_pretokenized_memmap_split(
            str(root),
            split_name,
            docs_tokens,
            preview_text=preview_texts.get(split_name, ""),
        )

    manifest = {
        "format": PRETOKENIZED_MEMMAP_FORMAT,
        "version": PRETOKENIZED_MEMMAP_VERSION,
        "tokenizer": tokenizer_meta,
        "tokenization": {
            "add_bos": bool(add_bos),
            "add_eos": bool(add_eos),
        },
        "source": source_meta or {},
        "splits": splits_meta,
    }
    manifest_path = root / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path), manifest


def load_pretokenized_memmap_manifest(manifest_path: str) -> Dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != PRETOKENIZED_MEMMAP_FORMAT:
        raise ValueError(f"不支持的预分词数据格式: {manifest.get('format')}")
    manifest["_manifest_path"] = str(path.resolve())
    manifest["_root_dir"] = str(path.parent.resolve())
    return manifest


class PretokenizedMemmapSplit:
    def __init__(self, root_dir: str, split_meta: Dict[str, Any]):
        self.root_dir = Path(root_dir)
        self.meta = dict(split_meta)
        self.name = split_meta["name"]
        self.doc_count = int(split_meta["doc_count"])
        self.token_count = int(split_meta["token_count"])
        self.preview_text = split_meta.get("preview_text", "")

        self.tokens_path = self.root_dir / split_meta["tokens_bin"]
        self.offsets_path = self.root_dir / split_meta["offsets_npy"]
        self.lengths_path = self.root_dir / split_meta["lengths_npy"]

        self.offsets = np.load(self.offsets_path, mmap_mode="r")
        self.lengths = np.load(self.lengths_path, mmap_mode="r")
        if self.token_count > 0:
            self.tokens = np.memmap(self.tokens_path, dtype=np.int32, mode="r", shape=(self.token_count,))
        else:
            self.tokens = np.empty((0,), dtype=np.int32)

    @staticmethod
    def _close_memmap_array(arr: Any) -> None:
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def close(self) -> None:
        self._close_memmap_array(self.tokens)
        self._close_memmap_array(self.offsets)
        self._close_memmap_array(self.lengths)

    def get_doc_length(self, doc_idx: int) -> int:
        return int(self.lengths[doc_idx])

    def get_doc_offset(self, doc_idx: int) -> int:
        return int(self.offsets[doc_idx])

    def slice_doc_tokens(self, doc_idx: int, start: int, end: int) -> np.ndarray:
        doc_len = self.get_doc_length(doc_idx)
        if doc_len <= 0:
            return np.empty((0,), dtype=np.int32)
        bounded_start = max(0, min(start, doc_len))
        bounded_end = max(bounded_start, min(end, doc_len))
        if bounded_end <= bounded_start:
            return np.empty((0,), dtype=np.int32)
        offset = self.get_doc_offset(doc_idx)
        return np.asarray(self.tokens[offset + bounded_start: offset + bounded_end], dtype=np.int32)

    def get_doc_tokens(self, doc_idx: int) -> np.ndarray:
        return self.slice_doc_tokens(doc_idx, 0, self.get_doc_length(doc_idx))

    def tail_before_doc(self, doc_idx: int, max_tokens: int) -> np.ndarray:
        if max_tokens <= 0:
            return np.empty((0,), dtype=np.int32)
        offset = self.get_doc_offset(doc_idx)
        start = max(0, offset - max_tokens)
        return np.asarray(self.tokens[start:offset], dtype=np.int32)

    def tail_tokens(self, max_tokens: int) -> np.ndarray:
        if max_tokens <= 0 or self.token_count <= 0:
            return np.empty((0,), dtype=np.int32)
        start = max(0, self.token_count - max_tokens)
        return np.asarray(self.tokens[start:self.token_count], dtype=np.int32)


class MemmapDocChunkDataset(Dataset):
    def __init__(
        self,
        split: PretokenizedMemmapSplit,
        *,
        seq_len: int,
        pad_id: int,
        stride: Optional[int] = None,
        rosa_memory_tokens: int = 512,
        global_memory_tokens: int = 0,
        full_doc_memory: bool = False,
        doc_global_prefix_source: Optional[PretokenizedMemmapSplit] = None,
        shared_global_memory: Optional[Sequence[int]] = None,
    ):
        self.split = split
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.stride = stride or seq_len
        self.rosa_memory_tokens = rosa_memory_tokens
        self.global_memory_tokens = global_memory_tokens
        self.full_doc_memory = full_doc_memory
        self.doc_global_prefix_source = doc_global_prefix_source
        if shared_global_memory is None:
            self.shared_global_memory = np.empty((0,), dtype=np.int32)
        else:
            self.shared_global_memory = np.asarray(list(shared_global_memory), dtype=np.int32)

        self.sample_counts = np.asarray(
            [count_doc_samples(self.split.get_doc_length(doc_idx), self.seq_len, self.stride) for doc_idx in range(self.split.doc_count)],
            dtype=np.int64,
        )
        self.sample_offsets = np.cumsum(self.sample_counts, dtype=np.int64)
        self.total_samples = int(self.sample_offsets[-1]) if len(self.sample_offsets) > 0 else 0

    def __len__(self) -> int:
        return self.total_samples

    def close(self) -> None:
        self.split.close()
        if self.doc_global_prefix_source is not None and self.doc_global_prefix_source is not self.split:
            self.doc_global_prefix_source.close()

    def _resolve_sample_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)
        doc_idx = int(np.searchsorted(self.sample_offsets, idx, side="right"))
        prev = int(self.sample_offsets[doc_idx - 1]) if doc_idx > 0 else 0
        local_sample_idx = idx - prev
        return doc_idx, int(local_sample_idx * self.stride)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_idx, start = self._resolve_sample_index(idx)
        doc_len = self.split.get_doc_length(doc_idx)

        chunk = self.split.slice_doc_tokens(doc_idx, start, start + self.seq_len + 1).tolist()
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.seq_len + 1 - len(chunk))
        x = chunk[:-1]
        y = chunk[1:]

        if self.full_doc_memory:
            local_mem = self.split.slice_doc_tokens(doc_idx, 0, start)
        else:
            mem_start = max(0, start - self.rosa_memory_tokens)
            local_mem = self.split.slice_doc_tokens(doc_idx, mem_start, start)

        if self.doc_global_prefix_source is not None:
            global_mem = self.doc_global_prefix_source.tail_before_doc(doc_idx, self.global_memory_tokens)
        else:
            global_mem = self.shared_global_memory

        if local_mem.size == 0 and len(global_mem) == 0:
            memory = torch.empty((0,), dtype=torch.long)
        else:
            memory = torch.tensor(
                np.concatenate([np.asarray(global_mem, dtype=np.int32), np.asarray(local_mem, dtype=np.int32)]),
                dtype=torch.long,
            )

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "rosa_memory_ids": memory,
        }


def build_memmap_chunk_datasets_from_manifest(
    manifest_path: str,
    *,
    seq_len: int,
    pad_id: int,
    stride: Optional[int],
    rosa_memory_tokens: int,
    rosa_memory_mode: str,
    rosa_global_memory_tokens: int,
    rosa_train_mode: str,
    enable_train_address_cache: bool = False,
    enable_train_state_snapshot: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any], Dict[str, Any]]:
    manifest = load_pretokenized_memmap_manifest(manifest_path)
    root_dir = manifest["_root_dir"]
    train_split = PretokenizedMemmapSplit(root_dir, manifest["splits"]["train"])
    val_split = PretokenizedMemmapSplit(root_dir, manifest["splits"]["val"])
    test_split = PretokenizedMemmapSplit(root_dir, manifest["splits"]["test"])

    if rosa_train_mode != "online_seq":
        raise NotImplementedError("预分词 memmap v1 当前仅支持 rosa_train_mode=online_seq。")
    if enable_train_address_cache:
        raise NotImplementedError("预分词 memmap v1 当前未接入训练地址缓存，请使用异步地址预取主线。")
    if enable_train_state_snapshot:
        raise NotImplementedError("预分词 memmap v1 当前未接入训练状态快照。")

    global_cap = rosa_global_memory_tokens if rosa_global_memory_tokens > 0 else rosa_memory_tokens

    if rosa_memory_mode == "doc_local":
        train_ds = MemmapDocChunkDataset(
            train_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
            full_doc_memory=True,
        )
        val_ds = MemmapDocChunkDataset(
            val_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
            full_doc_memory=True,
        )
        test_ds = MemmapDocChunkDataset(
            test_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
            full_doc_memory=True,
        )
        memory_meta = {
            "requested_train_mode": rosa_train_mode,
            "effective_train_mode": "online_seq",
            "rosa_memory_mode": rosa_memory_mode,
            "doc_local_memory_tokens": 0,
            "configured_doc_local_memory_tokens": rosa_memory_tokens,
            "global_train_memory_tokens": 0,
            "train_global_memory_size": 0,
            "precomputed_doc_local_sam": False,
            "cached_online_seq_addresses": False,
            "state_snapshot_online_seq": False,
            "train_state_snapshot_interval": 0,
            "uses_full_doc_memory": True,
            "address_build_policy": "sequence_online",
            "effective_history": "full_doc_prefix_online_seq",
            "dataset_backend": "pretokenized_memmap",
            "manifest_path": manifest["_manifest_path"],
        }
    elif rosa_memory_mode == "global_train":
        shared_train_tail = train_split.tail_tokens(global_cap)
        train_ds = MemmapDocChunkDataset(
            train_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            global_memory_tokens=global_cap,
            full_doc_memory=True,
            doc_global_prefix_source=train_split,
        )
        val_ds = MemmapDocChunkDataset(
            val_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            global_memory_tokens=global_cap,
            full_doc_memory=True,
            shared_global_memory=shared_train_tail,
        )
        test_ds = MemmapDocChunkDataset(
            test_split,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            global_memory_tokens=global_cap,
            full_doc_memory=True,
            shared_global_memory=shared_train_tail,
        )
        memory_meta = {
            "requested_train_mode": rosa_train_mode,
            "effective_train_mode": "online_seq",
            "rosa_memory_mode": rosa_memory_mode,
            "doc_local_memory_tokens": 0,
            "configured_doc_local_memory_tokens": rosa_memory_tokens,
            "global_train_memory_tokens": global_cap,
            "train_global_memory_size": int(shared_train_tail.size),
            "precomputed_doc_local_sam": False,
            "cached_online_seq_addresses": False,
            "state_snapshot_online_seq": False,
            "train_state_snapshot_interval": 0,
            "uses_full_doc_memory": True,
            "address_build_policy": "sequence_online",
            "effective_history": f"global_train_tail_{global_cap}_plus_full_doc_prefix",
            "dataset_backend": "pretokenized_memmap",
            "manifest_path": manifest["_manifest_path"],
        }
    else:
        raise ValueError(f"未知 rosa_memory_mode: {rosa_memory_mode}")

    dataset_meta = {
        "source": {
            "mode": "pretokenized_manifest",
            "manifest_path": manifest["_manifest_path"],
            "format": manifest["format"],
            "source": manifest.get("source", {}),
        },
        "train_docs": train_split.doc_count,
        "val_docs": val_split.doc_count,
        "test_docs": test_split.doc_count,
        "train_preview_text": train_split.preview_text,
        "val_preview_text": val_split.preview_text,
        "test_preview_text": test_split.preview_text,
        "tokenization": manifest.get("tokenization", {}),
        "tokenizer": manifest.get("tokenizer", {}),
    }
    return train_ds, val_ds, test_ds, memory_meta, dataset_meta
