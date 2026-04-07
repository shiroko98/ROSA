
import argparse
import glob
import gzip
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rosa_runtime import RosaAddressBatch, RosaHotAddressCache, RosaInjectionPayload, RosaPrefetcher

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ByteTokenizer:
    """Fallback tokenizer for offline smoke tests. Not Qwen/LLaMA compatible."""

    def __init__(self):
        self.vocab_size = 259
        self.pad_token_id = 256
        self.eos_token_id = 257
        self.bos_token_id = 258
        self.special_ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode_id(self, idx: int) -> str:
        if idx == self.pad_token_id:
            return "<PAD>"
        if idx == self.eos_token_id:
            return "<EOS>"
        if idx == self.bos_token_id:
            return "<BOS>"
        try:
            return bytes([idx]).decode("utf-8")
        except Exception:
            return repr(bytes([idx]))


class HFTokenizerWrapper:
    def __init__(self, name_or_path: str):
        if AutoTokenizer is None:
            raise RuntimeError("transformers 未安装，无法加载 HF tokenizer。")
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        if tok.pad_token_id is None:
            if tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "<pad>"})
        self.tok = tok
        self.vocab_size = len(tok)
        self.pad_token_id = tok.pad_token_id
        self.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        self.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else self.eos_token_id
        self.special_ids = set(
            x for x in [self.pad_token_id, self.eos_token_id, self.bos_token_id] if x is not None
        )
        if getattr(tok, "all_special_ids", None):
            self.special_ids.update(tok.all_special_ids)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode_id(self, idx: int) -> str:
        if idx < 0:
            return "<NA>"
        try:
            s = self.tok.decode([idx], skip_special_tokens=False)
            s = s.replace("\n", "\\n")
            return s if s else f"<id:{idx}>"
        except Exception:
            return f"<id:{idx}>"


def build_tokenizer(name_or_path: Optional[str]):
    if name_or_path:
        return HFTokenizerWrapper(name_or_path)
    return ByteTokenizer()


def read_text(path: str) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return f.read()


def split_docs(text: str, split_mode: str) -> List[str]:
    if split_mode == "paragraph":
        docs = [x.strip() for x in text.split("\n\n") if x.strip()]
    elif split_mode == "line":
        docs = [x.strip() for x in text.splitlines() if x.strip()]
    elif split_mode == "stream":
        docs = [text.strip()] if text.strip() else []
    else:
        raise ValueError(f"未知 split_mode: {split_mode}")
    return docs


def parse_csv_arg(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv_arg(text: Optional[str]) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def has_glob_magic(path_spec: str) -> bool:
    return any(ch in path_spec for ch in ["*", "?", "["])


def resolve_data_files(path_spec: str) -> List[str]:
    if not path_spec:
        raise ValueError("数据路径不能为空。")

    if has_glob_magic(path_spec):
        paths = sorted(glob.glob(path_spec, recursive=True))
    else:
        path = Path(path_spec)
        if path.is_dir():
            paths = sorted(str(p) for p in path.rglob("*") if p.is_file())
        elif path.exists():
            paths = [str(path)]
        else:
            paths = []

    files = [str(Path(p)) for p in paths if Path(p).is_file()]
    if not files:
        raise FileNotFoundError(f"未找到任何数据文件: {path_spec}")
    return files


def infer_data_format(path: str, data_format: str) -> str:
    if data_format != "auto":
        return data_format
    lower = path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        return "jsonl"
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        return "json"
    return "text"


def get_nested_value(obj: Any, dotted_key: str) -> Any:
    cur = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return "\n".join(x for x in value if x)
    return None


def extract_text_from_record(record: Any, json_text_keys: Sequence[str]) -> Optional[str]:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in json_text_keys:
            value = get_nested_value(record, key)
            text = coerce_text(value)
            if text is not None:
                return text
        if len(record) == 1:
            only_value = next(iter(record.values()))
            text = coerce_text(only_value)
            if text is not None:
                return text
    return None


def load_docs_from_path(
    path_spec: str,
    *,
    data_format: str,
    split_mode: str,
    json_text_keys: Sequence[str],
    max_docs: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    files = resolve_data_files(path_spec)
    docs: List[str] = []
    skipped_records = 0
    example_keys: Optional[List[str]] = None
    format_counts: Dict[str, int] = {}

    for path in files:
        cur_format = infer_data_format(path, data_format)
        format_counts[cur_format] = format_counts.get(cur_format, 0) + 1

        if cur_format == "text":
            for doc in split_docs(read_text(path), split_mode):
                docs.append(doc)
                if max_docs is not None and len(docs) >= max_docs:
                    break
        elif cur_format == "jsonl":
            opener = gzip.open if path.lower().endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL 解析失败: {path}:{line_no} -> {exc}") from exc
                    if example_keys is None and isinstance(record, dict):
                        example_keys = sorted(record.keys())
                    doc = extract_text_from_record(record, json_text_keys)
                    if doc is None or not doc.strip():
                        skipped_records += 1
                        continue
                    docs.append(doc.strip())
                    if max_docs is not None and len(docs) >= max_docs:
                        break
        elif cur_format == "json":
            opener = gzip.open if path.lower().endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload if isinstance(payload, list) else [payload]
            for idx, record in enumerate(records, start=1):
                if example_keys is None and isinstance(record, dict):
                    example_keys = sorted(record.keys())
                doc = extract_text_from_record(record, json_text_keys)
                if doc is None or not doc.strip():
                    skipped_records += 1
                    continue
                docs.append(doc.strip())
                if max_docs is not None and len(docs) >= max_docs:
                    break
        else:
            raise ValueError(f"不支持的数据格式: {cur_format}")

        if max_docs is not None and len(docs) >= max_docs:
            break

    if not docs:
        detail = f" 可用 JSON keys 示例: {example_keys}" if example_keys else ""
        raise ValueError(f"未从 `{path_spec}` 中解析出任何文档。{detail}")

    return docs, {
        "path_spec": path_spec,
        "resolved_files": files,
        "file_count": len(files),
        "doc_count": len(docs),
        "skipped_records": skipped_records,
        "format_counts": format_counts,
    }


def train_val_test_split(docs: List[str], train_ratio: float, val_ratio: float, seed: int):
    idx = list(range(len(docs)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    docs = [docs[i] for i in idx]
    n = len(docs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"文档数太少，当前 split 后得到 train={n_train}, val={n_val}, test={n_test}。")
    return docs[:n_train], docs[n_train:n_train + n_val], docs[n_train + n_val:]


def tokenize_docs(docs: Sequence[str], tokenizer, add_bos: bool, add_eos: bool) -> List[List[int]]:
    out = []
    for doc in docs:
        ids: List[int] = []
        if add_bos and tokenizer.bos_token_id is not None:
            ids.append(tokenizer.bos_token_id)
        ids.extend(tokenizer.encode(doc))
        if add_eos and tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        out.append(ids)
    return out


def tail_tokens(tokens: Sequence[int], max_tokens: int) -> List[int]:
    if max_tokens is None or max_tokens <= 0:
        return []
    if len(tokens) <= max_tokens:
        return list(tokens)
    return list(tokens[-max_tokens:])


def build_global_memory_prefixes(
    docs_tokens: Sequence[Sequence[int]],
    max_tokens: int,
) -> Tuple[List[List[int]], List[int]]:
    if max_tokens <= 0:
        return [[] for _ in docs_tokens], []

    prefixes: List[List[int]] = []
    running: List[int] = []
    for ids in docs_tokens:
        prefixes.append(list(running))
        if ids:
            merged = running + list(ids)
            running = merged[-max_tokens:]
    return prefixes, list(running)


def build_doc_local_precomputed_rosa(
    docs_tokens: Sequence[Sequence[int]],
    *,
    min_match_len: int,
    special_ids: Optional[set],
    forbid_special_target: bool,
) -> List[Dict[str, List[int]]]:
    special_ids = special_ids or set()
    out: List[Dict[str, List[int]]] = []
    for ids in docs_tokens:
        preds, match_lens = sam_rosa_predict(ids, min_match_len=min_match_len)
        rosa_ids: List[int] = []
        fired_match_lens: List[int] = []
        raw_best_lens: List[int] = []
        for pred, raw_m in zip(preds, match_lens):
            raw_best_lens.append(raw_m)
            if pred >= 0 and (not forbid_special_target or pred not in special_ids):
                rosa_ids.append(pred)
                fired_match_lens.append(raw_m)
            else:
                rosa_ids.append(-1)
                fired_match_lens.append(0)
        out.append(
            {
                "rosa_ids": rosa_ids,
                "fired_match_lens": fired_match_lens,
                "raw_best_lens": raw_best_lens,
            }
        )
    return out


class DocChunkDataset(Dataset):
    """
    以“文档连续流”的方式切 chunk，并为每个 chunk 返回其左侧历史 memory。
    memory 只来自同一文档的前文，不跨文档，不需要 SEP，因此更接近我们讨论的 ROSA memory 场景。
    """

    def __init__(
        self,
        docs_tokens: Sequence[Sequence[int]],
        seq_len: int,
        pad_id: int,
        stride: Optional[int] = None,
        rosa_memory_tokens: int = 512,
        global_memory_tokens: int = 0,
        doc_global_prefixes: Optional[Sequence[Sequence[int]]] = None,
        shared_global_memory: Optional[Sequence[int]] = None,
        doc_precomputed_rosa: Optional[Sequence[Dict[str, Sequence[int]]]] = None,
    ):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.rosa_memory_tokens = rosa_memory_tokens
        self.global_memory_tokens = global_memory_tokens
        self.samples: List[Dict[str, List[int]]] = []
        stride = stride or seq_len
        if doc_global_prefixes is not None and len(doc_global_prefixes) != len(docs_tokens):
            raise ValueError("doc_global_prefixes 长度必须与 docs_tokens 一致。")
        if doc_precomputed_rosa is not None and len(doc_precomputed_rosa) != len(docs_tokens):
            raise ValueError("doc_precomputed_rosa 长度必须与 docs_tokens 一致。")

        global_shared_tail = tail_tokens(shared_global_memory or [], global_memory_tokens)

        for doc_idx, ids in enumerate(docs_tokens):
            if len(ids) < 2:
                continue
            doc_global_prefix = (
                tail_tokens(doc_global_prefixes[doc_idx], global_memory_tokens)
                if doc_global_prefixes is not None
                else global_shared_tail
            )
            max_start = max(1, len(ids) - 1)
            for start in range(0, max_start, stride):
                chunk = list(ids[start:start + seq_len + 1])
                if len(chunk) < 2:
                    continue
                if len(chunk) < seq_len + 1:
                    chunk = chunk + [pad_id] * (seq_len + 1 - len(chunk))
                x = chunk[:-1]
                y = chunk[1:]

                mem_start = max(0, start - rosa_memory_tokens)
                local_mem = list(ids[mem_start:start])
                mem = doc_global_prefix + local_mem
                sample: Dict[str, List[int]] = {
                    "input_ids": x,
                    "labels": y,
                    "rosa_memory_ids": mem,
                }
                if doc_precomputed_rosa is not None:
                    pre = doc_precomputed_rosa[doc_idx]
                    pre_ids = list(pre["rosa_ids"][start:start + seq_len])
                    pre_fired = list(pre["fired_match_lens"][start:start + seq_len])
                    pre_raw = list(pre["raw_best_lens"][start:start + seq_len])
                    if len(pre_ids) < seq_len:
                        pad_n = seq_len - len(pre_ids)
                        pre_ids = pre_ids + [-1] * pad_n
                        pre_fired = pre_fired + [0] * pad_n
                        pre_raw = pre_raw + [0] * pad_n
                    sample["rosa_precomputed_ids"] = pre_ids
                    sample["rosa_precomputed_match_lens"] = pre_fired
                    sample["rosa_precomputed_raw_best_lens"] = pre_raw
                self.samples.append(sample)

                if start + seq_len + 1 >= len(ids):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        out = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "rosa_memory_ids": torch.tensor(sample["rosa_memory_ids"], dtype=torch.long),
        }
        if "rosa_precomputed_ids" in sample:
            out["rosa_precomputed_ids"] = torch.tensor(sample["rosa_precomputed_ids"], dtype=torch.long)
            out["rosa_precomputed_match_lens"] = torch.tensor(sample["rosa_precomputed_match_lens"], dtype=torch.long)
            out["rosa_precomputed_raw_best_lens"] = torch.tensor(
                sample["rosa_precomputed_raw_best_lens"], dtype=torch.long
            )
        return out


def make_collate_fn(pad_id: int):
    def collate(batch):
        xs = torch.stack([b["input_ids"] for b in batch], dim=0)
        ys = torch.stack([b["labels"] for b in batch], dim=0)
        max_mem = max((b["rosa_memory_ids"].numel() for b in batch), default=0)
        if max_mem == 0:
            mem = torch.empty((len(batch), 0), dtype=torch.long)
        else:
            mem = torch.full((len(batch), max_mem), pad_id, dtype=torch.long)
            for i, b in enumerate(batch):
                cur = b["rosa_memory_ids"]
                if cur.numel() > 0:
                    mem[i, -cur.numel():] = cur
        out = {"input_ids": xs, "labels": ys, "rosa_memory_ids": mem}
        if "rosa_precomputed_ids" in batch[0]:
            out["rosa_precomputed_ids"] = torch.stack([b["rosa_precomputed_ids"] for b in batch], dim=0)
            out["rosa_precomputed_match_lens"] = torch.stack(
                [b["rosa_precomputed_match_lens"] for b in batch], dim=0
            )
            out["rosa_precomputed_raw_best_lens"] = torch.stack(
                [b["rosa_precomputed_raw_best_lens"] for b in batch], dim=0
            )
        return out
    return collate


def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    *,
    batch_size: int,
    pad_id: int,
    train_seed: int,
):
    collate_fn = make_collate_fn(pad_id)
    train_generator = torch.Generator()
    train_generator.manual_seed(train_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def build_chunk_datasets(
    train_tok: Sequence[Sequence[int]],
    val_tok: Sequence[Sequence[int]],
    test_tok: Sequence[Sequence[int]],
    *,
    seq_len: int,
    pad_id: int,
    stride: Optional[int],
    rosa_memory_tokens: int,
    rosa_memory_mode: str,
    rosa_global_memory_tokens: int,
    rosa_backend: str,
    rosa_min_match_len: int,
    special_ids: Optional[set],
    forbid_special_target: bool,
):
    if rosa_memory_mode == "doc_local" and rosa_backend == "sam":
        train_pre = build_doc_local_precomputed_rosa(
            train_tok,
            min_match_len=rosa_min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        val_pre = build_doc_local_precomputed_rosa(
            val_tok,
            min_match_len=rosa_min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        test_pre = build_doc_local_precomputed_rosa(
            test_tok,
            min_match_len=rosa_min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        train_ds = DocChunkDataset(
            train_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            doc_precomputed_rosa=train_pre,
        )
        val_ds = DocChunkDataset(
            val_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            doc_precomputed_rosa=val_pre,
        )
        test_ds = DocChunkDataset(
            test_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=0,
            doc_precomputed_rosa=test_pre,
        )
        meta = {
            "rosa_memory_mode": rosa_memory_mode,
            "doc_local_memory_tokens": 0,
            "global_train_memory_tokens": 0,
            "train_global_memory_size": 0,
            "precomputed_doc_local_sam": True,
            "effective_history": "full_doc_prefix",
            "train_precomputed_docs": len(train_pre),
            "val_precomputed_docs": len(val_pre),
            "test_precomputed_docs": len(test_pre),
        }
        return train_ds, val_ds, test_ds, meta

    if rosa_memory_mode == "doc_local":
        train_ds = DocChunkDataset(
            train_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
        )
        val_ds = DocChunkDataset(
            val_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
        )
        test_ds = DocChunkDataset(
            test_tok,
            seq_len=seq_len,
            pad_id=pad_id,
            stride=stride,
            rosa_memory_tokens=rosa_memory_tokens,
        )
        meta = {
            "rosa_memory_mode": rosa_memory_mode,
            "doc_local_memory_tokens": rosa_memory_tokens,
            "global_train_memory_tokens": 0,
            "train_global_memory_size": 0,
            "precomputed_doc_local_sam": False,
            "effective_history": f"doc_local_tail_{rosa_memory_tokens}",
        }
        return train_ds, val_ds, test_ds, meta

    if rosa_memory_mode != "global_train":
        raise ValueError(f"未知 rosa_memory_mode: {rosa_memory_mode}")

    global_cap = rosa_global_memory_tokens if rosa_global_memory_tokens > 0 else rosa_memory_tokens
    train_prefixes, full_train_memory = build_global_memory_prefixes(train_tok, global_cap)

    train_ds = DocChunkDataset(
        train_tok,
        seq_len=seq_len,
        pad_id=pad_id,
        stride=stride,
        rosa_memory_tokens=rosa_memory_tokens,
        global_memory_tokens=global_cap,
        doc_global_prefixes=train_prefixes,
    )
    val_ds = DocChunkDataset(
        val_tok,
        seq_len=seq_len,
        pad_id=pad_id,
        stride=stride,
        rosa_memory_tokens=rosa_memory_tokens,
        global_memory_tokens=global_cap,
        shared_global_memory=full_train_memory,
    )
    test_ds = DocChunkDataset(
        test_tok,
        seq_len=seq_len,
        pad_id=pad_id,
        stride=stride,
        rosa_memory_tokens=rosa_memory_tokens,
        global_memory_tokens=global_cap,
        shared_global_memory=full_train_memory,
    )
    meta = {
        "rosa_memory_mode": rosa_memory_mode,
        "doc_local_memory_tokens": rosa_memory_tokens,
        "global_train_memory_tokens": global_cap,
        "train_global_memory_size": len(full_train_memory),
        "precomputed_doc_local_sam": False,
        "effective_history": f"global_train_tail_{global_cap}_plus_doc_local_tail_{rosa_memory_tokens}",
    }
    return train_ds, val_ds, test_ds, meta


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 256
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 8
    intermediate_size: int = 768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos[None, None, :, :], sin[None, None, :, :]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        assert cfg.dim % cfg.n_heads == 0
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.dropout = cfg.dropout

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(q, seqlen)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if attn_mask is None:
            attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            attn_mask = attn_mask[None, None, :, :]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.cfg.dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.post_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), attn_mask)
        x = x + self.mlp(self.post_norm(x))
        return x


def _pad_filtered_mem(mem_row: List[int], pad_id: int) -> List[int]:
    return [x for x in mem_row if x != pad_id]


class RosaValueStore(nn.Module):
    """ROSA value lookup 抽象层：shared embedding 或 per-layer value table。"""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        inject_layers: int,
        mode: str = "shared",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.inject_layers = inject_layers
        self.mode = mode

        if mode == "shared":
            self.per_layer_tables = None
        elif mode == "per_layer":
            self.per_layer_tables = nn.ModuleList(
                [nn.Embedding(vocab_size, dim) for _ in range(max(0, inject_layers))]
            )
        else:
            raise ValueError(f"未知 rosa_value_mode: {mode}")

    @property
    def is_per_layer(self) -> bool:
        return self.mode == "per_layer"

    def copy_shared_weights_(self, shared_embedding: nn.Embedding) -> None:
        if not self.is_per_layer or self.per_layer_tables is None:
            return
        with torch.no_grad():
            for table in self.per_layer_tables:
                table.weight.copy_(shared_embedding.weight)

    def lookup(
        self,
        layer_idx: int,
        addr_ids: torch.Tensor,
        *,
        shared_embedding: nn.Embedding,
    ) -> torch.Tensor:
        addr_ids_safe = addr_ids.clamp_min(0)
        if not self.is_per_layer:
            return shared_embedding(addr_ids_safe)
        if self.per_layer_tables is None or layer_idx >= len(self.per_layer_tables):
            raise IndexError(f"layer_idx={layer_idx} 超出 RosaValueStore 可用层数。")
        return self.per_layer_tables[layer_idx](addr_ids_safe)

    def lookup_with_hot_cache(
        self,
        layer_idx: int,
        addr_ids: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor],
        shared_embedding: nn.Embedding,
        hot_cache: Optional[RosaHotAddressCache],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        addr_ids_safe = addr_ids.clamp_min(0)

        def fetch_fn(ids: torch.Tensor) -> torch.Tensor:
            if not self.is_per_layer:
                return shared_embedding(ids)
            if self.per_layer_tables is None or layer_idx >= len(self.per_layer_tables):
                raise IndexError(f"layer_idx={layer_idx} 超出 RosaValueStore 可用层数。")
            return self.per_layer_tables[layer_idx](ids)

        if hot_cache is None:
            return fetch_fn(addr_ids_safe), {}
        return hot_cache.lookup(
            layer_idx,
            addr_ids_safe,
            valid_mask=valid_mask,
            value_dim=self.dim,
            fetch_fn=fetch_fn,
        )


def init_identity_linear_(proj: nn.Linear) -> None:
    if proj.weight.shape[0] != proj.weight.shape[1]:
        raise ValueError("仅支持方阵线性层做 identity 初始化。")
    with torch.no_grad():
        proj.weight.zero_()
        proj.weight.copy_(torch.eye(proj.weight.shape[0], device=proj.weight.device, dtype=proj.weight.dtype))


@dataclass(frozen=True)
class AddressMeta:
    addr_id: int
    raw_match_len: int
    fired_match_len: int
    valid_mask: bool
    special_mask: bool = False
    source_type: str = "token_exact"

    @property
    def hit_flag(self) -> bool:
        return self.valid_mask


@dataclass(frozen=True)
class RosaStateSnapshot:
    token_ids: Tuple[int, ...]
    num_tokens: int
    last_address: Optional[AddressMeta] = None


def build_address_meta(
    addr_id: int,
    raw_match_len: int,
    *,
    min_match_len: int,
    special_mask: bool = False,
    source_type: str = "token_exact",
) -> AddressMeta:
    valid = addr_id >= 0 and raw_match_len >= min_match_len and not special_mask
    fired = raw_match_len if valid else 0
    return AddressMeta(
        addr_id=addr_id if valid else -1,
        raw_match_len=raw_match_len,
        fired_match_len=fired,
        valid_mask=valid,
        special_mask=special_mask,
        source_type=source_type,
    )


def exact_match_step_address(
    history: Sequence[int],
    token_id: int,
    *,
    min_match_len: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    source_type: str = "token_exact",
) -> AddressMeta:
    special_ids = special_ids or set()
    gi = len(history)
    combined = list(history) + [token_id]

    best_len = 0
    best_j = -1
    for j in range(gi):
        if combined[j] != token_id:
            continue
        m = 1
        while j - m >= 0 and gi - m >= 0 and combined[j - m] == combined[gi - m]:
            m += 1
        if m > best_len or (m == best_len and j > best_j):
            best_len = m
            best_j = j

    addr_id = -1
    special_mask = False
    if best_j >= 0 and best_j + 1 < len(combined):
        candidate = combined[best_j + 1]
        if forbid_special_target and candidate in special_ids:
            special_mask = True
        else:
            addr_id = candidate

    return build_address_meta(
        addr_id,
        best_len,
        min_match_len=min_match_len,
        special_mask=special_mask,
        source_type=source_type,
    )


class OnlineRosaState:
    """
    参考级在线 ROSA 状态机。
    先保证 prefill / update_one 语义稳定，与现有整段 exact-match 定义逐 token 对齐；
    后续再在这个接口下替换为更高性能的状态结构。
    """

    def __init__(
        self,
        *,
        min_match_len: int = 1,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
    ):
        self.min_match_len = min_match_len
        self.special_ids = set(special_ids or set())
        self.forbid_special_target = forbid_special_target
        self.source_type = source_type
        self.reset()

    def reset(self) -> None:
        self._token_ids: List[int] = []
        self._last_address: Optional[AddressMeta] = None

    def prefill(self, token_ids: Sequence[int]) -> List[AddressMeta]:
        return [self.update_one(token_id) for token_id in token_ids]

    def update_one(self, token_id: int) -> AddressMeta:
        meta = exact_match_step_address(
            self._token_ids,
            token_id,
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            source_type=self.source_type,
        )
        self._token_ids.append(token_id)
        self._last_address = meta
        return meta

    def snapshot(self) -> RosaStateSnapshot:
        return RosaStateSnapshot(
            token_ids=tuple(self._token_ids),
            num_tokens=len(self._token_ids),
            last_address=self._last_address,
        )


def _coerce_batch_token_rows(token_ids: Any) -> List[List[int]]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() == 1:
            return [token_ids.detach().cpu().tolist()]
        if token_ids.dim() == 2:
            return token_ids.detach().cpu().tolist()
        raise ValueError("token_ids tensor 只能是 1D 或 2D。")
    return [list(row) for row in token_ids]


class OnlineRosaBatchState:
    """批量在线 ROSA 状态，提供 prefill + decode/update 最小闭环。"""

    def __init__(self, states: Sequence[OnlineRosaState]):
        self.states = list(states)

    @classmethod
    def create(
        cls,
        batch_size: int,
        *,
        min_match_len: int = 1,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
    ) -> "OnlineRosaBatchState":
        return cls(
            [
                OnlineRosaState(
                    min_match_len=min_match_len,
                    special_ids=special_ids,
                    forbid_special_target=forbid_special_target,
                    source_type=source_type,
                )
                for _ in range(batch_size)
            ]
        )

    def reset(self) -> None:
        for state in self.states:
            state.reset()

    def prefill(self, token_ids: Any, *, pad_id: Optional[int] = None) -> None:
        rows = _coerce_batch_token_rows(token_ids)
        if len(rows) != len(self.states):
            raise ValueError("prefill 的 batch 大小必须与 OnlineRosaBatchState 中的 state 数量一致。")
        for state, row in zip(self.states, rows):
            filtered = [tok for tok in row if pad_id is None or tok != pad_id]
            if filtered:
                state.prefill(filtered)

    def address_tokens(
        self,
        token_ids: Any,
        *,
        pad_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        rows = _coerce_batch_token_rows(token_ids)
        if len(rows) != len(self.states):
            raise ValueError("address_tokens 的 batch 大小必须与 OnlineRosaBatchState 中的 state 数量一致。")

        batch_meta: List[List[AddressMeta]] = []
        for state, row in zip(self.states, rows):
            row_meta: List[AddressMeta] = []
            for tok in row:
                if pad_id is not None and tok == pad_id:
                    row_meta.append(
                        build_address_meta(
                            -1,
                            0,
                            min_match_len=state.min_match_len,
                            source_type=state.source_type,
                        )
                    )
                    continue
                row_meta.append(state.update_one(tok))
            batch_meta.append(row_meta)
        return pack_address_meta_batch(batch_meta, device=device)

    def snapshot(self) -> List[RosaStateSnapshot]:
        return [state.snapshot() for state in self.states]


def build_online_rosa_batch_state(
    batch_size: int,
    *,
    min_match_len: int = 1,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    source_type: str = "token_exact",
) -> OnlineRosaBatchState:
    return OnlineRosaBatchState.create(
        batch_size,
        min_match_len=min_match_len,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
        source_type=source_type,
    )


def pack_address_meta_batch(
    batch_meta: Sequence[Sequence[AddressMeta]],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    bsz = len(batch_meta)
    seqlen = max((len(row) for row in batch_meta), default=0)

    addr_ids = torch.full((bsz, seqlen), -1, dtype=torch.long, device=device)
    raw_match_lens = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
    fired_match_lens = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
    valid_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)
    special_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)

    for b, row in enumerate(batch_meta):
        for i, meta in enumerate(row):
            addr_ids[b, i] = meta.addr_id
            raw_match_lens[b, i] = meta.raw_match_len
            fired_match_lens[b, i] = meta.fired_match_len
            valid_mask[b, i] = meta.valid_mask
            special_mask[b, i] = meta.special_mask

    return {
        "addr_ids": addr_ids,
        "raw_match_lens": raw_match_lens,
        "fired_match_lens": fired_match_lens,
        "valid_mask": valid_mask,
        "special_mask": special_mask,
    }


def make_rosa_address_batch(addressed: Dict[str, torch.Tensor], *, source: str = "unknown") -> RosaAddressBatch:
    return RosaAddressBatch(
        addr_ids=addressed["addr_ids"],
        raw_match_lens=addressed["raw_match_lens"],
        fired_match_lens=addressed["fired_match_lens"],
        valid_mask=addressed["valid_mask"],
        special_mask=addressed["special_mask"],
        source=source,
    )


def online_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, seqlen = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        row: List[AddressMeta] = []
        state = OnlineRosaState(
            min_match_len=min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        if mem:
            state.prefill(mem)
        row.extend(state.prefill(seq))
        out.append(row)
    return out


def naive_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, seqlen = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        combined = mem + seq
        offset = len(mem)
        row: List[AddressMeta] = []

        for i in range(seqlen):
            gi = offset + i
            cur = combined[gi]
            if cur < 0:
                row.append(build_address_meta(-1, 0, min_match_len=min_match_len))
                continue

            best_len = 0
            best_j = -1
            for j in range(gi):
                if combined[j] != cur:
                    continue
                m = 1
                while j - m >= 0 and gi - m >= 0 and combined[j - m] == combined[gi - m]:
                    m += 1
                if m > best_len or (m == best_len and j > best_j):
                    best_len = m
                    best_j = j

            addr_id = -1
            special_mask = False
            if best_j >= 0 and best_j + 1 < len(combined):
                candidate = combined[best_j + 1]
                if forbid_special_target and candidate in special_ids:
                    special_mask = True
                else:
                    addr_id = candidate
            row.append(
                build_address_meta(
                    addr_id,
                    best_len,
                    min_match_len=min_match_len,
                    special_mask=special_mask,
                )
            )
        out.append(row)

    return out


def sam_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, seqlen = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        combined = mem + seq
        offset = len(mem)

        preds, match_lens = sam_rosa_predict(combined, min_match_len=min_match_len)
        sliced_preds = preds[offset:offset + seqlen]
        sliced_match_lens = match_lens[offset:offset + seqlen]

        row: List[AddressMeta] = []
        for pred, raw_m in zip(sliced_preds, sliced_match_lens):
            special_mask = pred >= 0 and forbid_special_target and pred in special_ids
            row.append(
                build_address_meta(
                    pred,
                    raw_m,
                    min_match_len=min_match_len,
                    special_mask=special_mask,
                )
            )
        out.append(row)

    return out


def rosa_addressing_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    backend: str = "sam",
) -> Dict[str, torch.Tensor]:
    if backend == "naive":
        batch_meta = naive_rosa_address_meta_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    elif backend == "sam":
        batch_meta = sam_rosa_address_meta_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    else:
        raise ValueError(f"未知 rosa backend: {backend}")
    return pack_address_meta_batch(batch_meta, device=input_ids.device)


def sam_rosa_predict(seq: Sequence[int], min_match_len: int = 1) -> Tuple[List[int], List[int]]:
    n = len(seq)
    pred = [-1] * n
    match_len = [0] * n
    if n == 0:
        return pred, match_len

    s = 2 * n + 1
    trans: List[Optional[Dict[int, int]]] = [None] * s
    link = [-1] * s
    length = [0] * s
    endpos = [-1] * s

    trans[0] = {}
    last = 0
    z = 1

    for i, t in enumerate(seq):
        r = z
        z += 1
        trans[r] = {}
        length[r] = length[last] + 1
        p = last

        while p != -1 and t not in trans[p]:
            trans[p][t] = r
            p = link[p]

        if p == -1:
            link[r] = 0
        else:
            q = trans[p][t]
            if length[p] + 1 == length[q]:
                link[r] = q
            else:
                u = z
                z += 1
                trans[u] = trans[q].copy()
                length[u] = length[p] + 1
                link[u] = link[q]
                endpos[u] = endpos[q]

                while p != -1 and trans[p].get(t) == q:
                    trans[p][t] = u
                    p = link[p]

                link[q] = u
                link[r] = u

        v = r
        a = -1
        best_m = 0
        while v != -1:
            if length[v] > 0 and endpos[v] >= 0:
                best_m = length[v]
                if best_m >= min_match_len:
                    idx = endpos[v] + 1
                    if 0 <= idx < n:
                        a = seq[idx]
                break
            v = link[v]

        pred[i] = a
        match_len[i] = best_m
        last = r

        v = last
        while v != -1 and endpos[v] < i:
            endpos[v] = i
            v = link[v]

    return pred, match_len


def naive_rosa_retrieval_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    addressed = rosa_addressing_with_memory(
        input_ids=input_ids,
        memory_ids=memory_ids,
        min_match_len=min_match_len,
        pad_id=pad_id,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
        backend="naive",
    )
    return addressed["addr_ids"], addressed["fired_match_lens"], addressed["raw_match_lens"]


def sam_rosa_retrieval_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    addressed = rosa_addressing_with_memory(
        input_ids=input_ids,
        memory_ids=memory_ids,
        min_match_len=min_match_len,
        pad_id=pad_id,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
        backend="sam",
    )
    return addressed["addr_ids"], addressed["fired_match_lens"], addressed["raw_match_lens"]


def rosa_retrieval_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    backend: str = "sam",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend == "naive":
        return naive_rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    if backend == "sam":
        return sam_rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    raise ValueError(f"未知 rosa backend: {backend}")


class BaseLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        seqlen = input_ids.shape[1]
        attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)[None, None, :, :]
        for blk in self.layers:
            x = blk(x, attn_mask)
        x = self.norm(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ):
        hidden = self.forward_hidden(
            input_ids,
            rosa_memory_ids=rosa_memory_ids,
            rosa_online_state=rosa_online_state,
            rosa_precomputed_ids=rosa_precomputed_ids,
            rosa_precomputed_match_lens=rosa_precomputed_match_lens,
            rosa_precomputed_raw_best_lens=rosa_precomputed_raw_best_lens,
        )
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            out["loss"] = loss
        return out


class RosaFusedLM(BaseLM):
    def __init__(
        self,
        cfg: ModelConfig,
        pad_id: int,
        rosa_backend: str = "sam",
        min_match_len: int = 1,
        inject_layers: int = 2,
        inject_layer_ids: Optional[Sequence[int]] = None,
        rosa_scale: float = 0.25,
        rosa_value_mode: str = "shared",
        use_context_gate: bool = False,
        rosa_hot_cache_size: int = 0,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        use_match_len_gate: bool = True,
    ):
        super().__init__(cfg)
        self.pad_id = pad_id
        self.rosa_backend = rosa_backend
        self.min_match_len = min_match_len
        if inject_layer_ids:
            normalized_ids = sorted({int(x) for x in inject_layer_ids})
        else:
            normalized_ids = list(range(max(0, min(inject_layers, cfg.n_layers))))
        for layer_id in normalized_ids:
            if layer_id < 0 or layer_id >= cfg.n_layers:
                raise ValueError(f"inject layer id 超出范围: {layer_id}, n_layers={cfg.n_layers}")
        self.inject_layer_ids = tuple(normalized_ids)
        self.inject_layer_index = {layer_id: slot_idx for slot_idx, layer_id in enumerate(self.inject_layer_ids)}
        self.inject_layers = len(self.inject_layer_ids)
        self.rosa_scale = rosa_scale
        self.rosa_value_mode = rosa_value_mode
        self.use_context_gate = use_context_gate
        self.rosa_hot_cache_size = max(0, int(rosa_hot_cache_size))
        self.special_ids = special_ids or set()
        self.forbid_special_target = forbid_special_target
        self.use_match_len_gate = use_match_len_gate
        self.rosa_value_store = RosaValueStore(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            inject_layers=self.inject_layers,
            mode=rosa_value_mode,
        )
        self.rosa_value_store.copy_shared_weights_(self.embed_tokens)
        self.rosa_hot_cache = (
            RosaHotAddressCache(
                num_layers=self.inject_layers,
                max_entries_per_layer=self.rosa_hot_cache_size,
            )
            if self.rosa_hot_cache_size > 0
            else None
        )
        if use_context_gate:
            self.rosa_gate_hidden_norms = nn.ModuleList(
                [RMSNorm(cfg.dim, cfg.rms_norm_eps) for _ in range(self.inject_layers)]
            )
            self.rosa_gate_value_norms = nn.ModuleList(
                [RMSNorm(cfg.dim, cfg.rms_norm_eps) for _ in range(self.inject_layers)]
            )
            self.rosa_gate_key_projs = nn.ModuleList(
                [nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(self.inject_layers)]
            )
            self.rosa_gate_value_projs = nn.ModuleList(
                [nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(self.inject_layers)]
            )
            for proj in self.rosa_gate_key_projs:
                init_identity_linear_(proj)
            for proj in self.rosa_gate_value_projs:
                init_identity_linear_(proj)
            self.rosa_gate_match_len_scale = nn.Parameter(torch.ones(self.inject_layers))
            self.rosa_gate_bias = nn.Parameter(torch.zeros(self.inject_layers))
        else:
            self.rosa_gate_hidden_norms = None
            self.rosa_gate_value_norms = None
            self.rosa_gate_key_projs = None
            self.rosa_gate_value_projs = None
            self.rosa_gate_match_len_scale = None
            self.rosa_gate_bias = None

    def reset_hot_cache(self, *, clear_cache: bool = True) -> None:
        if self.rosa_hot_cache is None:
            return
        if clear_cache:
            self.rosa_hot_cache.reset()
        else:
            self.rosa_hot_cache.reset_stats(clear_cache=False)

    def get_hot_cache_stats(self, *, top_k: int = 5) -> Dict[str, Any]:
        if self.rosa_hot_cache is None:
            return {
                "enabled": False,
                "max_entries_per_layer": 0,
                "token_requests": 0.0,
                "token_hits": 0.0,
                "unique_requests": 0.0,
                "unique_hits": 0.0,
                "fills": 0.0,
                "evictions": 0.0,
                "token_hit_rate": 0.0,
                "unique_hit_rate": 0.0,
                "active_entries": 0,
                "layer_stats": [],
            }
        return self.rosa_hot_cache.stats(top_k=top_k)

    def init_online_state(self, batch_size: int) -> OnlineRosaBatchState:
        return build_online_rosa_batch_state(
            batch_size,
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
        )

    def init_prefetcher(
        self,
        *,
        use_async: bool = True,
        use_pinned_memory: bool = False,
        max_workers: int = 1,
    ) -> RosaPrefetcher:
        model_device = next(self.parameters()).device
        return RosaPrefetcher(
            builder=lambda address_batch: self.build_rosa_injection_payload(address_batch, device=model_device),
            use_async=use_async,
            supports_async=model_device.type == "cpu",
            use_pinned_memory=use_pinned_memory and model_device.type == "cpu",
            max_workers=max_workers,
        )

    def compute_rosa_address_batch(
        self,
        input_ids: torch.Tensor,
        *,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ) -> RosaAddressBatch:
        if rosa_precomputed_ids is not None:
            fired_match_lens = (
                rosa_precomputed_match_lens
                if rosa_precomputed_match_lens is not None
                else torch.zeros_like(rosa_precomputed_ids)
            )
            raw_match_lens = (
                rosa_precomputed_raw_best_lens
                if rosa_precomputed_raw_best_lens is not None
                else fired_match_lens
            )
            return RosaAddressBatch(
                addr_ids=rosa_precomputed_ids,
                raw_match_lens=raw_match_lens,
                fired_match_lens=fired_match_lens,
                valid_mask=rosa_precomputed_ids.ge(0),
                special_mask=torch.zeros_like(rosa_precomputed_ids, dtype=torch.bool),
                source="precomputed",
            )

        if rosa_online_state is not None:
            addressed = rosa_online_state.address_tokens(
                input_ids,
                pad_id=self.pad_id,
                device=input_ids.device,
            )
            return make_rosa_address_batch(addressed, source="online")

        addressed = rosa_addressing_with_memory(
            input_ids=input_ids,
            memory_ids=rosa_memory_ids,
            min_match_len=self.min_match_len,
            pad_id=self.pad_id,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            backend=self.rosa_backend,
        )
        return make_rosa_address_batch(addressed, source=f"memory:{self.rosa_backend}")

    def build_rosa_injection_payload(
        self,
        address_batch: RosaAddressBatch,
        *,
        device: Optional[torch.device] = None,
        source: Optional[str] = None,
    ) -> RosaInjectionPayload:
        batch = address_batch.to(device) if device is not None else address_batch
        hot_cache = self.rosa_hot_cache if (self.rosa_hot_cache is not None and not self.training) else None
        layer_values: List[torch.Tensor] = []
        cache_rows: List[Dict[str, float]] = []
        for layer_idx in range(self.inject_layers):
            values, cache_stats = self.rosa_value_store.lookup_with_hot_cache(
                layer_idx,
                batch.addr_ids,
                valid_mask=batch.valid_mask,
                shared_embedding=self.embed_tokens,
                hot_cache=hot_cache,
            )
            layer_values.append(values)
            if cache_stats:
                cache_rows.append(cache_stats)
        payload_stats: Dict[str, float] = {}
        if cache_rows:
            token_requests = sum(row.get("token_requests", 0.0) for row in cache_rows)
            token_hits = sum(row.get("token_hits", 0.0) for row in cache_rows)
            unique_requests = sum(row.get("unique_requests", 0.0) for row in cache_rows)
            unique_hits = sum(row.get("unique_hits", 0.0) for row in cache_rows)
            layers_with_requests = sum(1 for row in cache_rows if row.get("token_requests", 0.0) > 0.0)
            payload_stats.update(
                {
                    "rosa_hot_cache_size": float(self.rosa_hot_cache_size),
                    "rosa_hot_cache_active_entries": sum(row.get("active_entries", 0.0) for row in cache_rows),
                    "rosa_hot_cache_token_hit_rate": token_hits / token_requests if token_requests > 0 else 0.0,
                    "rosa_hot_cache_unique_hit_rate": unique_hits / unique_requests if unique_requests > 0 else 0.0,
                    "rosa_hot_cache_fill_rate": (
                        sum(row.get("fills", 0.0) for row in cache_rows) / unique_requests
                        if unique_requests > 0
                        else 0.0
                    ),
                    "rosa_hot_cache_evictions": sum(row.get("evictions", 0.0) for row in cache_rows),
                    "rosa_hot_cache_layers_used": float(layers_with_requests),
                }
            )
        return RosaInjectionPayload(
            address=batch,
            layer_values=tuple(layer_values),
            source=source or batch.source,
            stats=payload_stats or None,
        )

    def prepare_rosa_injection_payload(
        self,
        input_ids: torch.Tensor,
        *,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ) -> RosaInjectionPayload:
        address_batch = self.compute_rosa_address_batch(
            input_ids,
            rosa_memory_ids=rosa_memory_ids,
            rosa_online_state=rosa_online_state,
            rosa_precomputed_ids=rosa_precomputed_ids,
            rosa_precomputed_match_lens=rosa_precomputed_match_lens,
            rosa_precomputed_raw_best_lens=rosa_precomputed_raw_best_lens,
        )
        return self.build_rosa_injection_payload(address_batch, device=input_ids.device)

    def schedule_rosa_prefetch(
        self,
        prefetcher: RosaPrefetcher,
        request_key: str,
        input_ids: torch.Tensor,
        *,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ) -> RosaAddressBatch:
        address_batch = self.compute_rosa_address_batch(
            input_ids,
            rosa_memory_ids=rosa_memory_ids,
            rosa_online_state=rosa_online_state,
            rosa_precomputed_ids=rosa_precomputed_ids,
            rosa_precomputed_match_lens=rosa_precomputed_match_lens,
            rosa_precomputed_raw_best_lens=rosa_precomputed_raw_best_lens,
        )
        prefetcher.submit(request_key, address_batch)
        return address_batch

    def consume_rosa_prefetch(
        self,
        prefetcher: RosaPrefetcher,
        request_key: str,
        *,
        device: torch.device,
        fallback_address_batch: Optional[RosaAddressBatch] = None,
    ) -> RosaInjectionPayload:
        payload = prefetcher.consume(request_key, device=device)
        if payload is not None:
            return payload
        if fallback_address_batch is None:
            raise KeyError(f"未找到 request_key={request_key} 对应的预取 payload。")
        return self.build_rosa_injection_payload(fallback_address_batch, device=device)

    def compute_rosa_context_gate(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        rosa_value: torch.Tensor,
        fired_match_lens: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_context_gate:
            raise RuntimeError("use_context_gate=False 时不能调用 compute_rosa_context_gate。")
        hidden_norm = self.rosa_gate_hidden_norms[layer_idx](hidden_states)
        value_key = self.rosa_gate_key_projs[layer_idx](rosa_value)
        value_key = self.rosa_gate_value_norms[layer_idx](value_key)
        gate_logits = (hidden_norm * value_key).sum(dim=-1, keepdim=True) / math.sqrt(self.cfg.dim)
        if self.use_match_len_gate:
            len_prior = torch.log1p(fired_match_lens.float()).unsqueeze(-1)
            gate_logits = gate_logits + self.rosa_gate_match_len_scale[layer_idx] * len_prior
        gate_logits = gate_logits + self.rosa_gate_bias[layer_idx]
        return torch.sigmoid(gate_logits)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_payload: Optional[RosaInjectionPayload] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed_tokens(input_ids)
        seqlen = input_ids.shape[1]
        attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)[None, None, :, :]

        if rosa_payload is None:
            rosa_payload = self.prepare_rosa_injection_payload(
                input_ids,
                rosa_memory_ids=rosa_memory_ids,
                rosa_online_state=rosa_online_state,
                rosa_precomputed_ids=rosa_precomputed_ids,
                rosa_precomputed_match_lens=rosa_precomputed_match_lens,
                rosa_precomputed_raw_best_lens=rosa_precomputed_raw_best_lens,
            )
        else:
            rosa_payload = rosa_payload.to(input_ids.device)

        rosa_ids = rosa_payload.address.addr_ids
        fired_match_lens = rosa_payload.address.fired_match_lens
        raw_best_lens = rosa_payload.address.raw_match_lens
        active = (rosa_ids >= 0)

        if self.use_match_len_gate:
            # 比硬阈值更稳的软门控：m 越大，权重越强
            len_scale = torch.log1p(fired_match_lens.float()).unsqueeze(-1)
        else:
            len_scale = torch.ones_like(fired_match_lens, dtype=torch.float).unsqueeze(-1)
        active_mask = active.unsqueeze(-1)
        gate_values: List[torch.Tensor] = []
        gate_open_flags: List[torch.Tensor] = []

        for layer_idx, blk in enumerate(self.layers):
            slot_idx = self.inject_layer_index.get(layer_idx)
            if slot_idx is not None:
                rosa_value = rosa_payload.layer_values[slot_idx]
                if self.use_context_gate:
                    gate = self.compute_rosa_context_gate(
                        slot_idx,
                        x,
                        rosa_value,
                        fired_match_lens,
                    )
                    value_out = self.rosa_gate_value_projs[slot_idx](rosa_value)
                    rosa_resid = value_out * gate * self.rosa_scale
                    gate_values.append(gate)
                    gate_open_flags.append(gate.gt(0.5))
                else:
                    rosa_resid = rosa_value * len_scale * self.rosa_scale
                rosa_resid = rosa_resid * active_mask
                x = x + rosa_resid
            x = blk(x, attn_mask)
        x = self.norm(x)

        raw_has_match = raw_best_lens.gt(0)
        stats = {
            "rosa_value_per_layer": 1.0 if self.rosa_value_store.is_per_layer else 0.0,
            "rosa_inject_slots": float(self.inject_layers),
            "rosa_fire_coverage": active.float().mean().item(),
            "rosa_fired_avg_match_len": fired_match_lens[active].float().mean().item() if active.any() else 0.0,
            "rosa_raw_match_coverage": raw_has_match.float().mean().item(),
            "rosa_raw_avg_best_len": raw_best_lens[raw_has_match].float().mean().item() if raw_has_match.any() else 0.0,
        }
        if rosa_payload.stats:
            stats.update(rosa_payload.stats)
        if self.use_context_gate:
            if gate_values:
                gate_tensor = torch.cat([g.reshape(-1) for g in gate_values], dim=0)
                gate_open = torch.cat([g.reshape(-1) for g in gate_open_flags], dim=0)
                expanded_active = torch.cat([active.reshape(-1) for _ in gate_values], dim=0)
                active_gate_values = gate_tensor[expanded_active]
                active_gate_open = gate_open[expanded_active]
                stats.update(
                    {
                        "rosa_avg_gate": active_gate_values.float().mean().item() if active_gate_values.numel() > 0 else 0.0,
                        "rosa_gate_coverage": gate_open.float().mean().item() if gate_open.numel() > 0 else 0.0,
                        "rosa_gate_hit": active_gate_open.float().mean().item() if active_gate_open.numel() > 0 else 0.0,
                    }
                )
            else:
                stats.update(
                    {
                        "rosa_avg_gate": 0.0,
                        "rosa_gate_coverage": 0.0,
                        "rosa_gate_hit": 0.0,
                    }
                )
        return x, stats

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rosa_memory_ids: Optional[torch.Tensor] = None,
        rosa_online_state: Optional[OnlineRosaBatchState] = None,
        rosa_payload: Optional[RosaInjectionPayload] = None,
        rosa_precomputed_ids: Optional[torch.Tensor] = None,
        rosa_precomputed_match_lens: Optional[torch.Tensor] = None,
        rosa_precomputed_raw_best_lens: Optional[torch.Tensor] = None,
    ):
        hidden, stats = self.forward_hidden(
            input_ids,
            rosa_memory_ids=rosa_memory_ids,
            rosa_online_state=rosa_online_state,
            rosa_payload=rosa_payload,
            rosa_precomputed_ids=rosa_precomputed_ids,
            rosa_precomputed_match_lens=rosa_precomputed_match_lens,
            rosa_precomputed_raw_best_lens=rosa_precomputed_raw_best_lens,
        )
        logits = self.lm_head(hidden)
        out = {"logits": logits, **stats}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            out["loss"] = loss
        return out

    def forward_online(
        self,
        input_ids: torch.Tensor,
        rosa_online_state: OnlineRosaBatchState,
        labels: Optional[torch.Tensor] = None,
    ):
        rosa_payload = self.prepare_rosa_injection_payload(
            input_ids,
            rosa_online_state=rosa_online_state,
        )
        return self.forward(
            input_ids=input_ids,
            labels=labels,
            rosa_payload=rosa_payload,
        )

    def forward_prefetched(
        self,
        input_ids: torch.Tensor,
        rosa_payload: RosaInjectionPayload,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.forward(
            input_ids=input_ids,
            labels=labels,
            rosa_payload=rosa_payload,
        )


def labels_with_ignore(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    y = labels.clone()
    y[y == pad_id] = -100
    return y


def safe_ppl(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    if loss > 80:
        return float("inf")
    return math.exp(loss)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    correct = 0

    rosa_stats_sum: Dict[str, float] = {}
    rosa_steps = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            mem = batch["rosa_memory_ids"].to(device)
            pre_ids = batch.get("rosa_precomputed_ids")
            pre_match = batch.get("rosa_precomputed_match_lens")
            pre_raw = batch.get("rosa_precomputed_raw_best_lens")
            y = labels_with_ignore(batch["labels"].to(device), pad_id)
            out = model(
                x,
                y,
                rosa_memory_ids=mem,
                rosa_precomputed_ids=pre_ids.to(device) if pre_ids is not None else None,
                rosa_precomputed_match_lens=pre_match.to(device) if pre_match is not None else None,
                rosa_precomputed_raw_best_lens=pre_raw.to(device) if pre_raw is not None else None,
            )

            loss = out["loss"]  # per-token mean CE
            mask = y.ne(-100)
            num = mask.sum().item()

            total_nll += float(loss.item()) * num
            total_tokens += num

            pred = out["logits"].argmax(dim=-1)
            correct += ((pred == y) & mask).sum().item()

            rosa_keys = [k for k in out.keys() if k.startswith("rosa_")]
            if rosa_keys:
                for k in rosa_keys:
                    rosa_stats_sum[k] = rosa_stats_sum.get(k, 0.0) + float(out[k])
                rosa_steps += 1

    loss = total_nll / max(1, total_tokens)
    metrics = {
        "loss": loss,
        "ppl": safe_ppl(loss),
        "token_acc": correct / max(1, total_tokens),
        "valid_tokens": total_tokens,
    }
    if rosa_steps > 0:
        for k, v in rosa_stats_sum.items():
            metrics[k] = v / rosa_steps
    return metrics


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pad_id: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    use_bf16: bool,
) -> Dict[str, List[Dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    amp_enabled = use_bf16 and device.type == "cuda"

    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_nll = 0.0
        total_tokens = 0
        correct = 0

        rosa_stats_sum: Dict[str, float] = {}
        rosa_steps = 0

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            mem = batch["rosa_memory_ids"].to(device)
            pre_ids = batch.get("rosa_precomputed_ids")
            pre_match = batch.get("rosa_precomputed_match_lens")
            pre_raw = batch.get("rosa_precomputed_raw_best_lens")
            y = labels_with_ignore(batch["labels"].to(device), pad_id)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(
                        x,
                        y,
                        rosa_memory_ids=mem,
                        rosa_precomputed_ids=pre_ids.to(device) if pre_ids is not None else None,
                        rosa_precomputed_match_lens=pre_match.to(device) if pre_match is not None else None,
                        rosa_precomputed_raw_best_lens=pre_raw.to(device) if pre_raw is not None else None,
                    )
                    loss = out["loss"]
            else:
                out = model(
                    x,
                    y,
                    rosa_memory_ids=mem,
                    rosa_precomputed_ids=pre_ids.to(device) if pre_ids is not None else None,
                    rosa_precomputed_match_lens=pre_match.to(device) if pre_match is not None else None,
                    rosa_precomputed_raw_best_lens=pre_raw.to(device) if pre_raw is not None else None,
                )
                loss = out["loss"]

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            mask = y.ne(-100)
            num = mask.sum().item()
            total_nll += float(loss.item()) * num
            total_tokens += num

            pred = out["logits"].argmax(dim=-1)
            correct += ((pred == y) & mask).sum().item()

            rosa_keys = [k for k in out.keys() if k.startswith("rosa_")]
            if rosa_keys:
                for k in rosa_keys:
                    rosa_stats_sum[k] = rosa_stats_sum.get(k, 0.0) + float(out[k])
                rosa_steps += 1

        train_loss = total_nll / max(1, total_tokens)
        train_metrics = {
            "epoch": epoch,
            "loss": train_loss,
            "ppl": safe_ppl(train_loss),
            "token_acc": correct / max(1, total_tokens),
            "valid_tokens": total_tokens,
        }
        if rosa_steps > 0:
            for k, v in rosa_stats_sum.items():
                train_metrics[k] = v / rosa_steps

        val_metrics = evaluate(model, val_loader, device, pad_id)
        val_metrics["epoch"] = epoch

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        train_extra = ""
        val_extra = ""
        if "rosa_fire_coverage" in train_metrics:
            train_extra = f" | train fire_cov {train_metrics['rosa_fire_coverage']:.4f} fired_m {train_metrics['rosa_fired_avg_match_len']:.2f}"
        if "rosa_fire_coverage" in val_metrics:
            val_extra = f" | val fire_cov {val_metrics['rosa_fire_coverage']:.4f} fired_m {val_metrics['rosa_fired_avg_match_len']:.2f}"

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} ppl {train_metrics['ppl']:.4f} acc {train_metrics['token_acc']:.4f}{train_extra} | "
            f"val loss {val_metrics['loss']:.4f} ppl {val_metrics['ppl']:.4f} acc {val_metrics['token_acc']:.4f}{val_extra}"
        )
    return history


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def preview_doc(doc: str, n: int = 120) -> str:
    return doc[:n].replace("\n", " | ")


def build_model_config(args, tokenizer) -> ModelConfig:
    rope_theta = 1_000_000.0 if args.arch_style == "qwen" else 10_000.0
    rms_eps = 1e-6 if args.arch_style == "qwen" else 1e-5
    n_kv_heads = args.n_kv_heads if args.n_kv_heads > 0 else args.n_heads
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=n_kv_heads,
        intermediate_size=args.intermediate_size,
        rope_theta=rope_theta,
        rms_norm_eps=rms_eps,
        dropout=args.dropout,
        tie_word_embeddings=not args.no_tie_word_embeddings,
    )


def main():
    parser = argparse.ArgumentParser(description="训练标准 Qwen/LLaMA 风格基线模型 与 Emb(ROSA(x)) 融合模型（文档前文 memory 版）进行对比。")
    parser.add_argument("--data_path", type=str, default=None,
                        help="单一数据源路径，可为文件、目录或通配符。未显式提供 train/val/test 时使用随机切分。")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="显式训练集路径，可为文件、目录或通配符。")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="显式验证集路径，可为文件、目录或通配符。")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="显式测试集路径，可为文件、目录或通配符。")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="HF tokenizer 路径或名称。为空时使用 byte fallback，仅用于烟雾测试。")
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["paragraph", "line", "stream"])
    parser.add_argument("--data_format", type=str, default="auto", choices=["auto", "text", "jsonl", "json"],
                        help="auto 会按扩展名自动识别；json/jsonl 默认每条记录视为一个文档。")
    parser.add_argument("--json_text_keys", type=str, default="text,content,body,message",
                        help="JSON/JSONL 中按优先级查找文本的字段名，逗号分隔，支持 a.b.c。")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="单一数据源模式下，最多读取多少个文档后再做 train/val/test 切分。")
    parser.add_argument("--max_train_docs", type=int, default=None)
    parser.add_argument("--max_val_docs", type=int, default=None)
    parser.add_argument("--max_test_docs", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--rosa_memory_tokens", type=int, default=512,
                        help="每个 chunk 可看的同文档左侧历史 token 数。")
    parser.add_argument("--rosa_memory_mode", type=str, default="doc_local", choices=["doc_local", "global_train"],
                        help="doc_local 只看同文档前文；global_train 额外拼接全局 train memory。")
    parser.add_argument("--rosa_global_memory_tokens", type=int, default=0,
                        help="global_train 模式下可见的全局 train memory token 数；0 表示退化为与 rosa_memory_tokens 相同。")
    parser.add_argument("--rosa_backend", type=str, default="sam", choices=["sam", "naive"],
                        help="ROSA 检索后端。sam 更接近原版 ROSA；naive 用于回归对照。")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--arch_style", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=0)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_tie_word_embeddings", action="store_true")
    parser.add_argument("--rosa_min_match_len", type=int, default=1,
                        help="建议训练时先从 1 开始，让 side-branch 先学会使用 ROSA 信号。")
    parser.add_argument("--rosa_inject_layers", type=int, default=2)
    parser.add_argument("--rosa_inject_layer_ids", type=str, default="",
                        help="显式指定注入层位，例如 0,2；为空时默认使用前 rosa_inject_layers 层。")
    parser.add_argument("--rosa_scale", type=float, default=0.25)
    parser.add_argument("--rosa_value_mode", type=str, default="shared", choices=["shared", "per_layer"],
                        help="shared 复用词嵌入；per_layer 为每个注入层使用独立 value table。")
    parser.add_argument("--rosa_context_gate", action="store_true",
                        help="启用 Engram 风格的 context-aware gate。")
    parser.add_argument("--rosa_hot_cache_size", type=int, default=0,
                        help="每个注入层的热点地址缓存条目数；0 表示关闭，仅在 eval/profile 路径启用。")
    parser.add_argument("--rosa_allow_special_target", action="store_true")
    parser.add_argument("--rosa_disable_match_len_gate", action="store_true",
                        help="默认按 match len 软门控；加上此开关则不使用长度缩放。")
    parser.add_argument("--out_dir", type=str, default="outputs/rosa_compare")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

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
        dataset_source = {
            "mode": "explicit_splits",
            "train": train_source,
            "val": val_source,
            "test": test_source,
        }
        total_docs = len(train_docs) + len(val_docs) + len(test_docs)
    else:
        if not args.data_path:
            raise ValueError("未提供数据路径。请传 --data_path，或同时传 --train_data_path/--val_data_path/--test_data_path。")
        docs, source_meta = load_docs_from_path(
            args.data_path,
            data_format=args.data_format,
            split_mode=args.split_mode,
            json_text_keys=json_text_keys,
            max_docs=args.max_docs,
        )
        train_docs, val_docs, test_docs = train_val_test_split(docs, args.train_ratio, args.val_ratio, args.seed)
        dataset_source = {
            "mode": "single_source",
            "source": source_meta,
        }
        total_docs = len(docs)

    print(f"总文档数: {total_docs}")
    print(f"train/val/test: {len(train_docs)} / {len(val_docs)} / {len(test_docs)}")
    print(f"架构风格: {args.arch_style}")
    print(f"tokenizer: {args.tokenizer_name_or_path or 'byte-fallback'}")
    print(f"data format: {args.data_format}")
    print(f"json text keys: {json_text_keys}")
    print(f"ROSA 最小匹配长度阈值: {args.rosa_min_match_len}")
    print(f"ROSA backend: {args.rosa_backend}")
    print(f"ROSA memory mode: {args.rosa_memory_mode}")
    print(f"ROSA value mode: {args.rosa_value_mode}")
    print(f"ROSA context gate: {args.rosa_context_gate}")
    print(f"ROSA hot cache size: {args.rosa_hot_cache_size}")
    print(f"ROSA inject layer ids: {parse_int_csv_arg(args.rosa_inject_layer_ids) or list(range(args.rosa_inject_layers))}")
    print(f"示例 train doc: {preview_doc(train_docs[0]) if train_docs else '<empty>'}")

    train_tok = tokenize_docs(train_docs, tokenizer, add_bos=True, add_eos=True)
    val_tok = tokenize_docs(val_docs, tokenizer, add_bos=True, add_eos=True)
    test_tok = tokenize_docs(test_docs, tokenizer, add_bos=True, add_eos=True)

    train_ds, val_ds, test_ds, memory_meta = build_chunk_datasets(
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
        rosa_min_match_len=args.rosa_min_match_len,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if memory_meta.get("precomputed_doc_local_sam"):
        print("ROSA 文档内历史: full doc prefix (SAM precompute)")
    else:
        print(f"ROSA 文档内 memory tokens: {memory_meta['doc_local_memory_tokens']}")
    if memory_meta["rosa_memory_mode"] == "global_train":
        print(f"ROSA 全局 train memory tokens: {memory_meta['global_train_memory_tokens']}")
        print(f"训练集全局 memory 实际长度: {memory_meta['train_global_memory_size']}")
    if memory_meta.get("precomputed_doc_local_sam"):
        print("ROSA 特征: doc-local SAM 预计算模式")
    cfg = build_model_config(args, tokenizer)

    set_seed(args.seed)
    baseline = BaseLM(cfg)
    set_seed(args.seed)
    rosa_model = RosaFusedLM(
        cfg,
        pad_id=tokenizer.pad_token_id,
        rosa_backend=args.rosa_backend,
        min_match_len=args.rosa_min_match_len,
        inject_layers=args.rosa_inject_layers,
        inject_layer_ids=parse_int_csv_arg(args.rosa_inject_layer_ids),
        rosa_scale=args.rosa_scale,
        rosa_value_mode=args.rosa_value_mode,
        use_context_gate=args.rosa_context_gate,
        rosa_hot_cache_size=args.rosa_hot_cache_size,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        use_match_len_gate=not args.rosa_disable_match_len_gate,
    )

    base_params = count_params(baseline)
    rosa_params = count_params(rosa_model)
    print(f"baseline params: {base_params:,}")
    print(f"rosa params    : {rosa_params:,}")
    if base_params == rosa_params:
        print("参数量完全一致（ROSA 分支未引入额外可训练参数）。")
    else:
        print("注意：参数量不一致。")

    print("\n==== 训练 baseline ====")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        train_seed=args.seed,
    )
    base_hist = train_one_model(
        baseline, train_loader, val_loader, device,
        pad_id=tokenizer.pad_token_id,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, use_bf16=args.bf16,
    )
    base_test = evaluate(baseline.to(device), test_loader, device, tokenizer.pad_token_id)

    print("\n==== 训练 rosa-fused ====")
    set_seed(args.seed)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        train_seed=args.seed,
    )
    rosa_model = RosaFusedLM(
        cfg,
        pad_id=tokenizer.pad_token_id,
        rosa_backend=args.rosa_backend,
        min_match_len=args.rosa_min_match_len,
        inject_layers=args.rosa_inject_layers,
        inject_layer_ids=parse_int_csv_arg(args.rosa_inject_layer_ids),
        rosa_scale=args.rosa_scale,
        rosa_value_mode=args.rosa_value_mode,
        use_context_gate=args.rosa_context_gate,
        rosa_hot_cache_size=args.rosa_hot_cache_size,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
        use_match_len_gate=not args.rosa_disable_match_len_gate,
    )
    rosa_hist = train_one_model(
        rosa_model, train_loader, val_loader, device,
        pad_id=tokenizer.pad_token_id,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, use_bf16=args.bf16,
    )
    rosa_test = evaluate(rosa_model.to(device), test_loader, device, tokenizer.pad_token_id)

    summary = {
        "args": vars(args),
        "model_config": asdict(cfg),
        "tokenizer": {
            "name_or_path": args.tokenizer_name_or_path or "byte-fallback",
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "rosa": {
            "backend": args.rosa_backend,
            "memory_mode": args.rosa_memory_mode,
            "memory_tokens": memory_meta["doc_local_memory_tokens"],
            "global_memory_tokens": memory_meta["global_train_memory_tokens"],
            "effective_history": memory_meta["effective_history"],
            "min_match_len": args.rosa_min_match_len,
            "inject_layers": args.rosa_inject_layers,
            "inject_layer_ids": parse_int_csv_arg(args.rosa_inject_layer_ids),
            "scale": args.rosa_scale,
            "value_mode": args.rosa_value_mode,
            "context_gate": args.rosa_context_gate,
            "hot_cache_size": args.rosa_hot_cache_size,
        },
        "dataset": {
            "source": dataset_source,
            "memory": memory_meta,
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
        "param_count": {
            "baseline": base_params,
            "rosa_fused": rosa_params,
        },
        "baseline": {
            "history": base_hist,
            "test": base_test,
        },
        "rosa_fused": {
            "history": rosa_hist,
            "test": rosa_test,
        },
    }
    save_json(summary, os.path.join(args.out_dir, "comparison.json"))
    torch.save(baseline.state_dict(), os.path.join(args.out_dir, "baseline.pt"))
    torch.save(rosa_model.state_dict(), os.path.join(args.out_dir, "rosa_fused.pt"))

    print("\n==== 最终对比 ====")
    print(json.dumps({
        "baseline_test": base_test,
        "rosa_fused_test": rosa_test,
    }, ensure_ascii=False, indent=2))
    print(f"结果已保存到: {args.out_dir}")


if __name__ == "__main__":
    main()
