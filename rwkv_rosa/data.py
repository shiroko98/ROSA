"""Document-disjoint SFT data with equal visible history for every experiment arm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from third_party.rwkv_lm_v7.sft_binidx import EncodedDocument, build_document_from_record, load_chat_template
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "third_party/rwkv_lm_v7/chat_template.jinja"


class ByteTokenizer:
    """Offline toy tokenizer: 0 is padding, byte b is token b+1."""
    vocab_size = 257
    special_ids = {0}

    def encode(self, text):
        return [b + 1 for b in text.encode("utf-8")]

    def encodeBytesWithSpans(self, data):
        return [b + 1 for b in data], [(i, i + 1) for i in range(len(data))]

    def decode(self, ids):
        return bytes(i - 1 for i in ids if 1 <= i <= 256).decode("utf-8", errors="replace")


def build_tokenizer(kind="rwkv", vocab_path=None):
    if kind == "byte":
        return ByteTokenizer()
    tokenizer = TRIE_TOKENIZER(str(vocab_path or ROOT / "rwkv_vocab_v20230424.txt"))
    tokenizer.vocab_size = max(tokenizer.idx2token) + 1
    tokenizer.special_ids = {0, *tokenizer.special_token2idx.values()}
    # SFT needs the actual marker IDs, not byte sequences accidentally used as EOD.
    for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        if marker.encode("utf-8") not in tokenizer.special_token2idx or len(tokenizer.encode(marker)) != 1:
            raise ValueError(f"SFT RWKV vocabulary must contain single-token {marker}.")
    return tokenizer


def load_documents(path, tokenizer, *, template_path=DEFAULT_TEMPLATE, max_docs=0):
    path = Path(path)
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("JSON corpus must be a list of records.")
    else:
        lines = [s for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        records = ([{"text": s} for s in lines] if path.suffix == ".txt"
                   else [json.loads(s) for s in lines])
    template = load_chat_template(str(template_path))
    docs = []
    for record in records[:max_docs or None]:
        if "messages" in record:
            doc = build_document_from_record(record, tokenizer=tokenizer, template=template)
        elif "text" in record:
            ids = tokenizer.encode(record["text"])
            doc = EncodedDocument(ids, [1] * len(ids))
        else:
            raise ValueError("Each record needs messages (SFT) or text (causal LM).")
        if len(doc.input_ids) < 2 or not any(doc.loss_mask[1:]):
            raise ValueError("Each document needs at least two tokens and a supervised next-token target.")
        docs.append(doc)
    if not docs:
        raise ValueError(f"No usable documents in {path}")
    return docs


def document_hash(doc):
    if hasattr(doc, "content_hash"):
        return doc.content_hash
    return hashlib.sha256(json.dumps(doc.input_ids, separators=(",", ":")).encode()).hexdigest()


def split_documents(docs, seed):
    if len(docs) < 3:
        raise ValueError("Automatic document-disjoint split needs >=3 documents.")
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    n_eval = max(1, len(docs) // 10)
    return ([docs[i] for i in order[2 * n_eval:]],
            [docs[i] for i in order[:n_eval]], [docs[i] for i in order[n_eval:2 * n_eval]])


def validate_splits(splits):
    seen = set()
    for name in ("train", "val", "test"):
        ids = {document_hash(d) for d in splits[name]}
        if not ids or seen & ids:
            raise ValueError(f"Empty {name} split or identical document leaked across splits.")
        seen |= ids


class EqualHistoryDataset(Dataset):
    """Chunked targets plus a bounded left context; all arms see the exact same IDs.

    No train-document global memory and no cross-document state. Targets are
    supervised at most once per document; left-context warmup tokens have zero loss.
    """
    def __init__(self, documents, *, chunk_len=64, history_tokens=64):
        if chunk_len < 1 or history_tokens < 0:
            raise ValueError("chunk_len must be positive and history_tokens nonnegative.")
        self.documents = documents
        self.chunk_len, self.history_tokens = chunk_len, history_tokens
        self.windows = []
        for d, doc in enumerate(documents):
            # Binidx rows are NumPy views. Scan active target positions in C,
            # avoiding a Python loop over hundreds of millions of masked tokens.
            active = np.flatnonzero(np.asarray(doc.loss_mask)[1:])
            for block in np.unique(active // chunk_len):
                start = int(block) * chunk_len
                end = min(len(doc.input_ids) - 1, start + chunk_len)
                self.windows.append((d, max(0, start - history_tokens), start, end))
        if not self.windows:
            raise ValueError("No supervised windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        d, left, start, end = self.windows[idx]
        doc = self.documents[d]
        mask = list(doc.loss_mask[left + 1:end + 1])
        mask[:start - left] = [0] * (start - left)
        return {"input_ids": torch.tensor(doc.input_ids[left:end], dtype=torch.long),
                "targets": torch.tensor(doc.input_ids[left + 1:end + 1], dtype=torch.long),
                "loss_mask": torch.tensor(mask, dtype=torch.float32)}


def collate(batch):
    length = max(len(row["input_ids"]) for row in batch)
    result = {}
    for key in ("input_ids", "targets", "loss_mask"):
        dtype = torch.float32 if key == "loss_mask" else torch.long
        out = torch.zeros((len(batch), length), dtype=dtype)
        for i, row in enumerate(batch):
            out[i, :len(row[key])] = row[key]
        result[key] = out
    return result


def split_summary(splits, datasets):
    return {name: {"documents": len(docs), "windows": len(datasets[name]),
                   "supervised_tokens": sum(d.supervised_tokens if hasattr(d, "supervised_tokens") else sum(d.loss_mask[1:]) for d in docs),
                   "document_hashes": [document_hash(d) for d in docs]}
            for name, docs in splits.items()}
