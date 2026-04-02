#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 ROSA 非参数实验脚本（RWKV tokenizer 版）。

改动：
1) 使用 RWKV 词表做 greedy longest-match tokenization。
2) 支持 ROSA 最小匹配长度阈值 --min_match_len。
3) 增加一个简单 baseline：bigram most-frequent-next。
4) 保留 online / memory 两种评测模式。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


# =========================
# 0. RWKV tokenizer
# =========================

class RWKVTokenizer:
    """
    基于 rwkv_vocab_v20230424.txt 的 greedy longest-prefix tokenizer。
    词表每行格式类似：
        33 b' ' 1
        257 b'\\t\\t' 2
        3025 b'\\xe4\\xb8' 2
    """

    def __init__(self, vocab_path: Path):
        self.vocab_path = Path(vocab_path)
        self.id2token: Dict[int, bytes] = {}
        self.token2id: Dict[bytes, int] = {}

        with self.vocab_path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if not line:
                    continue

                first_sp = line.find(" ")
                last_sp = line.rfind(" ")
                if first_sp <= 0 or last_sp <= first_sp:
                    raise ValueError(f"词表第 {line_no} 行格式错误: {line!r}")

                idx = int(line[:first_sp])
                literal = line[first_sp + 1:last_sp]

                try:
                    token_bytes = ast.literal_eval(literal)
                except Exception as e:
                    raise ValueError(f"词表第 {line_no} 行 bytes literal 解析失败: {literal!r}") from e

                if not isinstance(token_bytes, (bytes, bytearray)):
                    raise ValueError(f"词表第 {line_no} 行不是 bytes: {literal!r}")

                token_bytes = bytes(token_bytes)
                self.id2token[idx] = token_bytes
                self.token2id[token_bytes] = idx

        if not self.id2token:
            raise ValueError("RWKV 词表为空")

        self.base_vocab_size = max(self.id2token.keys()) + 1
        self.sep_id = self.base_vocab_size
        self.vocab_size_with_sep = self.base_vocab_size + 1

        # 建 trie，按 byte 做最长前缀匹配
        self.trie: Dict[int, dict] = {}
        TERM = None
        for idx, tok in self.id2token.items():
            node = self.trie
            for b in tok:
                node = node.setdefault(b, {})
            node[TERM] = idx

    def encode(self, text: str) -> List[int]:
        buf = text.encode("utf-8")
        n = len(buf)
        ids: List[int] = []
        i = 0
        TERM = None

        while i < n:
            node = self.trie
            j = i
            last_id = None
            last_j = i

            while j < n and buf[j] in node:
                node = node[buf[j]]
                j += 1
                if TERM in node:
                    last_id = node[TERM]
                    last_j = j

            if last_id is None:
                # 正常情况下不该发生，因为词表覆盖了所有单 byte
                bad = buf[i:i + 16]
                raise ValueError(f"RWKV tokenizer 在字节位置 {i} 处无法编码: {bad!r}")

            ids.append(last_id)
            i = last_j

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chunks = []
        for x in ids:
            if x == self.sep_id:
                continue
            if x not in self.id2token:
                continue
            chunks.append(self.id2token[x])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_to_pretty(self, idx: int) -> str:
        if idx == -1:
            return "<NA>"
        if idx == self.sep_id:
            return "<SEP>"
        if idx not in self.id2token:
            return f"<UNK_ID:{idx}>"

        b = self.id2token[idx]
        try:
            s = b.decode("utf-8", errors="strict")
            s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            # 控制字符或空串时回退到 bytes repr
            if not s or any(ord(ch) < 32 for ch in s):
                return repr(b)
            return s
        except UnicodeDecodeError:
            return repr(b)


# =========================
# 1. ROSA 本体
# =========================

def rosa_predict(seq: Sequence[int], min_match_len: int = 1) -> Tuple[List[int], List[int]]:
    """
    在线 ROSA。
    返回:
        pred[i]: 用于预测 seq[i+1] 的 token，若无匹配则为 -1
        match_len[i]: 当前位置命中的最长匹配长度
    """
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


# =========================
# 2. 数据处理
# =========================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_docs(text: str, split_mode: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if split_mode == "stream":
        docs = [text]
    elif split_mode == "line":
        docs = [line.strip() for line in text.split("\n") if line.strip()]
    elif split_mode == "paragraph":
        docs = [c.strip() for c in text.split("\n\n") if c.strip()]
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")

    return docs


def limit_doc_length(docs: List[List[int]], max_doc_tokens: Optional[int]) -> List[List[int]]:
    if not max_doc_tokens or max_doc_tokens <= 0:
        return docs
    out = []
    for d in docs:
        if len(d) >= 2:
            out.append(d[:max_doc_tokens])
    return out


def train_val_test_split(
    docs: List[List[int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    docs = [docs[i] for i in idx]

    n = len(docs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"文档数太少，当前 split 后得到 train={n_train}, val={n_val}, test={n_test}。"
        )

    train_docs = docs[:n_train]
    val_docs = docs[n_train:n_train + n_val]
    test_docs = docs[n_train + n_val:]
    return train_docs, val_docs, test_docs


# =========================
# 3. toy 数据
# =========================

def make_repeat_toy_docs(
    num_docs: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    vocab = [
        "alpha", "beta", "gamma", "delta", "omega", "river", "stone", "light",
        "north", "south", "east", "west", "red", "blue", "green", "white",
        "book", "music", "paper", "cloud", "water", "wind", "fire", "earth",
    ]

    docs = []
    for _ in range(num_docs):
        target_len = rng.randint(min_len, max_len)
        phrases = []
        motif_bank = []
        while len(" ".join(phrases)) < target_len:
            if motif_bank and rng.random() < 0.70:
                motif = rng.choice(motif_bank)
            else:
                motif_words = rng.randint(2, 5)
                motif = " ".join(rng.choice(vocab) for _ in range(motif_words))
                motif_bank.append(motif)
            phrases.append(motif)
        docs.append(" | ".join(phrases))
    return docs


# =========================
# 4. 指标
# =========================

@dataclass
class Metrics:
    positions: int
    covered: int
    correct: int
    overall_acc: float
    covered_acc: float
    coverage: float
    avg_nll: float
    ppl: float


def calc_metrics(
    preds: Sequence[int],
    targets: Sequence[int],
    vocab_size: int,
    smooth_eps: float = 1e-4,
) -> Metrics:
    if len(preds) != len(targets):
        raise ValueError("preds / targets 长度不一致")

    total = len(targets)
    if total == 0:
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0, float("inf"), float("inf"))

    covered = 0
    correct = 0
    nll = 0.0

    off_prob = smooth_eps / max(1, vocab_size - 1)
    on_prob = max(1e-12, 1.0 - smooth_eps)
    uniform_prob = 1.0 / vocab_size

    for p, y in zip(preds, targets):
        if p == -1:
            prob = uniform_prob
        else:
            covered += 1
            if p == y:
                correct += 1
                prob = on_prob
            else:
                prob = off_prob
        nll -= math.log(max(prob, 1e-12))

    overall_acc = correct / total
    covered_acc = correct / covered if covered > 0 else 0.0
    coverage = covered / total
    avg_nll = nll / total
    ppl = math.exp(avg_nll)

    return Metrics(
        positions=total,
        covered=covered,
        correct=correct,
        overall_acc=overall_acc,
        covered_acc=covered_acc,
        coverage=coverage,
        avg_nll=avg_nll,
        ppl=ppl,
    )


def merge_metrics(items: Sequence[Metrics]) -> Metrics:
    positions = sum(x.positions for x in items)
    covered = sum(x.covered for x in items)
    correct = sum(x.correct for x in items)
    if positions == 0:
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0, float("inf"), float("inf"))

    total_nll = sum(x.avg_nll * x.positions for x in items)
    avg_nll = total_nll / positions
    ppl = math.exp(avg_nll)
    overall_acc = correct / positions
    covered_acc = correct / covered if covered > 0 else 0.0
    coverage = covered / positions

    return Metrics(
        positions=positions,
        covered=covered,
        correct=correct,
        overall_acc=overall_acc,
        covered_acc=covered_acc,
        coverage=coverage,
        avg_nll=avg_nll,
        ppl=ppl,
    )


# =========================
# 5. ROSA 评测
# =========================

def build_train_memory(train_docs: Sequence[Sequence[int]], sep_id: int, max_train_tokens: Optional[int]) -> List[int]:
    mem: List[int] = []
    for d in train_docs:
        mem.extend(d)
        mem.append(sep_id)
    if max_train_tokens and max_train_tokens > 0 and len(mem) > max_train_tokens:
        mem = mem[-max_train_tokens:]
    return mem


def eval_online_rosa(
    docs: Sequence[Sequence[int]],
    vocab_size: int,
    smooth_eps: float,
    min_match_len: int,
) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in docs:
        if len(doc) < 2:
            continue
        preds, _ = rosa_predict(doc, min_match_len=min_match_len)
        local_preds = preds[:-1]
        targets = doc[1:]
        m = calc_metrics(local_preds, targets, vocab_size=vocab_size, smooth_eps=smooth_eps)
        per_doc.append(m)

        for pos, (p, y) in enumerate(zip(local_preds, targets)):
            pos_stat_total[pos] = pos_stat_total.get(pos, 0) + 1
            if p == y:
                pos_stat_correct[pos] = pos_stat_correct.get(pos, 0) + 1

    merged = merge_metrics(per_doc)
    curve = []
    for pos in sorted(pos_stat_total):
        acc = pos_stat_correct.get(pos, 0) / pos_stat_total[pos]
        curve.append((pos, acc))
    return merged, per_doc, curve


def eval_with_memory_rosa(
    train_docs: Sequence[Sequence[int]],
    eval_docs: Sequence[Sequence[int]],
    sep_id: int,
    vocab_size: int,
    smooth_eps: float,
    max_train_tokens: Optional[int],
    min_match_len: int,
) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    memory = build_train_memory(train_docs, sep_id=sep_id, max_train_tokens=max_train_tokens)
    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in eval_docs:
        if len(doc) < 2:
            continue

        seq = memory + [sep_id] + list(doc)
        preds, _ = rosa_predict(seq, min_match_len=min_match_len)

        start = len(memory) + 1
        local_preds = preds[start:start + len(doc) - 1]
        targets = doc[1:]
        m = calc_metrics(local_preds, targets, vocab_size=vocab_size, smooth_eps=smooth_eps)
        per_doc.append(m)

        for pos, (p, y) in enumerate(zip(local_preds, targets)):
            pos_stat_total[pos] = pos_stat_total.get(pos, 0) + 1
            if p == y:
                pos_stat_correct[pos] = pos_stat_correct.get(pos, 0) + 1

    merged = merge_metrics(per_doc)
    curve = []
    for pos in sorted(pos_stat_total):
        acc = pos_stat_correct.get(pos, 0) / pos_stat_total[pos]
        curve.append((pos, acc))
    return merged, per_doc, curve


# =========================
# 6. baseline: bigram most-frequent-next
# =========================

def _best_next(counter: Dict[int, int]) -> int:
    # 频次高优先；频次相同则 token id 小优先，保证稳定。
    return max(counter.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _update_bigram_counts(counts: DefaultDict[int, Dict[int, int]], a: int, b: int) -> None:
    if a not in counts:
        counts[a] = {}
    counts[a][b] = counts[a].get(b, 0) + 1


def _copy_counts(counts: DefaultDict[int, Dict[int, int]]) -> DefaultDict[int, Dict[int, int]]:
    out: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
    for k, v in counts.items():
        out[k] = dict(v)
    return out


def eval_online_bigram(
    docs: Sequence[Sequence[int]],
    vocab_size: int,
    smooth_eps: float,
) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in docs:
        if len(doc) < 2:
            continue

        counts: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
        preds: List[int] = []

        for i in range(len(doc) - 1):
            ctx = doc[i]
            target = doc[i + 1]

            if ctx in counts and counts[ctx]:
                pred = _best_next(counts[ctx])
            else:
                pred = -1
            preds.append(pred)

            _update_bigram_counts(counts, ctx, target)

        targets = doc[1:]
        m = calc_metrics(preds, targets, vocab_size=vocab_size, smooth_eps=smooth_eps)
        per_doc.append(m)

        for pos, (p, y) in enumerate(zip(preds, targets)):
            pos_stat_total[pos] = pos_stat_total.get(pos, 0) + 1
            if p == y:
                pos_stat_correct[pos] = pos_stat_correct.get(pos, 0) + 1

    merged = merge_metrics(per_doc)
    curve = []
    for pos in sorted(pos_stat_total):
        acc = pos_stat_correct.get(pos, 0) / pos_stat_total[pos]
        curve.append((pos, acc))
    return merged, per_doc, curve


def _build_memory_bigram_counts(memory: Sequence[int]) -> DefaultDict[int, Dict[int, int]]:
    counts: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
    for i in range(len(memory) - 1):
        _update_bigram_counts(counts, memory[i], memory[i + 1])
    return counts


def eval_with_memory_bigram(
    train_docs: Sequence[Sequence[int]],
    eval_docs: Sequence[Sequence[int]],
    sep_id: int,
    vocab_size: int,
    smooth_eps: float,
    max_train_tokens: Optional[int],
) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    memory = build_train_memory(train_docs, sep_id=sep_id, max_train_tokens=max_train_tokens)
    base_counts = _build_memory_bigram_counts(memory)

    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in eval_docs:
        if len(doc) < 2:
            continue

        counts = _copy_counts(base_counts)
        preds: List[int] = []

        for i in range(len(doc) - 1):
            ctx = doc[i]
            target = doc[i + 1]

            if ctx in counts and counts[ctx]:
                pred = _best_next(counts[ctx])
            else:
                pred = -1
            preds.append(pred)

            _update_bigram_counts(counts, ctx, target)

        targets = doc[1:]
        m = calc_metrics(preds, targets, vocab_size=vocab_size, smooth_eps=smooth_eps)
        per_doc.append(m)

        for pos, (p, y) in enumerate(zip(preds, targets)):
            pos_stat_total[pos] = pos_stat_total.get(pos, 0) + 1
            if p == y:
                pos_stat_correct[pos] = pos_stat_correct.get(pos, 0) + 1

    merged = merge_metrics(per_doc)
    curve = []
    for pos in sorted(pos_stat_total):
        acc = pos_stat_correct.get(pos, 0) / pos_stat_total[pos]
        curve.append((pos, acc))
    return merged, per_doc, curve


# =========================
# 7. 输出辅助
# =========================

def print_metrics(title: str, m: Metrics) -> None:
    print(f"\n[{title}]")
    print(f"positions    : {m.positions}")
    print(f"covered      : {m.covered}")
    print(f"correct      : {m.correct}")
    print(f"coverage     : {m.coverage:.4f}")
    print(f"overall_acc  : {m.overall_acc:.4f}")
    print(f"covered_acc  : {m.covered_acc:.4f}")
    print(f"avg_nll      : {m.avg_nll:.6f}")
    print(f"ppl          : {m.ppl:.4f}")


def save_curve_csv(path: Path, curve: Sequence[Tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "accuracy"])
        for pos, acc in curve:
            writer.writerow([pos, acc])


def save_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 8. main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="纯 ROSA 非参数实验脚本（RWKV tokenizer 版）")
    parser.add_argument("--rwkv_vocab_path", type=str, required=True, help="RWKV 词表路径，例如 rwkv_vocab_v20230424.txt")
    parser.add_argument("--data_path", type=str, default=None, help="输入文本文件路径")
    parser.add_argument("--toy", type=str, default=None, choices=[None, "repeat"], help="使用 toy 数据")
    parser.add_argument("--mode", type=str, default="online", choices=["online", "memory"], help="评测模式")
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["stream", "line", "paragraph"], help="文档切分方式")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_eps", type=float, default=1e-4, help="为了能计算 NLL/PPL，对 one-hot 预测做固定平滑")
    parser.add_argument("--min_match_len", type=int, default=2, help="ROSA 最小匹配长度阈值，小于该长度时输出 <NA>")
    parser.add_argument("--max_doc_tokens", "--max_doc_bytes", dest="max_doc_tokens", type=int, default=1024, help="每个文档最多保留多少 token，0 表示不限制")
    parser.add_argument("--max_train_tokens", "--max_train_bytes", dest="max_train_tokens", type=int, default=100000, help="memory 模式下最多保留多少训练记忆 token，0 表示不限制")
    parser.add_argument("--num_toy_docs", type=int, default=300)
    parser.add_argument("--toy_min_len", type=int, default=128)
    parser.add_argument("--toy_max_len", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="./rosa_runs_rwkv")
    parser.add_argument("--dump_examples", type=int, default=3, help="打印多少个预测示例")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = RWKVTokenizer(Path(args.rwkv_vocab_path))

    if args.toy == "repeat":
        raw_docs = make_repeat_toy_docs(
            num_docs=args.num_toy_docs,
            min_len=args.toy_min_len,
            max_len=args.toy_max_len,
            seed=args.seed,
        )
    elif args.data_path:
        text = read_text(Path(args.data_path))
        raw_docs = split_docs(text, split_mode=args.split_mode)
    else:
        raise ValueError("请提供 --data_path，或使用 --toy repeat")

    token_docs = [tokenizer.encode(x) for x in raw_docs]
    token_docs = limit_doc_length(token_docs, max_doc_tokens=args.max_doc_tokens)
    token_docs = [d for d in token_docs if len(d) >= 2]

    train_docs, val_docs, test_docs = train_val_test_split(
        token_docs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"总文档数: {len(token_docs)}")
    print(f"train/val/test: {len(train_docs)} / {len(val_docs)} / {len(test_docs)}")
    print(f"模式: {args.mode}")
    print(f"tokenizer: RWKV @ {args.rwkv_vocab_path}")
    print(f"ROSA 最小匹配长度阈值: {args.min_match_len}")

    # ROSA
    if args.mode == "online":
        train_m, _, train_curve = eval_online_rosa(
            train_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps, min_match_len=args.min_match_len
        )
        val_m, _, val_curve = eval_online_rosa(
            val_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps, min_match_len=args.min_match_len
        )
        test_m, _, test_curve = eval_online_rosa(
            test_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps, min_match_len=args.min_match_len
        )

        base_train_m, _, base_train_curve = eval_online_bigram(
            train_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps
        )
        base_val_m, _, base_val_curve = eval_online_bigram(
            val_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps
        )
        base_test_m, _, base_test_curve = eval_online_bigram(
            test_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps
        )
    else:
        train_m, _, train_curve = eval_online_rosa(
            train_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps, min_match_len=args.min_match_len
        )
        val_m, _, val_curve = eval_with_memory_rosa(
            train_docs, val_docs,
            sep_id=tokenizer.sep_id,
            vocab_size=tokenizer.vocab_size_with_sep,
            smooth_eps=args.smooth_eps,
            max_train_tokens=args.max_train_tokens,
            min_match_len=args.min_match_len,
        )
        test_m, _, test_curve = eval_with_memory_rosa(
            train_docs, test_docs,
            sep_id=tokenizer.sep_id,
            vocab_size=tokenizer.vocab_size_with_sep,
            smooth_eps=args.smooth_eps,
            max_train_tokens=args.max_train_tokens,
            min_match_len=args.min_match_len,
        )

        base_train_m, _, base_train_curve = eval_online_bigram(
            train_docs, vocab_size=tokenizer.vocab_size_with_sep, smooth_eps=args.smooth_eps
        )
        base_val_m, _, base_val_curve = eval_with_memory_bigram(
            train_docs, val_docs,
            sep_id=tokenizer.sep_id,
            vocab_size=tokenizer.vocab_size_with_sep,
            smooth_eps=args.smooth_eps,
            max_train_tokens=args.max_train_tokens,
        )
        base_test_m, _, base_test_curve = eval_with_memory_bigram(
            train_docs, test_docs,
            sep_id=tokenizer.sep_id,
            vocab_size=tokenizer.vocab_size_with_sep,
            smooth_eps=args.smooth_eps,
            max_train_tokens=args.max_train_tokens,
        )

    print_metrics("ROSA train", train_m)
    print_metrics("ROSA val", val_m)
    print_metrics("ROSA test", test_m)

    print_metrics("Baseline(train) bigram_most_frequent", base_train_m)
    print_metrics("Baseline(val) bigram_most_frequent", base_val_m)
    print_metrics("Baseline(test) bigram_most_frequent", base_test_m)

    run_info = {
        "args": vars(args),
        "tokenizer": {
            "vocab_path": str(Path(args.rwkv_vocab_path).resolve()),
            "base_vocab_size": tokenizer.base_vocab_size,
            "sep_id": tokenizer.sep_id,
            "vocab_size_with_sep": tokenizer.vocab_size_with_sep,
        },
        "rosa": {
            "train": asdict(train_m),
            "val": asdict(val_m),
            "test": asdict(test_m),
        },
        "baseline": {
            "name": "bigram_most_frequent",
            "train": asdict(base_train_m),
            "val": asdict(base_val_m),
            "test": asdict(base_test_m),
        },
    }
    save_json(out_dir / "metrics.json", run_info)

    save_curve_csv(out_dir / "rosa_train_curve.csv", train_curve)
    save_curve_csv(out_dir / "rosa_val_curve.csv", val_curve)
    save_curve_csv(out_dir / "rosa_test_curve.csv", test_curve)

    save_curve_csv(out_dir / "baseline_train_curve.csv", base_train_curve)
    save_curve_csv(out_dir / "baseline_val_curve.csv", base_val_curve)
    save_curve_csv(out_dir / "baseline_test_curve.csv", base_test_curve)

    dump_n = min(args.dump_examples, len(test_docs))
    for k in range(dump_n):
        doc = test_docs[k]
        if args.mode == "online":
            preds, match_lens = rosa_predict(doc, min_match_len=args.min_match_len)
            preds = preds[:-1]
            match_lens = match_lens[:-1]
        else:
            seq = build_train_memory(train_docs, tokenizer.sep_id, args.max_train_tokens) + [tokenizer.sep_id] + list(doc)
            full_preds, full_match_lens = rosa_predict(seq, min_match_len=args.min_match_len)
            start = len(seq) - len(doc)
            preds = full_preds[start:start + len(doc) - 1]
            match_lens = full_match_lens[start:start + len(doc) - 1]
        targets = doc[1:]

        show_len = min(80, len(doc))
        show_doc = tokenizer.decode(doc[:show_len])
        print(f"\n[example {k}] doc prefix:")
        print(show_doc.replace("\n", "\\n"))

        print("[pos] m pred -> target")
        for i, (m, p, y) in enumerate(zip(match_lens[:40], preds[:40], targets[:40])):
            ps = tokenizer.token_to_pretty(p)
            ys = tokenizer.token_to_pretty(y)
            flag = "✓" if p == y else "x"
            print(f"{i:03d}  m={m:02d}  {ps!r} -> {ys!r}  {flag}")

    print(f"\n结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
