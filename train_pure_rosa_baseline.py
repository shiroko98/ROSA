#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 ROSA 非参数语言模型实验脚本。

说明：
1) 严格的 ROSA 本体是“非参数 / 非梯度”的，因此这里的“训练”本质上是：
   - 构建训练语料的离散记忆（可选）
   - 在验证/测试集上做 next-token 评测
2) 为了尽量贴近你发来的定义，这里直接实现了图中的在线 ROSA 版本。
3) 默认使用 utf-8 byte 级 token，避免额外 tokenizer 依赖，也能处理中英文混合文本。

支持两种实验模式：
- online:    每个文档只使用该文档自身前缀做在线预测（严格 ROSA）
- memory:    先把 train 文档拼成记忆，再在每个 eval 文档上评测（仍然是纯 ROSA，非参数）

示例：
  python train_pure_rosa_baseline.py --toy repeat --mode online
  python train_pure_rosa_baseline.py --toy repeat --mode memory
  python train_pure_rosa_baseline.py --data_path ./tiny_shakespeare.txt --split_mode paragraph --mode online
  python train_pure_rosa_baseline.py --data_path ./corpus.txt --split_mode line --mode memory
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SEP_ID = 256  # byte tokenizer 的文档分隔符，正常字节范围是 0~255
VOCAB_SIZE = 257


# =========================
# 1. ROSA 本体（按图中定义实现）
# =========================

def rosa_predict(seq: Sequence[int]) -> List[int]:
    """
    按用户给出的 RWKV-8 ROSA 伪代码/示意代码实现。

    输入:
        seq: token id 序列 x[0..n-1]
    输出:
        pred: 长度 n 的列表，其中 pred[i] 用于预测 seq[i+1]
              若没有匹配，则 pred[i] = -1。

    说明：
    - 这是“在线”版本：处理到 i 时，只能用 <= i 的历史。
    - pred[n-1] 没有可对齐的 next token，评测时会自然忽略。
    """
    n = len(seq)
    pred = [-1] * n
    if n == 0:
        return pred

    # SAM 容量按图里写法开 2*n+1
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

        # 从当前状态开始，沿 suffix link 找最近一个可用历史匹配
        v = r
        a = -1
        while v != -1:
            if length[v] > 0 and endpos[v] >= 0:
                idx = endpos[v] + 1
                if 0 <= idx < n:
                    a = seq[idx]
                break
            v = link[v]

        pred[i] = a
        last = r

        # 更新当前前缀覆盖到的所有 suffix state 的最近结束位置
        v = last
        while v != -1 and endpos[v] < i:
            endpos[v] = i
            v = link[v]

    return pred


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
        chunks = text.split("\n\n")
        docs = [c.strip() for c in chunks if c.strip()]
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")

    return docs


def encode_doc_byte(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def decode_bytes(ids: Sequence[int]) -> str:
    buf = bytes([x for x in ids if 0 <= x <= 255])
    return buf.decode("utf-8", errors="replace")


def limit_doc_length(docs: List[List[int]], max_doc_bytes: Optional[int]) -> List[List[int]]:
    if not max_doc_bytes or max_doc_bytes <= 0:
        return docs
    out = []
    for d in docs:
        if len(d) >= 2:
            out.append(d[:max_doc_bytes])
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
# 3. toy 数据集
# =========================

def make_repeat_toy_docs(
    num_docs: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> List[str]:
    """
    生成一个偏“重复模式”的 toy 数据集，方便 ROSA 验证。
    这里会反复复用短语，ROSA 往往能获得较高覆盖率和准确率。
    """
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
# 4. 评测
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
    vocab_size: int = VOCAB_SIZE,
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

    # 对 NLL 做按 token 数加权平均
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


def build_train_memory(train_docs: Sequence[Sequence[int]], max_train_bytes: Optional[int]) -> List[int]:
    mem: List[int] = []
    for d in train_docs:
        mem.extend(d)
        mem.append(SEP_ID)
    if max_train_bytes and max_train_bytes > 0 and len(mem) > max_train_bytes:
        mem = mem[-max_train_bytes:]
    return mem


def eval_online(docs: Sequence[Sequence[int]], smooth_eps: float) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in docs:
        if len(doc) < 2:
            continue
        preds = rosa_predict(doc)
        local_preds = preds[:-1]
        targets = doc[1:]
        m = calc_metrics(local_preds, targets, smooth_eps=smooth_eps)
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


def eval_with_memory(
    train_docs: Sequence[Sequence[int]],
    eval_docs: Sequence[Sequence[int]],
    smooth_eps: float,
    max_train_bytes: Optional[int],
) -> Tuple[Metrics, List[Metrics], List[Tuple[int, float]]]:
    memory = build_train_memory(train_docs, max_train_bytes=max_train_bytes)
    per_doc = []
    pos_stat_correct: Dict[int, int] = {}
    pos_stat_total: Dict[int, int] = {}

    for doc in eval_docs:
        if len(doc) < 2:
            continue

        seq = memory + [SEP_ID] + list(doc)
        preds = rosa_predict(seq)

        start = len(memory) + 1
        local_preds = preds[start:start + len(doc) - 1]
        targets = doc[1:]
        m = calc_metrics(local_preds, targets, smooth_eps=smooth_eps)
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
# 5. 输出辅助
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
# 6. main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="纯 ROSA 非参数实验脚本")
    parser.add_argument("--data_path", type=str, default=None, help="输入文本文件路径")
    parser.add_argument("--toy", type=str, default=None, choices=[None, "repeat"], help="使用 toy 数据")
    parser.add_argument("--mode", type=str, default="online", choices=["online", "memory"], help="评测模式")
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["stream", "line", "paragraph"], help="文档切分方式")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_eps", type=float, default=1e-4, help="为了能计算 NLL/PPL，对 one-hot 预测做固定平滑")
    parser.add_argument("--max_doc_bytes", type=int, default=2048, help="每个文档最多保留多少 byte，0 表示不限制")
    parser.add_argument("--max_train_bytes", type=int, default=200000, help="memory 模式下最多保留多少训练记忆字节，0 表示不限制")
    parser.add_argument("--num_toy_docs", type=int, default=300)
    parser.add_argument("--toy_min_len", type=int, default=128)
    parser.add_argument("--toy_max_len", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="./rosa_runs")
    parser.add_argument("--dump_examples", type=int, default=3, help="打印多少个预测示例")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    token_docs = [encode_doc_byte(x) for x in raw_docs]
    token_docs = limit_doc_length(token_docs, max_doc_bytes=args.max_doc_bytes)
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

    if args.mode == "online":
        train_m, _, train_curve = eval_online(train_docs, smooth_eps=args.smooth_eps)
        val_m, _, val_curve = eval_online(val_docs, smooth_eps=args.smooth_eps)
        test_m, _, test_curve = eval_online(test_docs, smooth_eps=args.smooth_eps)
    else:
        # train 自己仍然可以用在线模式看“自记忆”能力；
        # val/test 则用 train 记忆进行评测。
        train_m, _, train_curve = eval_online(train_docs, smooth_eps=args.smooth_eps)
        val_m, _, val_curve = eval_with_memory(
            train_docs, val_docs, smooth_eps=args.smooth_eps, max_train_bytes=args.max_train_bytes
        )
        test_m, _, test_curve = eval_with_memory(
            train_docs, test_docs, smooth_eps=args.smooth_eps, max_train_bytes=args.max_train_bytes
        )

    print_metrics("train", train_m)
    print_metrics("val", val_m)
    print_metrics("test", test_m)

    run_info = {
        "args": vars(args),
        "train": asdict(train_m),
        "val": asdict(val_m),
        "test": asdict(test_m),
    }
    save_json(out_dir / "metrics.json", run_info)
    save_curve_csv(out_dir / "train_curve.csv", train_curve)
    save_curve_csv(out_dir / "val_curve.csv", val_curve)
    save_curve_csv(out_dir / "test_curve.csv", test_curve)

    # 打印几个预测例子
    dump_n = min(args.dump_examples, len(test_docs))
    for k in range(dump_n):
        doc = test_docs[k]
        if args.mode == "online":
            preds = rosa_predict(doc)[:-1]
        else:
            seq = build_train_memory(train_docs, args.max_train_bytes) + [SEP_ID] + list(doc)
            full_preds = rosa_predict(seq)
            start = len(seq) - len(doc)
            preds = full_preds[start:start + len(doc) - 1]
        targets = doc[1:]

        # 只展示前 120 byte，避免刷屏
        show_len = min(120, len(doc))
        show_doc = decode_bytes(doc[:show_len])
        print(f"\n[example {k}] doc prefix:")
        print(show_doc.replace("\n", "\\n"))

        # 打印前 40 个位置的预测/标签（字节转可读字符）
        print("[pos] pred -> target")
        for i, (p, y) in enumerate(zip(preds[:40], targets[:40])):
            if p == -1:
                ps = "<NA>"
            elif p == SEP_ID:
                ps = "<SEP>"
            else:
                ps = decode_bytes([p]).replace("\n", "\\n")
            ys = "<SEP>" if y == SEP_ID else decode_bytes([y]).replace("\n", "\\n")
            flag = "✓" if p == y else "x"
            print(f"{i:03d}  {ps!r} -> {ys!r}  {flag}")

    print(f"\n结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
