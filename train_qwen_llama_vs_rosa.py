import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
        self.special_ids = set(x for x in [self.pad_token_id, self.eos_token_id, self.bos_token_id] if x is not None)
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
    with open(path, "r", encoding="utf-8") as f:
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


class LMDataset(Dataset):
    def __init__(self, docs_tokens: Sequence[Sequence[int]], seq_len: int, pad_id: int, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.samples: List[Tuple[List[int], List[int]]] = []
        stride = stride or seq_len
        for ids in docs_tokens:
            if len(ids) < 2:
                continue
            for start in range(0, max(1, len(ids) - 1), stride):
                chunk = list(ids[start:start + seq_len + 1])
                if len(chunk) < 2:
                    continue
                if len(chunk) < seq_len + 1:
                    chunk = chunk + [pad_id] * (seq_len + 1 - len(chunk))
                x = chunk[:-1]
                y = chunk[1:]
                self.samples.append((x, y))
                if start + seq_len + 1 >= len(ids):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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


def naive_rosa_retrieval(
    input_ids: torch.Tensor,
    min_match_len: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive per-sequence ROSA retrieval on token ids. Returns retrieved token ids and match lengths.
    Match length is number of matched tokens in the suffix. O(B*T^2*avg_match).
    """
    bsz, seqlen = input_ids.shape
    retrieved = torch.full_like(input_ids, -1)
    match_lens = torch.zeros_like(input_ids)
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    for b in range(bsz):
        seq = seqs[b]
        for i in range(seqlen):
            cur = seq[i]
            if cur < 0:
                continue
            best_len = 0
            best_j = -1
            for j in range(i):
                if seq[j] != cur:
                    continue
                m = 1
                while j - m >= 0 and i - m >= 0 and seq[j - m] == seq[i - m]:
                    m += 1
                if m > best_len or (m == best_len and j > best_j):
                    best_len = m
                    best_j = j
            if best_j >= 0 and best_len >= min_match_len and best_j + 1 < seqlen:
                target = seq[best_j + 1]
                if forbid_special_target and target in special_ids:
                    continue
                retrieved[b, i] = target
                match_lens[b, i] = best_len
    return retrieved.to(input_ids.device), match_lens.to(input_ids.device)


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

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        seqlen = input_ids.shape[1]
        attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)[None, None, :, :]
        for blk in self.layers:
            x = blk(x, attn_mask)
        x = self.norm(x)
        return x

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        hidden = self.forward_hidden(input_ids)
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


class RosaFusedLM(BaseLM):
    def __init__(
        self,
        cfg: ModelConfig,
        min_match_len: int = 2,
        inject_layers: int = 2,
        rosa_scale: float = 0.25,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
    ):
        super().__init__(cfg)
        self.min_match_len = min_match_len
        self.inject_layers = inject_layers
        self.rosa_scale = rosa_scale
        self.special_ids = special_ids or set()
        self.forbid_special_target = forbid_special_target

    def forward_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed_tokens(input_ids)
        seqlen = input_ids.shape[1]
        attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)[None, None, :, :]

        rosa_ids, match_lens = naive_rosa_retrieval(
            input_ids, min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
        )
        active = (rosa_ids >= 0)
        rosa_ids_safe = rosa_ids.clamp_min(0)
        rosa_emb = self.embed_tokens(rosa_ids_safe)
        len_scale = torch.log1p(match_lens.float()).unsqueeze(-1)
        rosa_resid = rosa_emb * len_scale * self.rosa_scale
        rosa_resid = rosa_resid * active.unsqueeze(-1)

        for layer_idx, blk in enumerate(self.layers):
            if layer_idx < self.inject_layers:
                x = x + rosa_resid
            x = blk(x, attn_mask)
        x = self.norm(x)
        stats = {
            "rosa_coverage": active.float().mean().item(),
            "rosa_avg_match_len": match_lens[active].float().mean().item() if active.any() else 0.0,
        }
        return x, stats

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        hidden, stats = self.forward_hidden(input_ids)
        logits = self.lm_head(hidden)
        out = {"logits": logits, **stats}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


def labels_with_ignore(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    y = labels.clone()
    y[y == pad_id] = -100
    return y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    rosa_cov = 0.0
    rosa_m = 0.0
    rosa_steps = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = labels_with_ignore(y.to(device), pad_id)
            out = model(x, y)
            loss = out["loss"]
            mask = y.ne(-100)
            num = mask.sum().item()
            total_loss += loss.item() * num
            total_tokens += num
            pred = out["logits"].argmax(dim=-1)
            correct += ((pred == y) & mask).sum().item()
            if "rosa_coverage" in out:
                rosa_cov += float(out["rosa_coverage"])
                rosa_m += float(out["rosa_avg_match_len"])
                rosa_steps += 1
    metrics = {
        "loss": total_loss / max(1, total_tokens),
        "ppl": math.exp(min(20.0, total_loss / max(1, total_tokens))),
        "token_acc": correct / max(1, total_tokens),
    }
    if rosa_steps > 0:
        metrics["rosa_coverage"] = rosa_cov / rosa_steps
        metrics["rosa_avg_match_len"] = rosa_m / rosa_steps
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
    scaler_enabled = use_bf16 and device.type == "cuda"
    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        rosa_cov = 0.0
        rosa_m = 0.0
        rosa_steps = 0
        for x, y in train_loader:
            x = x.to(device)
            y = labels_with_ignore(y.to(device), pad_id)
            optimizer.zero_grad(set_to_none=True)
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler_enabled):
                out = model(x, y)
                loss = out["loss"]
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            mask = y.ne(-100)
            num = mask.sum().item()
            total_loss += loss.item() * num
            total_tokens += num
            pred = out["logits"].argmax(dim=-1)
            correct += ((pred == y) & mask).sum().item()
            if "rosa_coverage" in out:
                rosa_cov += float(out["rosa_coverage"])
                rosa_m += float(out["rosa_avg_match_len"])
                rosa_steps += 1

        train_metrics = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_tokens),
            "ppl": math.exp(min(20.0, total_loss / max(1, total_tokens))),
            "token_acc": correct / max(1, total_tokens),
        }
        if rosa_steps > 0:
            train_metrics["rosa_coverage"] = rosa_cov / rosa_steps
            train_metrics["rosa_avg_match_len"] = rosa_m / rosa_steps
        val_metrics = evaluate(model, val_loader, device, pad_id)
        val_metrics["epoch"] = epoch
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['token_acc']:.4f} | val loss {val_metrics['loss']:.4f} acc {val_metrics['token_acc']:.4f}")
    return history


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def preview_doc(doc: str, n: int = 120) -> str:
    s = doc[:n].replace("\n", " | ")
    return s


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
    parser = argparse.ArgumentParser(description="训练 Qwen/LLaMA 风格基线模型 与 Emb(ROSA(x)) 融合模型，对比效果。")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="HF tokenizer 路径或名称。为空时使用 byte fallback，仅用于烟雾测试。")
    parser.add_argument("--split_mode", type=str, default="paragraph", choices=["paragraph", "line", "stream"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
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
    parser.add_argument("--rosa_min_match_len", type=int, default=2)
    parser.add_argument("--rosa_inject_layers", type=int, default=2)
    parser.add_argument("--rosa_scale", type=float, default=0.25)
    parser.add_argument("--rosa_allow_special_target", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/rosa_compare")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.tokenizer_name_or_path)
    text = read_text(args.data_path)
    docs = split_docs(text, args.split_mode)
    train_docs, val_docs, test_docs = train_val_test_split(docs, args.train_ratio, args.val_ratio, args.seed)

    print(f"总文档数: {len(docs)}")
    print(f"train/val/test: {len(train_docs)} / {len(val_docs)} / {len(test_docs)}")
    print(f"架构风格: {args.arch_style}")
    print(f"tokenizer: {args.tokenizer_name_or_path or 'byte-fallback'}")
    print(f"ROSA 最小匹配长度阈值: {args.rosa_min_match_len}")
    print(f"示例 train doc: {preview_doc(train_docs[0]) if train_docs else '<empty>'}")

    train_tok = tokenize_docs(train_docs, tokenizer, add_bos=True, add_eos=True)
    val_tok = tokenize_docs(val_docs, tokenizer, add_bos=True, add_eos=True)
    test_tok = tokenize_docs(test_docs, tokenizer, add_bos=True, add_eos=True)

    train_ds = LMDataset(train_tok, seq_len=args.seq_len, pad_id=tokenizer.pad_token_id, stride=args.stride)
    val_ds = LMDataset(val_tok, seq_len=args.seq_len, pad_id=tokenizer.pad_token_id, stride=args.stride)
    test_ds = LMDataset(test_tok, seq_len=args.seq_len, pad_id=tokenizer.pad_token_id, stride=args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    cfg = build_model_config(args, tokenizer)

    set_seed(args.seed)
    baseline = BaseLM(cfg)
    set_seed(args.seed)
    rosa_model = RosaFusedLM(
        cfg,
        min_match_len=args.rosa_min_match_len,
        inject_layers=args.rosa_inject_layers,
        rosa_scale=args.rosa_scale,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
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
    base_hist = train_one_model(
        baseline, train_loader, val_loader, device,
        pad_id=tokenizer.pad_token_id,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, use_bf16=args.bf16,
    )
    base_test = evaluate(baseline.to(device), test_loader, device, tokenizer.pad_token_id)

    print("\n==== 训练 rosa-fused ====")
    set_seed(args.seed)
    rosa_model = RosaFusedLM(
        cfg,
        min_match_len=args.rosa_min_match_len,
        inject_layers=args.rosa_inject_layers,
        rosa_scale=args.rosa_scale,
        special_ids=tokenizer.special_ids,
        forbid_special_target=not args.rosa_allow_special_target,
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
