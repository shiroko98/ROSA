"""Checkpoint-strict adapter around the pinned upstream RWKV-7 reference runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import nn

from .precision import fence_wkv_fp32

UPSTREAM_COMMIT = "1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1"
RUNTIME_PATH = Path(__file__).resolve().parents[1] / "third_party/rwkv_lm_v7/runtime.py"


@dataclass(frozen=True)
class RWKVConfig:
    vocab_size: int = 259
    n_embd: int = 32
    n_layer: int = 2
    head_size: int = 16
    d_decay_lora: int = 16
    d_aaa_lora: int = 16
    d_mv_lora: int = 16
    d_gate_lora: int = 16

    def validate(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError("All RWKV dimensions must be positive.")
        if self.n_embd % self.head_size:
            raise ValueError("n_embd must be divisible by head_size.")


def load_base_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load tensor-only RWKV .pth; refuse silent partial or adapter-only loading."""
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping) or not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError("Expected an RWKV tensor state_dict (.pth), not an optimizer/DeepSpeed directory.")
    state = dict(obj)
    for prefix in ("_forward_module.", "module.", "backbone."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    if "emb.weight" not in state:
        raise ValueError("RWKV checkpoint must contain emb.weight.")
    return state


def infer_config(state: Mapping[str, torch.Tensor]) -> RWKVConfig:
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("blocks.")})
    if not layers or layers != list(range(len(layers))):
        raise ValueError("RWKV block indices must be contiguous starting from zero.")
    vocab, dim = state["emb.weight"].shape
    try:
        cfg = RWKVConfig(vocab, dim, len(layers), state["blocks.0.att.r_k"].shape[1],
                         state["blocks.0.att.w1"].shape[1], state["blocks.0.att.a1"].shape[1],
                         state["blocks.0.att.v1"].shape[1], state["blocks.0.att.g1"].shape[1])
        if state["blocks.0.ffn.key.weight"].shape != (dim * 4, dim):
            raise ValueError("This pinned x070 backbone requires dim_ffn == 4 * n_embd.")
    except KeyError as exc:
        raise ValueError(f"Not a supported RWKV-7 x070 checkpoint: missing {exc}") from exc
    cfg.validate()
    return cfg


def load_runtime(cfg: RWKVConfig, dtype: torch.dtype):
    # Each model gets isolated runtime globals. Different model widths / dtypes
    # can coexist without configure_runtime() changing another model's recurrence.
    spec = importlib.util.spec_from_file_location("rosa_private_rwkv_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.configure_runtime(dtype=dtype, head_size=cfg.head_size,
                             d_decay_lora=cfg.d_decay_lora, d_aaa_lora=cfg.d_aaa_lora,
                             d_mv_lora=cfg.d_mv_lora, d_gate_lora=cfg.d_gate_lora)
    module._unfenced_wkv_op = module.RWKV7_OP
    module._fp32_wkv_op = fence_wkv_fp32(module.RWKV7_OP)
    module.RWKV7_OP = module._fp32_wkv_op
    return module


def initialize_reference(model: nn.Module) -> None:
    """Initialize the runtime's empty tensors for *toy* training, not official pretraining."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".weight") and ("ln" in name):
                p.fill_(1)
            elif name.endswith(".bias"):
                p.zero_()
            elif name.endswith(".w0"):
                p.fill_(-2)
            elif name.endswith((".k_k", ".k_a")):
                p.fill_(1)
            elif name.endswith((".x_r", ".x_w", ".x_k", ".x_v", ".x_a", ".x_g")):
                p.fill_(0.5)
            else:
                p.normal_(0, 0.02)


def build_backbone(cfg: RWKVConfig, *, dtype=torch.float32, state_dict=None):
    cfg.validate()
    runtime = load_runtime(cfg, dtype)
    args = SimpleNamespace(vocab_size=cfg.vocab_size, n_embd=cfg.n_embd,
                           n_layer=cfg.n_layer, head_size_a=cfg.head_size)
    model = runtime.RWKV(args)
    if state_dict is None:
        initialize_reference(model)
    else:
        # Strict loading is intentional: never silently train a mostly random model.
        model.load_state_dict(state_dict, strict=True)
    model.to(dtype=dtype)
    # Keep the private module alive and allow explicit reference precision checks.
    model._reference_runtime = runtime
    return model
