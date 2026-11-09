"""RWKV-7 backbone with explicit original / improved ROSA fusion definitions."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rosa_addressing import OnlineRosaBatchState, RosaAddressEngine
from rosa_runtime import RosaAddressBatch
from rosa_value_store import RosaValueStore
from .backbone import RWKVConfig, build_backbone


@dataclass
class RWKVRosaState:
    rwkv: list[dict[str, Any]]
    rosa: OnlineRosaBatchState | None
    batch_size: int
    tokens_seen: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (y * self.weight.float()).to(x.dtype)


class RWKVRosaLM(nn.Module):
    """Original = exact SAM + shared token embedding residual, no learned gate.

    This is an explicitly defined *minimal neural integration* of original ROSA,
    not a claim that original ROSA itself has trainable parameters or that this
    residual formula is an upstream official RWKV+ROSA implementation.
    Improved = same SAM addresses + per-layer learned values + context/length gate.
    """
    def __init__(self, cfg: RWKVConfig, *, variant="improved", inject_layers=(0,),
                 scale=0.15, min_match_len=1, sparse_values=False, value_shards=1,
                 sam_impl="fast", special_ids=(0,), dtype=torch.float32,
                 base_state=None, freeze_backbone=False, activation_checkpointing=False,
                 capacity_width=0):
        super().__init__()
        if variant not in {"baseline", "original", "improved", "capacity"}:
            raise ValueError(f"Unknown variant: {variant}")
        ids = tuple(sorted(set(inject_layers)))
        if not ids or any(i < 0 or i >= cfg.n_layer for i in ids):
            raise ValueError("inject_layers must name valid RWKV blocks.")
        if scale < 0 or min_match_len < 1 or value_shards < 1:
            raise ValueError("Invalid ROSA scale / match threshold / shard count.")
        if freeze_backbone and variant == "original":
            raise ValueError("Original ROSA has no trainable branch; frozen backbone leaves nothing to train.")
        self.cfg, self.variant = cfg, variant
        self.scale = float(scale)
        self.inject_layers = ids
        self.layer_to_slot = {i: s for s, i in enumerate(ids)}
        self.activation_checkpointing = activation_checkpointing
        self.backbone = build_backbone(cfg, dtype=dtype, state_dict=base_state)
        self.address_engine = RosaAddressEngine(min_match_len=min_match_len, pad_id=0,
                                               special_ids=set(special_ids), forbid_special_target=True,
                                               backend="sam", sequence_mode="online_sam",
                                               online_sam_impl=sam_impl)
        self.rosa_value_store = RosaValueStore(vocab_size=cfg.vocab_size, dim=cfg.n_embd,
                                              inject_layers=len(ids),
                                              mode="per_layer" if variant == "improved" else "shared",
                                              sparse_training=sparse_values and variant == "improved",
                                              num_shards=value_shards if variant == "improved" else 1)
        if variant == "improved":
            self.rosa_value_store.copy_shared_weights_(self.backbone.emb)
            self.hidden_norms = nn.ModuleList([RMSNorm(cfg.n_embd) for _ in ids])
            self.value_norms = nn.ModuleList([RMSNorm(cfg.n_embd) for _ in ids])
            self.key_projs = nn.ModuleList([nn.Linear(cfg.n_embd, cfg.n_embd, bias=False) for _ in ids])
            self.value_projs = nn.ModuleList([nn.Linear(cfg.n_embd, cfg.n_embd, bias=False) for _ in ids])
            for proj in [*self.key_projs, *self.value_projs]:
                nn.init.eye_(proj.weight)
            self.gate_bias = nn.Parameter(torch.zeros(len(ids)))
            self.length_scale = nn.Parameter(torch.ones(len(ids)))
        if variant == "capacity":
            if capacity_width < 1:
                raise ValueError("capacity variant needs positive capacity_width.")
            self.adapter_down = nn.ModuleList([nn.Linear(cfg.n_embd, capacity_width, bias=False) for _ in ids])
            self.adapter_up = nn.ModuleList([nn.Linear(capacity_width, cfg.n_embd, bias=False) for _ in ids])
            for layer in self.adapter_up:
                nn.init.zeros_(layer.weight)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.to(dtype=dtype)

    @property
    def has_rosa(self):
        return self.variant in {"original", "improved"}

    @property
    def rosa_device(self):
        # The original arm reuses backbone embeddings. Only improved ROSA has
        # independent parameters that can execute on a different device.
        return self.gate_bias.device if self.variant == 'improved' else self.backbone.emb.weight.device

    def place_devices(self, backbone_device, rosa_device=None):
        """Place the whole learned ROSA branch, not just SAM lookup, on CPU.

        Call before optimizer construction. Do NOT call model.to(cuda) afterwards:
        that would move CPU ROSA back onto the GPU.
        """
        backbone_device = torch.device(backbone_device)
        branch_device = torch.device(rosa_device) if rosa_device is not None else backbone_device
        if branch_device != backbone_device and self.variant != 'improved':
            raise ValueError('Separate ROSA placement requires variant=improved.')
        self.backbone.to(backbone_device)
        for name, module in self.named_children():
            if name != 'backbone':
                module.to(branch_device)
                if branch_device.type == 'cpu' and self.variant == 'improved':
                    module.float()
        for parameter in self.parameters(recurse=False):
            parameter.data = parameter.data.to(device=branch_device,
                                               dtype=torch.float32 if branch_device.type == 'cpu' else parameter.dtype)
        return self

    def _improved_residual(self, slot, hidden, address):
        context = (torch.autocast(device_type='cpu', enabled=False)
                   if self.rosa_device.type == 'cpu' else nullcontext())
        with context:
            return self._local_improved_residual(slot, hidden, address)

    def _local_improved_residual(self, slot, hidden, address):
        # Tensor.to is differentiable. Never detach hidden/residual: the gradient
        # must cross CPU->GPU and return through the frozen RWKV blocks to ROSA.
        address = address.to(self.rosa_device)
        hidden_local = hidden.to(device=self.rosa_device,
                                 dtype=self.gate_bias.dtype if self.rosa_device.type == 'cpu' else hidden.dtype)
        value = self.rosa_value_store.lookup(slot, address.addr_ids, shared_embedding=None)
        key = self.value_norms[slot](self.key_projs[slot](value))
        score = (self.hidden_norms[slot](hidden_local).float() * key.float()).sum(-1) / math.sqrt(self.cfg.n_embd)
        score = score + self.length_scale[slot] * torch.log1p(address.fired_match_lens.float())
        gate = torch.sigmoid(score + self.gate_bias[slot])
        residual = self.value_projs[slot](value) * gate.unsqueeze(-1).to(value.dtype)
        residual = self.scale * residual * address.valid_mask.unsqueeze(-1)
        return residual.to(device=hidden.device, dtype=hidden.dtype), gate[address.valid_mask].detach()

    def init_state(self, batch_size: int) -> RWKVRosaState:
        return RWKVRosaState(self.backbone.init_state(batch_size),
                             self.address_engine.init_state(batch_size) if self.has_rosa else None,
                             batch_size)

    def forward_features(self, input_ids, *, addresses: RosaAddressBatch | None = None,
                         state: RWKVRosaState | None = None, return_state=False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty time].")
        use_state = state is not None or return_state
        if use_state:
            if input_ids.eq(0).any():
                raise ValueError("Stateful batches must be unpadded; use separate sessions for unequal lengths.")
            state = self.init_state(input_ids.shape[0]) if state is None else state
            if state.batch_size != input_ids.shape[0]:
                raise ValueError("Online state batch size mismatch.")
            if addresses is not None:
                raise ValueError("Stateful calls must update their own ROSA state, not use detached addresses.")
        address = None
        stats = {}
        if self.has_rosa:
            if use_state:
                address = self.address_engine.forward_step(input_ids.detach().cpu(), state.rosa, device=self.rosa_device)
            else:
                address = addresses if addresses is not None else self.address_engine.forward_seq(
                    input_ids.detach().cpu(), device=self.rosa_device)
                address = address.to(self.rosa_device)
            if address.addr_ids.shape != input_ids.shape:
                raise ValueError("ROSA address shape does not match input.")
            valid = address.valid_mask
            nonpad = input_ids.detach().to(self.rosa_device).ne(0)
            stats["rosa_coverage"] = float((valid & nonpad).sum()) / max(1, int(nonpad.sum()))
            stats["rosa_active_addresses"] = float(torch.unique(address.addr_ids[valid]).numel())
        x = self.backbone.emb(input_ids)
        v_first = torch.empty_like(x)
        next_states = []
        gates = []
        for i, block in enumerate(self.backbone.blocks):
            slot = self.layer_to_slot.get(i)
            if slot is not None and self.has_rosa:
                if self.variant == 'improved':
                    residual, gate = self._improved_residual(slot, x, address)
                    gates.append(gate)
                    x = x + residual
                else:
                    value = self.rosa_value_store.lookup(slot, address.addr_ids, shared_embedding=self.backbone.emb)
                    x = x + (self.scale * value * address.valid_mask.unsqueeze(-1)).to(x.dtype)
            elif slot is not None and self.variant == "capacity":
                x = x + self.adapter_up[slot](F.silu(self.adapter_down[slot](x)))
            if use_state:
                x, v_first, block_state = block(x, v_first, state=state.rwkv[i], return_state=True)
                next_states.append(block_state)
            elif self.activation_checkpointing and self.training:
                x, v_first = checkpoint(block, x, v_first, use_reentrant=False)
            else:
                x, v_first = block(x, v_first)
        if gates:
            g = torch.cat(gates)
            stats["rosa_avg_gate"] = float(g.mean()) if g.numel() else 0.0
        x = self.backbone.ln_out(x)
        final_state = (RWKVRosaState(next_states, state.rosa, state.batch_size,
                                    state.tokens_seen + input_ids.shape[1]) if use_state else None)
        return x, stats, final_state

    def forward(self, input_ids, *, addresses=None, state=None, return_state=False):
        hidden, stats, final_state = self.forward_features(input_ids, addresses=addresses,
                                                          state=state, return_state=return_state)
        result = {"logits": self.backbone.head(hidden), "stats": stats}
        if final_state is not None:
            result["state"] = final_state
        return result


def configure_train_scope(model: RWKVRosaLM, scope="full", last_blocks=1):
    """Identical backbone training scope across arms; trainables use FP32 AdamW states.

    Frozen weights can remain BF16. Autocast handles mixed storage during forward.
    """
    if scope not in {"full", "last_block", "head", "rosa_only"}:
        raise ValueError(f"Unknown training scope: {scope}")
    if last_blocks < 1 or last_blocks > model.cfg.n_layer:
        raise ValueError("last_blocks must be between 1 and n_layer.")
    if scope == 'rosa_only' and model.variant != 'improved':
        raise ValueError('rosa_only training requires improved ROSA with independent trainable parameters.')
    for name, p in model.named_parameters():
        train = not name.startswith("backbone.") or scope == "full"
        if scope != "rosa_only" and name.startswith("backbone.head."):
            train = True
        if scope == "last_block" and name.startswith("backbone.blocks."):
            train = int(name.split(".")[2]) >= model.cfg.n_layer - last_blocks
        if scope == "last_block" and name.startswith("backbone.ln_out."):
            train = True
        p.requires_grad_(train)
        if train and p.dtype != torch.float32:
            p.data = p.data.float()
    return {"scope": scope, "last_blocks": last_blocks,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_backbone_parameters": sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad),
            "trainable_cpu_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad and p.device.type == 'cpu'),
            "trainable_cuda_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad and p.device.type == 'cuda'),
            "backbone_device": str(model.backbone.emb.weight.device), "rosa_device": str(model.rosa_device),
            "backbone_trainable_names": [n.removeprefix("backbone.") for n, p in model.named_parameters()
                                         if n.startswith("backbone.") and p.requires_grad]}
