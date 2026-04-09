from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def set_activation_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    setattr(model, "activation_checkpointing_enabled", bool(enabled))


def activation_checkpointing_enabled(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "activation_checkpointing_enabled", False))


def block_forward_with_activation_checkpoint(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or not block.training or not hidden_states.requires_grad:
        return block(hidden_states, attn_mask)

    def forward_fn(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return block(x, mask)

    return checkpoint(forward_fn, hidden_states, attn_mask, use_reentrant=False)


def maybe_no_sync(model: torch.nn.Module, *, enabled: bool):
    if not enabled:
        return nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if callable(no_sync):
        return no_sync()
    return nullcontext()
