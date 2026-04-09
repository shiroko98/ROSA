from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import math

import torch
import torch.nn as nn


@dataclass
class RosaOptimizerBundle:
    dense_optimizer: torch.optim.Optimizer
    sparse_optimizer: torch.optim.Optimizer | None
    dense_params: List[nn.Parameter]
    sparse_params: List[nn.Parameter]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.dense_optimizer.zero_grad(set_to_none=set_to_none)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

    def clip_grad_norm_(self, max_norm: float) -> float:
        if max_norm <= 0 or not self.dense_params:
            return 0.0
        total = torch.nn.utils.clip_grad_norm_(self.dense_params, max_norm)
        return float(total)

    def stats(self) -> Dict[str, float]:
        return {
            "rosa_sparse_value_optimizer": 1.0 if self.sparse_optimizer is not None else 0.0,
            "rosa_dense_optimizer_groups": float(len(self.dense_optimizer.param_groups)),
            "rosa_sparse_optimizer_groups": float(len(self.sparse_optimizer.param_groups)) if self.sparse_optimizer is not None else 0.0,
            "rosa_dense_optimizer_params": float(sum(p.numel() for p in self.dense_params)),
            "rosa_sparse_optimizer_params": float(sum(p.numel() for p in self.sparse_params)),
        }

    def current_lrs(self) -> Dict[str, float]:
        return {
            "optimizer_lr_dense": float(self.dense_optimizer.param_groups[0]["lr"]) if self.dense_optimizer.param_groups else 0.0,
            "optimizer_lr_sparse": (
                float(self.sparse_optimizer.param_groups[0]["lr"])
                if self.sparse_optimizer is not None and self.sparse_optimizer.param_groups
                else 0.0
            ),
        }

    def grad_norms(self) -> Dict[str, float]:
        return {
            "grad_norm_dense": _param_group_grad_norm(self.dense_params),
            "grad_norm_sparse": _param_group_grad_norm(self.sparse_params),
        }


def _trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def _param_group_grad_norm(params: Iterable[nn.Parameter]) -> float:
    sq_sum = 0.0
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        if grad.is_sparse:
            values = grad.coalesce().values()
            if values.numel() == 0:
                continue
            sq_sum += float(values.detach().float().pow(2).sum().item())
        else:
            sq_sum += float(grad.detach().float().pow(2).sum().item())
    return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0


def build_training_optimizers(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> RosaOptimizerBundle:
    sparse_params: List[nn.Parameter] = []
    if hasattr(model, "rosa_value_store") and hasattr(model.rosa_value_store, "sparse_parameters"):
        sparse_params = list(model.rosa_value_store.sparse_parameters())
    sparse_ids = {id(param) for param in sparse_params}

    dense_params = [
        param
        for param in _trainable_parameters(model)
        if id(param) not in sparse_ids
    ]
    dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, weight_decay=weight_decay, betas=betas)
    sparse_optimizer = None
    if sparse_params:
        sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr, betas=betas)
    return RosaOptimizerBundle(
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_params=dense_params,
        sparse_params=sparse_params,
    )
