from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rosa_runtime import RosaHotAddressCache


class RosaValueStore(nn.Module):
    """ROSA value lookup 抽象层：shared embedding、per-layer value table，以及可选稀疏梯度训练。"""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        inject_layers: int,
        mode: str = "shared",
        sparse_training: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.inject_layers = inject_layers
        self.mode = mode
        self.sparse_training = bool(sparse_training and mode == "per_layer")

        if mode == "shared":
            self.per_layer_tables = None
        elif mode == "per_layer":
            self.per_layer_tables = nn.ModuleList(
                [nn.Embedding(vocab_size, dim, sparse=self.sparse_training) for _ in range(max(0, inject_layers))]
            )
        else:
            raise ValueError(f"未知 rosa_value_mode: {mode}")

    @property
    def is_per_layer(self) -> bool:
        return self.mode == "per_layer"

    @property
    def uses_sparse_training(self) -> bool:
        return self.sparse_training

    def copy_shared_weights_(self, shared_embedding: nn.Embedding) -> None:
        if not self.is_per_layer or self.per_layer_tables is None:
            return
        with torch.no_grad():
            for table in self.per_layer_tables:
                table.weight.copy_(shared_embedding.weight)

    def sparse_parameters(self) -> List[nn.Parameter]:
        if not self.uses_sparse_training or self.per_layer_tables is None:
            return []
        return [table.weight for table in self.per_layer_tables]

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
