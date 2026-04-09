from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rosa_runtime import RosaHotAddressCache


class RowShardedEmbeddingTable(nn.Module):
    """按词表行切分的本地分片 embedding 表。"""

    def __init__(self, vocab_size: int, dim: int, *, num_shards: int, sparse: bool):
        super().__init__()
        if num_shards <= 1:
            raise ValueError("RowShardedEmbeddingTable 仅在 num_shards > 1 时使用。")
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_shards = num_shards
        self.sparse = sparse
        self.shard_rows = math.ceil(vocab_size / num_shards)
        self.shards = nn.ModuleList()
        self.shard_offsets: List[int] = []
        for shard_idx in range(num_shards):
            start = shard_idx * self.shard_rows
            if start >= vocab_size:
                break
            rows = min(self.shard_rows, vocab_size - start)
            self.shard_offsets.append(start)
            self.shards.append(nn.Embedding(rows, dim, sparse=sparse))

    def copy_from_embedding_(self, shared_embedding: nn.Embedding) -> None:
        with torch.no_grad():
            for shard, start in zip(self.shards, self.shard_offsets):
                end = start + shard.num_embeddings
                shard.weight.copy_(shared_embedding.weight[start:end])

    def sparse_parameters(self) -> List[nn.Parameter]:
        if not self.sparse:
            return []
        return [shard.weight for shard in self.shards]

    def value_shard_ids(self, addr_ids: torch.Tensor) -> torch.Tensor:
        safe = addr_ids.clamp_min(0)
        shard_ids = torch.div(safe, self.shard_rows, rounding_mode="floor")
        return shard_ids.clamp_max(len(self.shards) - 1)

    def forward(self, addr_ids: torch.Tensor) -> torch.Tensor:
        safe = addr_ids.clamp_min(0)
        flat_ids = safe.reshape(-1)
        out = torch.zeros(
            (flat_ids.numel(), self.dim),
            device=addr_ids.device,
            dtype=self.shards[0].weight.dtype,
        )
        if flat_ids.numel() == 0:
            return out.reshape(*safe.shape, self.dim)

        shard_ids = self.value_shard_ids(flat_ids)
        for shard_idx, (shard, start) in enumerate(zip(self.shards, self.shard_offsets)):
            mask = shard_ids.eq(shard_idx)
            if not mask.any():
                continue
            local_ids = flat_ids[mask] - start
            out[mask] = shard(local_ids)
        return out.reshape(*safe.shape, self.dim)


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
        num_shards: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.inject_layers = inject_layers
        self.mode = mode
        self.num_shards = max(1, int(num_shards))
        self.sparse_training = bool(sparse_training and mode == "per_layer")

        if mode == "shared":
            self.per_layer_tables = None
        elif mode == "per_layer":
            if self.num_shards == 1:
                self.per_layer_tables = nn.ModuleList(
                    [nn.Embedding(vocab_size, dim, sparse=self.sparse_training) for _ in range(max(0, inject_layers))]
                )
            else:
                self.per_layer_tables = nn.ModuleList(
                    [
                        RowShardedEmbeddingTable(
                            vocab_size,
                            dim,
                            num_shards=self.num_shards,
                            sparse=self.sparse_training,
                        )
                        for _ in range(max(0, inject_layers))
                    ]
                )
        else:
            raise ValueError(f"未知 rosa_value_mode: {mode}")

    @property
    def is_per_layer(self) -> bool:
        return self.mode == "per_layer"

    @property
    def uses_sparse_training(self) -> bool:
        return self.sparse_training

    @property
    def is_sharded(self) -> bool:
        return self.is_per_layer and self.num_shards > 1

    def copy_shared_weights_(self, shared_embedding: nn.Embedding) -> None:
        if not self.is_per_layer or self.per_layer_tables is None:
            return
        with torch.no_grad():
            for table in self.per_layer_tables:
                if isinstance(table, RowShardedEmbeddingTable):
                    table.copy_from_embedding_(shared_embedding)
                else:
                    table.weight.copy_(shared_embedding.weight)

    def sparse_parameters(self) -> List[nn.Parameter]:
        if not self.uses_sparse_training or self.per_layer_tables is None:
            return []
        params: List[nn.Parameter] = []
        for table in self.per_layer_tables:
            if isinstance(table, RowShardedEmbeddingTable):
                params.extend(table.sparse_parameters())
            else:
                params.append(table.weight)
        return params

    def active_shard_count(self, addr_ids: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> int:
        if not self.is_sharded or self.per_layer_tables is None or len(self.per_layer_tables) == 0:
            return 0
        flat_ids = addr_ids.reshape(-1)
        if valid_mask is not None:
            mask = valid_mask.reshape(-1)
            flat_ids = flat_ids[mask]
        if flat_ids.numel() == 0:
            return 0
        table = self.per_layer_tables[0]
        if not isinstance(table, RowShardedEmbeddingTable):
            return 0
        shard_ids = table.value_shard_ids(flat_ids)
        return int(torch.unique(shard_ids).numel())

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
