import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class RosaAddressBatch:
    addr_ids: torch.Tensor
    raw_match_lens: torch.Tensor
    fired_match_lens: torch.Tensor
    valid_mask: torch.Tensor
    special_mask: torch.Tensor
    source: str = "unknown"

    def to(self, device: torch.device) -> "RosaAddressBatch":
        return RosaAddressBatch(
            addr_ids=self.addr_ids.to(device),
            raw_match_lens=self.raw_match_lens.to(device),
            fired_match_lens=self.fired_match_lens.to(device),
            valid_mask=self.valid_mask.to(device),
            special_mask=self.special_mask.to(device),
            source=self.source,
        )


@dataclass(frozen=True)
class RosaInjectionPayload:
    address: RosaAddressBatch
    layer_values: Tuple[torch.Tensor, ...]
    source: str = "unknown"
    stats: Optional[Dict[str, float]] = None

    def to(self, device: torch.device) -> "RosaInjectionPayload":
        return RosaInjectionPayload(
            address=self.address.to(device),
            layer_values=tuple(value.to(device) for value in self.layer_values),
            source=self.source,
            stats=dict(self.stats or {}),
        )


def _stage_tensor(tensor: torch.Tensor, *, use_pinned_memory: bool) -> torch.Tensor:
    staged = tensor.detach().to("cpu")
    if use_pinned_memory and staged.device.type == "cpu":
        staged = staged.pin_memory()
    return staged


def stage_rosa_payload(payload: RosaInjectionPayload, *, use_pinned_memory: bool) -> RosaInjectionPayload:
    address = RosaAddressBatch(
        addr_ids=_stage_tensor(payload.address.addr_ids, use_pinned_memory=use_pinned_memory),
        raw_match_lens=_stage_tensor(payload.address.raw_match_lens, use_pinned_memory=use_pinned_memory),
        fired_match_lens=_stage_tensor(payload.address.fired_match_lens, use_pinned_memory=use_pinned_memory),
        valid_mask=_stage_tensor(payload.address.valid_mask, use_pinned_memory=use_pinned_memory),
        special_mask=_stage_tensor(payload.address.special_mask, use_pinned_memory=use_pinned_memory),
        source=payload.address.source,
    )
    return RosaInjectionPayload(
        address=address,
        layer_values=tuple(
            _stage_tensor(value, use_pinned_memory=use_pinned_memory) for value in payload.layer_values
        ),
        source=payload.source,
        stats=dict(payload.stats or {}),
    )


def rosa_payload_nbytes(payload: RosaInjectionPayload) -> int:
    total = 0
    tensors = [
        payload.address.addr_ids,
        payload.address.raw_match_lens,
        payload.address.fired_match_lens,
        payload.address.valid_mask,
        payload.address.special_mask,
        *payload.layer_values,
    ]
    for tensor in tensors:
        total += tensor.element_size() * tensor.numel()
    return total


class RosaHotAddressCache:
    def __init__(self, *, num_layers: int, max_entries_per_layer: int):
        self.num_layers = max(0, num_layers)
        self.max_entries_per_layer = max(0, max_entries_per_layer)
        self.enabled = self.num_layers > 0 and self.max_entries_per_layer > 0
        self._layer_caches: List[OrderedDict[int, torch.Tensor]] = [
            OrderedDict() for _ in range(self.num_layers)
        ]
        self._layer_freqs: List[Counter[int]] = [Counter() for _ in range(self.num_layers)]
        self._layer_stats: List[Dict[str, float]] = [self._empty_stats() for _ in range(self.num_layers)]
        self._stats: Dict[str, float] = self._empty_stats()
        self._lock = threading.Lock()

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {
            "token_requests": 0.0,
            "token_hits": 0.0,
            "unique_requests": 0.0,
            "unique_hits": 0.0,
            "fills": 0.0,
            "evictions": 0.0,
        }

    def reset(self) -> None:
        with self._lock:
            for cache in self._layer_caches:
                cache.clear()
            for freq in self._layer_freqs:
                freq.clear()
            self._stats = self._empty_stats()
            self._layer_stats = [self._empty_stats() for _ in range(self.num_layers)]

    def reset_stats(self, *, clear_cache: bool = False) -> None:
        with self._lock:
            self._stats = self._empty_stats()
            self._layer_stats = [self._empty_stats() for _ in range(self.num_layers)]
            for freq in self._layer_freqs:
                freq.clear()
            if clear_cache:
                for cache in self._layer_caches:
                    cache.clear()

    def _update_running_stats(self, layer_idx: int, stats: Dict[str, float]) -> None:
        with self._lock:
            for key, value in stats.items():
                if key not in self._stats:
                    continue
                self._stats[key] += float(value)
                self._layer_stats[layer_idx][key] += float(value)

    def lookup(
        self,
        layer_idx: int,
        addr_ids: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor],
        value_dim: int,
        fetch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx={layer_idx} 超出 RosaHotAddressCache 范围。")
        if not self.enabled:
            return fetch_fn(addr_ids), {}

        valid = valid_mask
        if valid is None:
            valid = torch.ones_like(addr_ids, dtype=torch.bool)
        else:
            valid = valid_mask.bool()

        flat_ids = addr_ids.reshape(-1)
        flat_valid = valid.reshape(-1)
        valid_positions = flat_valid.nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            dummy = fetch_fn(addr_ids.reshape(-1)[:1] if addr_ids.numel() > 0 else torch.zeros(1, dtype=addr_ids.dtype, device=addr_ids.device))
            zeros = torch.zeros(
                (*addr_ids.shape, value_dim),
                device=dummy.device,
                dtype=dummy.dtype,
            )
            return zeros, {
                "token_requests": 0.0,
                "token_hits": 0.0,
                "unique_requests": 0.0,
                "unique_hits": 0.0,
                "fills": 0.0,
                "evictions": 0.0,
                "active_entries": float(len(self._layer_caches[layer_idx])),
                "token_hit_rate": 0.0,
                "unique_hit_rate": 0.0,
            }

        valid_ids = flat_ids.index_select(0, valid_positions)
        unique_ids, inverse, counts = torch.unique(
            valid_ids,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        unique_id_list = [int(x) for x in unique_ids.detach().cpu().tolist()]
        count_list = [int(x) for x in counts.detach().cpu().tolist()]

        hit_rows: Dict[int, torch.Tensor] = {}
        miss_ids: List[int] = []
        miss_rows: List[int] = []
        token_hits = 0
        unique_hits = 0

        with self._lock:
            cache = self._layer_caches[layer_idx]
            freqs = self._layer_freqs[layer_idx]
            for row_idx, (addr_id, token_count) in enumerate(zip(unique_id_list, count_list)):
                freqs[addr_id] += token_count
                cached = cache.pop(addr_id, None)
                if cached is not None and cached.device == addr_ids.device:
                    cache[addr_id] = cached
                    hit_rows[row_idx] = cached
                    unique_hits += 1
                    token_hits += token_count
                else:
                    miss_ids.append(addr_id)
                    miss_rows.append(row_idx)

        miss_value_rows: Dict[int, torch.Tensor] = {}
        fills = 0
        evictions = 0
        if miss_ids:
            miss_tensor = torch.tensor(miss_ids, dtype=addr_ids.dtype, device=addr_ids.device)
            miss_values = fetch_fn(miss_tensor)
            for offset, row_idx in enumerate(miss_rows):
                miss_value_rows[row_idx] = miss_values[offset]
            staged_rows = [value.detach().clone() for value in miss_values]
            with self._lock:
                cache = self._layer_caches[layer_idx]
                for addr_id, staged in zip(miss_ids, staged_rows):
                    if addr_id not in cache:
                        fills += 1
                    cache[addr_id] = staged
                    cache.move_to_end(addr_id)
                    while len(cache) > self.max_entries_per_layer:
                        cache.popitem(last=False)
                        evictions += 1

        sample_value = None
        if hit_rows:
            sample_value = next(iter(hit_rows.values()))
        elif miss_value_rows:
            sample_value = next(iter(miss_value_rows.values()))
        else:
            sample_value = fetch_fn(valid_ids[:1])

        ordered_values = []
        for row_idx in range(len(unique_id_list)):
            value = hit_rows.get(row_idx)
            if value is None:
                value = miss_value_rows[row_idx]
            ordered_values.append(value.to(sample_value.device))
        unique_values = torch.stack(ordered_values, dim=0)
        valid_values = unique_values.index_select(0, inverse)

        out_flat = torch.zeros(
            (flat_ids.shape[0], value_dim),
            device=sample_value.device,
            dtype=sample_value.dtype,
        )
        out_flat.index_copy_(0, valid_positions, valid_values)

        local_stats = {
            "token_requests": float(valid_ids.numel()),
            "token_hits": float(token_hits),
            "unique_requests": float(len(unique_id_list)),
            "unique_hits": float(unique_hits),
            "fills": float(fills),
            "evictions": float(evictions),
        }
        self._update_running_stats(layer_idx, local_stats)
        with self._lock:
            active_entries = float(len(self._layer_caches[layer_idx]))
        local_stats.update(
            {
                "active_entries": active_entries,
                "token_hit_rate": local_stats["token_hits"] / local_stats["token_requests"]
                if local_stats["token_requests"] > 0
                else 0.0,
                "unique_hit_rate": local_stats["unique_hits"] / local_stats["unique_requests"]
                if local_stats["unique_requests"] > 0
                else 0.0,
            }
        )
        return out_flat.view(*addr_ids.shape, value_dim), local_stats

    def stats(self, *, top_k: int = 5) -> Dict[str, Any]:
        with self._lock:
            total = dict(self._stats)
            layer_stats = [dict(row) for row in self._layer_stats]
            cache_sizes = [len(cache) for cache in self._layer_caches]
            top_addrs = [freq.most_common(top_k) for freq in self._layer_freqs]

        token_requests = total["token_requests"]
        unique_requests = total["unique_requests"]
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "max_entries_per_layer": self.max_entries_per_layer,
            **total,
            "token_hit_rate": total["token_hits"] / token_requests if token_requests > 0 else 0.0,
            "unique_hit_rate": total["unique_hits"] / unique_requests if unique_requests > 0 else 0.0,
            "active_entries": sum(cache_sizes),
            "layer_stats": [],
        }
        for layer_idx, row in enumerate(layer_stats):
            layer_token_requests = row["token_requests"]
            layer_unique_requests = row["unique_requests"]
            summary["layer_stats"].append(
                {
                    "layer_idx": layer_idx,
                    **row,
                    "token_hit_rate": row["token_hits"] / layer_token_requests if layer_token_requests > 0 else 0.0,
                    "unique_hit_rate": row["unique_hits"] / layer_unique_requests if layer_unique_requests > 0 else 0.0,
                    "active_entries": cache_sizes[layer_idx],
                    "top_addresses": [
                        {"addr_id": addr_id, "count": count} for addr_id, count in top_addrs[layer_idx]
                    ],
                }
            )
        return summary


class RosaStagingBuffer:
    def __init__(self, *, use_pinned_memory: bool = False):
        self.use_pinned_memory = use_pinned_memory
        self._store: Dict[str, RosaInjectionPayload] = {}
        self._lock = threading.Lock()

    def stage(self, request_key: str, payload: RosaInjectionPayload) -> RosaInjectionPayload:
        staged = stage_rosa_payload(payload, use_pinned_memory=self.use_pinned_memory)
        with self._lock:
            self._store[request_key] = staged
        return staged

    def pop(self, request_key: str) -> Optional[RosaInjectionPayload]:
        with self._lock:
            return self._store.pop(request_key, None)

    def has(self, request_key: str) -> bool:
        with self._lock:
            return request_key in self._store


class RosaPrefetcher:
    def __init__(
        self,
        builder: Callable[[RosaAddressBatch], RosaInjectionPayload],
        *,
        use_async: bool = True,
        supports_async: bool = True,
        use_pinned_memory: bool = False,
        max_workers: int = 1,
    ):
        self.builder = builder
        self.use_async = use_async
        self.supports_async = supports_async
        self.staging_buffer = RosaStagingBuffer(use_pinned_memory=use_pinned_memory)
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rosa-prefetch")
            if use_async and supports_async
            else None
        )
        self.pending: Dict[str, Future] = {}
        self.lock = threading.Lock()
        self.stats_data = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "waits": 0,
            "wait_ms": 0.0,
            "staged_payloads": 0,
            "staged_bytes": 0,
            "sync_fallbacks": 0,
        }

    def _finalize_future(self, request_key: str, future: Future) -> RosaInjectionPayload:
        payload = future.result()
        staged = self.staging_buffer.stage(request_key, payload)
        self.stats_data["staged_payloads"] += 1
        self.stats_data["staged_bytes"] += rosa_payload_nbytes(staged)
        with self.lock:
            self.pending.pop(request_key, None)
        return staged

    def submit(self, request_key: str, address_batch: RosaAddressBatch) -> None:
        self.stats_data["requests"] += 1
        if self.staging_buffer.has(request_key):
            return
        with self.lock:
            if request_key in self.pending:
                return
            if self.executor is None:
                self.stats_data["sync_fallbacks"] += 1
                payload = self.builder(address_batch)
                staged = self.staging_buffer.stage(request_key, payload)
                self.stats_data["staged_payloads"] += 1
                self.stats_data["staged_bytes"] += rosa_payload_nbytes(staged)
                return
            self.pending[request_key] = self.executor.submit(self.builder, address_batch)

    def ready(self, request_key: str) -> bool:
        if self.staging_buffer.has(request_key):
            return True
        with self.lock:
            future = self.pending.get(request_key)
        if future is None:
            return False
        if future.done():
            self._finalize_future(request_key, future)
            return True
        return False

    def consume(self, request_key: str, *, device: Optional[torch.device] = None) -> Optional[RosaInjectionPayload]:
        payload = self.staging_buffer.pop(request_key)
        if payload is not None:
            self.stats_data["hits"] += 1
            return payload.to(device) if device is not None else payload

        with self.lock:
            future = self.pending.get(request_key)
        if future is None:
            self.stats_data["misses"] += 1
            return None

        started = time.perf_counter()
        if not future.done():
            self.stats_data["waits"] += 1
        payload = self._finalize_future(request_key, future)
        self.stats_data["wait_ms"] += (time.perf_counter() - started) * 1000.0
        self.stats_data["hits"] += 1
        payload = self.staging_buffer.pop(request_key) or payload
        return payload.to(device) if device is not None else payload

    def stats(self) -> Dict[str, float]:
        hits = self.stats_data["hits"]
        requests = self.stats_data["requests"]
        return {
            **self.stats_data,
            "hit_rate": hits / requests if requests > 0 else 0.0,
            "avg_wait_ms": self.stats_data["wait_ms"] / self.stats_data["waits"]
            if self.stats_data["waits"] > 0
            else 0.0,
        }

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
