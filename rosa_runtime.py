import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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

    def to(self, device: torch.device) -> "RosaInjectionPayload":
        return RosaInjectionPayload(
            address=self.address.to(device),
            layer_values=tuple(value.to(device) for value in self.layer_values),
            source=self.source,
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
