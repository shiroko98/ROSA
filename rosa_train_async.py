from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque, Dict, Iterable, Iterator, Optional

import torch


def prepare_online_seq_batch(
    batch: Dict[str, Any],
    *,
    address_engine: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    if batch.get("rosa_precomputed_ids") is not None:
        prepared = dict(batch)
        prepared["rosa_async_prefetch_prepare_s"] = time.perf_counter() - started_at
        return prepared

    input_ids = batch["input_ids"]
    memory_ids = batch.get("rosa_memory_ids")
    state_snapshots = batch.get("rosa_state_snapshots")
    replay_ids = batch.get("rosa_replay_ids")
    if state_snapshots is not None:
        address_batch = address_engine.forward_seq_from_snapshots(
            input_ids=input_ids,
            state_snapshots=state_snapshots,
            replay_ids=replay_ids,
            device=torch.device("cpu"),
        )
    else:
        address_batch = address_engine.forward_seq(
            input_ids=input_ids,
            memory_ids=memory_ids,
            device=torch.device("cpu"),
        )
    prepared = dict(batch)
    prepared["rosa_precomputed_ids"] = address_batch.addr_ids.cpu()
    prepared["rosa_precomputed_match_lens"] = address_batch.fired_match_lens.cpu()
    prepared["rosa_precomputed_raw_best_lens"] = address_batch.raw_match_lens.cpu()
    prepared["rosa_precomputed_source"] = address_batch.source
    prepared["rosa_async_prefetch_prepare_s"] = time.perf_counter() - started_at
    return prepared


def read_async_prefetch_batch_stats(batch: Dict[str, Any]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for key in [
        "rosa_async_prefetch_prepare_s",
        "rosa_async_prefetch_wait_s",
        "rosa_async_prefetch_depth",
        "rosa_async_prefetch_inflight",
        "rosa_async_prefetch_queue_fill",
    ]:
        value = batch.get(key)
        if value is not None:
            stats[key] = float(value)
    return stats


class RosaTrainAddressPrefetchLoader:
    def __init__(
        self,
        base_loader: Iterable[Dict[str, Any]],
        *,
        address_engine: Any,
        enabled: bool = True,
        max_workers: int = 1,
        prefetch_batches: int = 1,
    ):
        self.base_loader = base_loader
        self.address_engine = address_engine
        self.enabled = enabled
        self.max_workers = max(1, int(max_workers))
        self.prefetch_batches = max(1, int(prefetch_batches))

    def __len__(self) -> int:
        return len(self.base_loader)  # type: ignore[arg-type]

    def _submit(self, executor: ThreadPoolExecutor, batch: Dict[str, Any]) -> Future:
        return executor.submit(
            prepare_online_seq_batch,
            batch,
            address_engine=self.address_engine,
        )

    def _fill_pending(
        self,
        pending: Deque[Future],
        base_iter: Iterator[Dict[str, Any]],
        executor: ThreadPoolExecutor,
    ) -> None:
        while len(pending) < self.prefetch_batches:
            try:
                batch = next(base_iter)
            except StopIteration:
                break
            pending.append(self._submit(executor, batch))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        base_iter = iter(self.base_loader)
        if not self.enabled:
            yield from base_iter
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending: Deque[Future] = deque()
            self._fill_pending(pending, base_iter, executor)
            while pending:
                inflight = len(pending)
                wait_started_at = time.perf_counter()
                prepared = pending.popleft().result()
                wait_s = time.perf_counter() - wait_started_at
                self._fill_pending(pending, base_iter, executor)
                prepared = dict(prepared)
                prepared["rosa_async_prefetch_wait_s"] = wait_s
                prepared["rosa_async_prefetch_depth"] = float(self.prefetch_batches)
                prepared["rosa_async_prefetch_inflight"] = float(inflight)
                prepared["rosa_async_prefetch_queue_fill"] = float(len(pending)) / float(self.prefetch_batches)
                yield prepared


def maybe_wrap_train_address_prefetch(
    loader: Iterable[Dict[str, Any]],
    *,
    address_engine: Any,
    enabled: bool,
    max_workers: int,
    prefetch_batches: int,
):
    if not enabled:
        return loader
    return RosaTrainAddressPrefetchLoader(
        loader,
        address_engine=address_engine,
        enabled=enabled,
        max_workers=max_workers,
        prefetch_batches=prefetch_batches,
    )
