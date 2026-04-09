from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, Iterator, Optional

import torch


def prepare_online_seq_batch(
    batch: Dict[str, Any],
    *,
    address_engine: Any,
) -> Dict[str, Any]:
    if batch.get("rosa_precomputed_ids") is not None:
        return batch

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
    return prepared


class RosaTrainAddressPrefetchLoader:
    def __init__(
        self,
        base_loader: Iterable[Dict[str, Any]],
        *,
        address_engine: Any,
        enabled: bool = True,
        max_workers: int = 1,
    ):
        self.base_loader = base_loader
        self.address_engine = address_engine
        self.enabled = enabled
        self.max_workers = max(1, int(max_workers))

    def __len__(self) -> int:
        return len(self.base_loader)  # type: ignore[arg-type]

    def _submit(self, executor: ThreadPoolExecutor, batch: Dict[str, Any]) -> Future:
        return executor.submit(
            prepare_online_seq_batch,
            batch,
            address_engine=self.address_engine,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        base_iter = iter(self.base_loader)
        if not self.enabled:
            yield from base_iter
            return

        try:
            first_batch = next(base_iter)
        except StopIteration:
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            current_future = self._submit(executor, first_batch)
            while True:
                current_prepared = current_future.result()
                try:
                    next_batch = next(base_iter)
                except StopIteration:
                    yield current_prepared
                    break
                next_future = self._submit(executor, next_batch)
                yield current_prepared
                current_future = next_future


def maybe_wrap_train_address_prefetch(
    loader: Iterable[Dict[str, Any]],
    *,
    address_engine: Any,
    enabled: bool,
    max_workers: int,
):
    if not enabled:
        return loader
    return RosaTrainAddressPrefetchLoader(
        loader,
        address_engine=address_engine,
        enabled=enabled,
        max_workers=max_workers,
    )
