from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from rosa_addressing import OnlineRosaBatchState, RosaStateSnapshot
from rosa_runtime import RosaInjectionPayload


@dataclass(frozen=True)
class RosaSessionSnapshot:
    batch_size: int
    total_prefill_tokens: int
    total_decode_tokens: int
    decode_calls: int
    state_snapshots: List[RosaStateSnapshot]
    prefetch_enabled: bool = False
    scheduled_requests: int = 0
    last_address_source: Optional[str] = None


class RosaBatchSession:
    """统一 prefill/decode 生命周期的在线 ROSA session。"""

    def __init__(
        self,
        model: Any,
        batch_size: int,
        *,
        use_prefetch: bool = False,
        use_pinned_prefetch: bool = False,
        prefetch_workers: int = 1,
    ):
        self.model = model
        self.batch_size = int(batch_size)
        self.state: OnlineRosaBatchState = model.init_online_state(self.batch_size)
        self.use_prefetch = bool(use_prefetch)
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.decode_calls = 0
        self.last_payload: Optional[RosaInjectionPayload] = None
        self.last_output: Optional[Dict[str, Any]] = None
        self.prefetcher = (
            model.init_prefetcher(
                use_async=True,
                use_pinned_memory=use_pinned_prefetch,
                max_workers=prefetch_workers,
            )
            if self.use_prefetch
            else None
        )
        self._prefetch_state: Optional[OnlineRosaBatchState] = self.state.clone() if self.use_prefetch else None
        self._scheduled: Dict[str, Any] = {}
        self._scheduled_counter = 0
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RosaBatchSession 已关闭，不能继续使用。")

    def _run_online_phase(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self._ensure_open()
        payload = self.model.prepare_rosa_injection_payload(
            input_ids,
            rosa_online_state=self.state,
        )
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            rosa_payload=payload,
        )
        self.last_payload = payload
        self.last_output = out
        return out

    def _count_non_pad_tokens(self, input_ids: torch.Tensor) -> int:
        if hasattr(self.model, "pad_id"):
            return int(input_ids.ne(int(self.model.pad_id)).sum().item())
        return int(input_ids.numel())

    def prefill_seq(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        out = self._run_online_phase(input_ids, labels=labels)
        self.total_prefill_tokens += self._count_non_pad_tokens(input_ids)
        if self._prefetch_state is not None:
            self._prefetch_state = self.state.clone()
        return out

    def schedule_decode_step(self, input_ids: torch.Tensor, *, request_key: Optional[str] = None) -> str:
        self._ensure_open()
        if self.prefetcher is None or self._prefetch_state is None:
            raise RuntimeError("当前 session 未启用 prefetch。")
        key = request_key or f"decode-{self._scheduled_counter}"
        self._scheduled_counter += 1
        address_batch = self.model.schedule_rosa_prefetch(
            self.prefetcher,
            key,
            input_ids,
            rosa_online_state=self._prefetch_state,
        )
        self._scheduled[key] = {
            "address_batch": address_batch,
            "next_state": self._prefetch_state.clone(),
        }
        self._prefetch_state = self._prefetch_state.clone()
        return key

    def decode_step(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        request_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.prefetcher is None:
            out = self._run_online_phase(input_ids, labels=labels)
        else:
            key = request_key or self.schedule_decode_step(input_ids)
            scheduled = self._scheduled.pop(key)
            payload = self.model.consume_rosa_prefetch(
                self.prefetcher,
                key,
                device=input_ids.device,
                fallback_address_batch=scheduled["address_batch"],
            )
            out = self.model(
                input_ids=input_ids,
                labels=labels,
                rosa_payload=payload,
            )
            self.last_payload = payload
            self.last_output = out
            self.state = scheduled["next_state"]
            self._prefetch_state = self.state.clone()
        self.total_decode_tokens += self._count_non_pad_tokens(input_ids)
        self.decode_calls += 1
        return out

    def reset(self) -> None:
        self.state = self.model.init_online_state(self.batch_size)
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.decode_calls = 0
        self.last_payload = None
        self.last_output = None
        self._scheduled.clear()
        self._scheduled_counter = 0
        if self.use_prefetch:
            self._prefetch_state = self.state.clone()
        self._closed = False

    def snapshot(self) -> RosaSessionSnapshot:
        return RosaSessionSnapshot(
            batch_size=self.batch_size,
            total_prefill_tokens=self.total_prefill_tokens,
            total_decode_tokens=self.total_decode_tokens,
            decode_calls=self.decode_calls,
            state_snapshots=self.state.snapshot(),
            prefetch_enabled=self.prefetcher is not None,
            scheduled_requests=len(self._scheduled),
            last_address_source=self.last_payload.address.source if self.last_payload is not None else None,
        )

    def prefetch_stats(self) -> Dict[str, float]:
        if self.prefetcher is None:
            return {}
        return self.prefetcher.stats()

    def close(self) -> None:
        if self.prefetcher is not None:
            self.prefetcher.shutdown()
        self._closed = True
