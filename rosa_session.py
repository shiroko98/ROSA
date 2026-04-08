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
    last_address_source: Optional[str] = None


class RosaBatchSession:
    """统一 prefill/decode 生命周期的在线 ROSA session。"""

    def __init__(self, model: Any, batch_size: int):
        self.model = model
        self.batch_size = int(batch_size)
        self.state: OnlineRosaBatchState = model.init_online_state(self.batch_size)
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.decode_calls = 0
        self.last_payload: Optional[RosaInjectionPayload] = None
        self.last_output: Optional[Dict[str, Any]] = None
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
        return out

    def decode_step(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        out = self._run_online_phase(input_ids, labels=labels)
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
        self._closed = False

    def snapshot(self) -> RosaSessionSnapshot:
        return RosaSessionSnapshot(
            batch_size=self.batch_size,
            total_prefill_tokens=self.total_prefill_tokens,
            total_decode_tokens=self.total_decode_tokens,
            decode_calls=self.decode_calls,
            state_snapshots=self.state.snapshot(),
            last_address_source=self.last_payload.address.source if self.last_payload is not None else None,
        )

    def close(self) -> None:
        self._closed = True
