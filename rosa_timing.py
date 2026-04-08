from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch


def sync_timing_device(device: Optional[torch.device], *, enabled: bool) -> None:
    if not enabled or device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TimingCollector:
    def __init__(self, *, enabled: bool, device: Optional[torch.device] = None):
        self.enabled = bool(enabled)
        self.device = device
        self._stats_s: Dict[str, float] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        sync_timing_device(self.device, enabled=True)
        start = time.perf_counter()
        try:
            yield
        finally:
            sync_timing_device(self.device, enabled=True)
            self._stats_s[name] = self._stats_s.get(name, 0.0) + (time.perf_counter() - start)

    def add_seconds(self, name: str, value_s: float) -> None:
        if not self.enabled:
            return
        self._stats_s[name] = self._stats_s.get(name, 0.0) + max(0.0, float(value_s))

    def export_ms(self, *, prefix: str = "timing_") -> Dict[str, float]:
        if not self.enabled:
            return {}
        return {
            f"{prefix}{name}_ms": value_s * 1000.0
            for name, value_s in self._stats_s.items()
        }
