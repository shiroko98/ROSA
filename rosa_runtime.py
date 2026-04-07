from dataclasses import dataclass
from typing import Tuple

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
