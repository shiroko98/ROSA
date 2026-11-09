"""Label-free, bounded post-norm gain from frozen base logits, in token chunks."""
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def logit_gap_scales(hidden, direction, candidates, valid, head, *, maximum=.2, margin=.5, chunk_size=128):
    """Return detached gain BEFORE the learned retrieval gate is applied.

    Only the original strongest noncandidate is used: this is not a top-1/copy
    guarantee. No labels enter the controller. Nonpositive directions are muted.
    At most chunk_size x vocabulary FP32 logits are live; never T x vocabulary.
    """
    if head.weight.requires_grad or (head.bias is not None and head.bias.requires_grad):
        raise ValueError('Dynamic ROSA scale requires a frozen output head.')
    if hidden.shape != direction.shape or hidden.shape[:-1] != candidates.shape or valid.shape != candidates.shape:
        raise ValueError('Dynamic ROSA hidden/direction/address shapes differ.')
    device = head.weight.device
    z = hidden.detach().reshape(-1, hidden.shape[-1])
    u = direction.detach().reshape_as(z)
    c = candidates.reshape(-1)
    mask = valid.reshape(-1)
    gain = torch.zeros(c.shape, dtype=torch.float32, device=direction.device)
    useful = torch.zeros(c.shape, dtype=torch.bool, device=direction.device)
    # A temporary FP32 weight view/copy; the stored base bytes are never changed.
    with torch.autocast(device_type=device.type, enabled=False):
        weight = head.weight.detach().float()
        bias = None if head.bias is None else head.bias.detach().float()
        for start in range(0, len(c), chunk_size):
            end = min(len(c), start + chunk_size)
            active = mask[start:end].to(device=device)
            if not bool(active.any()):
                continue
            candidate = c[start:end].to(device=device)
            # Invalid addresses must not cause an out-of-range head-row lookup.
            candidate = torch.where(active, candidate, torch.zeros_like(candidate))
            if bool(((candidate < 0) | (candidate >= weight.shape[0])).any()):
                raise ValueError('Valid ROSA candidate is outside the output vocabulary.')
            logits = F.linear(z[start:end].to(device=device, dtype=torch.float32), weight, bias)
            top = logits.topk(2, dim=-1)
            competitor = torch.where(top.indices[:, 0] == candidate, top.indices[:, 1], top.indices[:, 0])
            rival_score = logits.gather(1, competitor[:, None]).squeeze(1)
            candidate_score = logits.gather(1, candidate[:, None]).squeeze(1)
            gap = rival_score - candidate_score + margin
            d = u[start:end].to(device=device, dtype=torch.float32)
            improvement = ((weight[candidate] - weight[competitor]) * d).sum(-1)
            good = active & torch.isfinite(gap) & torch.isfinite(improvement) & (improvement > 1e-6)
            needed = gap.clamp_min(0) / improvement.clamp_min(1e-6)
            bounded = torch.where(good, needed.clamp(max=maximum), torch.zeros_like(needed))
            gain[start:end] = bounded.to(gain.device)
            useful[start:end] = good.to(useful.device)
            # Do not retain the previous vocabulary chunk while building the next.
            del logits, top
    return gain.reshape(candidates.shape), useful.reshape(candidates.shape)
