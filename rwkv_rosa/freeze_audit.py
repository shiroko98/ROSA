"""Exact frozen-parameter integrity checks without retaining a second 3B model."""
from __future__ import annotations

import hashlib

import torch


def parameter_hash(module, *, chunk_elements=4 * 1024 * 1024):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            # uint8 viewing also supports BF16 NumPy conversion. Transfers are
            # bounded to one chunk, rather than duplicating the complete model.
            raw = flat[start:start + chunk_elements].contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
    return digest.hexdigest()


def check_frozen_backbone(model, expected_hash=None):
    if any(p.requires_grad or p.grad is not None for p in model.backbone.parameters()):
        raise AssertionError('Frozen RWKV backbone has a trainable parameter or a gradient.')
    current = parameter_hash(model.backbone)
    if expected_hash is not None and current != expected_hash:
        raise AssertionError('Frozen RWKV backbone parameter bytes changed.')
    return current
