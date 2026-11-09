"""Explicit precision boundary for the upstream differentiable WKV recurrence."""
from contextlib import contextmanager
from functools import wraps

import torch

WKV_PRECISION_POLICY = "fp32_autocast_disabled"


def fence_wkv_fp32(operation):
    """A float32 state is insufficient: autocast also downcasts float32 matmul.

    Keep Q/K/V projections in the outer precision policy, but perform every WKV
    recurrence matmul in FP32. The upstream operation explicitly casts its inputs
    and states to float32 and casts only its output back to the runtime dtype.
    """
    @wraps(operation)
    def forward(r, *args, **kwargs):
        with torch.autocast(device_type=r.device.type, enabled=False):
            return operation(r, *args, **kwargs)
    return forward


@contextmanager
def wkv_policy(runtime, *, legacy=False):
    """Legacy policy is an audit-only counterfactual, never a training default."""
    previous = runtime.RWKV7_OP
    try:
        runtime.RWKV7_OP = runtime._unfenced_wkv_op if legacy else runtime._fp32_wkv_op
        yield
    finally:
        runtime.RWKV7_OP = previous
