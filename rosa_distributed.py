import functools
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:  # pragma: no cover - import guard
    FSDP = None
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None
    transformer_auto_wrap_policy = None


@dataclass
class DistributedContext:
    strategy: str
    backend: str
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def init_distributed_context(*, strategy: str, backend: Optional[str] = None) -> DistributedContext:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = strategy != "none"
    if world_size > 1 and strategy == "none":
        raise ValueError("检测到 torchrun/distributed 环境，但 --distributed_strategy=none。请改用 ddp 或 fsdp。")
    if backend is None:
        if torch.cuda.is_available() and hasattr(dist, "is_nccl_available") and dist.is_nccl_available() and os.name != "nt":
            backend = "nccl"
        else:
            backend = "gloo"
    if enabled and not dist.is_initialized():
        if os.name == "nt" and "USE_LIBUV" not in os.environ:
            os.environ["USE_LIBUV"] = "0"
        dist.init_process_group(backend=backend)
    backend = backend or "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(
        strategy=strategy,
        backend=backend,
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def destroy_distributed_context(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def maybe_barrier(ctx: Optional[DistributedContext]) -> None:
    if ctx is not None and ctx.enabled and dist.is_initialized():
        dist.barrier()


def build_distributed_sampler(
    dataset: Dataset,
    *,
    ctx: Optional[DistributedContext],
    shuffle: bool,
    seed: int,
) -> Optional[DistributedSampler]:
    if ctx is None or not ctx.enabled or ctx.world_size <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def set_sampler_epoch_if_needed(sampler: Any, epoch: int) -> None:
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def wrap_model_for_distributed(
    model: torch.nn.Module,
    *,
    ctx: DistributedContext,
    block_cls: Type[torch.nn.Module],
    use_bf16: bool,
    sync_module_states: bool = True,
) -> torch.nn.Module:
    model.to(ctx.device)
    if not ctx.enabled or ctx.world_size <= 1:
        return model
    if ctx.strategy == "ddp":
        if ctx.device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
        return DistributedDataParallel(model)
    if ctx.strategy != "fsdp":
        raise ValueError(f"未知 distributed strategy: {ctx.strategy}")
    if FSDP is None or transformer_auto_wrap_policy is None:
        raise RuntimeError("当前环境未提供 torch.distributed.fsdp，无法启用 FSDP。")
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_cls},
    )
    mixed_precision = None
    if use_bf16 and ctx.device.type == "cuda" and MixedPrecision is not None:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=ctx.local_rank if ctx.device.type == "cuda" else None,
        sync_module_states=sync_module_states and ctx.world_size > 1,
        use_orig_params=True,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def is_fsdp_model(model: torch.nn.Module) -> bool:
    return FSDP is not None and isinstance(model, FSDP)


def extract_model_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    if not is_fsdp_model(model):
        return unwrap_model(model).state_dict()
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        return model.state_dict()


def load_model_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    if not is_fsdp_model(model):
        unwrap_model(model).load_state_dict(state_dict)
        return
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model.load_state_dict(state_dict)


def extract_optimizer_state(model: torch.nn.Module, optimizer_bundle: Any) -> Dict[str, Any]:
    dense_state = None
    sparse_state = None
    if optimizer_bundle.dense_optimizer is not None:
        if is_fsdp_model(model):
            optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg, optim_cfg):
                dense_state = FSDP.optim_state_dict(model, optimizer_bundle.dense_optimizer)
        else:
            dense_state = optimizer_bundle.dense_optimizer.state_dict()
    if optimizer_bundle.sparse_optimizer is not None:
        sparse_state = optimizer_bundle.sparse_optimizer.state_dict()
    return {
        "dense_optimizer": dense_state,
        "sparse_optimizer": sparse_state,
    }


def load_optimizer_state(model: torch.nn.Module, optimizer_bundle: Any, optimizer_state: Dict[str, Any]) -> None:
    dense_state = optimizer_state.get("dense_optimizer")
    sparse_state = optimizer_state.get("sparse_optimizer")
    if dense_state is not None and optimizer_bundle.dense_optimizer is not None:
        if is_fsdp_model(model):
            dense_state = FSDP.optim_state_dict_to_load(model, optimizer_bundle.dense_optimizer, dense_state)
        optimizer_bundle.dense_optimizer.load_state_dict(dense_state)
    if sparse_state is not None and optimizer_bundle.sparse_optimizer is not None:
        optimizer_bundle.sparse_optimizer.load_state_dict(sparse_state)


def reduce_scalar_sums(
    values: Sequence[float],
    *,
    ctx: Optional[DistributedContext],
    device: torch.device,
) -> Tuple[float, ...]:
    if ctx is None or not ctx.enabled or ctx.world_size <= 1:
        return tuple(float(v) for v in values)
    tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tuple(float(v) for v in tensor.cpu().tolist())


def reduce_mean_metric_dict(
    metrics: Dict[str, float],
    *,
    ctx: Optional[DistributedContext],
    device: torch.device,
) -> Dict[str, float]:
    if ctx is None or not ctx.enabled or ctx.world_size <= 1 or not metrics:
        return dict(metrics)
    keys = sorted(metrics.keys())
    tensor = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(ctx.world_size)
    values = tensor.cpu().tolist()
    return {k: float(v) for k, v in zip(keys, values)}


def reduce_sum_metric_dict(
    metrics: Dict[str, float],
    *,
    ctx: Optional[DistributedContext],
    device: torch.device,
) -> Dict[str, float]:
    if ctx is None or not ctx.enabled or ctx.world_size <= 1 or not metrics:
        return dict(metrics)
    keys = sorted(metrics.keys())
    tensor = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    values = tensor.cpu().tolist()
    return {k: float(v) for k, v in zip(keys, values)}
