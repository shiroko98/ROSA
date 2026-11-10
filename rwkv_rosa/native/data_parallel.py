"""Replicated DDP with exact sparse-row averaging, without optimizer sharding."""
from datetime import timedelta

import torch
from torch import distributed as dist
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Callback


@torch.no_grad()
def average_sparse_gradients(parameters, chunk_rows=1024):
    """Average the union of touched rows, including explicitly touched zero rows.

    NCCL does not reduce sparse tensors. Reduce a row-presence bitmap first,
    then bounded dense chunks of the union. Missing local gradients contribute
    zeros; globally untouched parameters keep grad=None (SparseAdam semantics).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    world = dist.get_world_size()
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None and (not grad.is_sparse or grad.sparse_dim() != 1):
            raise ValueError('Expected row-sparse embedding gradient.')
        grad = None if grad is None else grad.coalesce()
        present = torch.zeros(parameter.shape[0], device=parameter.device, dtype=torch.int32)
        if grad is not None:
            present[grad.indices()[0]] = 1
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        rows = present.nonzero().flatten()
        if rows.numel() == 0:
            parameter.grad = None
            continue
        values = torch.empty((len(rows), *parameter.shape[1:]), device=parameter.device, dtype=parameter.dtype)
        local_rows = rows[:0] if grad is None else grad.indices()[0]
        for start in range(0, len(rows), chunk_rows):
            selected = rows[start:start + chunk_rows]
            block = values[start:start + len(selected)]
            block.zero_()
            lo = int(torch.searchsorted(local_rows, selected[0]))
            hi = int(torch.searchsorted(local_rows, selected[-1], right=True))
            if hi > lo:
                offsets = torch.searchsorted(selected, local_rows[lo:hi])
                block[offsets] = grad.values()[lo:hi]
            dist.all_reduce(block)
            block.div_(world)
        parameter.grad = torch.sparse_coo_tensor(rows.unsqueeze(0), values, parameter.shape,
                                                device=parameter.device, is_coalesced=True)


class RosaDDPStrategy(DDPStrategy):
    def _setup_model(self, model):
        sparse_names = [name + '.weight' for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Embedding) and module.sparse]
        # This model is Lightning's wrapper: names include its `module.` prefix.
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, sparse_names)
        parameters = dict(model.named_parameters())
        for name in sparse_names:
            dist.broadcast(parameters[name].data, src=0)
        return super()._setup_model(model)


class SparseGradientSync(Callback):
    def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0):
        average_sparse_gradients(model.rosa_cpu.rosa_value_store.sparse_parameters())


def trainer_strategy(args):
    if not getattr(args, 'rosa_dp', 0):
        return args.strategy
    return RosaDDPStrategy(find_unused_parameters=True, broadcast_buffers=False,
                           timeout=timedelta(seconds=args.dist_timeout_sec))
