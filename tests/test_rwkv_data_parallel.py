"""Real two-process Lightning DP agrees with a global weighted-batch oracle."""
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from rwkv_rosa.native.adapter import CPUROSAOptimizer, clip_cpu_gradients
from rwkv_rosa.native.data_parallel import RosaDDPStrategy, SparseGradientSync, average_sparse_gradients


class Store(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = torch.nn.ModuleList([torch.nn.Embedding(8, 4, sparse=True) for _ in range(3)])

    def sparse_parameters(self):
        return [t.weight for t in self.tables]


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rosa_value_store = Store()
        self.dense = torch.nn.Linear(4, 1, bias=False)

    def loss(self, rank, batch):
        # Disjoint tables across ranks; duplicate rows and an always-unused table.
        rows = torch.tensor([rank, (batch + rank) % 4, rank])
        output = self.dense(self.rosa_value_store.tables[rank](rows)).flatten()
        return (output - (batch + 1) / 10).square().sum() * (rank + 1), 3 * (rank + 1)


class Normalize(pl.Callback):
    def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0):
        for parameter in model.rosa_cpu.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(2 * trainer.world_size / model.tokens)
        model.tokens = 0


class Toy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.rosa_cpu = Branch()
        self.tokens = 0
        self.data_offset = 0

    def on_train_epoch_start(self):
        # The native MyDataset applies this offset in train_callback's epoch hook.
        # Lightning resumes counters but does not skip the underlying iterator.
        self.data_offset = self.global_step * 2

    def training_step(self, batch, batch_idx):
        loss, count = self.rosa_cpu.loss(self.global_rank, int(batch[0].item()) + self.data_offset)
        total = torch.tensor(count, device=loss.device)
        dist.all_reduce(total)
        self.tokens += int(total)
        return loss

    def configure_optimizers(self):
        return CPUROSAOptimizer(self.rosa_cpu, lr=.01, weight_decay=.02)

    def configure_gradient_clipping(self, optimizer, optimizer_idx=0, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        clip_cpu_gradients(self.rosa_cpu.parameters(), .7)


class Save(pl.Callback):
    def __init__(self, folder):
        self.folder = Path(folder)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.global_step == 1 and batch_idx == 1:
            from rwkv_rosa.native.integration import NativeROSACallback
            callback = NativeROSACallback(SimpleNamespace(proj_dir=str(self.folder), rosa_dp=1))
            callback.save_full_checkpoint(trainer, 'test')


def worker(rank, folder, port):
    import os
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=1)
    def trainer(callbacks):
        return pl.Trainer(accelerator='cpu', devices=2,
            strategy=RosaDDPStrategy(process_group_backend='gloo', find_unused_parameters=True),
            replace_sampler_ddp=False, accumulate_grad_batches=2, max_steps=2, max_epochs=1,
            callbacks=callbacks, logger=False, enable_checkpointing=False,
            enable_model_summary=False, enable_progress_bar=False, gradient_clip_val=.7)
    torch.manual_seed(42)
    model = Toy()
    first = trainer([SparseGradientSync(), Normalize(), Save(folder)])
    first.fit(model, loader)
    continuous = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(987)
    resumed = Toy()
    second = trainer([SparseGradientSync(), Normalize()])
    second.fit(resumed, loader, ckpt_path=str(Path(folder) / 'native-last.ckpt'))
    assert second.global_step == 2
    for key, value in continuous.items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0, msg=key)
    torch.save(continuous, Path(folder) / f'rank{rank}.pth')
    # Missing local grads and explicitly present zero rows must be distinguished.
    parameter = torch.nn.Parameter(torch.ones(8, 4))
    if rank == 0:
        parameter.grad = torch.sparse_coo_tensor(torch.tensor([[3, 3]]), torch.zeros(2, 4), (8, 4))
    average_sparse_gradients([parameter], chunk_rows=1)
    assert parameter.grad.is_sparse
    assert parameter.grad.coalesce().indices().tolist() == [[3]]
    torch.testing.assert_close(parameter.grad.values(), torch.zeros(1, 4))
    dist.destroy_process_group()


def test_two_rank_weighted_accumulation_sparse_union_and_resume():
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(folder, port), nprocs=2, join=True)
        torch.manual_seed(42)
        oracle = Branch()
        optimizer = CPUROSAOptimizer(oracle, lr=.01, weight_decay=.02)
        for step in range(2):
            optimizer.zero_grad()
            for batch in (step * 2, step * 2 + 1):
                for rank in (0, 1):
                    loss, _ = oracle.loss(rank, batch)
                    (loss / 18).backward()
            clip_cpu_gradients(oracle.parameters(), .7)
            optimizer.step()
        for rank in (0, 1):
            actual = torch.load(Path(folder) / f'rank{rank}.pth', weights_only=True)
            for name, expected in oracle.state_dict().items():
                torch.testing.assert_close(actual['rosa_cpu.' + name], expected, atol=2e-7, rtol=2e-6)
