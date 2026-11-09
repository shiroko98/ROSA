"""Test distributed routing at the transport boundary; no GPU or multi-host claims."""
import contextlib
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import rosa_distributed as distributed
import rosa_wandb as monitoring


class FakeFSDP(torch.nn.Module):
    def __init__(self, module, **options):
        super().__init__()
        self.module = module
        self.options = options

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    @staticmethod
    def state_dict_type(*args):
        return contextlib.nullcontext()

    @staticmethod
    def optim_state_dict(model, optimizer):
        return optimizer.state_dict()

    @staticmethod
    def optim_state_dict_to_load(model, optimizer, state):
        return state


class DistributedBoundaryTests(unittest.TestCase):
    def context(self, strategy="ddp", device="cpu"):
        return distributed.DistributedContext(strategy, "gloo", True, 0, 0, 2, torch.device(device))

    def test_context_lifecycle_and_multi_rank_sampler(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}), mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(distributed.dist, "is_initialized", return_value=False), mock.patch.object(distributed.dist, "init_process_group") as init:
            with self.assertRaises(ValueError):
                distributed.init_distributed_context(strategy="none")
            ctx = distributed.init_distributed_context(strategy="ddp")
            init.assert_called_once_with(backend="gloo")
        self.assertTrue(ctx.is_main_process)
        sampler = distributed.build_distributed_sampler(list(range(8)), ctx=ctx, shuffle=False, seed=42)
        self.assertEqual(list(sampler), [0, 2, 4, 6])
        with mock.patch.object(distributed.dist, "is_initialized", return_value=True), mock.patch.object(distributed.dist, "barrier") as barrier, mock.patch.object(distributed.dist, "destroy_process_group") as destroy:
            distributed.maybe_barrier(ctx)
            distributed.destroy_distributed_context(ctx)
            self.assertEqual(barrier.call_count, 2)
            destroy.assert_called_once()

    def test_ddp_and_fsdp_configuration_and_state_roundtrip(self):
        model = torch.nn.Linear(2, 2)
        cpu = self.context()
        with mock.patch.object(distributed, "DistributedDataParallel") as ddp:
            self.assertIs(distributed.wrap_model_for_distributed(model, ctx=cpu, block_cls=torch.nn.Linear, use_bf16=False), ddp.return_value)
            ddp.assert_called_once_with(model)
        fake_cuda_model = mock.Mock()
        cuda = self.context(device="cuda:0")
        with mock.patch.object(distributed, "DistributedDataParallel") as ddp:
            distributed.wrap_model_for_distributed(fake_cuda_model, ctx=cuda, block_cls=torch.nn.Linear, use_bf16=False)
            ddp.assert_called_once_with(fake_cuda_model, device_ids=[0], output_device=0)
        with self.assertRaises(ValueError):
            distributed.wrap_model_for_distributed(model, ctx=self.context("invalid"), block_cls=torch.nn.Linear, use_bf16=False)
        with mock.patch.object(distributed, "FSDP", None), self.assertRaises(RuntimeError):
            distributed.wrap_model_for_distributed(model, ctx=self.context("fsdp"), block_cls=torch.nn.Linear, use_bf16=False)
        with mock.patch.object(distributed, "FSDP", FakeFSDP):
            wrapped = distributed.wrap_model_for_distributed(model, ctx=self.context("fsdp"), block_cls=torch.nn.Linear, use_bf16=False)
            self.assertTrue(distributed.is_fsdp_model(wrapped))
            state = distributed.extract_model_state_dict(wrapped)
            distributed.load_model_state_dict(wrapped, state)
            self.assertTrue(torch.equal(state["weight"], model.weight))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            sparse = mock.Mock()
            sparse.state_dict.return_value = {"sparse": True}
            bundle = SimpleNamespace(dense_optimizer=optimizer, sparse_optimizer=sparse)
            optim_state = distributed.extract_optimizer_state(wrapped, bundle)
            distributed.load_optimizer_state(wrapped, bundle, optim_state)
            sparse.load_state_dict.assert_called_once_with({"sparse": True})
            fake = distributed.wrap_model_for_distributed(fake_cuda_model, ctx=self.context("fsdp", "cuda:0"), block_cls=torch.nn.Linear, use_bf16=True)
            self.assertIsNotNone(fake.options["mixed_precision"])
            self.assertEqual(fake.options["device_id"], 0)

    def test_collective_reductions_preserve_metric_keys(self):
        ctx = self.context()
        with mock.patch.object(distributed.dist, "all_reduce", side_effect=lambda tensor, **kwargs: tensor.mul_(2)):
            self.assertEqual(distributed.reduce_scalar_sums([1, 3], ctx=ctx, device=ctx.device), (2., 6.))
            self.assertEqual(distributed.reduce_mean_metric_dict({"z": 3, "a": 1}, ctx=ctx, device=ctx.device), {"a": 1., "z": 3.})
            self.assertEqual(distributed.reduce_sum_metric_dict({"z": 3, "a": 1}, ctx=ctx, device=ctx.device), {"a": 2., "z": 6.})


class MonitoringFallbackTests(unittest.TestCase):
    def test_disabled_logger_and_raw_payloads(self):
        disabled = monitoring.WandbLogger()
        disabled.log_metrics({"loss": 1})
        disabled.log_raw({"text": "example"})
        disabled.update_summary({"loss": 1})
        self.assertFalse(disabled.enabled)
        run = mock.Mock()
        logger = monitoring.WandbLogger(run)
        logger.log_metrics({"invalid": float("nan")})
        run.log.assert_not_called()
        logger.log_raw({"value": 3}, step=2)
        run.log.assert_called_once_with({"value": 3, "trainer/global_step": 2.}, step=2)

    def test_missing_wandb_and_optional_metadata_failures(self):
        kwargs = dict(enabled=True, is_main_process=True, project="test", mode="offline", out_dir="unused", config={}, log_print=lambda value: None)
        with mock.patch.dict(sys.modules, {"wandb": None}):
            self.assertFalse(monitoring.init_wandb_logger(**kwargs).enabled)
        class Run:
            @property
            def url(self):
                raise RuntimeError("url unavailable")
        wandb = mock.Mock()
        wandb.init.return_value = Run()
        wandb.define_metric.side_effect = RuntimeError("old client")
        with mock.patch.dict(sys.modules, {"wandb": wandb}):
            self.assertTrue(monitoring.init_wandb_logger(**kwargs).enabled)


if __name__ == "__main__":
    unittest.main()
