import unittest
from unittest import mock

import torch

from rosa_optim import RosaOptimizerBundle
from rosa_train_monitor import collect_step_system_metrics
from rosa_train_console import format_train_step_console_line
from rosa_wandb import WandbLogger, init_wandb_logger, prefixed_wandb_metrics


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = False
        self.url = "https://wandb.local/run/test"

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


class RosaWandbTests(unittest.TestCase):
    def test_prefixed_wandb_metrics_filters_nonscalars(self):
        payload = prefixed_wandb_metrics(
            {
                "loss": 1.25,
                "token_acc": 0.5,
                "bad_nan": float("nan"),
                "text": "skip",
                "nested": {"x": 1},
            },
            "train",
        )

        self.assertEqual(
            payload,
            {
                "train/loss": 1.25,
                "train/token_acc": 0.5,
            },
        )

    def test_wandb_logger_logs_step_metrics_and_summary(self):
        run = FakeRun()
        logger = WandbLogger(run)

        logger.log_metrics({"loss": 2.0, "token_acc": 0.25, "note": "skip"}, step=7, prefix="rosa_fused/train_step")
        logger.update_summary({"loss": 1.0, "ppl": 2.0}, prefix="rosa_fused/test")
        logger.finish()

        self.assertEqual(len(run.logged), 1)
        payload, step = run.logged[0]
        self.assertEqual(step, 7)
        self.assertEqual(payload["trainer/global_step"], 7.0)
        self.assertEqual(payload["rosa_fused/train_step/loss"], 2.0)
        self.assertEqual(payload["rosa_fused/train_step/token_acc"], 0.25)
        self.assertNotIn("note", payload)
        self.assertEqual(run.summary["rosa_fused/test/loss"], 1.0)
        self.assertEqual(run.summary["rosa_fused/test/ppl"], 2.0)
        self.assertTrue(run.finished)

    def test_init_wandb_logger_uses_explicit_wandb_dir(self):
        fake_run = FakeRun()

        class FakeWandbModule:
            def __init__(self):
                self.init_kwargs = None

            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return fake_run

            def define_metric(self, *args, **kwargs):
                return None

        fake_wandb = FakeWandbModule()
        printed = []
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            logger = init_wandb_logger(
                enabled=True,
                is_main_process=True,
                project="ROSA",
                mode="online",
                out_dir="/tmp/out",
                wandb_dir="/tmp/wandb",
                config={"a": 1},
                log_print=printed.append,
            )

        self.assertTrue(logger.enabled)
        self.assertEqual(fake_wandb.init_kwargs["dir"], "/tmp/wandb")
        self.assertTrue(any("正在初始化" in line for line in printed))
        self.assertTrue(any("运行已创建" in line for line in printed))

    def test_collect_step_system_metrics_cpu(self):
        metrics = collect_step_system_metrics(
            device=torch.device("cpu"),
            step_started_at=0.0,
            step_tokens=128,
            total_tokens_seen=1024,
            epoch=2,
            batch_idx=4,
            num_batches=8,
            log_cuda_memory=False,
        )

        self.assertEqual(metrics["epoch"], 2.0)
        self.assertEqual(metrics["epoch_progress"], 0.5)
        self.assertEqual(metrics["tokens_seen_total"], 1024.0)
        self.assertIn("step_wall_ms", metrics)
        self.assertIn("tokens_per_s_wall", metrics)
        self.assertNotIn("cuda_memory_allocated_mb", metrics)

    def test_optimizer_bundle_reports_lr_and_grad_norm(self):
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        sparse_param = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        dense_optim = torch.optim.AdamW([param], lr=1e-3)
        sparse_optim = torch.optim.SparseAdam([sparse_param], lr=2e-3)
        bundle = RosaOptimizerBundle(
            dense_optimizer=dense_optim,
            sparse_optimizer=sparse_optim,
            dense_params=[param],
            sparse_params=[sparse_param],
        )

        param.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
        sparse_grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1]], dtype=torch.long),
            values=torch.tensor([5.0, 12.0], dtype=torch.float32),
            size=(2,),
        )
        sparse_param.grad = sparse_grad

        self.assertEqual(
            bundle.current_lrs(),
            {
                "optimizer_lr_dense": 1e-3,
                "optimizer_lr_sparse": 2e-3,
            },
        )
        norms = bundle.grad_norms()
        self.assertAlmostEqual(norms["grad_norm_dense"], 5.0, places=5)
        self.assertAlmostEqual(norms["grad_norm_sparse"], 13.0, places=5)

    def test_format_train_step_console_line_includes_key_metrics(self):
        line = format_train_step_console_line(
            run_name="rosa_fused",
            epoch=1,
            global_step=10,
            grad_accum_steps=4,
            metrics={
                "loss": 1.25,
                "ppl": 3.5,
                "token_acc": 0.75,
                "valid_tokens": 1024,
                "tokens_per_s_wall": 2048.0,
                "step_wall_ms": 88.5,
                "optimizer_lr_dense": 3e-4,
                "grad_norm_dense": 1.2,
                "cuda_max_memory_allocated_mb": 4096.0,
                "rosa_fire_coverage": 0.5,
            },
        )

        self.assertIn("[rosa_fused]", line)
        self.assertIn("step 10", line)
        self.assertIn("tok/s 2048.0", line)
        self.assertIn("step_ms 88.5", line)
        self.assertIn("max_mem 4096MB", line)


if __name__ == "__main__":
    unittest.main()
