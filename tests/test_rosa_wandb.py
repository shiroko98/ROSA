import unittest

import torch

from rosa_optim import RosaOptimizerBundle
from rosa_train_monitor import collect_step_system_metrics
from rosa_wandb import WandbLogger, prefixed_wandb_metrics


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = False

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


if __name__ == "__main__":
    unittest.main()
