import unittest

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


if __name__ == "__main__":
    unittest.main()
