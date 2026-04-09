import tempfile
import unittest
from pathlib import Path

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod
from rosa_checkpointing import TrainingCheckpointManager, load_training_checkpoint
from rosa_distributed import build_distributed_sampler, reduce_mean_metric_dict, reduce_scalar_sums, wrap_model_for_distributed


class RosaScaleTrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = rosa_mod.ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            dim=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            intermediate_size=32,
        )
        docs = [
            [1, 2, 1, 2, 3, 4],
            [5, 6, 5, 6, 7, 8],
            [9, 10, 9, 10, 11, 12],
        ]
        self.train_ds, self.val_ds, self.test_ds, _ = rosa_mod.build_chunk_datasets(
            docs,
            docs[:1],
            docs[1:2],
            seq_len=3,
            pad_id=0,
            stride=3,
            rosa_memory_tokens=1,
            rosa_memory_mode="doc_local",
            rosa_global_memory_tokens=0,
            rosa_backend="sam",
            rosa_train_mode="online_seq",
            rosa_seq_address_mode="online_sam",
            rosa_online_sam_impl="fast",
            rosa_min_match_len=1,
            special_ids=set(),
            forbid_special_target=True,
        )

    def _build_loaders(self):
        return rosa_mod.build_dataloaders(
            self.train_ds,
            self.val_ds,
            self.test_ds,
            batch_size=2,
            pad_id=0,
            train_seed=2026,
        )

    def test_train_one_model_supports_grad_accum_and_checkpoint_save(self):
        train_loader, val_loader, _ = self._build_loaders()
        rosa_mod.set_seed(1234)
        model = rosa_mod.BaseLM(self.cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrainingCheckpointManager(
                out_dir=tmpdir,
                model_name="baseline",
                save_every_epochs=1,
                is_main_process=True,
            )
            history = rosa_mod.train_one_model(
                model,
                train_loader,
                val_loader,
                device=torch.device("cpu"),
                pad_id=0,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=1.0,
                use_bf16=False,
                grad_accum_steps=2,
                checkpoint_manager=manager,
                run_name="baseline",
            )

            self.assertEqual(len(history["train"]), 1)
            self.assertTrue((Path(tmpdir) / "checkpoints" / "baseline" / "epoch_001.pt").exists())
            self.assertTrue((Path(tmpdir) / "checkpoints" / "baseline" / "last.pt").exists())

    def test_train_one_model_can_resume_from_checkpoint(self):
        train_loader, val_loader, _ = self._build_loaders()
        with tempfile.TemporaryDirectory() as tmpdir:
            rosa_mod.set_seed(5678)
            model = rosa_mod.BaseLM(self.cfg)
            manager = TrainingCheckpointManager(
                out_dir=tmpdir,
                model_name="baseline",
                save_every_epochs=1,
                is_main_process=True,
            )
            rosa_mod.train_one_model(
                model,
                train_loader,
                val_loader,
                device=torch.device("cpu"),
                pad_id=0,
                epochs=1,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=1.0,
                use_bf16=False,
                grad_accum_steps=2,
                checkpoint_manager=manager,
                run_name="baseline",
            )

            checkpoint = load_training_checkpoint(str(Path(tmpdir) / "checkpoints" / "baseline" / "last.pt"))
            resumed_model = rosa_mod.BaseLM(self.cfg)
            resumed_history = rosa_mod.train_one_model(
                resumed_model,
                train_loader,
                val_loader,
                device=torch.device("cpu"),
                pad_id=0,
                epochs=2,
                lr=1e-3,
                weight_decay=0.01,
                grad_clip=1.0,
                use_bf16=False,
                grad_accum_steps=2,
                resume_state=checkpoint,
                resume_checkpoint=checkpoint,
                run_name="baseline",
            )

            self.assertEqual(len(resumed_history["train"]), 2)
            self.assertEqual(resumed_history["train"][0]["epoch"], 1)
            self.assertEqual(resumed_history["train"][1]["epoch"], 2)

    def test_distributed_helpers_safely_fallback_without_context(self):
        train_sampler = build_distributed_sampler(self.train_ds, ctx=None, shuffle=True, seed=42)
        self.assertIsNone(train_sampler)
        metrics = reduce_mean_metric_dict({"a": 1.0, "b": 2.0}, ctx=None, device=torch.device("cpu"))
        self.assertEqual(metrics, {"a": 1.0, "b": 2.0})
        totals = reduce_scalar_sums([1.0, 2.0, 3.0], ctx=None, device=torch.device("cpu"))
        self.assertEqual(totals, (1.0, 2.0, 3.0))

        class DummyCtx:
            enabled = False
            world_size = 1
            local_rank = 0
            device = torch.device("cpu")
            strategy = "none"

        model = rosa_mod.BaseLM(self.cfg)
        wrapped = wrap_model_for_distributed(
            model,
            ctx=DummyCtx(),
            block_cls=rosa_mod.DecoderBlock,
            use_bf16=False,
        )
        self.assertIs(wrapped, model)
