import math
import unittest

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod
from rosa_optim import build_training_optimizers


class RosaSparseValueTrainingTests(unittest.TestCase):
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

    def test_per_layer_sparse_value_store_uses_sparse_embeddings(self):
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            inject_layers=2,
            rosa_value_mode="per_layer",
            rosa_sparse_value_training=True,
        )

        self.assertTrue(model.rosa_value_store.uses_sparse_training)
        self.assertIsNotNone(model.rosa_value_store.per_layer_tables)
        for table in model.rosa_value_store.per_layer_tables:
            self.assertTrue(table.sparse)

    def test_build_training_optimizers_creates_sparse_optimizer_for_sparse_value_store(self):
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            inject_layers=1,
            rosa_value_mode="per_layer",
            rosa_sparse_value_training=True,
        )

        bundle = build_training_optimizers(model, lr=1e-3, weight_decay=0.01)

        self.assertIsNotNone(bundle.sparse_optimizer)
        self.assertEqual(bundle.sparse_optimizer.__class__.__name__, "SparseAdam")
        self.assertGreater(bundle.stats()["rosa_sparse_optimizer_params"], 0.0)

    def test_train_one_model_runs_with_sparse_value_training(self):
        docs = [
            [1, 2, 1, 2, 3, 4],
            [5, 6, 5, 6, 7, 8],
            [9, 10, 9, 10, 11, 12],
        ]
        train_ds, val_ds, test_ds, _ = rosa_mod.build_chunk_datasets(
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
        train_loader, val_loader, _ = rosa_mod.build_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=2,
            pad_id=0,
            train_seed=2026,
        )
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=1,
            inject_layers=1,
            rosa_value_mode="per_layer",
            rosa_sparse_value_training=True,
            rosa_seq_address_mode="online_sam",
            rosa_online_sam_impl="fast",
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
        )

        self.assertEqual(len(history["train"]), 1)
        self.assertEqual(history["train"][0]["rosa_sparse_value_optimizer"], 1.0)
        self.assertEqual(history["train"][0]["rosa_sparse_value_training"], 1.0)
        self.assertGreaterEqual(history["train"][0]["rosa_active_address_count"], 0.0)
        self.assertTrue(math.isfinite(history["train"][0]["loss"]))


if __name__ == "__main__":
    unittest.main()
