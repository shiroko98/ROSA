import json
import tempfile
import unittest
from pathlib import Path

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod


class JsonlLoadingTests(unittest.TestCase):
    def test_load_docs_from_jsonl_glob_and_nested_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "part_b.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"content": "doc-b1"}, ensure_ascii=False),
                        json.dumps({"meta": {"body": "doc-b2"}}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "part_a.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"text": "doc-a1"}, ensure_ascii=False),
                        json.dumps({"ignored": 123}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            docs, meta = rosa_mod.load_docs_from_path(
                str(root / "*.jsonl"),
                data_format="auto",
                split_mode="paragraph",
                json_text_keys=["text", "content", "meta.body"],
            )

            self.assertEqual(docs, ["doc-a1", "doc-b1", "doc-b2"])
            self.assertEqual(meta["file_count"], 2)
            self.assertEqual(meta["skipped_records"], 1)
            self.assertEqual(meta["format_counts"], {"jsonl": 2})


class RosaRetrievalTests(unittest.TestCase):
    def test_sam_and_naive_retrieval_match_on_same_memory(self):
        input_ids = torch.tensor(
            [
                [1, 2, 1, 2, 3],
                [7, 8, 7, 8, 9],
            ],
            dtype=torch.long,
        )
        memory_ids = torch.tensor(
            [
                [4, 5, 1, 2, 0, 0],
                [6, 7, 8, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        naive = rosa_mod.rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=2,
            pad_id=0,
            special_ids={9},
            forbid_special_target=True,
            backend="naive",
        )
        sam = rosa_mod.rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=2,
            pad_id=0,
            special_ids={9},
            forbid_special_target=True,
            backend="sam",
        )

        for naive_tensor, sam_tensor in zip(naive, sam):
            self.assertTrue(torch.equal(naive_tensor, sam_tensor))

    def test_rosa_retrieval_reads_from_left_memory(self):
        input_ids = torch.tensor([[1, 2, 4]], dtype=torch.long)
        memory_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)

        retrieved, fired_match_lens, raw_best_lens = rosa_mod.rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=2,
            pad_id=99,
            special_ids=set(),
            forbid_special_target=True,
        )

        self.assertEqual(retrieved.tolist(), [[-1, 3, -1]])
        self.assertEqual(fired_match_lens.tolist(), [[0, 2, 0]])
        self.assertEqual(raw_best_lens.tolist(), [[1, 2, 0]])

    def test_rosa_special_target_filter_only_blocks_injection(self):
        input_ids = torch.tensor([[7]], dtype=torch.long)
        memory_ids = torch.tensor([[7, 5]], dtype=torch.long)

        retrieved, fired_match_lens, raw_best_lens = rosa_mod.rosa_retrieval_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=1,
            pad_id=99,
            special_ids={5},
            forbid_special_target=True,
        )

        self.assertEqual(retrieved.tolist(), [[-1]])
        self.assertEqual(fired_match_lens.tolist(), [[0]])
        self.assertEqual(raw_best_lens.tolist(), [[1]])

    def test_sam_predict_returns_expected_match_length_and_target(self):
        preds, match_lens = rosa_mod.sam_rosa_predict([1, 2, 1, 2, 3], min_match_len=2)

        self.assertEqual(preds, [-1, -1, -1, 1, -1])
        self.assertEqual(match_lens, [0, 0, 1, 2, 0])


class FairComparisonTests(unittest.TestCase):
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

    def test_baseline_and_rosa_share_same_initial_parameters(self):
        rosa_mod.set_seed(1234)
        baseline = rosa_mod.BaseLM(self.cfg)
        rosa_mod.set_seed(1234)
        rosa_model = rosa_mod.RosaFusedLM(self.cfg, pad_id=0)

        self.assertEqual(rosa_mod.count_params(baseline), rosa_mod.count_params(rosa_model))
        self.assertEqual(list(baseline.state_dict().keys()), list(rosa_model.state_dict().keys()))
        for key, base_tensor in baseline.state_dict().items():
            self.assertTrue(
                torch.equal(base_tensor, rosa_model.state_dict()[key]),
                msg=f"parameter mismatch at {key}",
            )

    def test_train_loader_order_is_repeatable_for_same_seed(self):
        docs_tokens = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]
        dataset = rosa_mod.DocChunkDataset(
            docs_tokens,
            seq_len=2,
            pad_id=0,
            stride=1,
            rosa_memory_tokens=3,
        )

        loader_a, _, _ = rosa_mod.build_dataloaders(
            dataset,
            dataset,
            dataset,
            batch_size=2,
            pad_id=0,
            train_seed=2026,
        )
        loader_b, _, _ = rosa_mod.build_dataloaders(
            dataset,
            dataset,
            dataset,
            batch_size=2,
            pad_id=0,
            train_seed=2026,
        )

        def collect_batches(loader):
            out = []
            for batch in loader:
                out.append(
                    (
                        batch["input_ids"].tolist(),
                        batch["labels"].tolist(),
                        batch["rosa_memory_ids"].tolist(),
                    )
                )
            return out

        self.assertEqual(collect_batches(loader_a), collect_batches(loader_b))


class RosaValueStoreTests(unittest.TestCase):
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

    def test_per_layer_value_store_copies_shared_embeddings_on_init(self):
        rosa_mod.set_seed(2026)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            inject_layers=2,
            rosa_value_mode="per_layer",
        )

        self.assertIsNotNone(model.rosa_value_store.per_layer_tables)
        for table in model.rosa_value_store.per_layer_tables:
            self.assertTrue(torch.equal(table.weight, model.embed_tokens.weight))

    def test_per_layer_value_store_matches_shared_path_at_init(self):
        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        memory_ids = torch.empty((1, 0), dtype=torch.long)

        rosa_mod.set_seed(77)
        shared_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=2,
            rosa_scale=0.5,
            rosa_value_mode="shared",
        )
        rosa_mod.set_seed(77)
        per_layer_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=2,
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
        )

        with torch.no_grad():
            shared_out = shared_model(input_ids=input_ids, rosa_memory_ids=memory_ids)
            per_layer_out = per_layer_model(input_ids=input_ids, rosa_memory_ids=memory_ids)

        self.assertTrue(torch.allclose(shared_out["logits"], per_layer_out["logits"], atol=1e-6))
        self.assertEqual(shared_out["rosa_value_per_layer"], 0.0)
        self.assertEqual(per_layer_out["rosa_value_per_layer"], 1.0)

    def test_per_layer_value_store_supports_layer_specific_lookup(self):
        rosa_mod.set_seed(88)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            inject_layers=2,
            rosa_value_mode="per_layer",
        )
        token_ids = torch.tensor([[5]], dtype=torch.long)

        with torch.no_grad():
            model.rosa_value_store.per_layer_tables[1].weight[5].fill_(3.14)

        layer0_value = model.rosa_value_store.lookup(0, token_ids, shared_embedding=model.embed_tokens)
        layer1_value = model.rosa_value_store.lookup(1, token_ids, shared_embedding=model.embed_tokens)

        self.assertFalse(torch.equal(layer0_value, layer1_value))


class RosaContextGateTests(unittest.TestCase):
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

    def test_context_gate_reports_gate_stats(self):
        rosa_mod.set_seed(3030)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=2,
            rosa_scale=0.5,
            use_context_gate=True,
        )

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long),
                rosa_memory_ids=torch.empty((1, 0), dtype=torch.long),
            )

        self.assertIn("rosa_avg_gate", out)
        self.assertIn("rosa_gate_coverage", out)
        self.assertIn("rosa_gate_hit", out)
        self.assertGreaterEqual(out["rosa_avg_gate"], 0.0)
        self.assertLessEqual(out["rosa_avg_gate"], 1.0)
        self.assertGreaterEqual(out["rosa_gate_coverage"], 0.0)
        self.assertLessEqual(out["rosa_gate_coverage"], 1.0)
        self.assertGreaterEqual(out["rosa_gate_hit"], 0.0)
        self.assertLessEqual(out["rosa_gate_hit"], 1.0)

    def test_context_gate_can_approximate_open_gate_legacy_path(self):
        input_ids = torch.tensor([[2]], dtype=torch.long)
        memory_ids = torch.tensor([[1, 2, 1]], dtype=torch.long)

        rosa_mod.set_seed(4040)
        legacy_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
            use_match_len_gate=False,
            use_context_gate=False,
        )
        rosa_mod.set_seed(4040)
        gated_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
            use_match_len_gate=False,
            use_context_gate=True,
        )

        with torch.no_grad():
            gated_model.rosa_gate_key_projs[0].weight.zero_()
            gated_model.rosa_gate_bias[0].fill_(20.0)

            legacy_out = legacy_model(input_ids=input_ids, rosa_memory_ids=memory_ids)
            gated_out = gated_model(input_ids=input_ids, rosa_memory_ids=memory_ids)

        self.assertTrue(torch.allclose(legacy_out["logits"], gated_out["logits"], atol=1e-5))
        self.assertGreater(gated_out["rosa_avg_gate"], 0.9999)


class RosaInjectionLayerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = rosa_mod.ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            dim=16,
            n_layers=3,
            n_heads=4,
            n_kv_heads=4,
            intermediate_size=32,
        )

    def test_explicit_inject_layer_ids_override_front_layers(self):
        rosa_mod.set_seed(5050)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            inject_layers=1,
            inject_layer_ids=[2],
            rosa_value_mode="per_layer",
        )

        self.assertEqual(model.inject_layer_ids, (2,))
        self.assertEqual(model.inject_layer_index, {2: 0})
        self.assertEqual(model.inject_layers, 1)

    def test_single_explicit_layer_can_match_front_layer_when_weights_are_copied(self):
        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        memory_ids = torch.empty((1, 0), dtype=torch.long)

        rosa_mod.set_seed(6060)
        front_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            inject_layer_ids=[0],
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
        )
        rosa_mod.set_seed(6060)
        explicit_model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            inject_layer_ids=[0],
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
        )

        with torch.no_grad():
            front_out = front_model(input_ids=input_ids, rosa_memory_ids=memory_ids)
            explicit_out = explicit_model(input_ids=input_ids, rosa_memory_ids=memory_ids)

        self.assertTrue(torch.allclose(front_out["logits"], explicit_out["logits"], atol=1e-6))


class RosaHotCacheIntegrationTests(unittest.TestCase):
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

    def test_hot_cache_stats_appear_after_repeated_eval_lookup(self):
        rosa_mod.set_seed(7070)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_value_mode="per_layer",
            rosa_hot_cache_size=8,
        )
        model.eval()
        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        memory_ids = torch.empty((1, 0), dtype=torch.long)

        with torch.no_grad():
            first = model(input_ids=input_ids, rosa_memory_ids=memory_ids)
            second = model(input_ids=input_ids, rosa_memory_ids=memory_ids)

        self.assertIn("rosa_hot_cache_token_hit_rate", second)
        self.assertEqual(first["rosa_hot_cache_token_hit_rate"], 0.0)
        self.assertGreater(second["rosa_hot_cache_token_hit_rate"], 0.0)
        self.assertGreater(second["rosa_hot_cache_active_entries"], 0.0)

        report = model.get_hot_cache_stats(top_k=2)
        self.assertTrue(report["enabled"])
        self.assertGreater(report["token_requests"], 0.0)
        self.assertGreater(report["token_hit_rate"], 0.0)
        self.assertLessEqual(report["token_hit_rate"], second["rosa_hot_cache_token_hit_rate"])

    def test_hot_cache_is_ignored_in_train_mode(self):
        rosa_mod.set_seed(8080)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_value_mode="per_layer",
            rosa_hot_cache_size=4,
        )
        model.train()

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long),
                rosa_memory_ids=torch.empty((1, 0), dtype=torch.long),
            )

        self.assertNotIn("rosa_hot_cache_token_hit_rate", out)
        self.assertEqual(model.get_hot_cache_stats()["token_requests"], 0.0)


class GlobalTrainMemoryTests(unittest.TestCase):
    def test_doc_local_sam_precompute_uses_full_doc_history(self):
        docs = [[1, 2, 1, 2, 3]]
        pre = rosa_mod.build_doc_local_precomputed_rosa(
            docs,
            min_match_len=2,
            special_ids=set(),
            forbid_special_target=True,
        )

        self.assertEqual(pre[0]["rosa_ids"], [-1, -1, -1, 1, -1])
        self.assertEqual(pre[0]["fired_match_lens"], [0, 0, 0, 2, 0])
        self.assertEqual(pre[0]["raw_best_lens"], [0, 0, 1, 2, 0])

    def test_build_global_memory_prefixes_keeps_previous_docs_only(self):
        prefixes, full_memory = rosa_mod.build_global_memory_prefixes(
            [
                [1, 2, 3],
                [4, 5],
                [6, 7, 8],
            ],
            max_tokens=4,
        )

        self.assertEqual(prefixes, [[], [1, 2, 3], [2, 3, 4, 5]])
        self.assertEqual(full_memory, [4, 5, 6, 7, 8][-4:])

    def test_global_train_mode_uses_prefix_memory_for_train_and_full_train_memory_for_eval(self):
        train_tok = [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ]
        val_tok = [
            [30, 31, 32, 33],
        ]

        train_ds, val_ds, test_ds, meta = rosa_mod.build_chunk_datasets(
            train_tok,
            val_tok,
            val_tok,
            seq_len=2,
            pad_id=0,
            stride=2,
            rosa_memory_tokens=2,
            rosa_memory_mode="global_train",
            rosa_global_memory_tokens=3,
            rosa_backend="sam",
            rosa_min_match_len=2,
            special_ids=set(),
            forbid_special_target=True,
        )

        self.assertEqual(meta["rosa_memory_mode"], "global_train")
        self.assertEqual(meta["global_train_memory_tokens"], 3)
        self.assertEqual(meta["train_global_memory_size"], 3)

        first_train = train_ds[0]
        second_train = train_ds[1]
        third_train = train_ds[2]
        first_val = val_ds[0]

        self.assertEqual(first_train["rosa_memory_ids"].tolist(), [])
        self.assertEqual(second_train["rosa_memory_ids"].tolist(), [10, 11])
        self.assertEqual(third_train["rosa_memory_ids"].tolist(), [11, 12, 13])
        self.assertEqual(first_val["rosa_memory_ids"].tolist(), [21, 22, 23])
        self.assertEqual(len(test_ds), len(val_ds))

    def test_doc_local_sam_builds_precomputed_chunk_features(self):
        docs = [[1, 2, 1, 2, 3]]
        train_ds, val_ds, test_ds, meta = rosa_mod.build_chunk_datasets(
            docs,
            docs,
            docs,
            seq_len=2,
            pad_id=0,
            stride=2,
            rosa_memory_tokens=1,
            rosa_memory_mode="doc_local",
            rosa_global_memory_tokens=0,
            rosa_backend="sam",
            rosa_min_match_len=2,
            special_ids=set(),
            forbid_special_target=True,
        )

        self.assertTrue(meta["precomputed_doc_local_sam"])
        first = train_ds[0]
        second = train_ds[1]
        self.assertEqual(first["rosa_memory_ids"].tolist(), [])
        self.assertEqual(second["rosa_memory_ids"].tolist(), [])
        self.assertEqual(first["rosa_precomputed_ids"].tolist(), [-1, -1])
        self.assertEqual(second["rosa_precomputed_ids"].tolist(), [-1, 1])
        self.assertEqual(second["rosa_precomputed_match_lens"].tolist(), [0, 2])
        self.assertEqual(second["rosa_precomputed_raw_best_lens"].tolist(), [1, 2])
        self.assertEqual(len(val_ds), len(test_ds))


if __name__ == "__main__":
    unittest.main()
