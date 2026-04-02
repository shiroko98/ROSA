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


class GlobalTrainMemoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
