import math
import tempfile
import unittest
from pathlib import Path

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod
from rosa_memmap_dataset import (
    MemmapDocChunkDataset,
    PretokenizedMemmapSplit,
    build_memmap_chunk_datasets_from_manifest,
    load_pretokenized_memmap_manifest,
    write_pretokenized_memmap_dataset,
)


class RosaMemmapDatasetTests(unittest.TestCase):
    def _write_manifest(self, root: Path, *, train_docs, val_docs, test_docs):
        manifest_path, manifest = write_pretokenized_memmap_dataset(
            str(root),
            split_docs_tokens={
                "train": train_docs,
                "val": val_docs,
                "test": test_docs,
            },
            tokenizer_meta={
                "name_or_path": "byte-fallback",
                "vocab_size": 259,
                "pad_token_id": 256,
                "bos_token_id": 258,
                "eos_token_id": 257,
            },
            source_meta={"mode": "unit_test"},
            preview_texts={
                "train": "train-preview",
                "val": "val-preview",
                "test": "test-preview",
            },
        )
        return manifest_path, manifest

    def assert_dataset_samples_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for idx in range(len(left)):
            left_item = left[idx]
            right_item = right[idx]
            self.assertTrue(torch.equal(left_item["input_ids"], right_item["input_ids"]), msg=f"input_ids mismatch at {idx}")
            self.assertTrue(torch.equal(left_item["labels"], right_item["labels"]), msg=f"labels mismatch at {idx}")
            self.assertTrue(
                torch.equal(left_item["rosa_memory_ids"], right_item["rosa_memory_ids"]),
                msg=f"rosa_memory_ids mismatch at {idx}",
            )

    def test_doc_local_memmap_dataset_matches_in_memory_dataset(self):
        docs = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11],
        ]
        tmpdir = tempfile.TemporaryDirectory()
        split = None
        memmap_ds = None
        try:
            root = Path(tmpdir.name)
            manifest_path, _ = self._write_manifest(root, train_docs=docs, val_docs=docs[:1], test_docs=docs[1:])
            manifest = load_pretokenized_memmap_manifest(manifest_path)
            split = PretokenizedMemmapSplit(manifest["_root_dir"], manifest["splits"]["train"])

            in_memory = rosa_mod.DocChunkDataset(
                docs,
                seq_len=3,
                pad_id=0,
                stride=2,
                rosa_memory_tokens=4,
                full_doc_memory=True,
            )
            memmap_ds = MemmapDocChunkDataset(
                split,
                seq_len=3,
                pad_id=0,
                stride=2,
                rosa_memory_tokens=4,
                full_doc_memory=True,
            )

            self.assert_dataset_samples_equal(in_memory, memmap_ds)
        finally:
            if memmap_ds is not None:
                memmap_ds.close()
            elif split is not None:
                split.close()
            tmpdir.cleanup()

    def test_global_train_memmap_builder_matches_existing_dataset_logic(self):
        train_docs = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9],
        ]
        val_docs = [[10, 11, 12, 13]]
        test_docs = [[14, 15, 16, 17]]
        tmpdir = tempfile.TemporaryDirectory()
        mem_train = mem_val = mem_test = None
        try:
            root = Path(tmpdir.name)
            manifest_path, _ = self._write_manifest(root, train_docs=train_docs, val_docs=val_docs, test_docs=test_docs)

            raw_train, raw_val, raw_test, raw_meta = rosa_mod.build_chunk_datasets(
                train_docs,
                val_docs,
                test_docs,
                seq_len=2,
                pad_id=0,
                stride=1,
                rosa_memory_tokens=3,
                rosa_memory_mode="global_train",
                rosa_global_memory_tokens=3,
                rosa_backend="sam",
                rosa_train_mode="online_seq",
                rosa_seq_address_mode="online_exact",
                rosa_min_match_len=1,
                special_ids=set(),
                forbid_special_target=True,
            )
            mem_train, mem_val, mem_test, mem_meta, dataset_meta = build_memmap_chunk_datasets_from_manifest(
                manifest_path,
                seq_len=2,
                pad_id=0,
                stride=1,
                rosa_memory_tokens=3,
                rosa_memory_mode="global_train",
                rosa_global_memory_tokens=3,
                rosa_train_mode="online_seq",
            )

            self.assertEqual(raw_meta["effective_history"], mem_meta["effective_history"])
            self.assertEqual(dataset_meta["train_docs"], len(train_docs))
            self.assertEqual(dataset_meta["val_docs"], len(val_docs))
            self.assertEqual(dataset_meta["test_docs"], len(test_docs))
            self.assert_dataset_samples_equal(raw_train, mem_train)
            self.assert_dataset_samples_equal(raw_val, mem_val)
            self.assert_dataset_samples_equal(raw_test, mem_test)
        finally:
            for ds in [mem_train, mem_val, mem_test]:
                if ds is not None:
                    ds.close()
            tmpdir.cleanup()

    def test_online_training_smoke_runs_on_memmap_manifest(self):
        docs = [
            [1, 2, 1, 2, 3, 4],
            [5, 6, 5, 6, 7, 8],
            [9, 10, 9, 10, 11, 12],
        ]
        tmpdir = tempfile.TemporaryDirectory()
        train_ds = val_ds = test_ds = None
        try:
            root = Path(tmpdir.name)
            manifest_path, _ = self._write_manifest(root, train_docs=docs, val_docs=docs[:1], test_docs=docs[1:2])
            train_ds, val_ds, test_ds, memory_meta, dataset_meta = build_memmap_chunk_datasets_from_manifest(
                manifest_path,
                seq_len=3,
                pad_id=0,
                stride=3,
                rosa_memory_tokens=4,
                rosa_memory_mode="doc_local",
                rosa_global_memory_tokens=0,
                rosa_train_mode="online_seq",
            )
            self.assertEqual(memory_meta["dataset_backend"], "pretokenized_memmap")
            self.assertEqual(dataset_meta["source"]["mode"], "pretokenized_manifest")

            train_loader, val_loader, _ = rosa_mod.build_dataloaders(
                train_ds,
                val_ds,
                test_ds,
                batch_size=2,
                pad_id=0,
                train_seed=2026,
            )

            cfg = rosa_mod.ModelConfig(
                vocab_size=32,
                max_seq_len=8,
                dim=16,
                n_layers=2,
                n_heads=4,
                n_kv_heads=4,
                intermediate_size=32,
            )
            rosa_mod.set_seed(9191)
            model = rosa_mod.RosaFusedLM(
                cfg,
                pad_id=0,
                min_match_len=1,
                inject_layers=1,
                rosa_seq_address_mode="online_exact",
            )
            history = rosa_mod.train_one_model(
                model,
                train_loader,
                val_loader,
                device=torch.device("cpu"),
                pad_id=0,
                epochs=1,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                use_bf16=False,
            )

            self.assertEqual(len(history["train"]), 1)
            self.assertEqual(history["train"][0]["rosa_address_source_online_seq"], 1.0)
            self.assertTrue(math.isfinite(history["train"][0]["loss"]))
        finally:
            for ds in [train_ds, val_ds, test_ds]:
                if ds is not None:
                    ds.close()
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
