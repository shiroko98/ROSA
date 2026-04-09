import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rosa_memmap_builder import build_streaming_pretokenized_memmap_dataset_from_explicit_splits
from rosa_memmap_dataset import PretokenizedMemmapSplit, load_pretokenized_memmap_manifest, write_pretokenized_memmap_dataset
from train_qwen_llama_vs_rosa_v2 import build_tokenizer, tokenize_docs


class RosaMemmapBuilderTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, docs):
        with open(path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

    def _collect_split_docs(self, manifest_path: str, split_name: str):
        manifest = load_pretokenized_memmap_manifest(manifest_path)
        split = PretokenizedMemmapSplit(manifest["_root_dir"], manifest["splits"][split_name])
        try:
            return [split.get_doc_tokens(i).tolist() for i in range(split.doc_count)]
        finally:
            split.close()

    def test_streaming_builder_matches_in_memory_writer_single_worker(self):
        train_docs = ["alpha beta", "gamma delta", "epsilon zeta"]
        val_docs = ["theta iota"]
        test_docs = ["kappa lambda"]
        tokenizer = build_tokenizer(None)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_jsonl(root / "train-00000.jsonl", train_docs[:2])
            self._write_jsonl(root / "train-00001.jsonl", train_docs[2:])
            self._write_jsonl(root / "val.jsonl", val_docs)
            self._write_jsonl(root / "test.jsonl", test_docs)

            manifest_path, manifest = build_streaming_pretokenized_memmap_dataset_from_explicit_splits(
                str(root / "stream_out"),
                tokenizer_name_or_path=None,
                train_data_path=str(root / "train-*.jsonl"),
                val_data_path=str(root / "val.jsonl"),
                test_data_path=str(root / "test.jsonl"),
                data_format="jsonl",
                split_mode="paragraph",
                json_text_keys=["text"],
                add_bos=True,
                add_eos=True,
                tokenize_workers=1,
                tokenize_batch_docs=2,
                progress_docs=1,
            )

            expected_manifest_path, _ = write_pretokenized_memmap_dataset(
                str(root / "expected_out"),
                split_docs_tokens={
                    "train": tokenize_docs(train_docs, tokenizer, add_bos=True, add_eos=True),
                    "val": tokenize_docs(val_docs, tokenizer, add_bos=True, add_eos=True),
                    "test": tokenize_docs(test_docs, tokenizer, add_bos=True, add_eos=True),
                },
                tokenizer_meta={
                    "name_or_path": "byte-fallback",
                    "vocab_size": tokenizer.vocab_size,
                    "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                },
                source_meta={"mode": "unit_test"},
            )

            self.assertEqual(manifest["splits"]["train"]["doc_count"], 3)
            self.assertEqual(manifest["source"]["train"]["doc_count"], 3)
            self.assertEqual(
                self._collect_split_docs(manifest_path, "train"),
                self._collect_split_docs(expected_manifest_path, "train"),
            )
            self.assertEqual(
                self._collect_split_docs(manifest_path, "val"),
                self._collect_split_docs(expected_manifest_path, "val"),
            )
            self.assertEqual(
                self._collect_split_docs(manifest_path, "test"),
                self._collect_split_docs(expected_manifest_path, "test"),
            )

    def test_streaming_builder_supports_multiple_workers(self):
        train_docs = [f"doc {idx}" for idx in range(6)]
        val_docs = ["val a", "val b"]
        test_docs = ["test a"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_jsonl(root / "train-00000.jsonl", train_docs[:3])
            self._write_jsonl(root / "train-00001.jsonl", train_docs[3:])
            self._write_jsonl(root / "val.jsonl", val_docs)
            self._write_jsonl(root / "test.jsonl", test_docs)

            manifest_path, manifest = build_streaming_pretokenized_memmap_dataset_from_explicit_splits(
                str(root / "stream_multi"),
                tokenizer_name_or_path=None,
                train_data_path=str(root / "train-*.jsonl"),
                val_data_path=str(root / "val.jsonl"),
                test_data_path=str(root / "test.jsonl"),
                data_format="jsonl",
                split_mode="paragraph",
                json_text_keys=["text"],
                add_bos=True,
                add_eos=True,
                tokenize_workers=2,
                tokenize_batch_docs=2,
                progress_docs=0,
            )

            self.assertEqual(manifest["splits"]["train"]["doc_count"], len(train_docs))
            self.assertEqual(manifest["splits"]["val"]["doc_count"], len(val_docs))
            self.assertEqual(manifest["splits"]["test"]["doc_count"], len(test_docs))

            train_tokens = self._collect_split_docs(manifest_path, "train")
            self.assertEqual(len(train_tokens), len(train_docs))
            self.assertTrue(all(len(row) > 0 for row in train_tokens))


if __name__ == "__main__":
    unittest.main()
