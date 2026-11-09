"""Input validation, tokenizer compatibility and checkpoint persistence regressions."""
import contextlib
import gzip
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_pure_rosa_baseline as pure
import train_pure_rosa_rwkv_baseline as rwkv
import train_qwen_llama_vs_rosa as legacy
import train_qwen_llama_vs_rosa_v2 as current
import rosa_memmap_builder as builder
import rosa_training_cache as cache
import rosa_checkpointing as checkpointing
import profile_rosa_online_baseline as profile


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_byte_decode_and_hf_wrappers_without_network(self):
        for module in [legacy, current]:
            with self.subTest(module=module.__name__):
                byte = module.ByteTokenizer()
                self.assertEqual([byte.decode_id(x) for x in [256, 257, 258, 97, 255]], ["<PAD>", "<EOS>", "<BOS>", "a", "b'\\xff'"])
                with mock.patch.object(module, "AutoTokenizer", None), self.assertRaises(RuntimeError):
                    module.HFTokenizerWrapper("offline")
                for eos in [None, 2]:
                    tok = mock.MagicMock()
                    tok.__len__.return_value = 20
                    tok.pad_token_id = None
                    tok.eos_token_id = eos
                    tok.bos_token_id = None
                    tok.all_special_ids = [8]
                    tok.encode.return_value = [4, 5]
                    tok.decode.return_value = "a\nb"
                    def add_tokens(_):
                        tok.pad_token_id = 3
                    tok.add_special_tokens.side_effect = add_tokens
                    with mock.patch.object(module, "AutoTokenizer") as auto:
                        auto.from_pretrained.return_value = tok
                        wrapper = module.build_tokenizer("offline")
                    self.assertEqual(wrapper.encode("ab"), [4, 5])
                    self.assertEqual(wrapper.decode_id(4), "a\\nb")
                    self.assertEqual(wrapper.decode_id(-1), "<NA>")
                    tok.decode.return_value = ""
                    self.assertEqual(wrapper.decode_id(4), "<id:4>")
                    tok.decode.side_effect = ValueError("decode failed")
                    self.assertEqual(wrapper.decode_id(4), "<id:4>")
                    self.assertIn(8, wrapper.special_ids)

    def test_rwkv_vocabulary_validation_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vocab.txt"
            for text in ["", "invalid\n", "1 not-a-literal 1\n", "1 'text' 4\n"]:
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    rwkv.RWKVTokenizer(path)
            path.write_text("\n" + "\n".join(f"{i} {bytes([i])!r} 1" for i in range(256)) + "\n256 b'ab' 2\n", encoding="utf-8")
            tokenizer = rwkv.RWKVTokenizer(path)
            self.assertEqual(tokenizer.encode("ab"), [256])
            self.assertEqual(tokenizer.decode(tokenizer.encode("ab中")), "ab中")
            self.assertEqual(tokenizer.decode([tokenizer.sep_id, 999]), "")
            for idx in [-1, tokenizer.sep_id, 999, 0, 10, 255]:
                self.assertIsInstance(tokenizer.token_to_pretty(idx), str)
            path.write_text("1 b'a' 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rwkv.RWKVTokenizer(path).encode("z")

    def test_pure_metrics_and_document_boundaries(self):
        for module in [pure, rwkv]:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.rosa_predict([]), [] if module is pure else ([], []))
                self.assertEqual(module.split_docs("a\r\nb\r\n", "line"), ["a", "b"])
                self.assertEqual(module.split_docs("a\n\nb", "paragraph"), ["a", "b"])
                self.assertEqual(module.split_docs("a", "stream"), ["a"])
                with self.assertRaises(ValueError):
                    module.split_docs("a", "invalid")
                self.assertEqual(module.limit_doc_length([[1, 2]], None), [[1, 2]])
                with self.assertRaises(ValueError):
                    module.train_val_test_split([[1, 2]], .8, .1, 42)
                with self.assertRaises(ValueError):
                    module.calc_metrics([1], [], vocab_size=257)
                self.assertEqual(module.calc_metrics([], [], vocab_size=257).positions, 0)
                self.assertEqual(module.merge_metrics([]).positions, 0)
        for docs in [[], [[1], [2, 3, 2, 3]]]:
            self.assertGreaterEqual(pure.eval_online(docs, 1e-4)[0].positions, 0)
            self.assertGreaterEqual(pure.eval_with_memory([], docs, 1e-4, 0)[0].positions, 0)
            for fn in [rwkv.eval_online_rosa, rwkv.eval_online_bigram]:
                self.assertGreaterEqual(fn(docs, vocab_size=257, smooth_eps=1e-4, **({"min_match_len": 1} if fn is rwkv.eval_online_rosa else {}))[0].positions, 0)
            for fn in [rwkv.eval_with_memory_rosa, rwkv.eval_with_memory_bigram]:
                self.assertGreaterEqual(fn([], docs, sep_id=256, vocab_size=257, smooth_eps=1e-4, max_train_tokens=0, **({"min_match_len": 1} if fn is rwkv.eval_with_memory_rosa else {}))[0].positions, 0)


class InputFormatRegressionTests(unittest.TestCase):
    def test_json_gzip_jsonl_and_directory_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in ["docs.json", "docs.json.gz", "docs.jsonl.gz"]:
                payload = [{"text": "first"}, {"text": " "}, {"ignored": 3}, {"text": ["second", "third"]}]
                text = json.dumps(payload) if ".jsonl" not in filename else "\n" + "\n".join(map(json.dumps, payload)) + "\n"
                path = root / filename
                if filename.endswith(".gz"):
                    with gzip.open(path, "wt", encoding="utf-8") as stream:
                        stream.write(text)
                else:
                    path.write_text(text, encoding="utf-8")
                docs, meta = current.load_docs_from_path(str(path), data_format="auto", split_mode="line", json_text_keys=["text"])
                self.assertEqual(docs, ["first", "second\nthird"])
                self.assertEqual(meta["skipped_records"], 2)
                stream, stats = builder.iter_docs_from_path(str(path), data_format="auto", split_mode="line", json_text_keys=["text"], max_docs=1)
                self.assertEqual(list(stream), ["first"])
                self.assertEqual(stats.doc_count, 1)
                stream, stats = builder.iter_docs_from_path(str(path), data_format="auto", split_mode="line", json_text_keys=["text"])
                self.assertEqual(list(stream), docs)
                self.assertEqual(stats.skipped_records, 2)
            self.assertEqual(len(current.resolve_data_files(str(root))), 3)
            for value, error in [("", ValueError), (str(root / "missing"), FileNotFoundError)]:
                with self.assertRaises(error):
                    current.resolve_data_files(value)
            broken = root / "bad.jsonl"
            broken.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSONL"):
                current.load_docs_from_path(str(broken), data_format="auto", split_mode="line", json_text_keys=["text"])
            stream, _ = builder.iter_docs_from_path(str(broken), data_format="auto", split_mode="line", json_text_keys=["text"])
            with self.assertRaisesRegex(ValueError, "JSONL"):
                list(stream)
            empty = root / "empty.json"
            empty.write_text('{"ignored": 3}', encoding="utf-8")
            with self.assertRaises(ValueError):
                current.load_docs_from_path(str(empty), data_format="auto", split_mode="line", json_text_keys=["text"])
            for loader in [current.load_docs_from_path, builder.iter_docs_from_path]:
                with self.assertRaises(ValueError):
                    result = loader(str(empty), data_format="invalid", split_mode="line", json_text_keys=["text"])
                    if loader is builder.iter_docs_from_path:
                        list(result[0])
            self.assertEqual(current.extract_text_from_record("plain", []), "plain")
            self.assertEqual(current.extract_text_from_record({"other": "fallback"}, []), "fallback")
            self.assertEqual(current.coerce_text(["a", "", "b"]), "a\nb")
            self.assertIsNone(current.get_nested_value({"x": 1}, "x.y"))
            self.assertEqual(current.parse_int_csv_arg(None), [])

    def test_worker_tokenization_and_cached_exact_addresses(self):
        with mock.patch.object(builder, "_WORKER_TOKENIZER", None), self.assertRaises(RuntimeError):
            builder._tokenize_docs_batch_worker(["abc"])
        with mock.patch.object(builder, "_WORKER_TOKENIZER", None), mock.patch.object(builder, "_WORKER_ADD_BOS", False), mock.patch.object(builder, "_WORKER_ADD_EOS", False):
            builder._init_tokenize_worker(None, True, True)
            result = builder._tokenize_docs_batch_worker(["abc"])
            self.assertEqual(result, [[258, 97, 98, 99, 257]])
        kwargs = dict(min_match_len=1, special_ids=set(), forbid_special_target=True)
        docs = [[1, 2, 1, 2, 3]]
        exact = cache.build_sequence_online_precomputed_rosa(docs, sequence_mode="online_exact", memory_prefixes=[[1, 2, 3]], **kwargs)
        sam = cache.build_sequence_online_precomputed_rosa(docs, sequence_mode="online_sam", memory_prefixes=[[1, 2, 3]], **kwargs)
        self.assertEqual(exact, sam)
        with self.assertRaises(ValueError):
            cache.build_sequence_online_precomputed_rosa(docs, sequence_mode="bad", **kwargs)
        with self.assertRaises(ValueError):
            cache.build_sequence_online_precomputed_rosa(docs, sequence_mode="online_exact", memory_prefixes=[], **kwargs)


class CheckpointPersistenceTests(unittest.TestCase):
    def test_checkpoint_retention_and_disabled_saving(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = checkpointing.TrainingCheckpointManager(out_dir=temp, model_name="tiny", save_every_epochs=1, is_main_process=True, keep_last_k=1)
            kwargs = dict(global_step=1, history={}, model_state={"weight": torch.ones(1)}, optimizer_state={}, args={})
            for epoch in [1, 2, 3]:
                manager.save(epoch=epoch, **kwargs)
            self.assertEqual([p.name for p in manager.checkpoint_dir.glob("epoch_*.pt")], ["epoch_003.pt"])
            self.assertEqual(checkpointing.load_training_checkpoint(str(manager.last_path()))["epoch"], 3)
            manager.is_main_process = False
            manager.save(epoch=4, **kwargs)
            self.assertFalse(manager.epoch_path(4).exists())

    def test_profile_checkpoint_formats_and_invalid_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            model = torch.nn.Linear(2, 2)
            for wrapper in [None, "model", "state_dict"]:
                state = model.state_dict()
                torch.save(state if wrapper is None else {wrapper: state}, path)
                self.assertEqual(set(profile.load_state_dict(str(path))), set(state))
                target = torch.nn.Linear(2, 2)
                profile.maybe_load_checkpoint(target, str(path))
                self.assertTrue(torch.equal(target.weight, model.weight))
            torch.save([1, 2], path)
            with self.assertRaises(ValueError):
                profile.load_state_dict(str(path))
            path.write_bytes(b"existing file")
            with mock.patch.object(checkpointing.torch, "save", side_effect=RuntimeError("disk failure")), self.assertRaises(RuntimeError):
                checkpointing.atomic_torch_save({}, str(path))
            self.assertEqual(path.read_bytes(), b"existing file")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
