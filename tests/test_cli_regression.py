"""Offline end-to-end coverage for the public command-line entry points."""
import contextlib
import io
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "toy_corpus.txt"


def run_cli(script, *args):
    output = io.StringIO()
    with mock.patch.object(sys, "argv", [script, *map(str, args)]), contextlib.redirect_stdout(output), mock.patch("torch.cuda.is_available", return_value=False):
        runpy.run_path(str(ROOT / script), run_name="__main__")
    return output.getvalue()


SMALL_MODEL = ["--seq_len", "16", "--stride", "16", "--dim", "8", "--n_layers", "1", "--n_heads", "2", "--n_kv_heads", "1", "--intermediate_size", "16", "--batch_size", "16", "--epochs", "1"]


class PublicCliRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_pure_rosa_cli_online_and_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            for script in ["train_pure_rosa_baseline.py", "train_pure_rosa_rwkv_baseline.py"]:
                for mode in ["online", "memory"]:
                    with self.subTest(script=script, mode=mode):
                        out = Path(temp) / f"{script}-{mode}"
                        args = ["--mode", mode, "--out_dir", out, "--dump_examples", "1", "--max_doc_bytes", "48", "--max_train_bytes", "64"]
                        if mode == "online":
                            args += ["--toy", "repeat", "--num_toy_docs", "20", "--toy_min_len", "32", "--toy_max_len", "48"]
                        else:
                            args += ["--data_path", CORPUS, "--split_mode", "line"]
                        if "rwkv" in script:
                            args += ["--rwkv_vocab_path", ROOT / "rwkv_vocab_v20230424.txt"]
                        run_cli(script, *args)
                        result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                        metrics = result["rosa"]["test"] if "rosa" in result else result["test"]
                        self.assertGreater(metrics["positions"], 0)
                        self.assertTrue(math.isfinite(metrics["avg_nll"]))
                        self.assertGreater(len(list(out.glob("*curve.csv"))), 2)
                        if "baseline" in result:
                            self.assertEqual(result["baseline"]["name"], "bigram_most_frequent")

    def test_legacy_transformer_cli_trains_both_models(self):
        with tempfile.TemporaryDirectory() as temp:
            run_cli("train_qwen_llama_vs_rosa.py", "--data_path", CORPUS, "--split_mode", "line", *SMALL_MODEL, "--arch_style", "qwen", "--out_dir", temp)
            result = json.loads((Path(temp) / "comparison.json").read_text(encoding="utf-8"))
            for name in ["baseline", "rosa_fused"]:
                self.assertTrue(math.isfinite(result[name]["test"]["loss"]))
                self.assertTrue((Path(temp) / f"{name}.pt").is_file())

    def test_current_cli_saves_and_resumes_training(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "first"
            common = ["--data_path", CORPUS, "--split_mode", "line", *SMALL_MODEL, "--arch_style", "qwen", "--rosa_recipe", "online_v1"]
            run_cli("train_qwen_llama_vs_rosa_v2.py", *common, "--out_dir", out, "--train_timing", "--train_log_every_steps", "1", "--grad_accum_steps", "2", "--activation_checkpointing")
            result = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
            self.assertGreater(result["rosa_fused"]["test"]["valid_tokens"], 0)
            checkpoint = out / "checkpoints" / "rosa_fused" / "last.pt"
            first = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(first["epoch"], 1)
            resumed = Path(temp) / "resumed"
            run_cli("train_qwen_llama_vs_rosa_v2.py", *common, "--epochs", "2", "--run_models", "rosa_fused", "--resume_from", checkpoint, "--out_dir", resumed)
            last = torch.load(resumed / "checkpoints" / "rosa_fused" / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            self.assertGreater(last["global_step"], first["global_step"])

    def test_current_cli_reference_global_and_sparse_recipes(self):
        with tempfile.TemporaryDirectory() as temp:
            variants = [
                ["--rosa_train_mode", "reference_precompute", "--rosa_backend", "sam"],
                ["--rosa_train_mode", "online_seq", "--rosa_memory_mode", "global_train", "--rosa_global_memory_tokens", "32", "--rosa_seq_address_mode", "online_sam"],
                ["--rosa_recipe", "online_v2_sparse_sharded", "--baseline_capacity_match_rosa", "--train_timing", "--bf16"],
                ["--rosa_recipe", "online_v1", "--enable_rosa_train_address_cache"],
                ["--rosa_train_mode", "reference_precompute", "--rosa_memory_mode", "global_train", "--rosa_global_memory_tokens", "32"],
                ["--rosa_seq_address_mode", "online_sam", "--rosa_memory_mode", "global_train", "--enable_rosa_train_address_cache", "--rosa_global_memory_tokens", "32"],
                ["--rosa_seq_address_mode", "online_sam", "--rosa_memory_mode", "global_train", "--enable_rosa_train_state_snapshot", "--rosa_train_state_snapshot_interval", "8", "--rosa_global_memory_tokens", "32"],
                ["--rosa_recipe", "online_v1", "--enable_rosa_train_state_snapshot", "--rosa_train_state_snapshot_interval", "8"],
            ]
            for i, flags in enumerate(variants):
                with self.subTest(flags=flags):
                    out = Path(temp) / str(i)
                    run_cli("train_qwen_llama_vs_rosa_v2.py", "--data_path", CORPUS, "--split_mode", "line", *SMALL_MODEL, *flags, "--out_dir", out)
                    result = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
                    for name in ["baseline", "rosa_fused"]:
                        self.assertTrue(math.isfinite(result[name]["test"]["loss"]))

    def test_memmap_cli_single_source_explicit_splits_and_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in ["single", "split"]:
                out = root / mode
                args = ["--out_dir", out, "--split_mode", "line", "--progress_docs", "1"]
                if mode == "single":
                    args += ["--data_path", CORPUS]
                else:
                    for split in ["train", "val", "test"]:
                        args += [f"--{split}_data_path", CORPUS]
                run_cli("build_rosa_memmap_dataset.py", *args)
                manifest = out / "dataset_manifest.json"
                contents = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertGreater(contents["splits"]["train"]["doc_count"], 0)
                run_cli("train_qwen_llama_vs_rosa_v2.py", "--pretokenized_manifest", manifest, *SMALL_MODEL, "--rosa_recipe", "online_v1", "--out_dir", root / f"trained-{mode}")
                self.assertTrue((root / f"trained-{mode}" / "rosa_fused.pt").exists())

    def test_memmap_cli_rejects_incomplete_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for args in [[], ["--train_data_path", CORPUS], ["--json_text_keys", ",", "--data_path", CORPUS]]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    run_cli("build_rosa_memmap_dataset.py", "--out_dir", temp, *args)

    def test_profile_and_layer_scan_entry_points(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ["--data_path", CORPUS, "--split_mode", "line", "--seq_len", "8", "--dim", "8", "--n_layers", "1", "--n_heads", "2", "--n_kv_heads", "1", "--intermediate_size", "16", "--num_samples", "2", "--prefill_tokens", "8", "--measure_iters", "1", "--batch_size", "2", "--decode_steps", "1", "--warmup_iters", "0", "--train_consistency_samples", "1"]
            profile = Path(temp) / "profile"
            run_cli("profile_rosa_online_baseline.py", *common, "--rosa_prefetch", "--out_dir", profile)
            self.assertTrue((profile / "profile_report.json").exists())
            scan = Path(temp) / "scan"
            run_cli("scan_rosa_injection_layers.py", *common, "--scan_mode", "single", "--layer_candidates", "0", "--out_dir", scan)
            self.assertTrue((scan / "layer_scan_report.json").exists())

    def test_snapshot_sweep_runs_offline_training_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            calls = []
            def local_run(command, **kwargs):
                calls.append(command)
                # The sweep requires a tokenizer argument. Use the real offline
                # fallback in this integration test, without downloading a model.
                args = command[2:]
                index = args.index("--tokenizer_name_or_path")
                del args[index:index + 2]
                run_cli(command[1], *args)
                return mock.Mock(returncode=0)
            with mock.patch("subprocess.run", side_effect=local_run):
                run_cli("sweep_rosa_snapshot_intervals.py", "--train_data_path", CORPUS, "--val_data_path", CORPUS, "--test_data_path", CORPUS, "--data_format", "text", "--tokenizer_name_or_path", "offline-byte", *SMALL_MODEL, "--max_train_docs", "2", "--max_val_docs", "1", "--max_test_docs", "1", "--snapshot_intervals", "8", "--modes", "sync,async", "--out_dir", out)
            result = json.loads((out / "snapshot_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(set(result["best_by_mode"]), {"sync", "async"})


if __name__ == "__main__":
    unittest.main()
