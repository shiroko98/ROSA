import tempfile
import unittest
from pathlib import Path

import scan_rosa_injection_layers as scan_mod


class RosaInjectionLayerScanSmokeTests(unittest.TestCase):
    def test_run_scan_writes_ranked_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = scan_mod.build_arg_parser()
            args = parser.parse_args(
                [
                    "--data_path",
                    "data/rosa_demo.txt",
                    "--data_format",
                    "text",
                    "--split_mode",
                    "line",
                    "--max_docs",
                    "4",
                    "--num_samples",
                    "2",
                    "--prefill_tokens",
                    "8",
                    "--decode_steps",
                    "2",
                    "--sample_stride",
                    "2",
                    "--batch_size",
                    "1",
                    "--warmup_iters",
                    "0",
                    "--measure_iters",
                    "1",
                    "--device",
                    "cpu",
                    "--arch_style",
                    "qwen",
                    "--seq_len",
                    "8",
                    "--dim",
                    "16",
                    "--n_layers",
                    "2",
                    "--n_heads",
                    "4",
                    "--n_kv_heads",
                    "4",
                    "--intermediate_size",
                    "32",
                    "--rosa_value_mode",
                    "per_layer",
                    "--rosa_context_gate",
                    "--scan_mode",
                    "single",
                    "--layer_candidates",
                    "0,1",
                    "--out_dir",
                    tmpdir,
                ]
            )

            summary = scan_mod.run_scan(args)

            report_path = Path(tmpdir) / "layer_scan_report.json"
            self.assertTrue(report_path.exists())
            self.assertEqual(summary["meta"]["num_runs"], 2)
            self.assertEqual(len(summary["ranked_results"]), 2)
            self.assertIsNotNone(summary["best"])
            self.assertIn("layer_ids", summary["best"])

    def test_run_scan_both_mode_writes_train_and_profile_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = scan_mod.build_arg_parser()
            args = parser.parse_args(
                [
                    "--data_path",
                    "data/rosa_demo.txt",
                    "--train_data_path",
                    "data/rosa_demo.txt",
                    "--val_data_path",
                    "data/rosa_demo.txt",
                    "--test_data_path",
                    "data/rosa_demo.txt",
                    "--data_format",
                    "text",
                    "--split_mode",
                    "line",
                    "--max_train_docs",
                    "4",
                    "--max_val_docs",
                    "4",
                    "--max_test_docs",
                    "4",
                    "--num_samples",
                    "2",
                    "--prefill_tokens",
                    "8",
                    "--decode_steps",
                    "2",
                    "--sample_stride",
                    "2",
                    "--batch_size",
                    "1",
                    "--epochs",
                    "1",
                    "--warmup_iters",
                    "0",
                    "--measure_iters",
                    "1",
                    "--device",
                    "cpu",
                    "--arch_style",
                    "qwen",
                    "--seq_len",
                    "8",
                    "--dim",
                    "16",
                    "--n_layers",
                    "2",
                    "--n_heads",
                    "4",
                    "--n_kv_heads",
                    "4",
                    "--intermediate_size",
                    "32",
                    "--rosa_recipe",
                    "online_v1",
                    "--experiment_mode",
                    "both",
                    "--scan_mode",
                    "single",
                    "--layer_candidates",
                    "0",
                    "--out_dir",
                    tmpdir,
                ]
            )

            summary = scan_mod.run_scan(args)

            report_path = Path(tmpdir) / "layer_scan_report.json"
            layer_train_path = Path(tmpdir) / "layers_0" / "train" / "train_summary.json"
            layer_profile_path = Path(tmpdir) / "layers_0" / "profile" / "profile_report.json"
            self.assertTrue(report_path.exists())
            self.assertTrue(layer_train_path.exists())
            self.assertTrue(layer_profile_path.exists())
            self.assertEqual(summary["meta"]["experiment_mode"], "both")
            self.assertEqual(summary["meta"]["recipe"]["name"], "online_v1")
            self.assertIn("train", summary["best"])
            self.assertIn("profile", summary["best"])
            self.assertGreaterEqual(summary["best"]["train"]["test_address_source_online_seq"], 1.0)


if __name__ == "__main__":
    unittest.main()
