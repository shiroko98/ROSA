import tempfile
import unittest
from pathlib import Path

import profile_rosa_online_baseline as profile_mod


class ProfileRosaOnlineBaselineSmokeTests(unittest.TestCase):
    def test_run_profile_writes_report_and_keeps_online_reference_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = profile_mod.build_arg_parser()
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
                    "--rosa_hot_cache_size",
                    "8",
                    "--out_dir",
                    tmpdir,
                ]
            )

            report = profile_mod.run_profile(args)

            report_path = Path(tmpdir) / "profile_report.json"
            self.assertTrue(report_path.exists())
            self.assertIn("prefill", report)
            self.assertIn("decode_micro", report)
            self.assertIn("hot_cache", report)
            self.assertEqual(report["prefill"]["correctness"]["address_agreement"]["all_equal"], 1.0)
            self.assertEqual(report["decode_micro"]["correctness"]["address_agreement"]["all_equal"], 1.0)
            self.assertLessEqual(report["summary"]["prefill_online_logit_max_abs_diff"], 1e-6)
            self.assertLessEqual(report["summary"]["decode_online_logit_max_abs_diff"], 1e-6)
            self.assertTrue(report["hot_cache"]["enabled"])
            self.assertGreater(report["hot_cache"]["prefill"]["token_requests"], 0.0)


if __name__ == "__main__":
    unittest.main()
