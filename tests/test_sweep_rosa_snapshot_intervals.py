import unittest
from pathlib import Path

import sweep_rosa_snapshot_intervals as sweep_mod


class SnapshotSweepScriptTests(unittest.TestCase):
    def test_parse_int_csv(self):
        self.assertEqual(sweep_mod.parse_int_csv("64, 128,256"), [64, 128, 256])

    def test_parse_mode_csv_deduplicates_and_validates(self):
        self.assertEqual(sweep_mod.parse_mode_csv("sync,async,sync"), ["sync", "async"])
        with self.assertRaises(ValueError):
            sweep_mod.parse_mode_csv("sync,unknown")

    def test_extract_run_metrics_reads_expected_fields(self):
        comparison = {
            "baseline": {
                "history": {
                    "train": [
                        {
                            "timing_step_ms": 12.5,
                        }
                    ]
                }
            },
            "rosa_fused": {
                "history": {
                    "train": [
                        {
                            "timing_step_ms": 18.0,
                            "timing_forward_ms": 10.0,
                            "timing_model_rosa_address_ms": 4.0,
                        }
                    ]
                },
                "test": {
                    "loss": 1.5,
                    "token_acc": 0.25,
                    "rosa_fire_coverage": 0.75,
                },
            },
            "dataset": {
                "memory": {
                    "effective_history": "full_doc_prefix_online_seq_snapshot_256",
                    "address_build_policy": "sequence_online_snapshot_replay",
                }
            },
        }
        out_dir = Path("outputs/fake")
        row = sweep_mod.extract_run_metrics(
            comparison,
            mode="sync",
            snapshot_interval=256,
            out_dir=out_dir,
        )
        self.assertEqual(row["mode"], "sync")
        self.assertEqual(row["snapshot_interval"], 256)
        self.assertEqual(row["rosa_step_ms"], 18.0)
        self.assertEqual(row["rosa_addr_ms"], 4.0)
        self.assertEqual(row["baseline_step_ms"], 12.5)
        self.assertEqual(row["effective_history"], "full_doc_prefix_online_seq_snapshot_256")
        self.assertEqual(row["address_build_policy"], "sequence_online_snapshot_replay")


if __name__ == "__main__":
    unittest.main()
