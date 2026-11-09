import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.data import ByteTokenizer
from rwkv_rosa.model import RWKVRosaLM
from rwkv_rosa.numerics import (address_agreement, audit_model, cache_error, float32_model,
                               logits_error, main, make_cases, partition, stream_forward, tensor_error)


class PrecisionAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.cfg = RWKVConfig(vocab_size=24, n_embd=16, n_layer=2, head_size=8,
                              d_decay_lora=8, d_aaa_lora=8, d_mv_lora=8, d_gate_lora=8)
        self.base = build_backbone(self.cfg).state_dict()
        x = torch.tensor([[1, 2, 1, 2, 3, 4, 1, 2]])
        self.cases = [("short", x, torch.tensor([[2, 1, 2, 3, 4, 1, 2, 3]])),
                      ("batch", x.repeat(2, 1), x.repeat(2, 1))]

    def test_audit_runs_all_precision_policies_for_all_arms(self):
        for variant in ("baseline", "original", "improved"):
            model = RWKVRosaLM(self.cfg, variant=variant, base_state=self.base)
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit_model(model, self.cases, torch.device("cpu"))
            self.assertEqual(set(result["against_fp32_full"]), {"legacy_amp", "fp32_wkv"})
            for policy in ("legacy_amp", "fp32_wkv", "float32"):
                for name, _, _ in self.cases:
                    record = result[policy][name]
                    self.assertEqual(record["repeat_or_stateless_max_abs_diff"], 0)
                    for stats in record["partitions"].values():
                        self.assertTrue(stats["cache"]["wkv_storage_all_fp32"])
                        self.assertTrue(stats["cache"]["tokens_seen_equal"])
                        self.assertGreater(stats["predictions"], 0)
                        self.assertGreaterEqual(stats["top1_agreement"], 0)
            self.assertEqual(result["operator_probe"]["fp32_wkv"]["state_error"]["max_abs_diff"], 0)
            self.assertGreater(result["operator_probe"]["legacy_amp"]["state_error"]["max_abs_diff"], 0)
            if variant == "baseline":
                self.assertIn("blocks.0.att.receptance", result["fp32_wkv"]["layer_trace_sft64_halves"])

    def test_metric_detects_changed_predictions_and_nll(self):
        ref = torch.tensor([[[2., 1., 0.], [0., 2., 1.]]])
        other = ref.clone()
        other[0, 0] = torch.tensor([1., 2., 0.])
        metric = logits_error(ref, other, torch.tensor([[0, 1]]))
        self.assertEqual(metric["top1_changed"], 1)
        self.assertEqual(metric["predictions"], 2)
        self.assertEqual(metric["changed_examples"][0]["position"], 0)
        self.assertGreater(metric["nll_delta"], 0)
        self.assertGreater(metric["mean_kl_reference_to_actual"], 0)
        with self.assertRaises(ValueError):
            tensor_error(ref, other[..., :1])
        with self.assertRaises(FloatingPointError):
            tensor_error(torch.tensor([float("nan")]), torch.ones(1))

    def test_partitions_cover_short_and_long_sequences(self):
        for length in (2, 3, 8, 64, 256):
            for mode in ("halves", "uneven", "decode"):
                sizes = partition(length, mode)
                self.assertEqual(sum(sizes), length)
                self.assertTrue(all(s > 0 for s in sizes))
        for length, mode in ((1, "halves"), (8, "unknown")):
            with self.assertRaises(ValueError):
                partition(length, mode)
        model = RWKVRosaLM(self.cfg, base_state=self.base)
        with self.assertRaises(ValueError):
            stream_forward(model, self.cases[0][1], [2])
        with torch.no_grad():
            first = model(self.cases[0][1], return_state=True)
            cache = cache_error(first["state"], first["state"])
        self.assertEqual(cache["max_wkv_abs_diff"], 0)
        self.assertTrue(all(address_agreement(model, self.cases[0][1]).values()))

    def test_precision_context_restores_storage_and_runtime_on_error(self):
        model = RWKVRosaLM(self.cfg, base_state=self.base, dtype=torch.bfloat16)
        previous = {n: p.dtype for n, p in model.named_parameters()}
        with self.assertRaises(RuntimeError):
            with float32_model(model):
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                self.assertEqual(model.backbone._reference_runtime.DTYPE, torch.float32)
                raise RuntimeError("test")
        self.assertEqual(previous, {n: p.dtype for n, p in model.named_parameters()})
        self.assertEqual(model.backbone._reference_runtime.DTYPE, torch.bfloat16)

    def test_cases_and_cli_failure_paths(self):
        cases = make_cases(ByteTokenizer(), 24)
        self.assertEqual([x.shape[0] for _, x, _ in cases], [1, 1, 2])
        self.assertEqual(cases[1][1].shape[1], 24)
        with tempfile.TemporaryDirectory() as folder:
            argv = ["--checkpoint", "unused", "--out-dir", folder]
            with mock.patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
                main(argv)
            Path(folder, "exists").write_text("x")
            with self.assertRaises(FileExistsError):
                main(argv)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv + ["--long-tokens", "1"])


if __name__ == "__main__":
    unittest.main()
