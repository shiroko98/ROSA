"""Optional extension failures, cache reset and sweep configuration regression tests."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import rosa_cpp_extension as extension
import rosa_layer_sweep as layers
import rosa_runtime as runtime
import sweep_rosa_snapshot_intervals as sweep


class ExtensionBuildBoundaryTests(unittest.TestCase):
    def test_cached_rebuilt_and_missing_extension_binaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "lib" / "rosa_sam_cpu_ext.pyd"
            with mock.patch.object(extension, "BUILD_LIB", root / "lib"), mock.patch.object(extension, "BUILD_TEMP", root / "temp"), mock.patch.object(extension, "_extension_binary_path", return_value=binary), mock.patch.object(extension, "_extension_suffix", return_value=".pyd"):
                def compile_binary(*args, **kwargs):
                    binary.write_bytes(b"test build product")
                with mock.patch.object(extension.subprocess, "run", side_effect=compile_binary) as build:
                    self.assertEqual(extension.build_rosa_sam_cpu_extension(force_rebuild=True, verbose=True), binary)
                    self.assertEqual(extension.build_rosa_sam_cpu_extension(), binary)
                    self.assertEqual(build.call_count, 1)
                binary.unlink()
                with mock.patch.object(extension.subprocess, "run"), self.assertRaises(FileNotFoundError):
                    extension.build_rosa_sam_cpu_extension(force_rebuild=True)
                alternate = root / "lib" / "rosa_sam_cpu_ext.alternate.pyd"
                alternate.write_bytes(b"alternate build product")
                with mock.patch.object(extension.subprocess, "run"):
                    self.assertEqual(extension.build_rosa_sam_cpu_extension(force_rebuild=True), alternate)

    def test_extension_import_failure_has_a_useful_error(self):
        with mock.patch.object(extension, "build_rosa_sam_cpu_extension", return_value=Path("test.pyd")), mock.patch.object(extension, "_ensure_torch_runtime_available"), mock.patch.object(extension.importlib, "import_module", side_effect=ImportError("missing dependency")), mock.patch.dict(extension.sys.modules):
            extension.sys.modules.pop(extension.EXTENSION_NAME, None)
            original_path = list(extension.sys.path)
            try:
                with self.assertRaisesRegex(ImportError, "test.pyd"):
                    extension.load_rosa_sam_cpu_extension(force_rebuild=True)
            finally:
                extension.sys.path[:] = original_path


class CacheAndSweepConfigurationTests(unittest.TestCase):
    def test_layer_sets_and_invalid_sweep_modes(self):
        self.assertEqual(layers.parse_custom_layer_sets("0,1; ;2"), [(0, 1), (2,)])
        self.assertEqual(layers.parse_custom_layer_sets(" "), [])
        expected = {"single": [(0,), (1,)], "pair": [(0, 1)], "single_and_pair": [(0,), (1,), (0, 1)], "custom": [(1,)]}
        for mode, result in expected.items():
            self.assertEqual(layers.build_layer_sets(scan_mode=mode, layer_candidates=[0, 1], custom_layer_sets=[(1,)]), result)
        with self.assertRaises(ValueError):
            layers.build_layer_sets(scan_mode="bad", layer_candidates=[], custom_layer_sets=[])
        self.assertEqual(sweep.parse_int_csv("8,,16"), [8, 16])
        self.assertEqual(sweep.parse_mode_csv("sync,,SYNC,async"), ["sync", "async"])
        for value in ["", "invalid"]:
            with self.assertRaises(ValueError):
                sweep.parse_mode_csv(value)
        with self.assertRaises(ValueError):
            sweep.parse_int_csv(",")

    def test_hot_cache_reset_and_disabled_lookup(self):
        cache = runtime.RosaHotAddressCache(num_layers=1, max_entries_per_layer=2)
        ids = torch.tensor([[1, 2]])
        def fetch(values):
            return values.float().unsqueeze(-1)
        first, _ = cache.lookup(0, ids, valid_mask=None, value_dim=1, fetch_fn=fetch)
        second, stats = cache.lookup(0, ids, valid_mask=None, value_dim=1, fetch_fn=fetch)
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(stats["token_hits"], 0)
        cache.reset_stats(clear_cache=True)
        _, stats = cache.lookup(0, ids, valid_mask=None, value_dim=1, fetch_fn=fetch)
        self.assertEqual(stats["token_hits"], 0)
        cache.reset()
        with self.assertRaises(IndexError):
            cache.lookup(1, ids, valid_mask=None, value_dim=1, fetch_fn=fetch)
        disabled = runtime.RosaHotAddressCache(num_layers=1, max_entries_per_layer=0)
        values, stats = disabled.lookup(0, ids, valid_mask=None, value_dim=1, fetch_fn=fetch)
        self.assertTrue(torch.equal(values, fetch(ids)))
        self.assertEqual(stats, {})


if __name__ == "__main__":
    unittest.main()
