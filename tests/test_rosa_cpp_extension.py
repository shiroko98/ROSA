import unittest

import torch

import rosa_addressing as addressing_mod
from rosa_cpp_extension import load_rosa_sam_cpu_extension


class RosaCompiledCpuExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ext = load_rosa_sam_cpu_extension()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"编译型 CPU 扩展不可用: {exc}") from exc

    def test_compiled_cpu_predict_matches_python(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4]
        py_pred, py_match = addressing_mod.sam_rosa_predict(seq, min_match_len=2)
        cpp_pred, cpp_match = addressing_mod.sam_rosa_predict_compiled_cpu(seq, min_match_len=2)

        self.assertEqual(cpp_pred, py_pred)
        self.assertEqual(cpp_match, py_match)

    def test_compiled_cpu_engine_matches_fast_engine(self):
        input_ids = torch.tensor(
            [
                [1, 2, 3, 1, 2],
                [4, 5, 4, 5, 6],
            ],
            dtype=torch.long,
        )
        memory_ids = torch.tensor(
            [
                [7, 1, 2, 0, 0],
                [8, 4, 5, 0, 0],
            ],
            dtype=torch.long,
        )

        fast_engine = addressing_mod.RosaAddressEngine(
            min_match_len=2,
            pad_id=0,
            backend="sam",
            sequence_mode="online_sam",
            online_sam_impl="fast",
            special_ids={9},
            forbid_special_target=True,
        )
        compiled_engine = addressing_mod.RosaAddressEngine(
            min_match_len=2,
            pad_id=0,
            backend="sam",
            sequence_mode="online_sam",
            online_sam_impl="compiled_cpu",
            special_ids={9},
            forbid_special_target=True,
        )

        fast = fast_engine.forward_seq(input_ids, memory_ids)
        compiled = compiled_engine.forward_seq(input_ids, memory_ids)

        self.assertTrue(torch.equal(compiled.addr_ids, fast.addr_ids))
        self.assertTrue(torch.equal(compiled.raw_match_lens, fast.raw_match_lens))
        self.assertTrue(torch.equal(compiled.fired_match_lens, fast.fired_match_lens))
        self.assertTrue(torch.equal(compiled.valid_mask, fast.valid_mask))
        self.assertTrue(torch.equal(compiled.special_mask, fast.special_mask))


if __name__ == "__main__":
    unittest.main()
