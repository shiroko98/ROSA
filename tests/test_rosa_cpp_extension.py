import unittest
import random

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

    def test_bulk_pack_preserves_ragged_empty_defaults_and_dtypes(self):
        a = addressing_mod.build_address_meta(5, 2, min_match_len=1)
        b = addressing_mod.build_address_meta(7, 3, min_match_len=1, special_mask=True)
        packed = addressing_mod.pack_address_meta_batch([[a, b], [], [a]])
        self.assertEqual(packed['addr_ids'].tolist(), [[5, -1], [-1, -1], [5, -1]])
        self.assertEqual(packed['raw_match_lens'].tolist(), [[2, 3], [0, 0], [2, 0]])
        self.assertEqual(packed['fired_match_lens'].tolist(), [[2, 0], [0, 0], [2, 0]])
        self.assertEqual(packed['special_mask'].tolist(), [[False, True], [False, False], [False, False]])
        self.assertEqual(packed['valid_mask'].dtype, torch.bool)
        self.assertEqual(packed['addr_ids'].dtype, torch.long)
        for rows, shape in (([], (0, 0)), ([[], []], (2, 0))):
            for value in addressing_mod.pack_address_meta_batch(rows).values():
                self.assertEqual(tuple(value.shape), shape)

    def test_compiled_cpu_predict_matches_python(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4]
        py_pred, py_match = addressing_mod.sam_rosa_predict(seq, min_match_len=2)
        cpp_pred, cpp_match = addressing_mod.sam_rosa_predict_compiled_cpu(seq, min_match_len=2)

        self.assertEqual(cpp_pred, py_pred)
        self.assertEqual(cpp_match, py_match)

    def test_latest_endpos_matches_eager_on_random_and_adversarial_sequences(self):
        rng = random.Random(73021)
        cases = [[], [1], [0]*1024, [1, 2]*512,
                 [1]*128+[2]+[1]*256+[2, 3]+[1]*256,
                 list(range(256))*3]
        for alphabet in (2, 3, 7, 64, 65536):
            cases.extend([[rng.randrange(alphabet) for _ in range(rng.randrange(1, 500))]
                          for _ in range(50)])
        for seq in cases:
            for minimum in (1, 2, 8, len(seq)+1):
                self.assertEqual(self.ext.sam_rosa_predict(seq, minimum),
                                 addressing_mod.sam_rosa_predict(seq, min_match_len=minimum))
        self.assertEqual(self.ext.algorithm_version, 'suffix_link_lct_v1')

    def test_all_fields_with_memory_specials_padding_and_causal_prefix(self):
        rng = random.Random(77)
        fields = ('addr_ids', 'raw_match_lens', 'fired_match_lens', 'valid_mask', 'special_mask')
        for forbidden in (True, False):
            for minimum in (1, 3, 19):
                engines = [addressing_mod.RosaAddressEngine(min_match_len=minimum,
                    pad_id=0, backend='sam', sequence_mode='online_sam',
                    online_sam_impl=impl, special_ids={0, 7}, forbid_special_target=forbidden)
                    for impl in ('fast', 'compiled_cpu')]
                ids = torch.tensor([[rng.randrange(8) for _ in range(250)]+[0]*128 for _ in range(3)])
                memory = torch.tensor([[1, 2, 0, 7, 0], [0]*5, [7, 7, 2, 3, 0]])
                reference, actual = [e.forward_seq(ids, memory) for e in engines]
                prefix = engines[1].forward_seq(ids[:, :127], memory)
                for field in fields:
                    self.assertTrue(torch.equal(getattr(reference, field), getattr(actual, field)), field)
                    self.assertTrue(torch.equal(getattr(actual, field)[:, :127], getattr(prefix, field)), field)

    def test_long_identical_tokens_have_exact_matches_without_suffix_walk(self):
        # Closed-form oracle is linear; don't run the old quadratic reference.
        n = 131072
        pred, lengths = self.ext.sam_rosa_predict([5]*n, 1)
        self.assertEqual(pred, [-1]+[5]*(n-1))
        self.assertEqual(lengths, list(range(n)))

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
