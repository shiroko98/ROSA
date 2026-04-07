import unittest

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod


class AddressInterfaceTests(unittest.TestCase):
    def test_rosa_addressing_matches_across_naive_and_sam(self):
        input_ids = torch.tensor(
            [
                [1, 2, 1, 2, 3],
                [7, 8, 7, 8, 9],
            ],
            dtype=torch.long,
        )
        memory_ids = torch.tensor(
            [
                [4, 5, 1, 2, 0, 0],
                [6, 7, 8, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        naive = rosa_mod.rosa_addressing_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=2,
            pad_id=0,
            special_ids={9},
            forbid_special_target=True,
            backend="naive",
        )
        sam = rosa_mod.rosa_addressing_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=2,
            pad_id=0,
            special_ids={9},
            forbid_special_target=True,
            backend="sam",
        )

        self.assertEqual(set(naive.keys()), set(sam.keys()))
        for key in naive.keys():
            self.assertTrue(torch.equal(naive[key], sam[key]), msg=f"mismatch at {key}")


class OnlineRosaStateTests(unittest.TestCase):
    def test_prefill_returns_expected_stepwise_meta(self):
        state = rosa_mod.OnlineRosaState(min_match_len=2)

        metas = state.prefill([1, 2, 1, 2, 3])

        self.assertEqual([m.addr_id for m in metas], [-1, -1, -1, 1, -1])
        self.assertEqual([m.raw_match_len for m in metas], [0, 0, 1, 2, 0])
        self.assertEqual([m.fired_match_len for m in metas], [0, 0, 0, 2, 0])
        self.assertEqual([m.hit_flag for m in metas], [False, False, False, True, False])

    def test_online_state_matches_sam_batch_with_memory(self):
        state = rosa_mod.OnlineRosaState(min_match_len=2)
        state.prefill([4, 5, 1, 2])
        metas = [state.update_one(token_id) for token_id in [1, 2, 3]]
        batch_metas = rosa_mod.online_rosa_address_meta_with_memory(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            memory_ids=torch.tensor([[4, 5, 1, 2]], dtype=torch.long),
            min_match_len=2,
            pad_id=0,
            special_ids=set(),
            forbid_special_target=True,
        )[0]

        expected = rosa_mod.sam_rosa_address_meta_with_memory(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            memory_ids=torch.tensor([[4, 5, 1, 2]], dtype=torch.long),
            min_match_len=2,
            pad_id=0,
            special_ids=set(),
            forbid_special_target=True,
        )[0]

        self.assertEqual(
            [(m.addr_id, m.raw_match_len, m.fired_match_len, m.valid_mask) for m in metas],
            [(m.addr_id, m.raw_match_len, m.fired_match_len, m.valid_mask) for m in expected],
        )
        self.assertEqual(
            [(m.addr_id, m.raw_match_len, m.fired_match_len, m.valid_mask) for m in batch_metas],
            [(m.addr_id, m.raw_match_len, m.fired_match_len, m.valid_mask) for m in expected],
        )

    def test_special_target_blocks_injection_but_preserves_raw_match(self):
        state = rosa_mod.OnlineRosaState(min_match_len=1, special_ids={5}, forbid_special_target=True)
        state.prefill([7, 5])

        meta = state.update_one(7)

        self.assertEqual(meta.addr_id, -1)
        self.assertEqual(meta.raw_match_len, 1)
        self.assertEqual(meta.fired_match_len, 0)
        self.assertFalse(meta.valid_mask)
        self.assertTrue(meta.special_mask)

    def test_reset_and_snapshot_clear_history(self):
        state = rosa_mod.OnlineRosaState(min_match_len=1)
        state.prefill([1, 2, 1])
        snap_before = state.snapshot()

        self.assertEqual(snap_before.token_ids, (1, 2, 1))
        self.assertEqual(snap_before.num_tokens, 3)
        self.assertIsNotNone(snap_before.last_address)

        state.reset()
        snap_after = state.snapshot()

        self.assertEqual(snap_after.token_ids, ())
        self.assertEqual(snap_after.num_tokens, 0)
        self.assertIsNone(snap_after.last_address)


if __name__ == "__main__":
    unittest.main()
