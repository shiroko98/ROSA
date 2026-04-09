import unittest

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod
from rosa_addressing import SuffixAutomatonRosaState
from rosa_training_snapshot import build_sequence_online_state_snapshots


class RosaTrainingSnapshotTests(unittest.TestCase):
    def test_online_sam_state_restores_from_snapshot(self):
        state = SuffixAutomatonRosaState(min_match_len=2)
        state.prefill([1, 2, 1, 2])
        snapshot = state.snapshot()

        restored = SuffixAutomatonRosaState.from_snapshot(
            snapshot,
            min_match_len=2,
        )

        self.assertEqual(restored.snapshot().token_ids, snapshot.token_ids)
        self.assertEqual(restored.snapshot().backend_name, "sam")
        self.assertEqual(restored.update_one(3), state.update_one(3))

    def test_forward_seq_from_snapshots_matches_direct_online_sam_with_replay(self):
        docs = [[1, 2, 1, 2, 3, 1]]
        snapshots = build_sequence_online_state_snapshots(
            docs,
            sequence_mode="online_sam",
            snapshot_interval=2,
            min_match_len=2,
            special_ids=set(),
            forbid_special_target=True,
        )
        engine = rosa_mod.RosaAddressEngine(
            min_match_len=2,
            pad_id=0,
            backend="sam",
            sequence_mode="online_sam",
        )
        input_ids = torch.tensor([[3, 1]], dtype=torch.long)
        replay_ids = torch.tensor([[1, 2]], dtype=torch.long)
        state_snapshot = [snapshots[0]["snapshots"][1]]

        restored = engine.forward_seq_from_snapshots(
            input_ids,
            state_snapshots=state_snapshot,
            replay_ids=replay_ids,
        )
        direct = engine.forward_seq(
            input_ids=input_ids,
            memory_ids=torch.tensor([[1, 2, 1, 2]], dtype=torch.long),
        )

        self.assertEqual(restored.source, "seq:online_sam:snapshot")
        self.assertTrue(torch.equal(restored.addr_ids, direct.addr_ids))
        self.assertTrue(torch.equal(restored.raw_match_lens, direct.raw_match_lens))
        self.assertTrue(torch.equal(restored.fired_match_lens, direct.fired_match_lens))
        self.assertTrue(torch.equal(restored.valid_mask, direct.valid_mask))
        self.assertTrue(torch.equal(restored.special_mask, direct.special_mask))

    def test_forward_seq_from_snapshots_matches_direct_online_exact(self):
        docs = [[4, 5, 4, 5, 6]]
        snapshots = build_sequence_online_state_snapshots(
            docs,
            sequence_mode="online_exact",
            snapshot_interval=2,
            min_match_len=2,
            special_ids=set(),
            forbid_special_target=True,
        )
        engine = rosa_mod.RosaAddressEngine(
            min_match_len=2,
            pad_id=0,
            backend="sam",
            sequence_mode="online_exact",
        )
        input_ids = torch.tensor([[4, 5]], dtype=torch.long)
        replay_ids = torch.tensor([[4]], dtype=torch.long)
        state_snapshot = [snapshots[0]["snapshots"][1]]

        restored = engine.forward_seq_from_snapshots(
            input_ids,
            state_snapshots=state_snapshot,
            replay_ids=replay_ids,
        )
        direct = engine.forward_seq(
            input_ids=input_ids,
            memory_ids=torch.tensor([[4, 5, 4]], dtype=torch.long),
        )

        self.assertEqual(restored.source, "seq:online_exact:snapshot")
        self.assertTrue(torch.equal(restored.addr_ids, direct.addr_ids))
        self.assertTrue(torch.equal(restored.raw_match_lens, direct.raw_match_lens))
        self.assertTrue(torch.equal(restored.fired_match_lens, direct.fired_match_lens))
        self.assertTrue(torch.equal(restored.valid_mask, direct.valid_mask))
        self.assertTrue(torch.equal(restored.special_mask, direct.special_mask))


if __name__ == "__main__":
    unittest.main()
