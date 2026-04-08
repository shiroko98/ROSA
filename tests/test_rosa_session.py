import unittest

import torch

import train_qwen_llama_vs_rosa_v2 as rosa_mod


class RosaBatchSessionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = rosa_mod.ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            dim=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            intermediate_size=32,
        )

    def test_prefill_seq_matches_forward_online_and_tracks_snapshot(self):
        rosa_mod.set_seed(3031)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_seq_address_mode="online_sam",
        )
        model.eval()
        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)

        with torch.no_grad():
            reference_state = model.init_online_state(batch_size=1)
            reference = model.forward_online(input_ids=input_ids, rosa_online_state=reference_state)

            session = model.init_online_session(batch_size=1)
            out = session.prefill_seq(input_ids)

        self.assertTrue(torch.allclose(reference["logits"], out["logits"], atol=1e-6))
        snap = session.snapshot()
        self.assertEqual(snap.batch_size, 1)
        self.assertEqual(snap.total_prefill_tokens, 5)
        self.assertEqual(snap.total_decode_tokens, 0)
        self.assertEqual(snap.decode_calls, 0)
        self.assertEqual(snap.state_snapshots[0].token_ids, (1, 2, 1, 2, 3))
        self.assertEqual(snap.last_address_source, "step:online_sam")

    def test_prefill_then_decode_step_matches_memory_reference_and_updates_counters(self):
        rosa_mod.set_seed(3032)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_seq_address_mode="online_sam",
        )
        model.eval()

        prefill = torch.tensor([[1, 2, 1]], dtype=torch.long)
        decode_step = torch.tensor([[2]], dtype=torch.long)
        memory_ids = torch.tensor([[1, 2, 1]], dtype=torch.long)

        with torch.no_grad():
            reference = model(input_ids=decode_step, rosa_memory_ids=memory_ids)

            session = model.init_online_session(batch_size=1)
            session.prefill_seq(prefill)
            out = session.decode_step(decode_step)

        self.assertTrue(torch.allclose(reference["logits"], out["logits"], atol=1e-6))
        snap = session.snapshot()
        self.assertEqual(snap.total_prefill_tokens, 3)
        self.assertEqual(snap.total_decode_tokens, 1)
        self.assertEqual(snap.decode_calls, 1)
        self.assertEqual(snap.state_snapshots[0].token_ids, (1, 2, 1, 2))

    def test_reset_and_close_manage_session_lifecycle(self):
        rosa_mod.set_seed(3033)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_seq_address_mode="online_sam",
        )
        model.eval()

        session = model.init_online_session(batch_size=1)
        session.prefill_seq(torch.tensor([[1, 2, 1]], dtype=torch.long))
        session.reset()

        snap = session.snapshot()
        self.assertEqual(snap.total_prefill_tokens, 0)
        self.assertEqual(snap.total_decode_tokens, 0)
        self.assertEqual(snap.decode_calls, 0)
        self.assertEqual(snap.state_snapshots[0].token_ids, ())

        session.close()
        with self.assertRaises(RuntimeError):
            session.decode_step(torch.tensor([[2]], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
