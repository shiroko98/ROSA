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


class OnlineInjectionLoopTests(unittest.TestCase):
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

    def test_online_batch_state_prefill_then_decode_matches_memory_mode(self):
        rosa_mod.set_seed(2026)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
        )
        model.eval()

        memory_ids = torch.tensor([[1, 2, 1]], dtype=torch.long)
        input_ids = torch.tensor([[2]], dtype=torch.long)

        with torch.no_grad():
            ref = model(input_ids=input_ids, rosa_memory_ids=memory_ids)

            online_state = model.init_online_state(batch_size=1)
            online_state.prefill(memory_ids, pad_id=0)
            online = model.forward_online(input_ids=input_ids, rosa_online_state=online_state)

        self.assertTrue(torch.allclose(ref["logits"], online["logits"], atol=1e-6))
        self.assertAlmostEqual(ref["rosa_fire_coverage"], online["rosa_fire_coverage"], places=6)
        self.assertAlmostEqual(ref["rosa_fired_avg_match_len"], online["rosa_fired_avg_match_len"], places=6)
        self.assertAlmostEqual(ref["rosa_raw_match_coverage"], online["rosa_raw_match_coverage"], places=6)
        self.assertAlmostEqual(ref["rosa_raw_avg_best_len"], online["rosa_raw_avg_best_len"], places=6)

        snap = online_state.snapshot()[0]
        self.assertEqual(snap.token_ids, (1, 2, 1, 2))
        self.assertIsNotNone(snap.last_address)
        self.assertEqual(snap.last_address.addr_id, 1)
        self.assertEqual(snap.last_address.fired_match_len, 2)

    def test_online_forward_can_process_prompt_sequence_from_empty_state(self):
        rosa_mod.set_seed(2027)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
        )
        model.eval()

        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        empty_memory = torch.empty((1, 0), dtype=torch.long)

        with torch.no_grad():
            ref = model(input_ids=input_ids, rosa_memory_ids=empty_memory)

            online_state = model.init_online_state(batch_size=1)
            online = model.forward_online(input_ids=input_ids, rosa_online_state=online_state)

        self.assertTrue(torch.allclose(ref["logits"], online["logits"], atol=1e-6))
        snap = online_state.snapshot()[0]
        self.assertEqual(snap.token_ids, (1, 2, 1, 2, 3))
        self.assertEqual(snap.last_address.raw_match_len, 0)

    def test_online_decode_state_keeps_advancing_across_steps(self):
        rosa_mod.set_seed(2028)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
        )
        model.eval()

        online_state = model.init_online_state(batch_size=1)
        online_state.prefill(torch.tensor([[1, 2, 1]], dtype=torch.long), pad_id=0)

        with torch.no_grad():
            model.forward_online(torch.tensor([[2]], dtype=torch.long), rosa_online_state=online_state)
            model.forward_online(torch.tensor([[3]], dtype=torch.long), rosa_online_state=online_state)

        snap = online_state.snapshot()[0]
        self.assertEqual(snap.token_ids, (1, 2, 1, 2, 3))
        self.assertIsNotNone(snap.last_address)
        self.assertEqual(snap.last_address.addr_id, -1)
        self.assertEqual(snap.last_address.raw_match_len, 0)

    def test_prepared_payload_matches_direct_forward(self):
        rosa_mod.set_seed(2029)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
            use_context_gate=True,
        )
        model.eval()

        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        memory_ids = torch.empty((1, 0), dtype=torch.long)

        with torch.no_grad():
            direct = model(input_ids=input_ids, rosa_memory_ids=memory_ids)
            address_batch = model.compute_rosa_address_batch(
                input_ids,
                rosa_memory_ids=memory_ids,
            )
            payload = model.build_rosa_injection_payload(address_batch, device=input_ids.device)
            staged = model(input_ids=input_ids, rosa_payload=payload)

        self.assertTrue(torch.allclose(direct["logits"], staged["logits"], atol=1e-6))
        self.assertAlmostEqual(direct["rosa_fire_coverage"], staged["rosa_fire_coverage"], places=6)
        self.assertAlmostEqual(direct["rosa_avg_gate"], staged["rosa_avg_gate"], places=6)

    def test_online_payload_path_matches_forward_online(self):
        rosa_mod.set_seed(2030)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
            use_context_gate=True,
        )
        model.eval()

        online_state_a = model.init_online_state(batch_size=1)
        online_state_b = model.init_online_state(batch_size=1)
        prefill = torch.tensor([[1, 2, 1]], dtype=torch.long)
        online_state_a.prefill(prefill, pad_id=0)
        online_state_b.prefill(prefill, pad_id=0)
        input_ids = torch.tensor([[2]], dtype=torch.long)

        with torch.no_grad():
            direct = model.forward_online(input_ids=input_ids, rosa_online_state=online_state_a)
            payload = model.prepare_rosa_injection_payload(
                input_ids,
                rosa_online_state=online_state_b,
            )
            staged = model(input_ids=input_ids, rosa_payload=payload)

        self.assertTrue(torch.allclose(direct["logits"], staged["logits"], atol=1e-6))
        self.assertAlmostEqual(direct["rosa_fire_coverage"], staged["rosa_fire_coverage"], places=6)
        self.assertAlmostEqual(direct["rosa_avg_gate"], staged["rosa_avg_gate"], places=6)

    def test_prefetcher_can_stage_and_consume_payload(self):
        rosa_mod.set_seed(2031)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
            rosa_value_mode="per_layer",
            use_context_gate=True,
        )
        model.eval()

        online_state = model.init_online_state(batch_size=1)
        online_state.prefill(torch.tensor([[1, 2, 1]], dtype=torch.long), pad_id=0)
        prefetch_state = model.init_online_state(batch_size=1)
        prefetch_state.prefill(torch.tensor([[1, 2, 1]], dtype=torch.long), pad_id=0)
        input_ids = torch.tensor([[2]], dtype=torch.long)
        prefetcher = model.init_prefetcher(use_async=True, max_workers=1)

        with torch.no_grad():
            direct = model.forward_online(input_ids=input_ids, rosa_online_state=online_state)
            address_batch = model.schedule_rosa_prefetch(
                prefetcher,
                "step-0",
                input_ids,
                rosa_online_state=prefetch_state,
            )
            payload = model.consume_rosa_prefetch(
                prefetcher,
                "step-0",
                device=input_ids.device,
                fallback_address_batch=address_batch,
            )
            staged = model.forward_prefetched(input_ids=input_ids, rosa_payload=payload)

        stats = prefetcher.stats()
        prefetcher.shutdown()

        self.assertTrue(torch.allclose(direct["logits"], staged["logits"], atol=1e-6))
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["staged_payloads"], 1)

    def test_prefetcher_sync_fallback_still_returns_payload(self):
        rosa_mod.set_seed(2032)
        model = rosa_mod.RosaFusedLM(
            self.cfg,
            pad_id=0,
            min_match_len=2,
            inject_layers=1,
            rosa_scale=0.5,
        )
        model.eval()

        input_ids = torch.tensor([[1, 2, 1, 2, 3]], dtype=torch.long)
        prefetcher = model.init_prefetcher(use_async=False)

        with torch.no_grad():
            address_batch = model.schedule_rosa_prefetch(
                prefetcher,
                "prefill-0",
                input_ids,
                rosa_memory_ids=torch.empty((1, 0), dtype=torch.long),
            )
            payload = model.consume_rosa_prefetch(
                prefetcher,
                "prefill-0",
                device=input_ids.device,
                fallback_address_batch=address_batch,
            )
            staged = model.forward_prefetched(input_ids=input_ids, rosa_payload=payload)
            direct = model(input_ids=input_ids, rosa_memory_ids=torch.empty((1, 0), dtype=torch.long))

        stats = prefetcher.stats()
        prefetcher.shutdown()

        self.assertTrue(torch.allclose(direct["logits"], staged["logits"], atol=1e-6))
        self.assertEqual(stats["sync_fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
