import unittest

import torch

import rosa_runtime as runtime_mod


class RosaHotAddressCacheTests(unittest.TestCase):
    def test_repeated_lookup_turns_into_hits(self):
        cache = runtime_mod.RosaHotAddressCache(num_layers=1, max_entries_per_layer=4)
        addr_ids = torch.tensor([[3, 5, 3, 0]], dtype=torch.long)
        valid_mask = torch.tensor([[True, True, True, False]])

        def fetch_fn(ids: torch.Tensor) -> torch.Tensor:
            return ids.float().unsqueeze(-1).repeat(1, 2)

        values_a, stats_a = cache.lookup(
            0,
            addr_ids,
            valid_mask=valid_mask,
            value_dim=2,
            fetch_fn=fetch_fn,
        )
        values_b, stats_b = cache.lookup(
            0,
            addr_ids,
            valid_mask=valid_mask,
            value_dim=2,
            fetch_fn=fetch_fn,
        )

        self.assertTrue(torch.equal(values_a, values_b))
        self.assertEqual(stats_a["token_hit_rate"], 0.0)
        self.assertEqual(stats_a["unique_hit_rate"], 0.0)
        self.assertEqual(stats_b["token_hit_rate"], 1.0)
        self.assertEqual(stats_b["unique_hit_rate"], 1.0)

        report = cache.stats(top_k=2)
        self.assertTrue(report["enabled"])
        self.assertGreater(report["token_requests"], 0.0)
        self.assertGreater(report["token_hit_rate"], 0.0)
        self.assertEqual(report["layer_stats"][0]["top_addresses"][0]["addr_id"], 3)

    def test_lru_eviction_keeps_cache_bounded(self):
        cache = runtime_mod.RosaHotAddressCache(num_layers=1, max_entries_per_layer=1)

        def fetch_fn(ids: torch.Tensor) -> torch.Tensor:
            return ids.float().unsqueeze(-1)

        cache.lookup(
            0,
            torch.tensor([[1]], dtype=torch.long),
            valid_mask=torch.tensor([[True]]),
            value_dim=1,
            fetch_fn=fetch_fn,
        )
        _, stats = cache.lookup(
            0,
            torch.tensor([[2]], dtype=torch.long),
            valid_mask=torch.tensor([[True]]),
            value_dim=1,
            fetch_fn=fetch_fn,
        )

        report = cache.stats(top_k=2)
        self.assertEqual(stats["evictions"], 1.0)
        self.assertEqual(report["active_entries"], 1)
        self.assertEqual(report["layer_stats"][0]["active_entries"], 1)


class RosaPayloadStageTests(unittest.TestCase):
    def test_stage_payload_preserves_scalar_stats(self):
        payload = runtime_mod.RosaInjectionPayload(
            address=runtime_mod.RosaAddressBatch(
                addr_ids=torch.tensor([[1]], dtype=torch.long),
                raw_match_lens=torch.tensor([[2]], dtype=torch.long),
                fired_match_lens=torch.tensor([[2]], dtype=torch.long),
                valid_mask=torch.tensor([[True]]),
                special_mask=torch.tensor([[False]]),
                source="unit",
            ),
            layer_values=(torch.ones(1, 1, 3),),
            source="unit",
            stats={"rosa_hot_cache_token_hit_rate": 0.5},
        )

        staged = runtime_mod.stage_rosa_payload(payload, use_pinned_memory=False)

        self.assertEqual(staged.stats["rosa_hot_cache_token_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
