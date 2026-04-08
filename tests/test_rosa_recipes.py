import unittest

import rosa_recipes as recipe_mod
import train_qwen_llama_vs_rosa_v2 as rosa_mod


class RosaRecipeTests(unittest.TestCase):
    def test_online_v1_recipe_applies_expected_defaults(self):
        parser = rosa_mod.build_arg_parser()
        args = parser.parse_args(
            [
                "--data_path",
                "data/rosa_demo.txt",
                "--rosa_recipe",
                "online_v1",
            ]
        )

        recipe_meta = rosa_mod.apply_rosa_recipe(args)

        self.assertTrue(recipe_meta["applied"])
        self.assertEqual(args.rosa_recipe, "online_v1")
        self.assertEqual(args.rosa_train_mode, "online_seq")
        self.assertEqual(args.rosa_memory_mode, "doc_local")
        self.assertEqual(args.rosa_backend, "sam")
        self.assertEqual(args.rosa_seq_address_mode, "online_sam")
        self.assertEqual(args.rosa_value_mode, "shared")
        self.assertTrue(args.rosa_context_gate)
        self.assertFalse(args.rosa_disable_match_len_gate)
        self.assertEqual(args.rosa_inject_layers, 1)
        self.assertEqual(args.rosa_inject_layer_ids, "0")
        self.assertEqual(args.rosa_min_match_len, 1)
        self.assertAlmostEqual(args.rosa_scale, 0.15)

    def test_online_v2_recipe_applies_expected_defaults(self):
        parser = rosa_mod.build_arg_parser()
        args = parser.parse_args(
            [
                "--data_path",
                "data/rosa_demo.txt",
                "--rosa_recipe",
                "online_v2",
            ]
        )

        recipe_meta = rosa_mod.apply_rosa_recipe(args)

        self.assertTrue(recipe_meta["applied"])
        self.assertEqual(args.rosa_recipe, "online_v2")
        self.assertEqual(args.rosa_train_mode, "online_seq")
        self.assertEqual(args.rosa_memory_mode, "doc_local")
        self.assertEqual(args.rosa_backend, "sam")
        self.assertEqual(args.rosa_seq_address_mode, "online_sam")
        self.assertEqual(args.rosa_value_mode, "per_layer")
        self.assertTrue(args.rosa_context_gate)
        self.assertFalse(args.rosa_disable_match_len_gate)
        self.assertEqual(args.rosa_inject_layers, 1)
        self.assertEqual(args.rosa_inject_layer_ids, "0")
        self.assertEqual(args.rosa_min_match_len, 1)
        self.assertAlmostEqual(args.rosa_scale, 0.15)

    def test_custom_recipe_keeps_existing_values(self):
        parser = rosa_mod.build_arg_parser()
        args = parser.parse_args(
            [
                "--data_path",
                "data/rosa_demo.txt",
                "--rosa_recipe",
                "custom",
                "--rosa_train_mode",
                "reference_precompute",
                "--rosa_memory_mode",
                "global_train",
                "--rosa_seq_address_mode",
                "reference_backend",
                "--rosa_value_mode",
                "per_layer",
                "--rosa_inject_layers",
                "2",
                "--rosa_inject_layer_ids",
                "1,2",
            ]
        )

        recipe_meta = rosa_mod.apply_rosa_recipe(args)

        self.assertFalse(recipe_meta["applied"])
        self.assertEqual(args.rosa_train_mode, "reference_precompute")
        self.assertEqual(args.rosa_memory_mode, "global_train")
        self.assertEqual(args.rosa_seq_address_mode, "reference_backend")
        self.assertEqual(args.rosa_value_mode, "per_layer")
        self.assertEqual(args.rosa_inject_layers, 2)
        self.assertEqual(args.rosa_inject_layer_ids, "1,2")

    def test_recipe_registry_lists_online_v1(self):
        names = recipe_mod.available_rosa_recipe_names()
        self.assertIn("custom", names)
        self.assertIn("online_v1", names)
        self.assertIn("online_v2", names)


if __name__ == "__main__":
    unittest.main()
