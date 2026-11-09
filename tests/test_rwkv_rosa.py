"""Real upstream RWKV backbone, causal ROSA fusion and paired experiment regression."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from rosa_optim import build_training_optimizers
from rwkv_rosa.backbone import RWKVConfig, build_backbone, infer_config, load_base_state, load_runtime
from rwkv_rosa.data import (ByteTokenizer, EqualHistoryDataset, build_tokenizer, collate,
                            load_documents, split_documents, validate_splits)
from rwkv_rosa.experiment import (clip_all_gradients, finish_metrics, main, parse_args, restore_checkpoint,
                                  score_hidden, unpack_batch)
from rwkv_rosa.model import RWKVRosaLM, configure_train_scope, RWKVRosaState
from third_party.rwkv_lm_v7.sft_binidx import EncodedDocument


class RWKVIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = RWKVConfig(vocab_size=24, n_embd=16, n_layer=2, head_size=8,
                              d_decay_lora=8, d_aaa_lora=8, d_mv_lora=8, d_gate_lora=8)
        self.base = build_backbone(self.cfg).state_dict()
        self.x = torch.tensor([[1, 2, 1, 2, 3, 1]])

    def model(self, variant="improved", **kwargs):
        return RWKVRosaLM(self.cfg, variant=variant, base_state=self.base, **kwargs)

    def test_checkpoint_dimensions_and_strict_load(self):
        self.assertEqual(infer_config(self.base), self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.pth"
            torch.save({"state_dict": {"module." + k: v for k, v in self.base.items()}}, path)
            loaded = load_base_state(path)
            self.assertTrue(torch.equal(loaded["emb.weight"], self.base["emb.weight"]))
            del loaded  # Release mmap before overwriting/removing the file (Windows).
            torch.save({"optimizer": 1}, path)
            with self.assertRaises(ValueError):
                load_base_state(path)
            torch.save({"unrelated": torch.ones(1)}, path)
            with self.assertRaises(ValueError):
                load_base_state(path)
        with self.assertRaises(RuntimeError):
            build_backbone(self.cfg, state_dict={k: v for k, v in self.base.items() if k != "head.weight"})
        with self.assertRaises(ValueError):
            infer_config({"emb.weight": torch.zeros(5, 8)})
        bad = dict(self.base)
        bad.pop("blocks.0.att.w1")
        with self.assertRaises(ValueError):
            infer_config(bad)
        bad = dict(self.base, **{"blocks.0.ffn.key.weight": torch.zeros(17, 16)})
        with self.assertRaises(ValueError):
            infer_config(bad)

    def test_baseline_exactly_matches_unmodified_upstream(self):
        runtime = load_runtime(self.cfg, torch.float32)
        runtime.RWKV7_OP = runtime._unfenced_wkv_op
        upstream = runtime.RWKV(
            type("Args", (), {"n_embd": 16, "vocab_size": 24, "n_layer": 2, "head_size_a": 8})())
        upstream.load_state_dict(self.base, strict=True)
        with torch.no_grad():
            expected = upstream(self.x)
            actual = self.model("baseline")(self.x)["logits"]
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_base_initialization_identical_and_not_aliased(self):
        models = [self.model(v) for v in ("baseline", "original", "improved")]
        for model in models:
            for key, value in model.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, self.base[key]), key)
        before = self.base["emb.weight"].clone()
        with torch.no_grad():
            models[0].backbone.emb.weight.add_(1)
        self.assertTrue(torch.equal(self.base["emb.weight"], before))
        self.assertTrue(torch.equal(models[1].backbone.emb.weight, before))

    def test_zero_scale_is_baseline(self):
        reference = self.model("baseline")(self.x)["logits"]
        for variant in ("original", "improved"):
            torch.testing.assert_close(self.model(variant, scale=0)(self.x)["logits"], reference, atol=0, rtol=0)

    def test_online_prefill_decode_and_snapshot_parity(self):
        for variant in ("baseline", "original", "improved"):
            model = self.model(variant, inject_layers=(0, 1)).eval()
            full = model(self.x)["logits"]
            first = model(self.x[:, :3], return_state=True)
            state = first["state"]
            if state.rosa is not None:
                restored = model.address_engine.init_state_from_snapshots(state.rosa.snapshot())
                state = RWKVRosaState(state.rwkv, restored, 1, state.tokens_seen)
            second = model(self.x[:, 3:], state=state)
            torch.testing.assert_close(torch.cat((first["logits"], second["logits"]), 1), full,
                                       rtol=2e-5, atol=2e-6)
            self.assertEqual(second["state"].tokens_seen, 6)

    def test_no_future_leakage_and_independent_batch_state(self):
        for variant in ("baseline", "original", "improved"):
            model = self.model(variant)
            changed = self.x.clone()
            changed[:, -2:] = torch.tensor([[9, 10]])
            torch.testing.assert_close(model(self.x)["logits"][:, :4], model(changed)["logits"][:, :4])
            batch = torch.cat((self.x, changed))
            actual = model(batch, return_state=True)["logits"]
            for i in range(2):
                torch.testing.assert_close(actual[i:i+1], model(batch[i:i+1])["logits"], rtol=2e-5, atol=2e-6)

    def test_per_layer_values_gate_gradients_sparse_optimizer(self):
        model = self.model(inject_layers=(0, 1), sparse_values=True, value_shards=3)
        out = model(self.x)
        out["logits"].square().mean().backward()
        self.assertIsNotNone(model.gate_bias.grad)
        self.assertTrue(model.rosa_value_store.is_sharded)
        self.assertTrue(all(p.grad.is_sparse for p in model.rosa_value_store.sparse_parameters() if p.grad is not None))
        optim = build_training_optimizers(model, lr=1e-3, weight_decay=0.01)
        self.assertIsNotNone(optim.sparse_optimizer)
        optim.step()
        self.assertGreater(out["stats"]["rosa_coverage"], 0)
        self.assertGreater(out["stats"]["rosa_avg_gate"], 0)
        tables = model.rosa_value_store.per_layer_tables
        self.assertIsNot(tables[0], tables[1])

    def test_capacity_adapter_is_active_and_starts_as_baseline(self):
        model = self.model("capacity", capacity_width=32)
        reference = self.model("baseline")(self.x)["logits"]
        torch.testing.assert_close(model(self.x)["logits"], reference)
        model(self.x)["logits"].square().mean().backward()
        self.assertGreater(float(model.adapter_up[0].weight.grad.abs().sum()), 0)

    def test_training_scope_and_checkpointing(self):
        model = self.model(activation_checkpointing=True)
        scope = configure_train_scope(model, "head")
        self.assertEqual(scope["backbone_trainable_names"], ["head.weight"])
        self.assertTrue(model.gate_bias.requires_grad)
        self.assertFalse(model.backbone.emb.weight.requires_grad)
        model(self.x)["logits"].sum().backward()
        self.assertIsNotNone(model.gate_bias.grad)
        scope = configure_train_scope(model, "last_block")
        self.assertTrue(model.backbone.blocks[-1].att.w0.requires_grad)
        self.assertFalse(model.backbone.blocks[0].att.w0.requires_grad)
        self.assertTrue(model.backbone.ln_out.weight.requires_grad)
        self.assertGreater(scope["trainable_parameters"], 0)
        self.assertEqual(configure_train_scope(model, "full")["trainable_parameters"], scope["total_parameters"])
        for kwargs in ({"scope": "oops"}, {"last_blocks": 0}):
            with self.assertRaises(ValueError):
                configure_train_scope(model, **kwargs)

    def test_runtime_configuration_is_isolated(self):
        model = self.model("baseline")
        first = model(self.x)["logits"].detach()
        other = RWKVConfig(vocab_size=24, n_embd=32, n_layer=1, head_size=16)
        RWKVRosaLM(other, variant="baseline")
        torch.testing.assert_close(first, model(self.x)["logits"], atol=0, rtol=0)

    def test_input_validation(self):
        for kwargs in ({"variant": "oops"}, {"inject_layers": (3,)}, {"scale": -1},
                       {"variant": "original", "freeze_backbone": True}, {"variant": "capacity"}):
            with self.assertRaises(ValueError):
                RWKVRosaLM(self.cfg, **kwargs)
        for cfg in (RWKVConfig(n_embd=0), RWKVConfig(n_embd=17, head_size=8)):
            with self.assertRaises(ValueError):
                build_backbone(cfg)
        model = self.model()
        with self.assertRaises(ValueError):
            model(torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.tensor([[1, 0]]), return_state=True)
        with self.assertRaises(ValueError):
            model(self.x, state=model.init_state(2))
        address = model.address_engine.forward_seq(self.x)
        with self.assertRaises(ValueError):
            model(self.x, addresses=address, return_state=True)
        with self.assertRaises(ValueError):
            model(self.x[:, :2], addresses=address)
        empty = model(torch.tensor([[1, 3, 4]]))
        self.assertEqual(empty["stats"]["rosa_avg_gate"], 0)


class RWKVDataTests(unittest.TestCase):
    def test_shifted_masks_context_and_unique_supervision(self):
        doc = EncodedDocument([1, 2, 3, 4, 5, 6, 7], [0, 1, 1, 0, 1, 1, 1])
        ds = EqualHistoryDataset([doc], chunk_len=2, history_tokens=2)
        self.assertEqual(len(ds), 3)
        second = ds[1]
        self.assertEqual(second["input_ids"].tolist(), [1, 2, 3, 4])
        self.assertEqual(second["targets"].tolist(), [2, 3, 4, 5])
        self.assertEqual(second["loss_mask"].tolist(), [0, 0, 0, 1])
        self.assertEqual(sum(float(ds[i]["loss_mask"].sum()) for i in range(len(ds))), 5)
        batch = collate([ds[0], ds[1]])
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["loss_mask"][0, -1], 0)
        for kwargs in ({"chunk_len": 0}, {"history_tokens": -1}):
            with self.assertRaises(ValueError):
                EqualHistoryDataset([doc], **kwargs)
        with self.assertRaises(ValueError):
            EqualHistoryDataset([EncodedDocument([1, 2], [0, 0])])

    def test_upstream_sft_mask_supervises_only_final_response(self):
        tok = ByteTokenizer()
        self.assertEqual(tok.decode(tok.encode("你好")), "你好")
        record = {"messages": [{"role": "user", "content": "UNSUPERVISED_USER_TEXT"},
                                {"role": "assistant", "content": "FINAL_ANSWER"}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sft.json"
            path.write_text(json.dumps([record]))
            doc = load_documents(path, tok)[0]
            selected = tok.decode([t for t, mask in zip(doc.input_ids, doc.loss_mask) if mask])
            self.assertIn("FINAL_ANSWER", selected)
            self.assertNotIn("UNSUPERVISED_USER_TEXT", selected)
            path.write_text(json.dumps({"not": "a list"}))
            with self.assertRaises(ValueError):
                load_documents(path, tok)
            path.write_text(json.dumps([{"text": "a"}]))
            with self.assertRaises(ValueError):
                load_documents(path, tok)
            path.write_text(json.dumps([{"unknown": "x"}]))
            with self.assertRaises(ValueError):
                load_documents(path, tok)
            path.write_text("[]")
            with self.assertRaises(ValueError):
                load_documents(path, tok)
            txt = Path(folder) / "text.txt"
            txt.write_text("abc\ndef\n")
            self.assertEqual(len(load_documents(txt, tok)), 2)
            jl = Path(folder) / "text.jsonl"
            jl.write_text('{"text":"abcd"}\n')
            self.assertEqual(len(load_documents(jl, tok)), 1)

    def test_real_rwkv_tokenizer_and_special_ids(self):
        tok = build_tokenizer("rwkv")
        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            self.assertEqual(len(tok.encode(marker)), 1)
            self.assertIn(tok.encode(marker)[0], tok.special_ids)
        self.assertEqual(tok.decode(tok.encode("测试字段 ABC-12")), "测试字段 ABC-12")
        self.assertIsInstance(build_tokenizer("byte"), ByteTokenizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad_vocab.txt"
            path.write_text("1 b'a' 1\n")
            with self.assertRaises(ValueError):
                build_tokenizer("rwkv", path)

    def test_document_split_is_disjoint_and_deterministic(self):
        docs = [EncodedDocument([1, 2, i], [1, 1, 1]) for i in range(3, 13)]
        a = split_documents(docs, 12)
        self.assertEqual(a, split_documents(docs, 12))
        validate_splits(dict(zip(("train", "val", "test"), a)))
        with self.assertRaises(ValueError):
            split_documents(docs[:2], 1)
        with self.assertRaises(ValueError):
            validate_splits({"train": docs, "val": [docs[0]], "test": [docs[1]]})


class RWKVExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "corpus.json"
        self.data.write_text(json.dumps([{"text": f"{i} ABAB C ABAB C"} for i in range(8)]))

    def tearDown(self):
        self.tmp.cleanup()

    def run_experiment(self, out="run", *extra):
        argv = ["--data", str(self.data), "--out-dir", str(self.root / out),
                "--chunk-len", "8", "--history-tokens", "4", "--toy-dim", "16", "--toy-head-size", "8",
                "--max-steps", "1", "--eval-batches", "1", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            return main(argv)

    def test_all_four_arms_train_evaluate_and_save(self):
        result = self.run_experiment("all", "--variants", "baseline,original,improved,capacity",
                                     "--sparse-values", "--value-shards", "3", "--activation-checkpointing",
                                     "--grad-accum-steps", "2", "--save-every-steps", "1", "--verify-online")
        self.assertEqual(set(result["runs"]), {"baseline", "original", "improved", "capacity"})
        scopes = [v["parameters"]["backbone_trainable_names"] for v in result["runs"].values()]
        self.assertTrue(all(s == scopes[0] for s in scopes))
        for variant, metrics in result["runs"].items():
            self.assertEqual(metrics["global_step"], 1)
            self.assertGreater(metrics["test"]["supervised_tokens"], 0)
            self.assertTrue(Path(metrics["checkpoint"]).exists())
        self.assertGreater(result["runs"]["improved"]["parameters"]["total_parameters"],
                           result["runs"]["original"]["parameters"]["total_parameters"])

    def test_resume_matches_uninterrupted_training_for_every_arm(self):
        self.run_experiment("continuous", "--max-steps", "2", "--no-address-async")
        self.run_experiment("resumed", "--max-steps", "1", "--no-address-async")
        result = self.run_experiment("resumed", "--resume", "--max-steps", "2", "--no-address-async")
        for variant in ("baseline", "original", "improved"):
            self.assertEqual(result["runs"][variant]["global_step"], 2)
            a = torch.load(self.root / "continuous" / variant / "last.pt", weights_only=False)
            b = torch.load(self.root / "resumed" / variant / "last.pt", weights_only=False)
            for name in a["trainable_state"]:
                torch.testing.assert_close(a["trainable_state"][name], b["trainable_state"][name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.run_experiment("resumed", "--resume", "--lr", "0.003", "--no-address-async")
        with self.assertRaises(FileExistsError):
            self.run_experiment("continuous")

    def test_sync_async_training_produces_same_updates(self):
        self.run_experiment("sync", "--variants", "original,improved", "--no-address-async")
        self.run_experiment("async", "--variants", "original,improved")
        for variant in ("original", "improved"):
            a = torch.load(self.root / "sync" / variant / "last.pt", weights_only=False)
            b = torch.load(self.root / "async" / variant / "last.pt", weights_only=False)
            for name in a["trainable_state"]:
                torch.testing.assert_close(a["trainable_state"][name], b["trainable_state"][name], atol=0, rtol=0)

    def test_full_epoch_explicit_splits_and_no_save(self):
        val, test = self.root / "val.txt", self.root / "test.txt"
        val.write_text("VAL ABAB ABAB")
        test.write_text("TEST ABAB ABAB")
        result = self.run_experiment("full", "--variants", "improved", "--max-steps", "0", "--epochs", "2",
                                     "--eval-batches", "0", "--batch-size", "2", "--grad-accum-steps", "3",
                                     "--val-data", str(val), "--test-data", str(test), "--no-save-checkpoints",
                                     "--train-scope", "head")
        arm = result["runs"]["improved"]
        self.assertIsNone(arm["checkpoint"])
        self.assertGreater(arm["global_step"], 1)
        self.assertFalse(arm["evaluation_is_truncated"])

    def test_parser_config_overrides_and_validation(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"batch_size": 2, "seed": 4}))
        args = parse_args(["--config", str(config), "--seed", "5"])
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.batch_size, 2)
        config.write_text('{"typo_key": 1}')
        for argv in (["--config", str(config)], ["--epochs", "0"], ["--history-tokens", "-1"],
                     ["--dtype", "bfloat16"], ["--checkpoint", "missing.pth"], ["--val-data", "only"],
                     ["--resume", "--no-save-checkpoints"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        with self.assertRaises(ValueError):
            self.run_experiment("invalid", "--variants", "baseline,baseline")

    def test_real_checkpoint_cli_and_process_guard(self):
        vocab = self.root / "vocab.txt"
        entries = [f"{i+1} {bytes([i])!r} 1" for i in range(256)]
        for i, token in enumerate(("<|im_start|>", "<|im_end|>", "<|endoftext|>"), 257):
            entries.append(f"{i} {token.encode()!r} {len(token)}")
        vocab.write_text("\n".join(entries) + "\n")
        cfg = RWKVConfig(vocab_size=260, n_embd=16, head_size=8)
        checkpoint = self.root / "base.pth"
        torch.save(build_backbone(cfg).state_dict(), checkpoint)
        result = self.run_experiment("pretrained", "--checkpoint", str(checkpoint), "--tokenizer", "rwkv",
                                     "--vocab", str(vocab), "--variants", "baseline", "--verify-online")
        self.assertEqual(result["runs"]["baseline"]["online_verification"]["scale_zero_vs_upstream_max_abs_diff"], 0)
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "2"}), self.assertRaises(ValueError):
            self.run_experiment("bad_ddp")
        with self.assertRaises(ValueError):
            self.run_experiment("no_data", "--data", "")

    def test_global_gradient_clip_includes_sparse_rows_and_rejects_nan(self):
        from types import SimpleNamespace
        dense = torch.nn.Parameter(torch.ones(2))
        sparse = torch.nn.Parameter(torch.ones(3, 2))
        optimizer = SimpleNamespace(dense_params=[dense], sparse_params=[sparse])
        self.assertEqual(clip_all_gradients(optimizer, 1), 0)
        dense.grad = torch.tensor([3., 0.])
        sparse.grad = torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([[4., 0.]]), (3, 2))
        self.assertAlmostEqual(clip_all_gradients(optimizer, 1), 5)
        norm = (dense.grad.square().sum() + sparse.grad.values().square().sum()).sqrt()
        self.assertLessEqual(float(norm), 1)
        self.assertAlmostEqual(clip_all_gradients(optimizer, 0), float(norm), places=5)
        dense.grad[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            clip_all_gradients(optimizer, 1)

    def test_metric_and_checkpoint_validation(self):
        with self.assertRaises(ValueError):
            finish_metrics(1, 1, 0)
        with self.assertRaises(FloatingPointError):
            finish_metrics(float("nan"), 1, 1)
        self.assertEqual(finish_metrics(2, 1, 2)["token_accuracy"], 0.5)
        result = self.run_experiment("checkpoint", "--variants", "baseline")
        path = Path(result["runs"]["baseline"]["checkpoint"])
        args = parse_args(["--toy-dim", "16", "--toy-head-size", "8"])
        model = RWKVRosaLM(RWKVConfig(vocab_size=257, n_embd=16, head_size=8), variant="baseline")
        optimizer = build_training_optimizers(model, lr=1e-4, weight_decay=0.01)
        with self.assertRaises(ValueError):
            restore_checkpoint(path, model, optimizer, "bad signature")
        ckpt = torch.load(path, weights_only=False)
        ckpt["trainable_state"].pop(next(iter(ckpt["trainable_state"])))
        torch.save(ckpt, path)
        with self.assertRaises(ValueError):
            restore_checkpoint(path, model, optimizer, ckpt["signature"])


if __name__ == "__main__":
    unittest.main()
