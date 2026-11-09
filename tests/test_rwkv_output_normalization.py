"""Post-norm output-value RMS normalization and precision/metadata regressions."""
import argparse
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.adapter import CPUImprovedROSA, configure_output_head
from rwkv_rosa.native.cpu_compare import CPUInference, convert, load_cpu, state_hash
from rwkv_rosa.native.integration import NativeROSACallback, add_arguments, install
from rwkv_rosa.native.position import normalize_metadata


class OutputNormalizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.embedding = nn.Embedding(32, 16).requires_grad_(False)
        with torch.no_grad():
            self.embedding.weight.fill_(.005)
        self.ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 3]])

    def branch(self, mode):
        branch = CPUImprovedROSA(self.embedding, position='post_norm', output_norm=mode)
        branch.begin(self.ids)
        return branch

    def test_legacy_default_and_invalid_modes(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertEqual(parser.parse_args([]).rosa_output_norm, 'none')
        self.assertEqual(normalize_metadata({'position': 'post_norm'})['output_norm'], 'none')
        legacy = CPUImprovedROSA(self.embedding, position='post_norm')
        explicit = self.branch('none')
        self.assertEqual(legacy.state_dict().keys(), explicit.state_dict().keys())
        legacy.load_state_dict(explicit.state_dict(), strict=True)
        legacy.begin(self.ids)
        x = torch.randn(1, 8, 16).bfloat16()
        torch.testing.assert_close(legacy.inject_post_norm(x), explicit.inject_post_norm(x), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            CPUImprovedROSA(self.embedding, layers=(0,), output_norm='rms')
        with self.assertRaises(ValueError):
            normalize_metadata({'position': 'post_norm', 'output_norm': 'bad'})
        fixed = self.branch('rms')
        with self.assertRaises(RuntimeError):
            explicit.load_state_dict(fixed.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            fixed.load_state_dict(explicit.state_dict(), strict=True)

    def test_small_bf16_residual_is_preserved_and_value_is_normalized(self):
        old, fixed = self.branch('none'), self.branch('rms')
        fixed.load_state_dict({**old.state_dict(), '_output_norm_version': fixed._output_norm_version}, strict=True)
        x = torch.full((1, 8, 16), 4., dtype=torch.bfloat16)
        before, after = old.inject_post_norm(x), fixed.inject_post_norm(x)
        valid = fixed.addresses.valid_mask
        self.assertTrue(valid.any())
        self.assertEqual(before.dtype, torch.bfloat16)
        self.assertEqual(after.dtype, torch.float32)
        self.assertTrue(torch.equal(before, x))
        delta = after - x.float()
        self.assertTrue(bool((delta[valid].abs() > .01).all()))
        self.assertTrue(torch.equal(delta[~valid], torch.zeros_like(delta[~valid])))
        old_float_delta = old.inject_post_norm(x.float()) - x.float()
        ratio = delta[valid] / old_float_delta[valid]
        torch.testing.assert_close(ratio, torch.full_like(ratio, 1 / (.005**2 + 1e-6)**.5), rtol=.002, atol=.01)
        # Norm precedes the output projection, preserving its learnable amplitude.
        with torch.no_grad():
            fixed.value_projs[0].weight.mul_(2)
        doubled = fixed.inject_post_norm(x) - x.float()
        torch.testing.assert_close(doubled, 2 * delta, atol=1e-6, rtol=1e-5)

    def test_output_head_is_fp32_under_amp_and_base_weights_are_unchanged(self):
        head = nn.Linear(16, 32, bias=False).bfloat16().requires_grad_(False)
        before = state_hash(head.state_dict())
        configure_output_head(head, 'rms')
        branch = self.branch('rms')
        x = torch.full((1, 8, 16), 4., dtype=torch.bfloat16)
        features = branch.inject_post_norm(x)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = head(features)
            base = head(x)
        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits, F.linear(features, head.weight.float()), atol=0, rtol=0)
        self.assertGreater(float((logits - base).abs().max()), 0)
        logits.square().mean().backward()
        for projection in (branch.key_projs[0], branch.value_projs[0]):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(bool(projection.weight.grad.isfinite().all()))
        self.assertTrue(any(p.grad is not None for p in branch.rosa_value_store.parameters()))
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(self.embedding.weight.grad)
        self.assertEqual(state_hash(head.state_dict()), before)
        self.assertEqual(head.weight.dtype, torch.bfloat16)

    def test_native_fused_ce_uses_fp32_head_view_without_casting_backbone(self):
        source = Path('third_party/rwkv_lm_v7_native/src/model.py')
        cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'RWKV')
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_sft_loss_from_hidden')
        seen = []
        def fused(hidden, weight, target, mask, chunk):
            seen.append((hidden.dtype, weight.dtype, chunk))
            return F.cross_entropy(F.linear(hidden, weight).reshape(-1, 32), target.reshape(-1))
        scope = {'head_masked_cross_entropy_cuda': fused}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        model = SimpleNamespace(args=SimpleNamespace(sft_masked_fused_ce_chunk=4),
            head=nn.Linear(16, 32, bias=False).bfloat16().requires_grad_(False),
            _head_weight_context=lambda gather: nullcontext())
        h = torch.randn(1, 8, 16, requires_grad=True)
        scope['_sft_loss_from_hidden'](model, h, self.ids, torch.ones(1, 8)).backward()
        self.assertEqual(seen, [(torch.float32, torch.float32, 4)])
        self.assertIsNotNone(h.grad)
        self.assertIsNone(model.head.weight.grad)
        self.assertEqual(model.head.weight.dtype, torch.bfloat16)

    def test_metadata_resume_export_and_cpu_streaming(self):
        body = build_backbone(RWKVConfig(vocab_size=32, n_embd=16, n_layer=2, head_size=8)).bfloat16()
        body.requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.pth'
            torch.save(body.state_dict(), base)
            args = SimpleNamespace(proj_dir=str(root), ctx_len=8, data_file='toy', epoch_steps=2,
                rosa_layers='', rosa_scale=.05, rosa_position='post_norm', rosa_output_norm='rms')
            callback = NativeROSACallback(args)
            branch = CPUImprovedROSA(body.emb, position='post_norm', output_norm='rms')
            checkpoint = {'state_dict': {**body.state_dict(), **{'rosa_cpu.' + k: v for k, v in branch.state_dict().items()}}, 'global_step': 1}
            callback.on_save_checkpoint(None, None, checkpoint)
            callback.on_load_checkpoint(None, None, copy.deepcopy(checkpoint))
            wrong = copy.deepcopy(checkpoint)
            wrong['native_rosa'].pop('output_norm')
            with self.assertRaises(ValueError):
                callback.on_load_checkpoint(None, None, wrong)
            path = root / 'model.ckpt'
            torch.save(checkpoint, path)
            convert(path, base, root / 'export')
            model = load_cpu(root / 'export')
            self.assertEqual(model.rosa.output_norm, 'rms')
            with torch.inference_mode():
                full = model(self.ids)
                _, state = model.features(self.ids[:, :5], return_state=True)
                last, state = model.features(self.ids[:, 5:], state=state, history=self.ids, return_state=True)
                torch.testing.assert_close(model.backbone.head(last), full[:, 5:], atol=2e-5, rtol=2e-5)

    def test_normalized_standalone_request_and_all_invalid_addresses(self):
        from rwkv_rosa.native.standalone_needle import request_from_checkpoint
        branch = self.branch('rms')
        branch.addresses.valid_mask.zero_()
        x = torch.randn(1, 8, 16).bfloat16()
        y = branch.inject_post_norm(x)
        torch.testing.assert_close(y, x.float(), atol=0, rtol=0)
        self.assertTrue(bool(y.isfinite().all()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'model.ckpt'
            torch.save({'native_rosa_checkpoint_kind': 'adapter_optimizer_loop_with_external_base_v1',
                'native_rosa': {'position': 'post_norm', 'layers': '', 'scale': .05, 'output_norm': 'rms'},
                'native_rosa_frozen_base': {'path': 'base.pth', 'sha256': 'base', 'parameter_names': ['emb.weight']},
                'global_step': 1, 'state_dict': {'rosa_cpu.' + k: v for k, v in branch.state_dict().items()}}, path)
            request = request_from_checkpoint(path, root / 'evaluation')
            self.assertEqual(request['output_norm'], 'rms')

    def test_installer_keeps_frozen_body_bytes_and_enables_output_head(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Module()
            model.emb = self.embedding
            model.head = nn.Linear(16, 32, bias=False)
            model.bfloat16()
            before = state_hash(model.state_dict())
            args = SimpleNamespace(rosa_cpu=1, devices=1, num_nodes=1, strategy='auto', precision='bf16',
                load_model='base', train_stage=0, load_partial=0, resume_ckpt_path=None,
                rosa_position='post_norm', rosa_layers='', rosa_output_norm='rms', n_layer=1,
                rosa_scale=.05, rosa_device='cpu', rosa_resume='', run_name='norm-test', proj_dir=directory,
                lr_init=1e-3, betas=(.9, .99), adam_eps=1e-8, weight_decay=0.)
            install(model, args)
            self.assertEqual(model.rosa_cpu.output_norm, 'rms')
            x = torch.full((1, 8, 16), 4., dtype=torch.bfloat16)
            model.rosa_cpu.begin(self.ids)
            model.head(model.rosa_cpu.inject_post_norm(x)).sum().backward()
            frozen = {k: v for k, v in model.state_dict().items() if not k.startswith('rosa_cpu.')}
            self.assertEqual(state_hash(frozen), before)
            self.assertFalse(model.head.weight.requires_grad)
            self.assertIsNone(model.head.weight.grad)


if __name__ == '__main__':
    unittest.main()
