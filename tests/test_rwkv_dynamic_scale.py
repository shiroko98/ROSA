"""Detached logit-gap controller: direction, cap, chunking, gradients and resume."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.adapter import CPUImprovedROSA, configure_output_head
from rwkv_rosa.native.cpu_compare import CPUInference, convert, load_cpu, state_hash
from rwkv_rosa.native.dynamic_scale import logit_gap_scales
from rwkv_rosa.native.integration import NativeROSACallback, add_arguments, install
from rwkv_rosa.native.position import normalize_metadata, scale_options


class DynamicScaleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.head = nn.Linear(2, 3, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))

    def scales(self, hidden, direction, candidates, valid=None, **kwargs):
        if valid is None:
            valid = torch.ones_like(candidates, dtype=torch.bool)
        return logit_gap_scales(hidden, direction, candidates, valid, self.head, **kwargs)

    def test_new_default_cap_is_point_two_without_reinterpreting_old_metadata(self):
        from rwkv_rosa.native.position import scale_options_from_args
        parser=argparse.ArgumentParser();add_arguments(parser)
        self.assertEqual(parser.parse_args([]).rosa_dynamic_max,.2)
        self.assertEqual(scale_options_from_args(SimpleNamespace())['dynamic_max'],.2)
        embedding=nn.Embedding(8,2).requires_grad_(False)
        branch=CPUImprovedROSA(embedding,position='post_norm',output_norm='rms_match',scale_mode='logit_gap')
        self.assertEqual(branch.scale_options['dynamic_max'],.2)
        gain,_=self.scales(torch.tensor([[[2.,0.]]]),torch.tensor([[[0.,2.]]]),torch.tensor([[1]]))
        self.assertAlmostEqual(float(gain),.2,places=6)
        legacy={'position':'post_norm','output_norm':'rms','scale_mode':'logit_gap'}
        self.assertEqual(normalize_metadata(legacy)['dynamic_max'],.5)
        self.assertEqual(normalize_metadata({**legacy,'dynamic_max':.2})['dynamic_max'],.2)
        self.assertEqual(normalize_metadata({**legacy,'dynamic_max':.5})['dynamic_max'],.5)

    def test_label_free_gap_calculation_cap_and_suppress_wrong_direction(self):
        # Base scores [2, 0, -2], candidate 1. Direction changes its relative score by 2/scale.
        h = torch.tensor([[[2., 0.], [2., 0.], [2., 0.]]])
        u = torch.tensor([[[0., 2.], [0., -2.], [0., 0.]]], requires_grad=True)
        c = torch.ones(1, 3, dtype=torch.long)
        gain, useful = self.scales(h, u, c, maximum=2., margin=.5)
        torch.testing.assert_close(gain, torch.tensor([[1.25, 0., 0.]]))
        self.assertEqual(useful.tolist(), [[True, False, False]])
        self.assertFalse(gain.requires_grad)
        capped, _ = self.scales(h, u, c, maximum=.5, margin=.5)
        self.assertEqual(float(capped.max()), .5)
        # The same controller can amplify a WRONG retrieval candidate. No oracle-label claim.
        scores = F.linear(h[:, :1] + gain[:, :1, None] * u[:, :1], self.head.weight)
        self.assertEqual(int(scores.argmax(-1)), 1)

    def test_existing_top_candidate_and_invalid_or_nonfinite_inputs(self):
        h = torch.tensor([[[2., 0.], [2., 0.], [2., 0.]]])
        u = torch.tensor([[[1., 0.], [0., 2.], [float('nan'), 0.]]])
        c = torch.tensor([[0, 999, 1]])
        gain, _ = self.scales(h, u, c, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(gain, torch.zeros_like(gain))
        # A bad address is rejected only when marked valid.
        with self.assertRaises(ValueError):
            self.scales(h[:, :1], u[:, :1], torch.tensor([[999]]))
        self.head.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            self.scales(h[:, :1], u[:, :1], torch.tensor([[0]]))

    def test_chunking_matches_full_and_bounds_vocabulary_allocation(self):
        h = torch.randn(2, 19, 2)
        u = torch.randn_like(h)
        c = torch.randint(0, 3, (2, 19))
        full = self.scales(h, u, c, chunk_size=100)
        sizes = []
        linear = F.linear
        def checked(x, w, bias=None):
            sizes.append(x.shape[0])
            return linear(x, w, bias)
        with patch('rwkv_rosa.native.dynamic_scale.F.linear', side_effect=checked):
            chunked = self.scales(h, u, c, chunk_size=4)
        torch.testing.assert_close(full, chunked)
        self.assertEqual(max(sizes), 4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp = self.scales(h, u, c, chunk_size=4)
        torch.testing.assert_close(amp, chunked)

    def test_adapter_gate_gradient_and_no_frozen_head_registration(self):
        embedding = nn.Embedding(8, 2).requires_grad_(False)
        with torch.no_grad():
            embedding.weight.copy_(torch.tensor([[0., 1.]] * 8))
        fixed = CPUImprovedROSA(embedding, position='post_norm', output_norm='rms')
        dynamic = CPUImprovedROSA(embedding, position='post_norm', output_norm='rms',
            scale_mode='logit_gap', dynamic_max=.5, dynamic_margin=.5, dynamic_chunk=2)
        dynamic.load_state_dict({**fixed.state_dict(), '_dynamic_scale_version': dynamic._dynamic_scale_version})
        ids = torch.tensor([[1, 2, 1, 2]])
        dynamic.begin(ids)
        x = torch.tensor([[[2., 0.]] * 4]).bfloat16()
        with self.assertRaises(RuntimeError):
            dynamic.inject_post_norm(x)
        dynamic.bind_output_head(self.head)
        # Choose candidate 1 everywhere an address is valid, without providing any label.
        dynamic.addresses.addr_ids.fill_(1)
        dynamic.addresses.fired_match_lens.fill_(14)
        before = state_hash(self.head.state_dict())
        y = dynamic.inject_post_norm(x)
        self.assertEqual(y.dtype, torch.float32)
        self.assertLessEqual(dynamic.stats['rosa_dynamic_scale_max'], .5)
        F.linear(y, self.head.weight).square().mean().backward()
        self.assertIsNotNone(dynamic.value_projs[0].weight.grad)
        self.assertIsNotNone(dynamic.gate_bias.grad)
        self.assertGreater(float(dynamic.value_projs[0].weight.grad.norm()), 0)
        self.assertIsNone(self.head.weight.grad)
        self.assertEqual(state_hash(self.head.state_dict()), before)
        self.assertFalse(any('head' in k for k in dynamic.state_dict()))
        with self.assertRaises(RuntimeError):
            fixed.load_state_dict(dynamic.state_dict(), strict=True)

    def test_short_matches_are_not_dynamically_amplified(self):
        embedding = nn.Embedding(8, 2).requires_grad_(False)
        branch = CPUImprovedROSA(embedding, position='post_norm', output_norm='rms', scale_mode='logit_gap')
        branch.bind_output_head(self.head)
        branch.begin(torch.tensor([[1, 2, 1, 2]]))
        branch.addresses.fired_match_lens.fill_(1)
        x = torch.randn(1, 4, 2).bfloat16()
        y = branch.inject_post_norm(x)
        torch.testing.assert_close(y, x.float(), atol=0, rtol=0)
        self.assertEqual(branch.stats['rosa_dynamic_eligible_fraction'], 0.)
        self.assertEqual(branch.stats['rosa_dynamic_scale_max'], 0.)

    def test_validation_metadata_and_resume_reject_semantic_changes(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertEqual(parser.parse_args([]).rosa_scale_mode, 'fixed')
        self.assertEqual(normalize_metadata({'position': 'post_norm'})['scale_mode'], 'fixed')
        for changes in ({'output_norm': 'none'}, {'position': 'block', 'layers': '0'},
                        {'dynamic_max': float('inf')}, {'dynamic_margin': -1}, {'dynamic_chunk': 0}, {'dynamic_min_match': 0}):
            meta = {'position': 'post_norm', 'output_norm': 'rms', 'scale_mode': 'logit_gap', **changes}
            with self.assertRaises(ValueError):
                normalize_metadata(meta)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(proj_dir=directory, ctx_len=8, data_file='toy', epoch_steps=2,
                rosa_position='post_norm', rosa_layers='', rosa_scale=.05, rosa_output_norm='rms',
                rosa_scale_mode='logit_gap', rosa_dynamic_max=.5, rosa_dynamic_margin=.5, rosa_dynamic_chunk=4)
            cb = NativeROSACallback(args)
            checkpoint = {'state_dict': {}, 'global_step': 1}
            cb.on_save_checkpoint(None, None, checkpoint)
            cb.on_load_checkpoint(None, None, copy.deepcopy(checkpoint))
            for change in ({'scale_mode': 'fixed'}, {'dynamic_max': .2}, {'dynamic_margin': 1.}):
                wrong = copy.deepcopy(checkpoint)
                wrong['native_rosa'].update(change)
                with self.assertRaises(ValueError):
                    cb.on_load_checkpoint(None, None, wrong)

    def test_export_cpu_streaming_and_installer_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = build_backbone(RWKVConfig(vocab_size=32, n_embd=16, n_layer=2, head_size=8)).bfloat16()
            body.requires_grad_(False)
            base = root / 'base.pth'
            torch.save(body.state_dict(), base)
            args = SimpleNamespace(rosa_cpu=1, devices=1, num_nodes=1, strategy='auto', precision='bf16',
                load_model=str(base), train_stage=0, load_partial=0, resume_ckpt_path=None,
                rosa_position='post_norm', rosa_layers='', rosa_output_norm='rms', rosa_scale_mode='logit_gap',
                rosa_dynamic_max=.5, rosa_dynamic_margin=.5, rosa_dynamic_chunk=4, n_layer=2,
                rosa_scale=.05, rosa_device='cpu', rosa_resume='', run_name='dynamic-test', proj_dir=str(root),
                lr_init=1e-3, betas=(.9,.99), adam_eps=1e-8, weight_decay=0.,ctx_len=8,data_file='toy',epoch_steps=2)
            cb = install(body, args)
            ids = torch.tensor([[1,2,1,2,3,1,2,3]])
            body.rosa_cpu.begin(ids)
            out = body.rosa_cpu.inject_post_norm(torch.randn(1,8,16).bfloat16())
            body.head(out).square().mean().backward()
            self.assertTrue(all(p.grad is None for n,p in body.named_parameters() if not n.startswith('rosa_cpu.')))
            checkpoint = {'global_step':1, 'state_dict':body.state_dict()}
            cb.on_save_checkpoint(None, body, checkpoint)
            path = root / 'model.ckpt'
            torch.save(checkpoint, path)
            manifest = convert(path, base, root / 'export')
            self.assertEqual(manifest['rosa_metadata']['scale_mode'], 'logit_gap')
            model = load_cpu(root / 'export')
            with torch.inference_mode():
                full = model(ids)
                _, state = model.features(ids[:,:5], return_state=True)
                tail, _ = model.features(ids[:,5:], state=state, history=ids, return_state=True)
                torch.testing.assert_close(model.backbone.head(tail), full[:,5:], atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
