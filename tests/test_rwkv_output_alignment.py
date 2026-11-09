"""Post-projection output RMS matching, before gate and dynamic gain."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from torch import nn
from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.adapter import CPUImprovedROSA, configure_output_head
from rwkv_rosa.native.cpu_compare import convert, load_cpu
from rwkv_rosa.native.integration import NativeROSACallback
from rwkv_rosa.native.position import normalize_metadata


class OutputAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        self.embedding = nn.Embedding(32,16).requires_grad_(False)
        self.ids = torch.tensor([[1,2,1,2,3,1,2,3]])

    def make(self, **options):
        branch = CPUImprovedROSA(self.embedding, position='post_norm', output_norm='rms_match', **options)
        branch.begin(self.ids)
        return branch

    def test_fixed_scale_is_relative_to_z_and_projection_amplitude_cancels(self):
        branch = self.make(scale=.2)
        x = torch.randn(1,8,16).bfloat16() * torch.arange(1,9).reshape(1,8,1)
        y = branch.inject_post_norm(x)
        valid = branch.addresses.valid_mask
        delta_rms = (y-x.float()).square().mean(-1).sqrt()
        ref = x.float().square().mean(-1).sqrt()
        # Use the exact gate from the original key path, not a calibrated probability.
        a = branch.addresses; v = branch.rosa_value_store.lookup(0,a.addr_ids,shared_embedding=None)
        key = branch.value_norms[0](branch.key_projs[0](v))
        score = (branch.hidden_norms[0](x.float())*key).sum(-1)/4 + branch.length_scale[0]*torch.log1p(a.fired_match_lens.float()) + branch.gate_bias[0]
        expected = .2*score.sigmoid()
        torch.testing.assert_close((delta_rms/ref)[valid],expected[valid],atol=2e-5,rtol=2e-4)
        with torch.no_grad():branch.value_projs[0].weight.mul_(7)
        after = branch.inject_post_norm(x)
        torch.testing.assert_close(after,y,atol=2e-5,rtol=2e-4)
        self.assertLessEqual(branch.stats['rosa_residual_relative_rms_max'],.20001)
        self.assertEqual(after.dtype,torch.float32)

    def test_dynamic_controller_sees_the_aligned_direction(self):
        from unittest.mock import patch
        head = nn.Linear(16,32,bias=False).requires_grad_(False)
        branch = self.make(scale_mode='logit_gap')
        branch.bind_output_head(head)
        x = torch.randn(1,8,16).bfloat16()*5
        captured=[]
        def controller(z,u,c,valid,head,**kwargs):
            captured.append(u.detach().clone())
            return torch.full_like(c,kwargs['maximum'],dtype=torch.float32),valid
        with patch('rwkv_rosa.native.dynamic_scale.logit_gap_scales',side_effect=controller):
            y=branch.inject_post_norm(x)
        torch.testing.assert_close(captured[0].square().mean(-1).sqrt(),x.float().square().mean(-1).sqrt(),atol=2e-5,rtol=2e-4)
        self.assertLessEqual(branch.stats['rosa_residual_relative_rms_max'],.20001)
        configure_output_head(head,'rms_match')
        logits=head(y);self.assertEqual(logits.dtype,torch.float32)
        logits.square().mean().backward()
        self.assertGreater(float(branch.value_projs[0].weight.grad.norm()),0)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(self.embedding.weight.grad)

    def test_zero_projection_and_zero_hidden_are_finite(self):
        branch=self.make()
        with torch.no_grad():branch.value_projs[0].weight.zero_()
        x=torch.randn(1,8,16).bfloat16()
        torch.testing.assert_close(branch.inject_post_norm(x),x.float(),atol=0,rtol=0)
        with torch.no_grad():branch.value_projs[0].weight.copy_(torch.eye(16))
        y=branch.inject_post_norm(torch.zeros_like(x))
        self.assertTrue(bool(y.isfinite().all()));self.assertEqual(float(y.abs().max()),0.)

    def test_zero_gain_first_batch_is_not_falsely_reported_as_updated(self):
        import time
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(proj_dir=directory,ctx_len=8,rosa_device='cpu',rosa_save=0,
                rosa_checkpoint_seconds=0.,rosa_checkpoint_initial=0)
            callback=NativeROSACallback(args)
            callback.audit={'backbone_sha256_before':'same','rosa_values_sha256_before':'before'}
            callback.started=time.perf_counter()-1;callback.mask_tokens=8.;callback.batch_hash='batch'
            branch=SimpleNamespace(scale_mode='logit_gap',rosa_value_store=nn.Embedding(8,2),stats={})
            model=SimpleNamespace(rosa_cpu=branch)
            trainer=SimpleNamespace(global_step=1,my_grad_norm=0.,my_loss=1.,optimizers=[SimpleNamespace(assert_states_device=Mock())])
            with patch('torch.cuda.synchronize'),patch('torch.cuda.max_memory_allocated',return_value=0), \
                 patch('torch.cuda.max_memory_reserved',return_value=0), \
                 patch('rwkv_rosa.native.integration.frozen_hash',return_value='same'), \
                 patch('rwkv_rosa.native.integration.parameter_hash',return_value='before'):
                callback.on_train_batch_end(trainer,model,None,(torch.ones(1,8),),0)
            self.assertFalse(callback.audit['rosa_values_changed'])
            callback.started=time.perf_counter()-1;trainer.global_step=2;trainer.my_grad_norm=.1
            with patch('torch.cuda.synchronize'),patch('torch.cuda.max_memory_allocated',return_value=0), \
                 patch('torch.cuda.max_memory_reserved',return_value=0), \
                 patch('rwkv_rosa.native.integration.parameter_hash',return_value='after'):
                callback.on_train_batch_end(trainer,model,None,(torch.ones(1,8),),1)
            self.assertTrue(callback.audit['rosa_values_changed'])
            self.assertEqual(callback.audit['rosa_values_first_observed_update'],2)

    def test_metadata_export_streaming_and_cross_formula_load_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            body=build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8)).bfloat16().requires_grad_(False)
            base=root/'base.pth';torch.save(body.state_dict(),base)
            branch=CPUImprovedROSA(body.emb,position='post_norm',output_norm='rms_match',scale_mode='logit_gap')
            args=SimpleNamespace(proj_dir=directory,ctx_len=8,data_file='toy',epoch_steps=2,rosa_position='post_norm',
                rosa_layers='',rosa_scale=.05,rosa_output_norm='rms_match',rosa_scale_mode='logit_gap')
            callback=NativeROSACallback(args)
            checkpoint={'state_dict':{**body.state_dict(),**{'rosa_cpu.'+k:v for k,v in branch.state_dict().items()}},'global_step':1}
            callback.on_save_checkpoint(None,None,checkpoint)
            callback.on_load_checkpoint(None,None,copy.deepcopy(checkpoint))
            wrong=copy.deepcopy(checkpoint);wrong['native_rosa']['output_norm']='rms'
            with self.assertRaises(ValueError):callback.on_load_checkpoint(None,None,wrong)
            old=CPUImprovedROSA(body.emb,position='post_norm',output_norm='rms',scale_mode='logit_gap')
            with self.assertRaises(RuntimeError):old.load_state_dict(branch.state_dict(),strict=True)
            path=root/'model.ckpt';torch.save(checkpoint,path);convert(path,base,root/'export');model=load_cpu(root/'export')
            with torch.inference_mode():
                full=model(self.ids)
                _,state=model.features(self.ids[:,:5],return_state=True)
                tail,_=model.features(self.ids[:,5:],state=state,history=self.ids,return_state=True)
                torch.testing.assert_close(model.backbone.head(tail),full[:,5:],atol=2e-5,rtol=2e-5)
            self.assertEqual(normalize_metadata(checkpoint['native_rosa'])['output_norm'],'rms_match')


if __name__=='__main__':unittest.main()
