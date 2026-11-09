"""Explicit output-branch ROSA: gradients, recurrent-state isolation, metadata."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from torch import nn
from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.cpu_compare import CPUInference, convert, load_cpu, generate
from rwkv_rosa.native.integration import add_arguments, install, NativeROSACallback
from rwkv_rosa.native.position import resolve_position, normalize_metadata
from rwkv_rosa.native.needle_eval import evaluate, summarize


class PostNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        self.body=build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8)).eval().requires_grad_(False)
        self.ids=torch.tensor([[1,2,1,2,3,1,2,3]])
        self.rosa=CPUImprovedROSA(self.body.emb,position='post_norm')

    def test_position_is_not_a_magic_block_layer(self):
        import argparse
        p=argparse.ArgumentParser();add_arguments(p)
        self.assertEqual(p.parse_args(['--rosa_position','post_norm']).rosa_layers,None)
        self.assertEqual(resolve_position('post_norm'),('post_norm',()))
        self.assertEqual(normalize_metadata({'layers':'1','scale':.05})['position'],'block')
        self.assertEqual(self.rosa.layers,());self.assertEqual(self.rosa.slots,{})
        for position,layers in (('head',None),('post_norm',(0,)),('block',()),('block',(1,1))):
            with self.assertRaises(ValueError):resolve_position(position,layers)
        block=CPUImprovedROSA(self.body.emb,layers=(1,))
        self.assertEqual(sum(p.numel() for p in block.parameters()),sum(p.numel() for p in self.rosa.parameters()))
        self.assertTrue(all(torch.equal(v,self.rosa.state_dict()[n]) for n,v in block.state_dict().items()))
        x=torch.randn(1,8,16)
        self.assertIs(self.rosa.inject(0,x),x)
        self.assertIs(block.inject_post_norm(x),x)
        with self.assertRaises(RuntimeError):self.rosa.inject_post_norm(x)

    def test_cpu_output_decomposition_states_streaming_and_gradients(self):
        base=CPUInference(self.body); model=CPUInference(self.body,self.rosa)
        # CPUInference is evaluation-frozen; reenable adapter for gradient test.
        model.rosa.requires_grad_(True)
        z,state0=base.features(self.ids,return_state=True)
        features,state1=model.features(self.ids,return_state=True)
        torch.testing.assert_close(state0,state1,atol=0,rtol=0)
        self.rosa.begin(self.ids)
        torch.testing.assert_close(features,self.rosa.inject_post_norm(z),atol=0,rtol=0)
        delta=features-z
        torch.testing.assert_close(self.body.head(features),self.body.head(z)+self.body.head(delta),atol=2e-6,rtol=2e-5)
        self.assertFalse(z.requires_grad);self.assertTrue(features.requires_grad)
        self.body.head(features).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in self.rosa.parameters()))
        self.assertTrue(all(p.grad is None for p in self.body.parameters()))
        with torch.inference_mode():
            first,state=model.features(self.ids[:,:5],return_state=True)
            last,state=model.features(self.ids[:,5:],state=state,history=self.ids,return_state=True)
            torch.testing.assert_close(torch.cat((first,last),1),features,atol=2e-5,rtol=2e-5)
            self.assertLessEqual(len(generate(model,self.ids[0].tolist(),max_new=3)),3)

    def test_actual_native_forward_has_no_reverse_path_through_any_block(self):
        source=Path('third_party/rwkv_lm_v7_native/src/model.py')
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RWKV')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_features')
        scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source.resolve()),'exec'),scope)
        class Block(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(16,16);self.output_grad=None
            def forward(self,x,v):
                y=self.linear(x);self.output_grad=y.requires_grad
                return y,y
        native=nn.Module();native.emb=copy.deepcopy(self.body.emb);native.blocks=nn.ModuleList([Block(),Block()]);native.ln_out=nn.LayerNorm(16)
        native.requires_grad_(False);native.rosa_cpu=CPUImprovedROSA(native.emb,position='post_norm')
        native.args=SimpleNamespace(ctx_len=8,grad_cp=1,rosa_no_checkpoint_layers=())
        norm_inputs=[];native.ln_out.register_forward_pre_hook(lambda module,args:norm_inputs.append(args[0].requires_grad))
        with patch('torch.utils.checkpoint.checkpoint') as cp:
            output=scope['_forward_features'](native,self.ids);output.square().mean().backward();cp.assert_not_called()
        self.assertEqual(norm_inputs,[False]);self.assertTrue(all(b.output_grad is False for b in native.blocks))
        self.assertTrue(any(p.grad is not None for p in native.rosa_cpu.parameters()))
        self.assertTrue(all(p.grad is None for n,p in native.named_parameters() if not n.startswith('rosa_cpu.')))

    def test_position_metadata_roundtrip_export_and_cross_position_resume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);base=tmp/'base.pth';self.body.bfloat16();torch.save(self.body.state_dict(),base)
            args=SimpleNamespace(proj_dir=str(tmp),ctx_len=8,data_file='data',epoch_steps=2,rosa_layers='',rosa_scale=.05,rosa_position='post_norm')
            callback=NativeROSACallback(args);checkpoint={'state_dict':dict(self.body.state_dict()),'global_step':3}
            checkpoint['state_dict'].update({'rosa_cpu.'+k:v for k,v in self.rosa.state_dict().items()})
            callback.on_save_checkpoint(None,None,checkpoint)
            self.assertEqual(checkpoint['native_rosa']['position'],'post_norm')
            callback.on_load_checkpoint(None,None,copy.deepcopy(checkpoint))
            missing=copy.deepcopy(checkpoint);missing.pop('native_rosa')
            with self.assertRaises(ValueError):callback.on_load_checkpoint(None,None,missing)
            wrong=copy.deepcopy(checkpoint);wrong['native_rosa'].update(position='block',layers='1')
            with self.assertRaises(ValueError):callback.on_load_checkpoint(None,None,wrong)
            legacy=copy.deepcopy(wrong);legacy['native_rosa'].pop('position')
            with self.assertRaises(ValueError):callback.on_load_checkpoint(None,None,legacy)
            path=tmp/'model.ckpt';torch.save(checkpoint,path)
            manifest=convert(path,base,tmp/'export');self.assertEqual(manifest['rosa_metadata']['position'],'post_norm')
            loaded=load_cpu(tmp/'export');self.assertEqual(loaded.rosa.position,'post_norm');self.assertEqual(loaded.rosa.layers,())
            reference=CPUInference(build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8),state_dict=self.body.state_dict()),self.rosa)
            with torch.inference_mode():torch.testing.assert_close(loaded(self.ids),reference(self.ids),atol=0,rtol=0)

    def test_periodic_cpu_worker_loads_explicit_post_norm_and_caches_only_base(self):
        from rwkv_rosa.native.periodic_needle import run_request
        from rwkv_rosa.native.cpu_compare import state_hash
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);self.body.bfloat16();base=tmp/'base.pth';torch.save(self.body.state_dict(),base)
            adapter=tmp/'adapter.pth';torch.save(self.rosa.state_dict(),adapter)
            case={'id':'tiny','difficulty':'unrelated','stem_ids':[1,2,3,1],
                  'suffix_ids':{'qa':[2],'exact_prefix':[2]},'answer_ids':[3],
                  'secret':'123456','needle_start':0,'length':4,'depth':.5}
            request={'step':1,'base':str(base),'adapter':str(adapter),'position':'post_norm','layers':[],
                     'scale':.05,'threads':2,'seed':42,'length':4,'depths':[.5],
                     'difficulties':['unrelated'],'max_new_tokens':2,
                     'backbone_sha256':state_hash(dict(self.body.state_dict())),
                     'baseline_cache':str(tmp/'cache.json'),'result':str(tmp/'report.json')}
            path=tmp/'request.json';path.write_text(json.dumps(request))
            tok=SimpleNamespace(idx2token={i:str(i).encode() for i in range(32)})
            with patch('rwkv_rosa.native.periodic_needle.TRIE_TOKENIZER',return_value=tok), \
                 patch('rwkv_rosa.native.periodic_needle.build_case',return_value=case), \
                 patch('torch.set_num_interop_threads'):
                first=run_request(path);second=run_request(path)
                compact=tmp/'compact.ckpt'
                torch.save({'native_rosa':{'position':'post_norm','layers':'','scale':.05},
                    'native_rosa_frozen_base':{'sha256':request['backbone_sha256']},
                    'global_step':1,'state_dict':{'rosa_cpu.'+k:v for k,v in self.rosa.state_dict().items()}},compact)
                request.update(adapter=str(compact),adapter_format='native_checkpoint');path.write_text(json.dumps(request))
                third=run_request(path)
                self.assertEqual(second['summary'],third['summary'])
                request['step']=2;path.write_text(json.dumps(request))
                with self.assertRaises(ValueError):run_request(path)
            self.assertEqual(first['position'],'post_norm');self.assertEqual(first['layers'],[])
            self.assertFalse(first['baseline_cache_used']);self.assertTrue(second['baseline_cache_used'])
            trained=[x for x in second['results'] if x['arm']=='trained']
            self.assertTrue(all('post_norm_target_logit_delta' in x for x in trained))

    def test_needle_reports_output_logit_effect_and_correct_wrong_candidate_groups(self):
        model=CPUInference(self.body,self.rosa)
        # Exact-prefix-like tiny input: long matching suffix predicts token 3.
        stem=[1,2,3,1];suffix=[2]
        case={'stem_ids':stem,'suffix_ids':{'qa':suffix},'answer_ids':[3], 'secret':'123456', 'needle_start':0}
        tokenizer=SimpleNamespace(idx2token={i:str(i).encode() for i in range(32)})
        with torch.inference_mode():_,state=model.features(torch.tensor([stem]),return_state=True)
        row=evaluate(model,tokenizer,case,'qa',state,2)
        self.assertTrue(row['post_norm_candidate_is_target']);self.assertLess(row['post_norm_logit_decomposition_max_abs_error'],2e-6)
        row.update(arm='trained',mode='qa',length=8)
        wrong={**row,'post_norm_candidate_is_target':False}
        summary=summarize([row,wrong])[0]
        self.assertEqual(summary['post_norm_correct_candidate_cases'],1)
        self.assertEqual(summary['post_norm_wrong_candidate_cases'],1)
        self.assertIn('post_norm_correct_candidate_target_top1_after',summary)


if __name__=='__main__':unittest.main()
