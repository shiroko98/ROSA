"""Native checkpoint export and paired CPU inference regression tests."""
from pathlib import Path
import tempfile
import unittest
import argparse
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
import torch

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.cpu_compare import CPUInference, convert, load_cpu, score, generate, prepare_windows, run


class CPUCompareTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.base=build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8)).eval().requires_grad_(False)
        self.rosa=CPUImprovedROSA(self.base.emb,layers=(0,),device='cpu')
        self.ids=torch.tensor([[1,2,1,2,3,1,2,3]])

    def test_plain_cpu_forward_matches_reference(self):
        with torch.inference_mode():
            torch.testing.assert_close(CPUInference(self.base)(self.ids),self.base(self.ids),atol=0,rtol=0)

    def test_rosa_streaming_and_causal_address_history(self):
        model=CPUInference(self.base,self.rosa)
        with torch.inference_mode():
            full=model(self.ids)
            a,state=model.features(self.ids[:,:5],return_state=True)
            b,state=model.features(self.ids[:,5:],state=state,history=self.ids,return_state=True)
            got=model.backbone.head(torch.cat((a,b),dim=1))
            torch.testing.assert_close(full,got,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(full[:,:5],model(self.ids[:,:5]),atol=2e-5,rtol=2e-5)
            with self.assertRaises(ValueError): model.features(self.ids[:,:0])
            with self.assertRaises(ValueError): model.features(self.ids[:,:1],history=self.ids)
            self.assertLessEqual(len(generate(model,self.ids[0,:3].tolist(),max_new=3)),3)

    def test_native_shuffled_holdout_and_end_to_end_cpu_report(self):
        from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset
        from third_party.rwkv_lm_v7_native.src.sft_split import compute_sft_shuffled_split_indices
        import json
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);prefix=str(tmp/'data')
            for suffix,dtype in [('',np.uint16),('.mask',np.uint8)]:
                a=np.tile(np.array([1,2,1,2,3,1,2,1,2,3,1,2,3,1,2,3,1],dtype=dtype),(30,1))
                if suffix:a[:]=1
                a.tofile(prefix+suffix+'.bin')
                with MMapIndexedDataset.Index.writer(prefix+suffix+'.idx',dtype) as w:
                    w.write([17]*30,list(range(31)))
            windows,metadata=prepare_windows(prefix,12,4,8,7)
            train,ev=compute_sft_shuffled_split_indices(30,12,seed=42)
            self.assertFalse(set(x['physical_row'] for x in metadata)&set(train))
            self.assertTrue(set(x['physical_row'] for x in metadata)<=set(ev[8:]))
            with self.assertRaises(ValueError):prepare_windows(prefix,8,1,8,7)
            base=tmp/'base.pth';self.base.bfloat16();torch.save(self.base.state_dict(),base)
            rosa=CPUImprovedROSA(self.base.emb,layers=(0,),device='cpu')
            state=dict(self.base.state_dict());state.update({'rosa_cpu.'+k:v for k,v in rosa.state_dict().items()})
            checkpoint=tmp/'native.ckpt'
            torch.save({'state_dict':state,'global_step':7,'native_rosa':{'layers':'0','scale':.05,'data_file':prefix}},checkpoint)
            args=argparse.Namespace(output=str(tmp/'out'),base=str(base),checkpoint=str(checkpoint),data=prefix,
                heldout=12,rows=4,length=8,seed=7,split_seed=42,threads=2,max_new_tokens=2,wandb=True)
            fake_tokenizer=SimpleNamespace(encode=lambda _: [1,2,1],idx2token={i:b'x' for i in range(32)})
            wb=Mock();wb.init.return_value.entity='mobius';wb.init.return_value.project='ROSA-RWKV7';wb.init.return_value.id='test'
            wb.init.return_value.url='http://example.test/run';wb.Api.return_value.run.return_value.summary={'cpu_eval/trained/loss':1.}
            with patch('third_party.rwkv_lm_v7.tokenizer.TRIE_TOKENIZER',return_value=fake_tokenizer), \
                 patch('torch.set_num_interop_threads'),patch.dict('sys.modules',{'wandb':wb}), \
                 patch('rwkv_rosa.native.wandb_auth.configure_wandb_auth'):
                report=run(args)
            self.assertEqual(report['status'],'complete');self.assertTrue(report['wandb']['verified'])
            self.assertEqual(report['comparisons']['trained_vs_initial']['loss_reduction'],0.)
            self.assertTrue((Path(args.output)/'generations.json').exists())
            with patch('torch.cuda.is_available',return_value=True):
                with self.assertRaises(RuntimeError):run(args)

    def test_full_checkpoint_export_reload_and_backbone_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);base=tmp/'base.pth';ckpt=tmp/'native.ckpt';out=tmp/'export'
            self.base.bfloat16();torch.save(self.base.state_dict(),base)
            self.rosa=CPUImprovedROSA(self.base.emb,layers=(0,),device='cpu')
            state=dict(self.base.state_dict());state.update({'rosa_cpu.'+k:v for k,v in self.rosa.state_dict().items()})
            meta={'layers':'0','scale':.05,'ctx_len':8,'data_file':'toy','epoch_steps':1}
            torch.save({'state_dict':state,'native_rosa':meta,'global_step':7},ckpt)
            manifest=convert(ckpt,base,out)
            self.assertTrue(manifest['serialized_weights_verified']);self.assertEqual(manifest['global_step'],7)
            self.assertTrue((out/'rosa-adapter.pth').exists())
            compact=tmp/'compact.ckpt'
            from rwkv_rosa.native.cpu_compare import state_hash
            torch.save({'state_dict':{k:v for k,v in state.items() if k.startswith('rosa_cpu.')},
                'native_rosa':meta,'global_step':7,'native_rosa_checkpoint_kind':'adapter_optimizer_loop_with_external_base_v1',
                'native_rosa_frozen_base':{'sha256':state_hash(dict(self.base.state_dict()))}},compact)
            compact_manifest=convert(compact,base,tmp/'compact_export')
            self.assertEqual(compact_manifest['full_state_sha256'],manifest['full_state_sha256'])
            wrong_base=tmp/'wrong_base.pth';torch.save({**self.base.state_dict(),'emb.weight':self.base.emb.weight+1},wrong_base)
            with self.assertRaises(ValueError):convert(compact,wrong_base,tmp/'wrong_export')

            model=load_cpu(out,'trained');model0=load_cpu(out,'initial');plain=load_cpu(out,'base')
            with torch.inference_mode():
                torch.testing.assert_close(model(self.ids),model0(self.ids),atol=0,rtol=0)
                torch.testing.assert_close(plain(self.ids),plain.backbone(self.ids),atol=0,rtol=0)
            weights=torch.tensor([[0.,1.,0.,1.,1.,0.,1.,0.]])
            result=score(model,(self.ids,self.ids,weights),chunk=2)
            self.assertEqual(result['mask_weight'],4)
            logits=model(self.ids);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,32),self.ids.flatten(),reduction='none')
            self.assertAlmostEqual(result['loss'],float((loss*weights.flatten()).sum()/4),places=5)
            with self.assertRaises(ValueError):score(model,(self.ids,self.ids,torch.zeros_like(weights)))
            with self.assertRaises(ValueError):load_cpu(out,'invalid')
            state['emb.weight']=state['emb.weight']+1
            torch.save({'state_dict':state,'native_rosa':meta,'global_step':7},ckpt)
            with self.assertRaises(ValueError):convert(ckpt,base,tmp/'bad')


if __name__=='__main__':unittest.main()
