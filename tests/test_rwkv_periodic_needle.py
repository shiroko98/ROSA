"""Physical-split virtual windows and every-eval CPU needle monitor integration."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import tempfile
import unittest
import numpy as np
import torch
from rwkv_rosa.backbone import build_backbone,RWKVConfig
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.cpu_compare import state_hash
from rwkv_rosa.native.periodic_needle import run_request,run_native_needle_eval
from rwkv_rosa.native.windowed_data import configure_windowed_dataset,read_window
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


def write_data(prefix):
    for suffix,dtype in [('',np.uint16),('.mask',np.uint8)]:
        a=np.arange(4*34,dtype=dtype).reshape(4,34)
        if suffix:a[:]=1
        a.tofile(str(prefix)+suffix+'.bin')
        with MMapIndexedDataset.Index.writer(str(prefix)+suffix+'.idx',dtype) as w:w.write([34]*4,list(range(5)))


class PeriodicNeedleTest(unittest.TestCase):
    def test_virtual_windows_cover_every_target_and_preserve_physical_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sft';write_data(p)
            body=MMapIndexedDataset(str(p));mask=MMapIndexedDataset(str(p)+'.mask')
            def make(indices):return SimpleNamespace(args=SimpleNamespace(rosa_windowed_data=1,ctx_len=16),data_type='sft_binidx',data=body,mask_data=mask,sft_doc_indices=indices,sft_doc_start=0,sft_doc_count=len(indices),sft_split='train')
            d=configure_windowed_dataset(make(np.array([1,3])));ev=configure_windowed_dataset(make(np.array([0,2])))
            self.assertEqual(d.sft_doc_count,6);self.assertFalse(set(d.rosa_physical_indices)&set(ev.rosa_physical_indices))
            for logical_row,physical_row in enumerate(d.rosa_physical_indices):
                got=[]
                for slot in range(3):
                    ids,w=read_window(d,logical_row*3+slot);got.extend(ids[1:].tolist());self.assertTrue(w.all())
                self.assertEqual(got,body[int(physical_row)][1:].tolist())
            plain=make(np.array([1]));plain.args.rosa_windowed_data=0;self.assertIs(configure_windowed_dataset(plain),plain)
            plain.args.rosa_windowed_data=1;plain.data_type='binidx'
            with self.assertRaises(ValueError):configure_windowed_dataset(plain)
            plain.data_type='sft_binidx';plain.data=SimpleNamespace(sizes=np.array([3,4]))
            with self.assertRaises(ValueError):configure_windowed_dataset(plain)
            plain.data=body;plain.args.ctx_len=0
            with self.assertRaises(ValueError):configure_windowed_dataset(plain)

    def test_cpu_worker_baseline_cache_and_native_hook(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);b=build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8)).bfloat16()
            base=tmp/'base.pth';torch.save(dict(reversed(list(b.state_dict().items()))),base)
            rosa=CPUImprovedROSA(b.emb,layers=(1,),device='cpu');adapter=tmp/'adapter.pth';torch.save(rosa.state_dict(),adapter)
            request={'step':2,'base':str(base),'adapter':str(adapter),'threads':2,'seed':1,'length':1024,'depths':[.5],
                'difficulties':['unrelated','similar_archives'],'max_new_tokens':2,'baseline_cache':str(tmp/'base_cache.json'),
                'result':str(tmp/'result.json'),'layers':[1],'scale':.05,'backbone_sha256':state_hash(dict(b.state_dict())), 'backbone_parameter_names':list(b.state_dict())}
            path=tmp/'request.json';path.write_text(json.dumps(request))
            def case(tok,length,depth,seed,difficulty,prompt_template='chatml'):return {'id':difficulty,'difficulty':difficulty,'length':length,'depth':depth,'stem_ids':[1,2,1,2],'secret':'123456','prompt_template':prompt_template}
            def evaluate(*args):return {'passkey_match':False,'strict_digits_only':False,'answer_nll':1.,'first_answer_token_rank':2}
            def patches():
                from contextlib import ExitStack
                stack=ExitStack();stack.enter_context(patch('torch.set_num_interop_threads'))
                stack.enter_context(patch('rwkv_rosa.native.periodic_needle.build_case',side_effect=case))
                stack.enter_context(patch('rwkv_rosa.native.periodic_needle.evaluate',side_effect=evaluate))
                stack.enter_context(patch('rwkv_rosa.native.periodic_needle.address_diagnostic',return_value={'hit':True}))
                return stack
            with patches():
                result=run_request(path);self.assertFalse(result['baseline_cache_used']);self.assertEqual(len(result['results']),8)
                result=run_request(path);self.assertTrue(result['baseline_cache_used'])
                request['max_new_tokens']=3;path.write_text(json.dumps(request))
                with self.assertRaises(ValueError):run_request(path)
            args=SimpleNamespace(proj_dir=str(tmp/'native'),load_model=str(base),rosa_scale=.05,rosa_needle_every_eval=1,
                rosa_needle_length=1024,rosa_needle_depths='0.5',rosa_needle_max_new_tokens=2,rosa_needle_cpu_threads=2,rosa_needle_timeout=120)
            model=SimpleNamespace(rosa_cpu=rosa,_rosa_frozen_sha256=request['backbone_sha256'],named_parameters=b.named_parameters);trainer=SimpleNamespace(my_wandb=Mock())
            def fake_subprocess(command,**kw):
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'')
                r=json.loads(Path(command[-1]).read_text());result['step']=r['step']
                Path(r['result']).write_text(json.dumps(result));return SimpleNamespace(returncode=0)
            with patch('rwkv_rosa.native.periodic_needle.subprocess.run',side_effect=fake_subprocess):run_native_needle_eval(trainer,model,args,7)
            self.assertTrue((Path(args.proj_dir)/'needle/history.jsonl').exists());trainer.my_wandb.log.assert_called_once()
            self.assertFalse((Path(args.proj_dir)/'needle/step000000007/adapter.pth').exists())
            args.rosa_needle_every_eval=0;run_native_needle_eval(trainer,model,args,8)
        with patch('torch.cuda.is_available',return_value=True):
            with self.assertRaises(RuntimeError):run_request('unused')


if __name__=='__main__':unittest.main()

class FrozenPrefixTest(unittest.TestCase):
    def test_native_forward_skips_only_checkpoint_for_no_grad_prefix(self):
        import ast
        from torch import nn
        from torch.utils.checkpoint import checkpoint
        source=Path('third_party/rwkv_lm_v7_native/src/model.py').read_text()
        tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RWKV')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_features')
        scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(Path('third_party/rwkv_lm_v7_native/src/model.py').resolve()),'exec'),scope)
        class Block(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(16,16)
            def forward(self,x,v):return torch.tanh(self.linear(x))+v*0.,x
        model=nn.Module();model.emb=nn.Embedding(32,16);model.blocks=nn.ModuleList([Block(),Block()]);model.ln_out=nn.LayerNorm(16)
        model.requires_grad_(False);model.rosa_cpu=CPUImprovedROSA(model.emb,layers=(1,),device='cpu')
        model.args=SimpleNamespace(ctx_len=8,grad_cp=1,rosa_no_checkpoint_layers=())
        # Avoid multiplying uninitialized v in this toy block, unlike native's
        # first block which always initializes its own v_first.
        class First(Block):
            def forward(self,x,v):return torch.tanh(self.linear(x)),x
        model.blocks[0]=First().requires_grad_(False)
        ids=torch.tensor([[1,2,1,2,3,1,2,3]])
        with patch('torch.utils.checkpoint.checkpoint',wraps=checkpoint) as cp:
            out=scope['_forward_features'](model,ids);out.square().mean().backward()
            self.assertEqual(cp.call_count,1)
        self.assertTrue(any(p.grad is not None for p in model.rosa_cpu.parameters()))
        self.assertTrue(all(p.grad is None for p in model.blocks.parameters()))
        model.rosa_cpu.zero_grad();model.args.rosa_no_checkpoint_layers=(1,)
        with patch('torch.utils.checkpoint.checkpoint',wraps=checkpoint) as cp:
            scope['_forward_features'](model,ids).square().mean().backward();self.assertEqual(cp.call_count,0)
