"""Convert native full-state ROSA checkpoints and run matched CPU-only diagnostics.

ROSA is nonlinear, input-dependent retrieval: it CANNOT be statically merged into
RWKV matrices. The flat .pth retains rosa_cpu.*; use the ROSA-aware loader below.
CPU FP32 reference inference is a paired diagnostic, not native BF16 CUDA parity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rosa_runtime import RosaAddressBatch
from rwkv_rosa.backbone import build_backbone, infer_config, load_base_state
from rwkv_rosa.freeze_audit import parameter_hash
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.position import normalize_metadata, scale_options
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset
from third_party.rwkv_lm_v7_native.src.sft_split import compute_sft_shuffled_split_indices


def write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def state_hash(state, parameter_names=None):
    # Native nn.Module registration order need not match a .pth mapping's order.
    # Persist the native order for external-base checks, without changing legacy
    # fingerprints or sorting/reinterpreting previously saved checkpoints.
    if parameter_names is not None:
        if len(parameter_names) != len(set(parameter_names)) or set(parameter_names) != set(state):
            raise ValueError('External base parameter names differ from native model.')
        items = lambda: ((name, state[name]) for name in parameter_names)
    else:
        items = lambda: iter(state.items())
    return parameter_hash(SimpleNamespace(named_parameters=items))


def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp); tmp.replace(path)


def convert(checkpoint, base_path, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    # Resolve once: live training may advance native-last.ckpt during evaluation.
    checkpoint = Path(checkpoint).resolve(strict=True)
    c = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = c['state_dict']; meta = normalize_metadata(c['native_rosa'])
    if c.get('native_rosa_checkpoint_kind') == 'adapter_optimizer_loop_with_external_base_v1':
        external = load_base_state(base_path)
        if state_hash(external, c['native_rosa_frozen_base'].get('parameter_names')) != c['native_rosa_frozen_base']['sha256']:
            raise ValueError('External base hash does not match the adapter checkpoint.')
        state = {**external, **state}
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a native tensor state_dict.')
    base = {k:v for k,v in state.items() if not k.startswith('rosa_cpu.')}
    rosa = {k[len('rosa_cpu.'):]:v for k,v in state.items() if k.startswith('rosa_cpu.')}
    original = load_base_state(base_path)
    if set(base) != set(original) or any(base[k].dtype != original[k].dtype or
            not torch.equal(base[k], original[k]) for k in base):
        raise ValueError('Checkpoint backbone differs from the specified frozen base model.')
    if not rosa or any(not bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()):
        raise ValueError('Missing/nonfinite ROSA weights.')
    cfg = infer_config(base)
    manifest = {'format':'native_rwkv_rosa_inference_v1', 'source_checkpoint':str(checkpoint),
                'global_step':int(c['global_step']), 'base_model':str(Path(base_path).resolve()),
                'base_sha256':state_hash(base), 'backbone_exactly_matches_base':True,
                'full_state_sha256':state_hash(state), 'rosa_state_sha256':state_hash(rosa),
                'config':asdict(cfg), 'rosa_metadata':meta,
                'full_model':'rwkv7-rosa.pth', 'adapter':'rosa-adapter.pth',
                'warning':'NOT a static merge: vanilla RWKV ignores/rejects rosa_cpu.*; requires ROSA-aware runtime.'}
    atomic_save(state, output/'rwkv7-rosa.pth')
    atomic_save({'format':manifest['format'], 'state_dict':rosa, 'config':asdict(cfg),
                 'base_sha256':manifest['base_sha256'], 'native_rosa':meta}, output/'rosa-adapter.pth')
    # Verify serialized output, not merely the in-memory source mapping.
    restored = torch.load(output/'rwkv7-rosa.pth', map_location='cpu', weights_only=True, mmap=True)
    assert state_hash(restored) == manifest['full_state_sha256']
    manifest['serialized_weights_verified'] = True
    write_json(output/'manifest.json', manifest)
    return manifest


class CPUInference(nn.Module):
    """Same layer-before-block injection as native training, using CPU FP32 RWKV."""
    def __init__(self, backbone, rosa=None):
        super().__init__(); self.backbone = backbone; self.rosa = rosa
        self.eval().requires_grad_(False)
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('Comparison requires every model parameter on CPU FP32.')

    def features(self, ids, *, state=None, history=None, return_state=False):
        if ids.device.type != 'cpu' or ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty CPU [batch,time] token IDs.')
        b = self.backbone
        if self.rosa is not None and getattr(self.rosa, 'scale_mode', 'fixed') == 'logit_gap':
            self.rosa.bind_output_head(b.head)
        if self.rosa is not None:
            full = ids if history is None else history
            if not torch.equal(full[:, -ids.shape[1]:], ids):
                raise ValueError('Address history must end in the current token chunk.')
            self.rosa.begin(full)
            if full.shape[1] != ids.shape[1]:
                a = self.rosa.addresses; sl = slice(-ids.shape[1], None)
                self.rosa.addresses = RosaAddressBatch(**{
                    key:getattr(a,key)[:,sl] for key in ('addr_ids','raw_match_lens','fired_match_lens','valid_mask','special_mask')
                }, source=a.source)
        if return_state and state is None: state = b.init_state(ids.shape[0], device='cpu')
        x = b.emb(ids); v = torch.empty_like(x); next_state = []
        for layer, block in enumerate(b.blocks):
            if self.rosa is not None: x = self.rosa.inject(layer, x)
            previous = None if state is None else state[layer]
            if return_state:
                x, v, new = block(x,v,state=previous,return_state=True); next_state.append(new)
            else: x,v = block(x,v,state=previous)
        x = b.ln_out(x)
        self.post_norm_probe = None
        if self.rosa is not None:
            before = x[:, -1].detach().clone() if self.rosa.position == 'post_norm' else None
            x = self.rosa.inject_post_norm(x)
            if before is not None:
                self.post_norm_probe = {'base_hidden': before, 'delta_hidden': (x[:, -1]-before).detach()}
        return (x,next_state) if return_state else x

    def forward(self, ids):
        return self.backbone.head(self.features(ids))


def load_cpu(export, variant='trained'):
    export = Path(export); manifest = json.loads((export/'manifest.json').read_text())
    state = torch.load(export/'rwkv7-rosa.pth', map_location='cpu', weights_only=True, mmap=True)
    base = {k:v for k,v in state.items() if not k.startswith('rosa_cpu.')}
    b = build_backbone(infer_config(base), dtype=torch.float32, state_dict=base)
    if variant == 'base':return CPUInference(b)
    if variant not in ('initial','trained'):raise ValueError('Unknown variant.')
    meta = normalize_metadata(manifest['rosa_metadata']); layers = tuple(int(i) for i in meta['layers'].split(',') if i)
    rosa = CPUImprovedROSA(b.emb,layers,meta['scale'],device='cpu',position=meta['position'],output_norm=meta['output_norm'], **scale_options(meta))
    if variant == 'trained':
        rosa.load_state_dict({k[len('rosa_cpu.'):]:v for k,v in state.items() if k.startswith('rosa_cpu.')},strict=True)
    return CPUInference(b,rosa)


def prepare_windows(prefix, heldout, rows, length, seed, split_seed=42):
    tokens = MMapIndexedDataset(prefix); masks = MMapIndexedDataset(prefix+'.mask')
    if not np.array_equal(tokens.sizes,masks.sizes): raise ValueError('Mask lengths mismatch.')
    if heldout <= 8 or rows > heldout-8: raise ValueError('Insufficient unseen held-out rows.')
    # Native training shuffles BEFORE splitting; the last physical rows are NOT
    # the held-out set. Reuse its exact seed-42 split, excluding the eight rows
    # repeatedly used in online monitoring.
    train_indices, eval_indices = compute_sft_shuffled_split_indices(len(tokens),heldout,seed=split_seed)
    choices = np.random.default_rng(seed).choice(eval_indices[8:],size=rows,replace=False)
    if set(choices) & set(train_indices): raise AssertionError('Evaluation/train row leakage.')
    windows=[]; manifest=[]
    for row in choices:
        mask = masks[int(row)]; active = np.flatnonzero(mask[1:])
        if not len(active):raise ValueError('Selected held-out row has no supervision.')
        # Deterministic supervision-centred window, not the training loader and
        # not full 192K context. Preserve every original token/mask in the span.
        center = int(active[len(active)//2]); left = max(0,min(center-length//2,len(mask)-length-1))
        ids = np.array(tokens[int(row)][left:left+length+1],dtype=np.int64)
        weights = np.array(mask[left+1:left+length+1],dtype=np.float32)
        if len(ids)!=length+1 or weights.sum()<=0:raise ValueError('Invalid diagnostic window.')
        windows.append((torch.from_numpy(ids[:-1]).unsqueeze(0),torch.from_numpy(ids[1:]).unsqueeze(0),torch.from_numpy(weights).unsqueeze(0)))
        manifest.append({'physical_row':int(row),'start':left,'input_tokens':length,
                         'supervised_tokens':float(weights.sum()),
                         'token_mask_sha256':hashlib.sha256(ids.tobytes()+weights.tobytes()).hexdigest()})
    return windows,manifest


@torch.inference_mode()
def score(model, window, chunk=128):
    ids,targets,mask = window; hidden = model.features(ids)
    hidden=hidden.reshape(-1,hidden.shape[-1]);y=targets.flatten();w=mask.flatten()
    active=torch.nonzero(w!=0,as_tuple=False).flatten(); nll=correct=0.
    for idx in active.split(chunk):
        logits=model.backbone.head(hidden[idx]).float()
        losses=F.cross_entropy(logits,y[idx],reduction='none')
        nll += float((losses*w[idx]).sum());correct+=float(((logits.argmax(-1)==y[idx])*w[idx]).sum())
    count=float(w.sum())
    if count<=0 or not math.isfinite(nll):raise ValueError('Invalid evaluation loss.')
    return {'nll_sum':nll,'mask_weight':count,'loss':nll/count,'correct_weight':correct,'accuracy':correct/count}


@torch.inference_mode()
def generate(model, ids, max_new=64):
    history=torch.tensor([ids],dtype=torch.long)
    hidden,state=model.features(history,return_state=True)
    logits=model.backbone.head(hidden[:,-1]);generated=[]
    for _ in range(max_new):
        token=int(logits.argmax(-1));generated.append(token)
        if token in (0,65531,65532): break
        current=torch.tensor([[token]]);history=torch.cat((history,current),dim=1)
        hidden,state=model.features(current,state=state,history=history,return_state=True)
        logits=model.backbone.head(hidden[:,-1])
    return generated


def run(args):
    if torch.cuda.is_available():raise RuntimeError('Launch with CUDA_VISIBLE_DEVICES=\"\"; CPU-only test required.')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','pid':os.getpid(),'device':'cpu','dtype':'float32','cuda_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'threads':args.threads,'seed':args.seed,'config':vars(args),
            'scope':'bounded held-out supervision-centred windows, context reset per window; NOT full-192K quality evaluation',
            'precision_note':'Both arms use identical CPU FP32 reference recurrence; not claimed numerically identical to native BF16 CUDA.',
            'arms':{},'comparisons':{}}
    write_json(out/'report.json',report)
    export=out/'export';manifest=convert(args.checkpoint,args.base,export);report['conversion']=manifest
    if Path(args.data).resolve() != Path(manifest['rosa_metadata']['data_file']).resolve():
        raise ValueError('Evaluation data prefix differs from native training metadata.')
    report['split_policy'] = {'method':'native shuffled split before epoch shuffling','seed':args.split_seed,
                              'heldout_rows':args.heldout,'exclude_first_monitoring_rows':8}
    windows,wm=prepare_windows(args.data,args.heldout,args.rows,args.length,args.seed,args.split_seed)
    report['windows']=wm
    from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER
    from jinja2 import Template
    root=Path(__file__).resolve().parents[2]
    tokenizer=TRIE_TOKENIZER(str(root/'rwkv_vocab_v20230424.txt'))
    template=Template((root/'third_party/rwkv_lm_v7/chat_template.jinja').read_text())
    prompts=[('计算 17 + 25，只输出数字。','42'),('计算 9 × 8，只输出数字。','72'),
             ('将英文 apple 翻译成中文，只输出译文。','苹果'),('中国的首都是哪座城市？只输出城市名。','北京'),
             ('请只输出下面这串字符，不要添加其他内容：ROSA-7391','ROSA-7391'),
             ('记住：小林的取件码是 583206。小林去了图书馆，然后回家。问：小林的取件码是多少？只输出取件码。','583206'),
             ('把列表 3, 1, 2 从小到大排序，只输出排序后的列表。','1, 2, 3'),
             ('用一句话解释什么是机器学习。',None)]
    generations={}
    for variant in ('base','initial','trained'):
        model=load_cpu(export,variant);results=[];started=time.monotonic()
        for i,window in enumerate(windows):
            t=time.monotonic();r=score(model,window);r.update(window=i,seconds=time.monotonic()-t);results.append(r)
            print('CPU EVAL',variant,i,json.dumps(r),flush=True)
            with (out/'window_metrics.jsonl').open('a') as f:f.write(json.dumps({'variant':variant,**r})+'\n')
        total=sum(x['mask_weight'] for x in results);nll=sum(x['nll_sum'] for x in results)
        report['arms'][variant]={'loss':nll/total,'ppl':math.exp(nll/total),'accuracy':sum(x['correct_weight'] for x in results)/total,
                                  'supervised_tokens':total,'windows':results,'evaluation_seconds':time.monotonic()-started}
        write_json(out/'report.json',report)
        samples=[]
        if variant in ('base','trained'):
            for question,expected in prompts:
                prompt=template.render(messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':question}],add_generation_prompt=True,no_add_thinking=True)
                t=time.monotonic();ids=generate(model,tokenizer.encode(prompt),args.max_new_tokens)
                text=b''.join(tokenizer.idx2token.get(i,f'<unk:{i}>'.encode()) for i in ids).decode('utf-8',errors='replace')
                sample={'question':question,'expected':expected,'generated_text':text,'seconds':time.monotonic()-t,'tokens':len(ids),'stop_token':ids[-1] if ids else None}
                samples.append(sample);print('CPU GENERATION',variant,json.dumps(sample,ensure_ascii=False),flush=True)
            generations[variant]=samples;write_json(out/'generations.json',generations)
        del model
    for other in ('base','initial'):
        a=report['arms'][other];b=report['arms']['trained'];delta=[x['loss']-y['loss'] for x,y in zip(a['windows'],b['windows'])]
        weights=np.array([x['mask_weight'] for x in a['windows']]);gains=np.array([x['nll_sum']-y['nll_sum'] for x,y in zip(a['windows'],b['windows'])])
        rng=np.random.default_rng(args.seed);indices=rng.integers(0,len(windows),size=(2000,len(windows)))
        boots=gains[indices].sum(1)/weights[indices].sum(1)
        report['comparisons'][f'trained_vs_{other}']={'loss_reduction':a['loss']-b['loss'],'relative_loss_reduction':1-b['loss']/a['loss'],
            'relative_ppl_reduction':1-b['ppl']/a['ppl'],'accuracy_delta':b['accuracy']-a['accuracy'],
            'windows_improved':sum(x>0 for x in delta),'windows_total':len(delta),'paired_window_bootstrap_loss_gain_95pct':np.quantile(boots,[.025,.975]).tolist()}
    report['status']='complete';write_json(out/'report.json',report)
    print('FINAL',json.dumps({'arms':{k:{a:v for a,v in r.items() if a!='windows'} for k,r in report['arms'].items()},'comparisons':report['comparisons']},indent=2),flush=True)
    if args.wandb:
        from rwkv_rosa.native.wandb_auth import configure_wandb_auth
        configure_wandb_auth()
        import wandb
        run=wandb.init(project='ROSA-RWKV7',entity=os.environ.get('WANDB_ENTITY','mobius'),job_type='cpu-evaluation',
            name=f'cpu-base-vs-rosa-step{manifest["global_step"]}',dir=str(out),config={**vars(args),'device':'cpu','dtype':'float32'},save_code=False,settings=wandb.Settings(disable_git=True))
        metrics={f'cpu_eval/{arm}/{k}':v for arm,r in report['arms'].items() for k,v in r.items() if isinstance(v,(float,int))}
        run.log(metrics);report['wandb']={'id':run.id,'url':run.url};run.finish()
        remote=wandb.Api(timeout=20).run(f'{run.entity}/{run.project}/{run.id}')
        report['wandb']['verified']=remote.summary.get('cpu_eval/trained/loss') is not None
        write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',required=True);p.add_argument('--heldout',type=int,default=2117)
    p.add_argument('--rows',type=int,default=16);p.add_argument('--length',type=int,default=2048)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--wandb',action='store_true')
    args=p.parse_args()
    if min(args.rows,args.length,args.threads,args.max_new_tokens)<=0:p.error('Positive evaluation sizes required.')
    run(args)
