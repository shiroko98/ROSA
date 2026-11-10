"""CPU passkey evaluation on every native SFT eval, without allocating a second GPU model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from rwkv_rosa.native.cpu_compare import CPUInference,write_json,state_hash
from rwkv_rosa.native.needle_eval import build_case,address_diagnostic,make_engine,evaluate,summarize
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.backbone import load_base_state,infer_config,build_backbone
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

from .position import scale_options

ROOT=Path(__file__).resolve().parents[2]


def run_request(request):
    if torch.cuda.is_available():raise RuntimeError('Periodic needle worker must be CPU only.')
    r=json.loads(Path(request).read_text());torch.set_num_threads(r['threads']);torch.set_num_interop_threads(1)
    tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    cases=[]
    for difficulty in r['difficulties']:
        for i,depth in enumerate(r['depths']):
            cases.append(build_case(tokenizer,r['length'],depth,r['seed']+i*997,difficulty,
                                    prompt_template=r.get('prompt_template', 'chatml')))
    # Base hash and complete token-case hashes key the baseline: never reuse an
    # incompatible base, length, query, seed or decoding rule.
    signature=hashlib.sha256(json.dumps({'backbone_sha256':r['backbone_sha256'],
        'cases':cases,'max_new_tokens':r['max_new_tokens'],'precision':'cpu_fp32_reference'},sort_keys=True).encode()).hexdigest()
    cache=Path(r['baseline_cache']);cached=json.loads(cache.read_text()) if cache.exists() else None
    if cached is not None and cached['signature']!=signature:raise ValueError('Periodic needle baseline signature mismatch.')
    base=load_base_state(r['base']);body=build_backbone(infer_config(base),dtype=torch.float32,state_dict=base)
    base_fingerprint=state_hash(base,r.get('backbone_parameter_names'))
    if base_fingerprint!=r['backbone_sha256']:raise ValueError('CPU worker base hash differs from frozen native model.')
    del base
    results=[];diag={};engine=make_engine()
    model=CPUInference(body)
    # Synthetic inputs are fixed and intentionally NOT used for training.
    for arm in ('base','trained'):
        if arm=='base' and cached is not None:
            results.extend(cached['results']);continue
        if arm=='trained':
            adapter=torch.load(r['adapter'],map_location='cpu',weights_only=r.get('adapter_format')!='native_checkpoint',mmap=True)
            model.rosa=CPUImprovedROSA(body.emb,tuple(r['layers']),r['scale'],device='cpu',address_impl='compiled_cpu',position=r.get('position','block'),output_norm=r.get('output_norm','none'),**scale_options(r))
            if r.get('adapter_format') == 'native_checkpoint':
                from .position import normalize_metadata
                meta=normalize_metadata(adapter['native_rosa'])
                if (int(adapter['global_step'])!=r['step'] or meta['position']!=r.get('position','block')
                        or meta['layers']!=','.join(map(str,r['layers'])) or meta['scale']!=r['scale']
                        or meta['output_norm']!=r.get('output_norm','none') or scale_options(meta)!=scale_options(r)
                        or adapter['native_rosa_frozen_base']['sha256']!=r['backbone_sha256']):
                    raise ValueError('Standalone checkpoint step/position/scale/base mismatch.')
                adapter={k[len('rosa_cpu.'):]:v for k,v in adapter['state_dict'].items() if k.startswith('rosa_cpu.')}
            model.rosa.load_state_dict(adapter,strict=True);model.eval().requires_grad_(False)
        for case in cases:
            _,state=model.features(torch.tensor([case['stem_ids']]),return_state=True)
            for mode in ('qa','exact_prefix'):
                t=time.monotonic();row=evaluate(model,tokenizer,case,mode,state,r['max_new_tokens'])
                row.update(arm=arm,mode=mode,difficulty=case['difficulty'],case=case['id'],length=case['length'],depth=case['depth'],secret=case['secret'],seconds=time.monotonic()-t)
                results.append(row)
                write_json(Path(r['result']).with_name('progress.json'),
                    {'status':'running','step':r['step'],'results':results,'completed_answers':len(results)})
                print('PERIODIC NEEDLE',json.dumps(row,ensure_ascii=False),flush=True)
        if arm=='base' and cached is None:
            write_json(cache,{'signature':signature,'results':[x for x in results if x['arm']=='base']})
    if cached is None:write_json(cache,{'signature':signature,'results':[x for x in results if x['arm']=='base']})
    for case in cases:
        for mode in ('qa','exact_prefix'):
            diag[case['difficulty']+'/'+case['id']+'/'+mode]=address_diagnostic(case,mode,engine)
    groups=[]
    for difficulty in r['difficulties']:
        groups.extend([{**x,'difficulty':difficulty} for x in summarize([y for y in results if y['difficulty']==difficulty])])
    report={'status':'complete','step':r['step'],'device':'cpu','dtype':'float32','signature':signature,
            'position':r.get('position','block'),'output_norm':r.get('output_norm','none'),**scale_options(r),'layers':r['layers'],'length':r['length'],
            'prompt_template':r.get('prompt_template', 'chatml'),
            'case_definition':{'depths':r['depths'],'difficulties':r['difficulties'],'seed':r['seed']},
            'baseline_cache_used':cached is not None,'results':results,'summary':groups,'address_diagnostics':diag,
            'scope':'Fixed synthetic CPU FP32 passkey generation at the recorded document length; not native BF16 numerical parity or comprehensive NIAH.'}
    write_json(r['result'],report)
    return report


def run_native_needle_eval(trainer,model,args,step):
    if not getattr(args,'rosa_needle_every_eval',0):return
    out=Path(args.proj_dir)/'needle';out.mkdir(parents=True,exist_ok=True)
    stepdir=out/f'step{int(step):09d}';stepdir.mkdir(exist_ok=False)
    # Save only learned adapter weights. Full periodic training checkpoints are
    # independent and preserve optimizer state; these evaluation snapshots do not.
    adapter=stepdir/'adapter.pth'
    torch.save({k:v.detach().cpu() for k,v in model.rosa_cpu.state_dict().items()},adapter)
    from rwkv_rosa.native.integration import frozen_parameters
    request={'backbone_parameter_names':[name for name, _ in frozen_parameters(model)],'step':int(step),'base':args.load_model,'adapter':str(adapter),
             'backbone_sha256':getattr(model,'_rosa_frozen_sha256'),
             'position':model.rosa_cpu.position,'layers':list(model.rosa_cpu.layers),'scale':args.rosa_scale,
             'output_norm':model.rosa_cpu.output_norm,**model.rosa_cpu.scale_options,
             'length':args.rosa_needle_length,'depths':[float(x) for x in args.rosa_needle_depths.split(',')],
             'difficulties':['unrelated','similar_archives'],'seed':20261109,
             'max_new_tokens':args.rosa_needle_max_new_tokens,'threads':args.rosa_needle_cpu_threads,
             'baseline_cache':str(out/'base_cache.json'),'result':str(stepdir/'report.json')}
    write_json(stepdir/'request.json',request)
    start=time.monotonic()
    with (stepdir/'worker.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-m','rwkv_rosa.native.periodic_needle','--request',str(stepdir/'request.json')],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(args.rosa_needle_cpu_threads),MKL_NUM_THREADS=str(args.rosa_needle_cpu_threads)),
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.rosa_needle_timeout)
    report=json.loads((stepdir/'report.json').read_text())
    report['wall_seconds']=time.monotonic()-start;write_json(stepdir/'report.json',report)
    metrics={'needle/evaluation_seconds':report['wall_seconds'],'needle/step':int(step)}
    for row in report['summary']:
        prefix=f'needle/{row["difficulty"]}/{row["arm"]}/{row["mode"]}'
        for key in ('passkey_accuracy','passkey_correct','strict_correct','mean_answer_nll','cases'):
            metrics[prefix+'/'+key]=row[key]
        for key,value in row.items():
            if key.startswith('post_norm_'):metrics[prefix+'/'+key]=value
    if getattr(trainer,'my_wandb',None) is not None:trainer.my_wandb.log(metrics,step=int(step))
    with (out/'history.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
    # The raw answers/diagnostics remain; redundant adapter snapshots are removed
    # after successful evaluation, not confused with hourly resumable checkpoints.
    adapter.unlink()
    print('NATIVE NEEDLE EVAL:',json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True)
    run_request(p.parse_args().request)
