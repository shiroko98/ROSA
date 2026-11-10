"""Extend the original short-passkey protocol to256K without changing its query."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from .cpu_compare import write_json
from .fast_gpu_needle import FastEvaluator
from .gpu_needle import NativeGPU, STOPS
from .needle_eval import build_case, clean_decode, answer_scores
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT=Path(__file__).resolve().parents[2]


def budget_scores(tokenizer, generated, secret, cap):
    ids=generated[:cap];text=clean_decode(tokenizer,ids)
    unclosed='<think>' in text and '</think>' not in text
    final='' if unclosed else text.rsplit('</think>',1)[-1]
    return {'budget':cap,'generated_tokens':len(ids),'generated_text':text,'final_answer_text':final,
            'content_hit':answer_scores(text,secret)['passkey_match'],
            'final_hit':answer_scores(final,secret)['passkey_match'],
            'strict_final':final.strip()==secret,'strict_whole_output':text.strip()==secret,
            'unclosed_thinking':unclosed,
            'hit_generation_limit':len(ids)==cap and ids[-1] not in STOPS}


@torch.inference_mode()
def worker(folder,slot,workers):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((folder/'manifest.json').read_text())
    capacity=max(s['length'] for s in manifest['cases'])+max(manifest['max_new_tokens_by_mode'].values())+2048
    model=NativeGPU(manifest['train_dir'],manifest['adapter_snapshot'],capacity)
    model.tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    evaluator=FastEvaluator(model)
    audit=evaluator.audit(model.tokenizer.encode('User: 你好。\n\nAssistant:'))
    write_json(folder/f'runtime_audit_{slot}.json',{'checkpoint_step':model.meta['step'],
        'base_sha256':model.meta['base_sha256_native_order'],'adapter_sha256':model.meta['adapter_state_sha256'],
        'audit':audit,'gpu':torch.cuda.get_device_name(),'configured_context_capacity':capacity,
        'training':False,'scale_options':model.adapter.scale_options})
    for index in range(slot,len(manifest['cases']),workers):
        spec=manifest['cases'][index];path=folder/'cases'/f'{index:04d}.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':continue
        write_json(path,{'status':'running','spec':spec,'pid':os.getpid()});started=time.monotonic()
        case=build_case(model.tokenizer,spec['length'],spec['depth'],spec['seed'],spec['difficulty'],prompt_template='rwkv')
        rows=[];traces={}
        for mode in ('qa','exact_prefix'):
            events=[]
            def observe(phase,emitted,address,stats):
                match=int(address.fired_match_lens[0,0]);valid=bool(address.valid_mask[0,0])
                gain=float(stats.get('rosa_dynamic_scale_max',0.));rms=float(stats.get('rosa_residual_rms',0.))
                if (not valid or match<6) and (gain!=0 or rms!=0):raise AssertionError('Below-threshold ROSA residual')
                if not 0<=gain<=.300001:raise AssertionError('Invalid dynamic gain')
                events.append({'phase':phase,'emitted_tokens':emitted,'match_tokens':match,
                               'candidate':int(address.addr_ids[0,0]),'valid':valid,'effective_scale':gain,'residual_rms':rms})
            seed=1729+spec['reference_case_index']*10007+(10000019 if mode=='exact_prefix' else 0)
            sampling={'temperature':0.,'top_p':1.,'top_k':0,'seed':seed}
            answers,seconds=evaluator.evaluate(case,mode,('base','trained'),manifest['max_new_tokens_by_mode'][mode],sampling=sampling,observer=observe)
            for row in answers:
                row.update(case_index=index,reference_case_index=spec['reference_case_index'],key=case['key'],
                    seconds=seconds,answer_tokens=len(case['answer_ids']),question_tokens=len(case['suffix_ids'][mode]),
                    max_new_tokens=manifest['max_new_tokens_by_mode'][mode],
                    by_budget={str(cap):budget_scores(model.tokenizer,row['generated_ids'],case['secret'],cap)
                               for cap in manifest['score_budgets'][mode]})
                if row['arm']=='trained':
                    active=[e for e in events if e['residual_rms']>0]
                    row.update(rosa_prediction_steps=len(events),rosa_nonzero_steps=len(active),
                               rosa_prompt_nonzero=events[0]['residual_rms']>0,
                               rosa_first_nonzero_step=active[0]['emitted_tokens'] if active else None)
                rows.append(row)
            traces[mode]=events
            print('PAIRED',json.dumps({'index':index,'depth':spec['depth'],'mode':mode,'seconds':seconds,
                  'scores':{r['arm']:{cap:s['final_hit'] for cap,s in r['by_budget'].items()} for r in answers}}),flush=True)
        record={k:v for k,v in case.items() if k not in ('stem_ids','suffix_ids')}
        record.update(status='complete',spec=spec,results=rows,
            prompt_sha256={mode:hashlib.sha256(torch.tensor(case['stem_ids']+suffix,dtype=torch.int64).numpy().tobytes()).hexdigest()
                           for mode,suffix in case['suffix_ids'].items()},
            suffix_text={mode:model.tokenizer.decode(ids) for mode,ids in case['suffix_ids'].items()},
            wall_seconds=time.monotonic()-started)
        write_json(folder/'traces'/path.name,traces);write_json(path,record)
    write_json(folder/f'worker_complete_{slot}.json',{'status':'complete','peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    folder=Path(a.output)
    try:worker(folder,a.slot,a.workers)
    except Exception:
        write_json(folder/f'failure_{a.slot}.json',{'error':traceback.format_exc()});raise


if __name__=='__main__':main()
