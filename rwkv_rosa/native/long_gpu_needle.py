"""Paired long-key/long-question/long-answer GPU retrieval with gate traces."""
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
from .gpu_needle import NativeGPU
from .long_answer_cases import build_long_case,long_answer_scores
from .needle_eval import clean_decode
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT=Path(__file__).resolve().parents[2]


@torch.inference_mode()
def worker(folder,slot,workers):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((folder/'manifest.json').read_text())
    max_context=max(x['length'] for x in manifest['cases'])+max(manifest['max_new_tokens_by_mode'].values())+2048
    model=NativeGPU(manifest['train_dir'],manifest['adapter_snapshot'],max_context)
    model.tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    evaluator=FastEvaluator(model)
    audit=evaluator.audit(model.tokenizer.encode('User: 你好。\n\nAssistant:'))
    write_json(folder/f'runtime_audit_{slot}.json',{'checkpoint_step':model.meta['step'],
        'base_sha256':model.meta['base_sha256_native_order'],'adapter_sha256':model.meta['adapter_state_sha256'],
        'audit':audit,'gpu':torch.cuda.get_device_name(),'dynamic_min_match':model.adapter.scale_options['dynamic_min_match'],
        'dynamic_max':model.adapter.scale_options['dynamic_max'],'training':False})
    for index in range(slot,len(manifest['cases']),workers):
        spec=manifest['cases'][index];path=folder/'cases'/f'{index:04d}.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':continue
        started=time.monotonic();write_json(path,{'status':'running','spec':spec,'pid':os.getpid()})
        case=build_long_case(model.tokenizer,spec['length'],spec['depth'],spec['seed'],spec['answer_kind'],spec['nominal_answer_tokens'])
        rows=[];diagnostics={};traces={}
        for mode in ('qa','exact_prefix'):
            events=[]
            def observe(phase,emitted,address,stats):
                match=int(address.fired_match_lens[0,0]);valid=bool(address.valid_mask[0,0])
                scale=float(stats.get('rosa_dynamic_scale_max',0.));rms=float(stats.get('rosa_residual_rms',0.))
                if (not valid or match<6) and (scale!=0 or rms!=0):
                    raise AssertionError('ROSA activated below the checkpoint match threshold')
                if scale>.3+1e-6:raise AssertionError('Dynamic gain exceeds checkpoint cap')
                events.append({'phase':phase,'emitted_tokens':emitted,'match_tokens':match,
                    'candidate':int(address.addr_ids[0,0]),'valid':valid,'effective_scale':scale,'residual_rms':rms})
            budget=manifest['max_new_tokens_by_mode'][mode]
            answers,seconds=evaluator.evaluate(case,mode,('base','trained'),budget,observer=observe)
            traces[mode]=events
            active=[x for x in events if x['residual_rms']>0]
            for row in answers:
                for key in ('passkey_match','strict_digits_only','first_six_digit_answer'):row.pop(key,None)
                row.update(case_index=index,key=case['key'],answer_kind=case['answer_kind'],
                    nominal_answer_tokens=case['nominal_answer_tokens'],actual_answer_tokens=case['actual_answer_tokens'],
                    answer_chars=case['answer_chars'],target_text=case['secret'],max_new_tokens=budget,
                    seconds=seconds,actual_document_depth=case['actual_document_depth'],
                    question_tokens=len(case['suffix_ids'][mode]),
                    **long_answer_scores(model.tokenizer,row['generated_text'],case['secret']))
                if row['arm']=='trained':
                    row.update(rosa_free_prediction_steps=len(events),rosa_nonzero_steps=len(active),
                        rosa_first_nonzero_step=active[0]['emitted_tokens'] if active else None,
                        rosa_max_match_tokens=max(x['match_tokens'] for x in events),
                        rosa_prompt_nonzero=events[0]['residual_rms']>0,
                        rosa_threshold_verified=True)
                rows.append(row)
                with (folder/f'answers_{slot}.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            first=answers[0]
            diagnostics[mode]={'suffix_only_exact_match':first['suffix_only_exact_match'],
                'teacher_forced_address_token_hits':first['address_token_hits'],'answer_tokens':len(first['answer_ids']),
                'candidate_ids':first['address_candidates'],'target_ids':first['answer_ids'],
                'prompt_last_match_len':first['query_last_match_len']}
            print('PAIRED',json.dumps({'index':index,'length':spec['length'],'answer_kind':spec['answer_kind'],
                'answer_tokens':case['actual_answer_tokens'],'mode':mode,'seconds':seconds,
                'results':[(x['arm'],x['final_contains_full_answer'],x['whole_final_exact_match'],x['best_correct_prefix_fraction']) for x in answers],
                'rosa_nonzero_steps':len(active)}),flush=True)
        record={k:v for k,v in case.items() if k not in ('stem_ids','suffix_ids')}
        record.update(status='complete',spec=spec,results=rows,address_diagnostics=diagnostics,
            prompt_sha256={mode:hashlib.sha256(torch.tensor(case['stem_ids']+suffix,dtype=torch.int64).numpy().tobytes()).hexdigest()
                for mode,suffix in case['suffix_ids'].items()},
            suffix_text={mode:model.tokenizer.decode(ids) for mode,ids in case['suffix_ids'].items()},
            wall_seconds=time.monotonic()-started)
        write_json(folder/'traces'/f'{index:04d}.json',traces)
        write_json(path,record)
    write_json(folder/f'worker_complete_{slot}.json',{'status':'complete','peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--slot',type=int,required=True);p.add_argument('--workers',type=int,required=True);args=p.parse_args()
    folder=Path(args.output)
    try:worker(folder,args.slot,args.workers)
    except Exception:
        write_json(folder/f'failure_{args.slot}.json',{'error':traceback.format_exc()});raise


if __name__=='__main__':main()
