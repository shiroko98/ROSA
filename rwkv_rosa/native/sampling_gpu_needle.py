"""Controlled sampling and decode-budget comparison on fixed NIAH inputs."""
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
from .needle_eval import build_case, clean_decode, answer_scores
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT=Path(__file__).resolve().parents[2]


@torch.inference_mode()
def worker(folder,slot,workers):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((folder/'manifest.json').read_text())
    model=NativeGPU(manifest['train_dir'],manifest['adapter_snapshot'],132096)
    model.tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    evaluator=FastEvaluator(model)
    audit=evaluator.audit(model.tokenizer.encode('User: 你好。\n\nAssistant:'))
    write_json(folder/f'runtime_audit_{slot}.json',{'checkpoint_step':model.meta['step'],
        'base_sha256':model.meta['base_sha256_native_order'],'adapter_sha256':model.meta['adapter_state_sha256'],
        'audit':audit,'gpu':torch.cuda.get_device_name(),'same_prompt_and_sampling_seed_across_arms':True})
    for index in range(slot,len(manifest['cases']),workers):
        spec=manifest['cases'][index]
        case=build_case(model.tokenizer,spec['length'],spec['depth'],spec['seed'],spec['difficulty'],prompt_template='rwkv')
        for policy in manifest['sampling_configurations']:
            path=folder/policy['id']/'cases'/f'{index:04d}.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='complete':continue
            started=time.monotonic();write_json(path,{'status':'running','spec':spec,'pid':os.getpid()})
            rows=[];diagnostics={}
            for mode in ('qa','exact_prefix'):
                seed=1729+index*10007+(10000019 if mode=='exact_prefix' else 0)
                sampling={'temperature':policy['temperature'],'top_p':policy['top_p'],'top_k':0,'seed':seed}
                budget=manifest['max_new_tokens_by_mode'][mode]
                answers,seconds=evaluator.evaluate(case,mode,('base','trained'),budget,sampling=sampling)
                for row in answers:
                    text=row['generated_text'];unclosed='<think>' in text and '</think>' not in text
                    final='' if unclosed else text.rsplit('</think>',1)[-1]
                    row.update(case_index=index,key=case['key'],seconds=seconds,sampling_configuration=policy['id'],
                        max_new_tokens=budget,unclosed_thinking=unclosed,final_answer_text=final,
                        final_answer_passkey_match=answer_scores(final,case['secret'])['passkey_match'])
                    for cap in (24,64,128,256,512):
                        prefix=clean_decode(model.tokenizer,row['generated_ids'][:cap])
                        row[f'passkey_match_at_{cap}']=answer_scores(prefix,case['secret'])['passkey_match']
                    rows.append(row)
                    with (folder/f'answers_{slot}.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                first=answers[0]
                diagnostics[mode]={'suffix_only_exact_match':first['suffix_only_exact_match'],
                    'teacher_forced_address_token_hits':first['address_token_hits'],'answer_tokens':len(first['answer_ids']),
                    'candidate_ids':first['address_candidates'],'target_ids':first['answer_ids'],
                    'prompt_last_match_len':first['query_last_match_len']}
                print('PAIRED',json.dumps({'policy':policy['id'],'index':index,'length':spec['length'],'mode':mode,
                    'seconds':seconds,'hits':[(row['arm'],row['passkey_match'],row['final_answer_passkey_match']) for row in answers]}),flush=True)
            write_json(path,{'status':'complete','spec':spec,'case_id':case['id'],'stem_sha256':case['stem_sha256'],
                'secret':case['secret'],'key':case['key'],'needle_start':case['needle_start'],'needle_tokens':case['needle_tokens'],
                'prompt_sha256':{mode:hashlib.sha256(torch.tensor(case['stem_ids']+suffix,dtype=torch.int64).numpy().tobytes()).hexdigest()
                    for mode,suffix in case['suffix_ids'].items()},'results':rows,'address_diagnostics':diagnostics,
                'sampling':policy,'wall_seconds':time.monotonic()-started})
    write_json(folder/f'worker_complete_{slot}.json',{'status':'complete','peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--slot',type=int,required=True);p.add_argument('--workers',type=int,required=True);args=p.parse_args()
    folder=Path(args.output)
    try:worker(folder,args.slot,args.workers)
    except Exception:
        write_json(folder/f'failure_{args.slot}.json',{'error':traceback.format_exc()});raise


if __name__=='__main__':main()
