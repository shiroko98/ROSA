"""Ten-case, three-variant GPU NIAH with persistent RWKV and SAM state."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

from rwkv_rosa.native.cpu_compare import write_json
from rwkv_rosa.native.gpu_needle import NativeGPU,grouped_summary,rank,tail_pad,STOPS
from rwkv_rosa.native.gpu_state import StatefulNative,IncrementalAddress,state_nbytes
from rwkv_rosa.native.needle_eval import build_case,answer_scores,clean_decode
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT=Path(__file__).resolve().parents[2]


def ten_cases(tokenizer,length,seed,template):
    for i,depth in enumerate((.1,.3,.5,.7,.9)):
        for j,mode in enumerate(('qa','exact_prefix')):
            # Fixed coverage, selected before seeing any result. Five cases per
            # difficulty, five per mode; both templates use the same identifiers.
            difficulty=('unrelated','similar_archives')[(i+j)%2]
            case=build_case(tokenizer,length,depth,seed+i*997,difficulty,prompt_template=template)
            yield case,mode


class FastEvaluator:
    def __init__(self,model):
        self.model=model;self.runtime=StatefulNative(model.body)

    @torch.inference_mode()
    def logits(self,hidden,address):
        body=self.model.body;adapter=self.model.adapter
        adapter.addresses=address
        trained=adapter.inject_post_norm(hidden)
        return {'base':body.head(hidden).float()[0,-1],
                'trained':body.head(trained).float()[0,-1]},trained-hidden

    @torch.inference_mode()
    def evaluate(self,case,mode,arms,max_tokens,heartbeat=None):
        prompt=case['stem_ids']+case['suffix_ids'][mode]
        torch.cuda.synchronize();began=time.perf_counter()
        hidden,prompt_state=self.runtime.features(torch.tensor([prompt],device='cuda'))
        last=hidden[:,-1:].clone();del hidden
        sam=IncrementalAddress();sam.prefill(prompt)
        logits,delta=self.logits(last,sam.last)
        torch.cuda.synchronize();prefill_seconds=time.perf_counter()-began
        first={a:logits[a].clone() for a in arms};state={a:prompt_state for a in arms}
        # These state tensors are immutable inputs. Native kernel allocates new
        # states, so the two arms can safely share prompt state before branching.
        sams={a:sam.clone() for a in arms};generated={a:[] for a in arms};done=set()
        calls=0;decode_seconds=0.;generated_steps=0
        for step in range(max_tokens):
            for arm in arms:
                if arm in done:continue
                token=int(logits[arm].argmax());generated[arm].append(token)
                if token in STOPS:done.add(arm)
            if len(done)==len(arms) or step+1==max_tokens:break
            # Share only when the full generated prefix is equal, never merely
            # when the latest token coincides after previously diverging.
            groups={}
            for arm in arms:
                if arm not in done:groups.setdefault(tuple(generated[arm]),[]).append(arm)
            for continuation,matching in groups.items():
                token=continuation[-1];torch.cuda.synchronize();start=time.perf_counter()
                h,next_state=self.runtime.features(torch.tensor([[token]],device='cuda'),state[matching[0]])
                calls+=1
                for arm in matching:
                    state[arm]=next_state
                    address=sams[arm].update_one(token)
                    if arm=='base':logits[arm]=self.model.body.head(h).float()[0,-1]
                    else:
                        self.model.adapter.addresses=address
                        logits[arm]=self.model.body.head(self.model.adapter.inject_post_norm(h)).float()[0,-1]
                torch.cuda.synchronize();decode_seconds+=time.perf_counter()-start
            generated_steps=step+1
            if heartbeat:heartbeat(step+1)
        # Teacher-forced probe runs AFTER generation, continuing an independent
        # prompt state/SAM branch. Target tokens never enter free generation.
        answer=case['answer_ids'];teacher={a:[first[a]] for a in arms};diag=sam.clone()
        candidates=[int(sam.last.addr_ids[0,0])];hits=[bool(sam.last.valid_mask[0,0]) and candidates[0]==answer[0]]
        if len(answer)>1:
            h,_=self.runtime.features(torch.tensor([answer[:-1]],device='cuda'),prompt_state)
            for i,token in enumerate(answer[:-1]):
                address=diag.update_one(token)
                scores,_=self.logits(h[:,i:i+1],address)
                candidates.append(int(address.addr_ids[0,0]));hits.append(bool(address.valid_mask[0,0]) and candidates[-1]==answer[i+1])
                for arm in arms:teacher[arm].append(scores[arm])
        target=torch.tensor(answer,device='cuda');rows=[]
        for arm in arms:
            ids=generated[arm];text=clean_decode(self.model.tokenizer,ids)
            scores=torch.stack(teacher[arm]);nll=float(F.cross_entropy(scores,target))
            if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite passkey logits.')
            row={'case':case['id'],'length':case['length'],'depth':case['depth'],'seed':case['seed'],
                'difficulty':case['difficulty'],'mode':mode,'prompt_template':case['prompt_template'],
                'arm':arm,'secret':case['secret'],'prompt_tokens':len(prompt),'stem_sha256':case['stem_sha256'],
                'answer_ids':answer,'address_candidates':candidates,'suffix_only_exact_match':all(hits),
                'query_last_match_len':int(sam.last.fired_match_lens[0,0]),'address_token_hits':sum(hits),
                'generated_ids':ids,'generated_text':text,**answer_scores(text,case['secret']),
                'hit_generation_limit':len(ids)==max_tokens and ids[-1] not in STOPS,
                'first_answer_token_rank':rank(first[arm],answer[0]),
                'first_answer_token_probability':float(first[arm].softmax(-1)[answer[0]]),'answer_nll':nll,
                'prefill_seconds_paired':prefill_seconds,'decode_seconds_paired':decode_seconds,
                'single_token_forward_calls_paired':calls,'decode_seconds_per_call':decode_seconds/max(1,calls),
                'rwkv_state_MiB_per_arm':state_nbytes(prompt_state)/2**20,
                'full_context_prefill_calls':1,'incremental_decode':True,'incremental_sam':True}
            if arm=='trained':
                # Recompute prompt logits only from saved final prompt vector;
                # do not replay the full history for these diagnostics.
                before,change=self.logits(last,sam.last)
                row.update(post_norm_target_logit_delta=float(before['trained'][answer[0]]-before['base'][answer[0]]),
                    post_norm_target_rank_before=rank(before['base'],answer[0]),
                    post_norm_target_rank_after=rank(before['trained'],answer[0]),
                    post_norm_residual_zero_fraction=float((change==0).float().mean()))
            rows.append(row)
        torch.cuda.synchronize()
        return rows,time.perf_counter()-began

    @torch.inference_mode()
    def audit(self,prompt):
        ids=torch.tensor([prompt],device='cuda');n=len(prompt)
        reference=self.model.body._forward_features(tail_pad(ids))[:,:n]
        full,full_state=self.runtime.features(ids)
        a,state=self.runtime.features(ids[:,:-1]);b,split_state=self.runtime.features(ids[:,-1:],state)
        expected=self.model.body.head(reference[:,-1]).float()
        got=self.model.body.head(full[:,-1]).float();split=self.model.body.head(b[:,-1]).float()
        metrics={'tokens':n,'native_hidden_max_abs':float((full-reference).abs().max()),
            'native_logits_max_abs':float((got-expected).abs().max()),
            'split_hidden_max_abs':float((torch.cat((a,b),1)-full).abs().max()),
            'split_logits_max_abs':float((split-got).abs().max()),
            'native_top1':int(expected.argmax()),'stateful_top1':int(got.argmax()),'split_top1':int(split.argmax()),
            'note':'Split GEMM shape changes may cause BF16 differences; no bitwise-parity promise.'}
        if metrics['native_hidden_max_abs']>.01:raise ValueError(f'Stateful prefill differs from native: {metrics}')
        return metrics


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--length',type=int,default=8192)
    p.add_argument('--seed',type=int,default=20261109);p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--wandb',action='store_true');args=p.parse_args(argv)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'request.json').exists():raise SystemExit('Use a fresh output directory.')
        torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        request={**vars(args),'training':False,'source_step':200,'incremental_rwkv':True,'incremental_sam':True,
            'variants':[{'arm':'base','prompt_template':'rwkv'},{'arm':'trained','prompt_template':'rwkv'},
                        {'arm':'trained','prompt_template':'chatml'}],
            'case_count_per_variant':10,'generated_answers_expected':30,
            'length_definition':'document tokens excluding query; 8K quick subset, not a completed 192K generation benchmark',
            'decode_policy':'greedy; identical cap across all three variants; generation-limit is reported separately',
            'started_at':datetime.now().astimezone().isoformat()}
        write_json(out/'request.json',request)
        status={'status':'loading','pid':os.getpid(),'completed_case_templates':0,'completed_answers':0}
        write_json(out/'status.json',status);rows=[];run=None;start=time.perf_counter()
        def publish():
            write_json(out/'status.json',status)
            write_json(out/'report.json',{'request':request,'status':dict(status),'wall_seconds':time.perf_counter()-start,
                'summary':grouped_summary(rows,('arm','prompt_template')),
                'by_mode':grouped_summary(rows,('arm','prompt_template','mode')),
                'by_difficulty':grouped_summary(rows,('arm','prompt_template','difficulty')),
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
        try:
            model=NativeGPU(args.train_dir,args.checkpoint,args.length+1024)
            model.tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'));evaluator=FastEvaluator(model)
            request.update(source_step=model.meta['step'],base_sha256=model.meta['base_sha256_native_order'],
                adapter_sha256=model.meta['adapter_state_sha256'],source_training_run_id=(Path(args.train_dir)/'wandb_run_id.txt').read_text().strip())
            if args.wandb:
                from rwkv_rosa.native.wandb_auth import configure_wandb_auth
                configure_wandb_auth()
                import wandb
                run=wandb.init(project='ROSA-RWKV7',entity=os.environ.get('WANDB_ENTITY','mobius'),
                    group='needle-of-'+request['source_training_run_id'],job_type='gpu-stateful-needle',
                    name=f'3B-step{model.meta["step"]}-stateful-10cases-3variants',config=request,dir=str(out),
                    settings=wandb.Settings(init_timeout=30))
                (out/'wandb_run_id.txt').write_text(run.id+'\n')
            write_json(out/'request.json',request)
            # Real checkpoint audit: short prompt + the complete 8K prompt.
            cases={template:list(ten_cases(model.tokenizer,args.length,args.seed,template)) for template in ('rwkv','chatml')}
            status['status']='auditing';publish()
            audit=[evaluator.audit(model.tokenizer.encode('User: 你好。\n\nAssistant:'))]
            first_case,mode=cases['rwkv'][0];audit.append(evaluator.audit(first_case['stem_ids']+first_case['suffix_ids'][mode]))
            write_json(out/'numerical_audit.json',audit)
            print('STATE AUDIT',json.dumps(audit),flush=True)
            # Interleave templates for each fixed case to produce three-way
            # answers immediately, rather than waiting for one entire matrix.
            for i in range(10):
                for template,arms in (('rwkv',('base','trained')),('chatml',('trained',))):
                    case,mode=cases[template][i]
                    status.update(status='evaluating',case_index=i+1,active_mode=mode,prompt_template=template,generated_tokens=0);publish()
                    def heartbeat(n):status['generated_tokens']=n;write_json(out/'status.json',status)
                    answer,seconds=evaluator.evaluate(case,mode,arms,args.max_new_tokens,heartbeat)
                    for row in answer:row['wall_seconds_paired']=seconds
                    with (out/'results.jsonl').open('a') as f:
                        for row in answer:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        f.flush();os.fsync(f.fileno())
                    rows.extend(answer);status['completed_case_templates']+=1;status['completed_answers']=len(rows);publish()
                    print('FAST ANSWERS',json.dumps(answer,ensure_ascii=False),flush=True)
                    if run:
                        metrics={'completed_answers':len(rows),'source_training_step':model.meta['step']}
                        for summary in grouped_summary(rows,('arm','prompt_template')):
                            for k,v in summary.items():
                                if isinstance(v,(float,int)):metrics[f'{summary["arm"]}/{summary["prompt_template"]}/{k}']=v
                        run.log(metrics)
            status.update(status='complete',finished_at=datetime.now().astimezone().isoformat());publish()
            if run:
                for name in ('request.json','numerical_audit.json','report.json','results.jsonl'):
                    run.save(str(out/name),base_path=str(out),policy='now')
        except BaseException:
            status.update(status='failed',error=traceback.format_exc());publish();raise
        finally:
            if run:run.finish(exit_code=0 if status['status']=='complete' else 1)


if __name__=='__main__':main()
