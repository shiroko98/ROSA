"""CPU-only paired passkey retrieval test: neural answers versus suffix hits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F

from rosa_addressing import RosaAddressEngine
from rwkv_rosa.native.cpu_compare import convert, load_cpu, write_json
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER


ROOT=Path(__file__).resolve().parents[2]


def make_engine():
    # Exactly the native adapter's literal-token SAM policy, not semantic search.
    return RosaAddressEngine(min_match_len=1,pad_id=0,
        special_ids={0,65530,65531,65532,65533,65534},forbid_special_target=True,
        backend='sam',sequence_mode='online_sam',online_sam_impl='fast')


def clean_decode(tokenizer, ids):
    visible=[]
    for token in ids:
        if token in (0,65531,65532):break
        visible.append(token)
    return b''.join(tokenizer.idx2token.get(i,f'<unk:{i}>'.encode()) for i in visible).decode('utf-8',errors='replace')


def answer_scores(text, secret):
    # Content retrieval is separate from obeying a digits-only output format.
    found=re.search(r'(?<!\d)\d{6}(?!\d)',text)
    return {'passkey_match':bool(found and found.group()==secret),
            'strict_digits_only':text.strip()==secret,
            'first_six_digit_answer':found.group() if found else None}


def build_case(tokenizer, length, depth, seed, difficulty='unrelated', prompt_template='chatml'):
    rng=random.Random(seed)
    key=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8))
    secret=str(rng.randrange(100000,1000000))
    anchor=f'\n档案[{key}]的口令是：'
    anchor_ids=tokenizer.encode(anchor);answer_ids=tokenizer.encode(secret)
    needle=tokenizer.encode(anchor+secret+'。\n')
    # Direct completion is naturally tokenized, not an artificially spliced
    # answer boundary that changes the suffix engine's matching difficulty.
    if needle[:len(anchor_ids)]!=anchor_ids or needle[len(anchor_ids):len(anchor_ids)+len(answer_ids)]!=answer_ids:
        raise ValueError('Tokenizer merged across the answer boundary.')
    if prompt_template not in ('chatml','rwkv'):
        raise ValueError('Unknown prompt template.')
    header=tokenizer.encode('<|im_start|>System: You are a helpful assistant.<|im_end|>\n<|im_start|>User: 请阅读文档并查找指定档案的口令。\n文档开始：\n'
        if prompt_template == 'chatml' else 'User: 请阅读文档并查找指定档案的口令。\n文档开始：\n')
    budget=length-len(header)-len(needle)
    if budget<32:raise ValueError('Context is too short for header and needle.')
    paragraphs=[];tokens=[];number=0;used_keys={key};used_secrets={secret}
    if difficulty not in ('unrelated','similar_archives'):raise ValueError('Unknown difficulty.')
    places=['图书馆','花园','车站','办公室','展览馆','仓库','公园','教学楼']
    actions=['整理资料','检查设备','记录天气','清点书籍','核对日程','安排会议','巡查道路','测量温度']
    while len(tokens)<budget:
        number+=1
        if difficulty=='similar_archives':
            other=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8))
            while other in used_keys:other=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8))
            code=str(rng.randrange(100000,1000000))
            while code in used_secrets:code=str(rng.randrange(100000,1000000))
            used_keys.add(other);used_secrets.add(code)
            paragraph=f'\n档案[{other}]的口令是：{code}。\n'
        else:
            paragraph=f'记录{number}：工作人员在{rng.choice(places)}{rng.choice(actions)}，共处理{rng.randrange(1,99)}项事务，随后完成交接。\n'
        paragraphs.append(paragraph);tokens.extend(tokenizer.encode(paragraph))
    filler=tokens[:budget];at=int(budget*depth)
    stem=header+filler[:at]+needle+filler[at:]
    assert len(stem)==length
    decoded=b''.join(tokenizer.idx2token[i] for i in stem).decode('utf-8',errors='replace')
    if decoded.count(secret)!=1:raise AssertionError('Secret must occur exactly once before querying.')
    assistant_boundary = '<|im_end|>\n<|im_start|>Assistant: ' if prompt_template == 'chatml' else '\n\nAssistant:'
    qa=tokenizer.encode(f'\n文档结束。\n问题：档案[{key}]的口令是什么？只输出六位数字，不要解释。'+assistant_boundary)
    completion=tokenizer.encode('\n文档结束。请补全下面的条目，只填写口令。'+assistant_boundary)+anchor_ids
    for suffix in (qa,completion):
        if secret in b''.join(tokenizer.idx2token[i] for i in suffix).decode('utf-8',errors='replace'):
            raise AssertionError('Answer leaked into query.')
    return {'id':f'L{length}_D{depth:g}_S{seed}','length':length,'depth':depth,'seed':seed,'difficulty':difficulty,
            'key':key,'secret':secret,'answer_ids':answer_ids,'stem_ids':stem,'prompt_template':prompt_template,
            'needle_start':len(header)+at,'needle_tokens':len(needle),
            'stem_sha256':hashlib.sha256(torch.tensor(stem,dtype=torch.int64).numpy().tobytes()).hexdigest(),
            'suffix_ids':{'qa':qa,'exact_prefix':completion}}


def address_diagnostic(case, mode, engine):
    prompt=case['stem_ids']+case['suffix_ids'][mode];target=case['answer_ids']
    # Teacher forcing is ONLY a diagnostic. All-hit is equivalent to exact
    # free-running suffix-only continuation at this fixed answer-token length:
    # if every earlier retrieved token is right, the history is identical.
    history=torch.tensor([prompt+target[:-1]],dtype=torch.long)
    address=engine.forward_seq(history,device=torch.device('cpu'))
    start=len(prompt)-1;end=start+len(target)
    got=address.addr_ids[0,start:end].tolist();valid=address.valid_mask[0,start:end].tolist()
    hits=[bool(v and a==b) for a,b,v in zip(got,target,valid)]
    return {'suffix_only_exact_match':all(hits),'teacher_forced_address_token_hits':sum(hits),
            'answer_tokens':len(target),'candidate_ids':got,'target_ids':target,
            'prompt_last_match_len':int(address.fired_match_lens[0,start]),
            'prompt_last_candidate_is_target':hits[0]}


@torch.inference_mode()
def evaluate(model, tokenizer, case, mode, prefix_state, max_new_tokens):
    suffix=case['suffix_ids'][mode];history=torch.tensor([case['stem_ids']+suffix])
    ids=torch.tensor([suffix]);hidden,state=model.features(ids,state=prefix_state,history=history,return_state=True)
    logits=model.backbone.head(hidden[:,-1]);first_logits=logits.clone()
    details={}
    if model.rosa is not None:
        a=model.rosa.addresses
        details={'query_last_address':int(a.addr_ids[0,-1]),'query_last_match_len':int(a.fired_match_lens[0,-1])}
        if model.rosa.position == 'post_norm':
            probe = model.post_norm_probe
            before = model.backbone.head(probe['base_hidden'])
            delta = model.backbone.head(probe['delta_hidden'])
            target_first = case['answer_ids'][0]
            candidate = int(a.addr_ids[0, -1]); valid = bool(a.valid_mask[0, -1])
            def rank(scores, token):
                return 1 + int((scores[0] > scores[0, token]).sum())
            details.update(post_norm_target_logit_before=float(before[0, target_first]),
                post_norm_target_logit_delta=float(delta[0, target_first]),
                post_norm_target_logit_after=float(logits[0, target_first]),
                post_norm_target_rank_before=rank(before, target_first),
                post_norm_target_rank_after=rank(logits, target_first),
                post_norm_candidate_valid=valid,
                post_norm_candidate_is_target=valid and candidate == target_first,
                post_norm_logit_decomposition_max_abs_error=float((logits-before-delta).abs().max()))
            if valid:
                details.update(post_norm_candidate_logit_delta=float(delta[0, candidate]),
                    post_norm_candidate_rank_before=rank(before, candidate),
                    post_norm_candidate_rank_after=rank(logits, candidate))
        if 0 in model.rosa.slots:
            embedding=model.backbone.emb(ids)
            residual=model.rosa.inject(0,embedding)-embedding
            details['query_last_injection_to_embedding_norm_ratio']=float(residual[:,-1].norm()/embedding[:,-1].norm().clamp_min(1e-12))
    # Preserve prompt state for the probability probe; the existing reference
    # recurrence returns new state tensors and does not mutate the input cache.
    prompt_state=state;generated=[];free_history=history
    for _ in range(max_new_tokens):
        token=int(logits.argmax(-1));generated.append(token)
        if token in (0,65531,65532):break
        current=torch.tensor([[token]]);free_history=torch.cat((free_history,current),dim=1)
        hidden,state=model.features(current,state=state,history=free_history,return_state=True)
        logits=model.backbone.head(hidden[:,-1])
    text=clean_decode(tokenizer,generated)
    target=torch.tensor(case['answer_ids'])
    teacher_logits=[first_logits]
    if len(target)>1:
        chunk=target[:-1].unsqueeze(0);full=torch.cat((history,chunk),dim=1)
        features,_=model.features(chunk,state=prompt_state,history=full,return_state=True)
        teacher_logits.append(model.backbone.head(features).squeeze(0))
    answer_logits=torch.cat(teacher_logits,dim=0)
    nll=float(F.cross_entropy(answer_logits,target))
    if not math.isfinite(nll):raise FloatingPointError('Nonfinite passkey NLL.')
    first=int(target[0]);rank=1+int((first_logits[0]>first_logits[0,first]).sum())
    return {**answer_scores(text,case['secret']),'generated_text':text,'generated_ids':generated,
            'hit_generation_limit':len(generated)==max_new_tokens and generated[-1] not in (0,65531,65532),
            'answer_nll':nll,'first_answer_token_rank':rank,
            'first_answer_token_probability':float(first_logits.softmax(-1)[0,first]),
            'prompt_tokens':len(history[0]),'needle_depth_actual':case['needle_start']/len(history[0]),**details}


def summarize(results):
    groups=defaultdict(list)
    for r in results:groups[(r['arm'],r['mode'],r['length'])].append(r)
    summary=[]
    for (a,m,l),rows in sorted(groups.items()):
        entry={'arm':a,'mode':m,'length':l,'cases':len(rows),
            'passkey_correct':sum(x['passkey_match'] for x in rows),
            'strict_correct':sum(x['strict_digits_only'] for x in rows),
            'passkey_accuracy':sum(x['passkey_match'] for x in rows)/len(rows),
            'mean_answer_nll':sum(x['answer_nll'] for x in rows)/len(rows),
            'mean_first_token_rank':sum(x['first_answer_token_rank'] for x in rows)/len(rows)}
        probes=[x for x in rows if 'post_norm_target_logit_delta' in x]
        if probes:
            entry['post_norm_mean_target_logit_delta']=sum(x['post_norm_target_logit_delta'] for x in probes)/len(probes)
            for label,subset in (('correct_candidate',[x for x in probes if x['post_norm_candidate_is_target']]),
                                 ('wrong_candidate',[x for x in probes if x['post_norm_candidate_valid'] and not x['post_norm_candidate_is_target']])):
                entry[f'post_norm_{label}_cases']=len(subset)
                if subset:
                    entry[f'post_norm_{label}_mean_target_logit_delta']=sum(x['post_norm_target_logit_delta'] for x in subset)/len(subset)
                    entry[f'post_norm_{label}_target_top1_before']=sum(x['post_norm_target_rank_before']==1 for x in subset)/len(subset)
                    entry[f'post_norm_{label}_target_top1_after']=sum(x['post_norm_target_rank_after']==1 for x in subset)/len(subset)
        summary.append(entry)
    return summary


def run(args):
    if torch.cuda.is_available():raise RuntimeError('CUDA must be hidden for this CPU evaluation.')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if getattr(args,'export',''):
        existing=Path(args.export).resolve(strict=True)
        manifest=json.loads((existing/'manifest.json').read_text())
        if Path(args.base).resolve()!=Path(manifest['base_model']).resolve() or Path(args.checkpoint).resolve()!=Path(manifest['source_checkpoint']).resolve():
            raise ValueError('Reused export does not match requested base/checkpoint.')
        (out/'export').symlink_to(existing,target_is_directory=True)
    else:
        manifest=convert(args.checkpoint,args.base,out/'export')
    tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    difficulty=getattr(args,'difficulty','unrelated')
    conditions=([(l,d) for l in (1024,8192) for d in (.1,.5,.9)]+[(32768,.5)] if difficulty=='unrelated'
                else [(8192,d) for d in (.1,.5,.9)]+[(32768,.5)])
    cases=[build_case(tokenizer,l,d,args.seed+i*997,difficulty=difficulty) for i,(l,d) in enumerate(conditions)]
    write_json(out/'cases.json',cases)
    engine=make_engine();diagnostics={}
    for case in cases:
        for mode in ('qa','exact_prefix'):
            diagnostics[case['id']+'/'+mode]=address_diagnostic(case,mode,engine)
    report={'status':'running','pid':os.getpid(),'device':'cpu','dtype':'float32',
            'checkpoint_step':manifest['global_step'],'conversion':manifest,'config':vars(args),
            'scope':f'Small synthetic {difficulty} passkey diagnostic; {len(cases)} unique target secrets, {2*len(cases)} paired mode/position cases; not full 192K or an official benchmark.',
            'prefix_cache_policy':'Reuse only common document prefill, no future query or answer in shared cache; same tokens/cache method in all arms.',
            'address_diagnostics':diagnostics,'results':[],'summary':[]}
    write_json(out/'report.json',report)
    for arm in ('base','initial','trained'):
        model=load_cpu(out/'export',arm)
        for case in cases:
            # Initialized ROSA is an explicitly labelled central-8K control,
            # not silently compared on a different aggregate evaluation set.
            if arm=='initial' and (case['length']!=8192 or case['depth']!=.5):continue
            started=time.monotonic();_,state=model.features(torch.tensor([case['stem_ids']]),return_state=True)
            prefill=time.monotonic()-started
            for mode in ('qa','exact_prefix'):
                started=time.monotonic();r=evaluate(model,tokenizer,case,mode,state,args.max_new_tokens)
                r.update(arm=arm,mode=mode,case=case['id'],length=case['length'],depth=case['depth'],key=case['key'],secret=case['secret'],prefill_seconds=prefill,query_decode_seconds=time.monotonic()-started)
                report['results'].append(r);report['summary']=summarize(report['results'])
                write_json(out/'report.json',report)
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                print('NEEDLE',json.dumps(r,ensure_ascii=False),flush=True)
        del model
    report['status']='complete';write_json(out/'report.json',report)
    if args.wandb:
        from rwkv_rosa.native.wandb_auth import configure_wandb_auth
        configure_wandb_auth()
        import wandb
        run=wandb.init(project='ROSA-RWKV7',entity=os.environ.get('WANDB_ENTITY','mobius'),job_type='cpu-needle-evaluation',
            name=f'cpu-needle-{difficulty}-base-vs-rosa-step{manifest["global_step"]}',dir=str(out),config=vars(args),save_code=False,settings=wandb.Settings(disable_git=True))
        metrics={f'needle/{r["arm"]}/{r["mode"]}/{r["length"]}/{k}':v for r in report['summary'] for k,v in r.items() if k not in ('length',) and isinstance(v,(float,int))}
        run.log(metrics);report['wandb']={'id':run.id,'url':run.url};run.finish()
        remote=wandb.Api(timeout=20).run(f'{run.entity}/{run.project}/{run.id}')
        report['wandb']['verified']=remote.summary.get('needle/trained/qa/8192/cases')==3
        write_json(out/'report.json',report)
    print('FINAL',json.dumps({'summary':report['summary'],'wandb':report.get('wandb')},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--max-new-tokens',type=int,default=24);p.add_argument('--wandb',action='store_true')
    p.add_argument('--difficulty',choices=('unrelated','similar_archives'),default='unrelated')
    p.add_argument('--export',default='',help='Reuse an already verified export of exactly this source checkpoint.')
    args=p.parse_args()
    if min(args.threads,args.max_new_tokens)<=0:p.error('Positive threads and decode length required.')
    torch.set_num_interop_threads(1)
    run(args)
