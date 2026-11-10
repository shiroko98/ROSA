#!/usr/bin/env python3
"""Verify, report and publish the original short-passkey256K extension."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def aggregate(rows,cap):
    groups=defaultdict(dict)
    for r in rows:groups[(r['length'],r['seed'])][r['arm']]=r
    assert all(set(g)=={'base','trained'} for g in groups.values())
    pairs=list(groups.values());n=len(pairs)
    result={'cases':n,'output_budget':cap}
    for arm in ('base','trained'):
        ss=[p[arm]['by_budget'][str(cap)] for p in pairs]
        result[arm]={k:sum(s[k] for s in ss) for k in ['final_hit','content_hit','strict_final','strict_whole_output','unclosed_thinking','hit_generation_limit']}
        result[arm]['accuracy']=result[arm]['final_hit']/n
        c=result[arm]['final_hit'];z=1.959963984540054;rate=c/n
        center=(rate+z*z/(2*n))/(1+z*z/n);radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        result[arm]['accuracy_wilson_95_percent']=[max(0.,100*(center-radius)),min(100.,100*(center+radius))]
    delta=np.array([int(p['trained']['by_budget'][str(cap)]['final_hit'])-int(p['base']['by_budget'][str(cap)]['final_hit']) for p in pairs])
    ids=np.random.default_rng(20261110).integers(0,n,(10000,n))
    result['gain_pp']=100*delta.mean();result['gain_bootstrap_95_pp']=(100*np.quantile(delta[ids].mean(1),[.025,.975])).tolist()
    if all('rosa_nonzero_steps' in p['trained'] for p in pairs):
        result['rosa_cases_with_nonzero_residual']=sum(p['trained']['rosa_nonzero_steps']>0 for p in pairs)
    return result


def finalize(folder):
    from rwkv_rosa.native.short_256k_needle import budget_scores
    from rwkv_rosa.native.needle_eval import build_case,clean_decode
    from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER
    tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    m=json.loads((folder/'manifest.json').read_text());prior=Path(m['short_reference_experiment'])
    audits=[json.loads(p.read_text()) for p in folder.glob('runtime_audit_*.json')]
    assert len(audits)==8 and len(list(folder.glob('worker_complete*')))==8 and not list(folder.glob('failure*'))
    assert len({a['base_sha256'] for a in audits})==len({a['adapter_sha256'] for a in audits})==1
    assert all(a['checkpoint_step']==1800 and not a['training'] and a['configured_context_capacity']==268288 for a in audits)
    assert all(a['scale_options']['dynamic_min_match']==6 and a['scale_options']['dynamic_max']==.3 for a in audits)
    previous_audits=[json.loads(p.read_text()) for p in prior.parent.glob('runtime_audit*')]
    assert all(a['base_sha256']==audits[0]['base_sha256'] and a['adapter_sha256']==audits[0]['adapter_sha256'] for a in previous_audits)
    fresh=[];trace_steps=0;active_steps=0;inputs=[]
    for index,spec in enumerate(m['cases']):
        path=folder/'cases'/f'{index:04d}.json';doc=json.loads(path.read_text());assert doc['status']=='complete' and doc['spec']==spec
        ref=json.loads((prior/'cases'/f"{spec['reference_case_index']:04d}.json").read_text())
        case=build_case(tokenizer,spec['length'],spec['depth'],spec['seed'],spec['difficulty'],prompt_template='rwkv')
        assert doc['stem_sha256']==case['stem_sha256'];assert doc['secret']==ref['secret']==case['secret'];assert doc['key']==ref['key']==case['key']
        assert len(case['answer_ids'])==3
        traces=json.loads((folder/'traces'/path.name).read_text())
        for mode in ('qa','exact_prefix'):
            prompt=case['stem_ids']+case['suffix_ids'][mode]
            assert hashlib.sha256(np.array(prompt,dtype=np.int64).tobytes()).hexdigest()==doc['prompt_sha256'][mode]
            suffix=tokenizer.decode(case['suffix_ids'][mode]);assert suffix==doc['suffix_text'][mode]
            inputs.append({'case_index':index,'mode':mode,'document_tokens':len(case['stem_ids']),'question_tokens':len(case['suffix_ids'][mode]),'input_tokens':len(prompt),'answer_tokens':3,'suffix':suffix})
            rr=[r for r in doc['results'] if r['mode']==mode];assert {r['arm'] for r in rr}=={'base','trained'} and len(rr)==2
            assert len({r['prompt_tokens'] for r in rr})==1 and rr[0]['prompt_tokens']==len(prompt)
            for row in rr:
                assert row['secret']==case['secret'] and row['answer_ids']==case['answer_ids']
                assert clean_decode(tokenizer,row['generated_ids'])==row['generated_text']
                for cap,scores in row['by_budget'].items():assert scores==budget_scores(tokenizer,row['generated_ids'],case['secret'],int(cap))
                row.update(prompt_sha256=doc['prompt_sha256'][mode]);fresh.append(row)
            trained=next(r for r in rr if r['arm']=='trained');events=traces[mode]
            assert len(events)==len(trained['generated_ids'])==trained['rosa_prediction_steps']
            assert [e['emitted_tokens'] for e in events]==list(range(len(events)))
            assert sum(e['residual_rms']>0 for e in events)==trained['rosa_nonzero_steps']
            for e in events:
                assert math.isfinite(e['residual_rms']) and 0<=e['effective_scale']<=.300001
                if not e['valid'] or e['match_tokens']<6:assert e['effective_scale']==e['residual_rms']==0
                trace_steps+=1;active_steps+=e['residual_rms']>0
    assert len(fresh)==60
    reference=[]
    for p in sorted((prior/'cases').glob('*.json')):
        d=json.loads(p.read_text());assert d['status']=='complete'
        for row in d['results']:
            cap=512 if row['mode']=='qa' else 128
            row['by_budget']={str(cap):budget_scores(tokenizer,row['generated_ids'],row['secret'],cap)}
            reference.append(row)
    comparison=[];depths=[]
    for length in [8192,16384,32768,65536,131072,262144]:
        source=fresh if length==262144 else reference
        for mode in ['qa','exact_prefix']:
            rr=[r for r in source if r['length']==length and r['mode']==mode]
            caps=m['score_budgets'][mode] if length==262144 else [512 if mode=='qa' else 128]
            for cap in caps:
                comparison.append(dict(length=length,mode=mode,**aggregate(rr,cap)))
                if length==262144:
                    for depth in [.1,.3,.5,.7,.9]:depths.append(dict(depth=depth,mode=mode,**aggregate([r for r in rr if r['depth']==depth],cap)))
    summary={'status':'complete','checkpoint_step':1800,'new_contexts':15,'new_generated_answers':60,
             'comparison':comparison,'by_depth':depths,'input_lengths':inputs,
             'trace_prediction_steps_verified':trace_steps,'nonzero_residual_steps':active_steps,
             'base_sha256':audits[0]['base_sha256'],'adapter_sha256':audits[0]['adapter_sha256'],
             'uncertainty':'15seed/position families;3cases per position. Paired10000-resample bootstrap; all-zero intervals can degenerate and do not prove equivalence. Wilson intervals are descriptive.',
             'reference':str(prior),'training_stopped':True,'input_truncation':False}
    write(folder/'summary.json',summary)
    write(folder/'verification.json',dict(status='complete',scores_recomputed=True,prompts_rebuilt_and_hash_verified=15,
        same_reference_keys_answers_and_query=True,checkpoint_hashes_equal_in_all_workers=True,
        trace_prediction_steps_verified=trace_steps,below_threshold_residuals=0,input_truncation=False))
    for name,rows in [('answers',fresh),('reference_answers',reference)]:
        (folder/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        with (folder/(name+'.csv')).open('w',newline='') as f:
            fields=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fields);w.writeheader()
            w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    flat=[]
    for g in comparison:
        row={k:json.dumps(v) if isinstance(v,list) else v for k,v in g.items() if k not in ('base','trained')}
        for arm in ['base','trained']:row.update({arm+'_'+k:json.dumps(v) if isinstance(v,list) else v for k,v in g[arm].items()})
        flat.append(row)
    with (folder/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
    lines=['# 原版短问答：256K上下文对照','',
      '使用此前相同的8位档案编号、六位数字口令（3tokens）、相似档案干扰及普通RWKV User/Assistant模板；5个位置10%/30%/50%/70%/90%，每位置3例，共15份256K文档。沿用128K那15份的种子、key、secret和问题后缀；只把文档延长。','',
      '冻结同一RWKV-7 2.9B本体与step1800 ROSA，动态min_match=6、dynamic_max=0.3、margin=0.5。greedy：temperature=0、top_p=1、top_k=0，无重复惩罚；问答实际输出上限4096、前缀补全1024。利用同一greedy生成前缀另按问答128/512、补全128评分，与旧预算保持同口径。不会从多个采样中挑最好结果。','',
      '主指标沿用旧版：最终答复中第一个独立六位数必须等于目标；若thinking未闭合则失败，已闭合时只取最后</think>之后的文本。思考中的内容命中、严格六位数输出均单列。不是只要任意位置曾出现目标就算成功。','',
      '## 分长度及输出上限的最终答复命中','',
      '|文档长度|任务|输出上限|Base|ROSA|提升(pp)|配对95%区间(pp)|','|---|---|---|---|---|---|---|']
    for g in comparison:
        ci=g['gain_bootstrap_95_pp'];n=g['cases']
        lines.append(f"|{g['length']//1024}K|{g['mode']}|{g['output_budget']}|{g['base']['final_hit']}/{n} ({100*g['base']['accuracy']:.2f}%)|{g['trained']['final_hit']}/{n} ({100*g['trained']['accuracy']:.2f}%)|{g['gain_pp']:+.2f}|[{ci[0]:.2f},{ci[1]:.2f}]|")
    lines+=['','## 256K按位置拆分','', '|位置|任务|输出上限|Base最终答复|ROSA最终答复|','|---|---|---|---|---|']
    for g in depths:lines.append(f"|{100*g['depth']:.0f}%|{g['mode']}|{g['output_budget']}|{g['base']['final_hit']}/3|{g['trained']['final_hit']}/3|")
    lines+=['','## 截断、thinking及格式','', '|任务|预算|内容命中Base/ROSA|未闭合thinking Base/ROSA|触及上限Base/ROSA|最终严格六位数Base/ROSA|','|---|---|---|---|---|---|']
    for g in comparison:
        if g['length']!=262144:continue
        a,b=g['base'],g['trained'];lines.append(f"|{g['mode']}|{g['output_budget']}|{a['content_hit']}/{b['content_hit']}|{a['unclosed_thinking']}/{b['unclosed_thinking']}|{a['hit_generation_limit']}/{b['hit_generation_limit']}|{a['strict_final']}/{b['strict_final']}|")
    lines+=['','256K文档为262,144tokens，实际输入还包括问题后缀，逐题长度见summary.json/answers.csv。完整输入已重建校验SHA256，没有截断。256K超过训练上下文192,512，属于同一旧测试家族的探索性外推，不是新选参后的独立确认集。','',
      '每任务仅15例、每位置3例。全零bootstrap区间退化不能解释为真实准确率或差值必为零；0/15的Wilson95%区间约为0%–20.39%。前缀补全是复制诊断，不能代替自然问答准确率；本组短答案不能与上一组16–137-token长答案混用。','']
    (folder/'RESULTS_zh.md').write_text('\n'.join(lines))
    source=folder/'evaluation_source';source.mkdir(exist_ok=True)
    names=['rwkv_rosa/native/short_256k_needle.py','rwkv_rosa/native/fast_gpu_needle.py','rwkv_rosa/native/gpu_needle.py','rwkv_rosa/native/gpu_state.py','rwkv_rosa/native/needle_eval.py','rwkv_rosa/native/adapter.py','rwkv_rosa/native/dynamic_scale.py','rwkv_rosa/native/position.py','scripts/report_short_256k_needle.py','scripts/plot_short_256k.py','tests/test_short_256k_needle.py']
    for name in names:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    test_log = Path('/tmp/rosa-short256-tests.log')
    if test_log.exists():
        shutil.copy2(test_log, folder/'tests.log')
    m.update(status='complete',finished_at=datetime.now().astimezone().isoformat());write(folder/'manifest.json',m)
    return summary


def publish(folder,r):
    os.environ['WANDB_BASE_URL']='http://dian.sea-group.org:9098'
    os.environ.pop('WANDB_RUN_ID',None);os.environ.pop('WANDB_RESUME',None)
    from rwkv_rosa.native.wandb_auth import configure_wandb_auth
    configure_wandb_auth()
    import wandb
    m=json.loads((folder/'manifest.json').read_text());identity=(folder/'wandb_run_id.txt').read_text().strip() if (folder/'wandb_run_id.txt').exists() else None
    run=wandb.init(id=identity,resume='allow',project='ROSA-RWKV7',entity='mobius',group='needle-of-ea5e442b',job_type='gpu-short-256K-needle',name='step1800-original-short-questions-256K',dir=str(folder),config=m,settings=wandb.Settings(disable_git=True))
    (folder/'wandb_run_id.txt').write_text(run.id+'\n');(folder/'wandb_url.txt').write_text(run.url+'\n')
    metrics={};table=[]
    for g in r['comparison']:
        key=f"{g['length']//1024}K/{g['mode']}/cap{g['output_budget']}"
        table.append([g['length'],g['mode'],g['output_budget'],g['base']['final_hit'],g['trained']['final_hit'],g['gain_pp']])
        metrics[key+'/gain_pp']=g['gain_pp']
        for arm in ['base','trained']:
            for k,v in g[arm].items():
                if isinstance(v,(int,float)):metrics[key+'/'+arm+'/'+k]=v
    metrics['comparison']=wandb.Table(columns=['document_tokens','task','budget','base_hits','ROSA_hits','gain_pp'],data=table)
    metrics['comparison_plot']=wandb.Image(str(folder/'comparison.png'));run.log(metrics)
    run.summary.update(dict(status='complete',new_contexts=15,new_generated_answers=60,checkpoint_step=1800,input_truncation=False,training_stopped=True))
    artifact=wandb.Artifact('step1800-original-short-256K-'+run.id,type='evaluation')
    allowed=lambda p:p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.png','.txt','.log') and not p.name.startswith(('worker_','report_'))
    for p in folder.iterdir():
        if allowed(p):artifact.add_file(str(p))
    for name in ['cases','traces','evaluation_source']:artifact.add_dir(str(folder/name),name=name)
    run.log_artifact(artifact);identity,url=run.id,run.url;run.finish()
    remote=wandb.Api(timeout=60).run('mobius/ROSA-RWKV7/'+identity)
    assert remote.summary['new_generated_answers']==60 and remote.summary['status']=='complete'
    v=json.loads((folder/'verification.json').read_text());v.update(wandb_verified=True,wandb_url=url);write(folder/'verification.json',v)
    with zipfile.ZipFile(folder/'data_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in folder.iterdir():
            if allowed(p):z.write(p,p.name)
        for name in ['cases','traces','evaluation_source']:
            for p in (folder/name).rglob('*'):
                if p.is_file():z.write(p,p.relative_to(folder))
    print('VERIFIED',url,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wait',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args();folder=Path(a.output)
    def status(state,**kw):write(folder/'report_status.json',dict(status=state,updated_at=datetime.now().astimezone().isoformat(),**kw))
    try:
        while len(list(folder.glob('worker_complete*')))!=8:
            if list(folder.glob('failure*')):raise RuntimeError('Inference worker failure')
            if not a.wait:raise RuntimeError('Inference not complete')
            status('waiting_for_inference');time.sleep(10)
        status('verifying');r=finalize(folder)
        status('plotting');subprocess.run([os.environ.get('ROSA_PLOT_PYTHON', sys.executable),str(ROOT/'scripts/plot_short_256k.py'),str(folder)],check=True)
        if a.publish:status('uploading_wandb');publish(folder,r)
        status('complete')
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
