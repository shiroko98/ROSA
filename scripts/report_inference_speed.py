#!/usr/bin/env python3
"""Summarize completed device-placement speed measurements and publish evidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def report(folder):
    m=json.loads((folder/'manifest.json').read_text())
    reports={p:json.loads((folder/p/'report.json').read_text()) for p in m['placements']}
    assert all(r['status']=='complete' and not r['training'] for r in reports.values())
    m.update(status='complete',finished_at=datetime.now().astimezone().isoformat())
    write(folder/'manifest.json',m)
    assert len({r['base_sha256'] for r in reports.values()})==len({r['adapter_sha256'] for r in reports.values()})==1
    assert len({r['body_parameters'] for r in reports.values()})==len({r['adapter_parameters'] for r in reports.values()})==1
    rows=[];summary=[]
    for placement,r in reports.items():
        assert len(r['measurements'])==6
        for mode in m['modes']:
            rr=[x for x in r['measurements'] if x['mode']==mode];assert len(rr)==3
            assert all(x['tokens']==128 and len(x['generated_ids'])==len(x['latency_seconds'])==128 for x in rr)
            assert all(x['generated_ids']==rr[0]['generated_ids'] and x['first_token']==rr[0]['first_token'] for x in rr)
            assert all(x['prompt_sha256']==rr[0]['prompt_sha256'] for x in rr)
            assert all(x['decode_seconds']>=sum(x['latency_seconds']) for x in rr)
            rate=sum(x['tokens'] for x in rr)/sum(x['decode_seconds'] for x in rr)
            entry=dict(placement=placement,mode=mode,repetitions=3,total_timed_tokens=384,
                tokens_per_second=rate,median_run_tokens_per_second=statistics.median(x['tokens_per_second'] for x in rr),
                min_run_tokens_per_second=min(x['tokens_per_second'] for x in rr),max_run_tokens_per_second=max(x['tokens_per_second'] for x in rr),
                mean_ms_per_token=1000/rate,p50_ms_per_token=1000*statistics.median(t for x in rr for t in x['latency_seconds']),
                p95_ms_per_token=1000*float(np.quantile([t for x in rr for t in x['latency_seconds']],.95)),
                document_prefill_seconds=rr[0]['document_body_prefill_seconds']+rr[0]['document_sam_prefill_seconds'],
                prefill_tokens_per_second=rr[0]['prefill_tokens_per_second'],estimated_ttft_seconds=rr[0]['estimated_ttft_seconds'],
                prompt_tokens=rr[0]['prompt_tokens'],prompt_sha256=rr[0]['prompt_sha256'],
                prefill_threads=rr[0]['prefill_threads'],decode_threads=rr[0]['decode_threads'],
                rosa_nonzero_steps=sum(x['rosa_nonzero_steps'] for x in rr),stop_tokens_seen=sum(x['stop_tokens_seen'] for x in rr),
                peak_gpu_allocated_GiB=r['peak_gpu_allocated_GiB'],peak_cpu_process_RSS_GiB=r['max_rss_GiB'],
                body_dtype=r['body_dtype'])
            summary.append(entry);rows.extend(rr)
    for mode in m['modes']:
        assert len({g['prompt_sha256'] for g in summary if g['mode']==mode})==1
    gpu=reports['gpu'];hybrid=reports['hybrid']
    parity={}
    for mode in m['modes']:
        a=next(r for r in gpu['measurements'] if r['mode']==mode);b=next(r for r in hybrid['measurements'] if r['mode']==mode)
        parity[mode]=dict(first_token_equal=a['first_token']==b['first_token'],all128_output_tokens_equal=a['generated_ids']==b['generated_ids'])
    result=dict(status='complete',manifest=m,summary=summary,placement_audits={p:r['audit'] for p,r in reports.items()},
        gpu_hybrid_generation_parity=parity,base_sha256=gpu['base_sha256'],adapter_sha256=gpu['adapter_sha256'],
        body_parameters=gpu['body_parameters'],adapter_parameters=gpu['adapter_parameters'],
        total_timed_decode_tokens=2304,total_decode_runs=18,independent_document_prefills=3,
        interpretation='Batch-one speed of the implemented runtimes,not a pure device-only comparison:BF16 native CUDA versus FP32 reference CPU. Three decode repeats share immutable document/prompt states; each placement has one timed document prefill. SAM remains CPU in all placements;hybrid controller uses GPU head. Fixed128steps,not an accuracy test or guaranteed useful output.')
    write(folder/'summary.json',result)
    (folder/'measurements.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    with (folder/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    labels={'gpu':'本体GPU＋ROSA分支GPU','hybrid':'本体GPU＋ROSA分支CPU','cpu':'本体CPU＋ROSA分支CPU'}
    text=['# RWKV-7 2.9B＋step1800 ROSA：128K单请求推理速度','',
      '同一冻结本体、同一ROSA权重、同一128K文档（131,072tokens）、同一自然问答/精确前缀提示，batch=1。使用一张H80080GB或Xeon Platinum8468的一路48个物理核（CPU0–47，固定亲和性）。不启动训练。','',
      'GPU/混合本体为原生BF16 CUDA；纯CPU为现有FP32参考运行时＋编译C++ WKV，ROSA和输出头计算均为FP32。结果代表当前实现，不能解释为相同精度下的纯硬件速度差，或CPU推理的性能上限。','',
      '所有部署的离散SAM/LCT寻址都在CPU，按token增量更新。混合部署把ROSA可训练参数放在CPU，logit_gap动态增益仍调用GPU上的冻结输出头；GPU/CPU传输、同步与选词全部计入生成耗时。保留min_match=6、dynamic_max=0.3、margin=0.5。','',
      '## 主结果','',
      '|部署|128K文档预填充(s)|自然问答生成(tokens/s)|前缀补全生成(tokens/s)|峰值GPU显存(GiB)|峰值进程CPU RSS(GiB)|','|---|---|---|---|---|---|']
    for placement in m['placements']:
        a=next(g for g in summary if g['placement']==placement and g['mode']=='qa');b=next(g for g in summary if g['placement']==placement and g['mode']=='exact_prefix')
        text.append(f"|{labels[placement]}|{a['document_prefill_seconds']:.3f}|{a['tokens_per_second']:.3f}|{b['tokens_per_second']:.3f}|{a['peak_gpu_allocated_GiB']:.3f}|{a['peak_cpu_process_RSS_GiB']:.3f}|")
    text+=['','生成吞吐按三轮总tokens/总耗时计算；每轮在首token之后测128个递归生成步，共18轮、2304个计时tokens。为固定工作量，即便出现EOS也继续到128步，EOS次数保存在comparison.csv。这是速度基准，不把这些输出当质量评测。','',
      '每个部署只计时一次完整文档预填充；随后以不修改的文档状态分支到两种问题、三轮生成，没有每token重放128K历史。权重加载、JIT、tokenizer和预热不计入速度。首token估算包含文档本体+SAM、问题后缀本体+SAM、第一次ROSA+head+argmax，不含缓存分支克隆或线程池切换；因此单列为已加载模型的TTFT估算，不提供启动端到端时延。','',
      '## 时延、激活及重复范围','',
      '|部署|任务|输入tokens|TTFT估算(s)|平均ms/token|P95ms/token|三轮tokens/s最小–最大|ROSA非零步/384|','|---|---|---|---|---|---|---|---|']
    for g in summary:
        text.append(f"|{labels[g['placement']]}|{g['mode']}|{g['prompt_tokens']}|{g['estimated_ttft_seconds']:.3f}|{g['mean_ms_per_token']:.3f}|{g['p95_ms_per_token']:.3f}|{g['min_run_tokens_per_second']:.3f}–{g['max_run_tokens_per_second']:.3f}|{g['rosa_nonzero_steps']}/384|")
    cpu=reports['cpu']
    text+=['','## CPU线程选择','',json.dumps(cpu['thread_tuning'],ensure_ascii=False,indent=2),'',
      '线程选择仅通过短预热进行；完整128K文档没有被反复试跑后挑最好结果。一个NUMA节点的48个物理核可用，未使用SMT或双路全部96个物理核。不同NUMA布局、量化和优化内核的速度需另测。','',
      '## 校验与限制','',
      '- 三种部署的本体、ROSA哈希一致，参数全部冻结；两种提示的输入SHA256逐项一致。','- 每部署同一提示三轮的全部输出token一致，验证没有跨轮累积状态。GPU/CPU混合分支另与GPU分支做真实warm prompt的logit对齐检查；CPU编译WKV与FP32参考作数值对齐。',
      '- CPU FP32与GPU BF16可能生成不同轨迹；实际ROSA激活次数已列出，不能把自由生成路径差异误认为纯设备差异。仅测一份128K相似档案文档，三轮生成不是三个独立文档样本；预填充仅一次，没有置信区间。',
      '- 峰值GPU显存是torch实际分配峰值；CPU RSS为进程历史峰值，可能含加载与校验阶段，不是稳定运行所需内存的下界。','',
      'GPU/混合生成对齐情况：`'+json.dumps(parity,ensure_ascii=False)+'`','']
    (folder/'RESULTS_zh.md').write_text('\n'.join(text))
    source=folder/'evaluation_source';source.mkdir(exist_ok=True)
    for name in ['rwkv_rosa/native/inference_speed.py','rwkv_rosa/native/adapter.py','rwkv_rosa/native/cpu_wkv_inference.py','cpp_extensions/rwkv_wkv_cpu_inference.cpp','rwkv_rosa/native/cpu_compare.py','rwkv_rosa/native/gpu_state.py','rwkv_rosa/native/gpu_needle.py','rwkv_rosa/native/dynamic_scale.py','rwkv_rosa/native/needle_eval.py','rwkv_rosa/backbone.py','scripts/report_inference_speed.py','scripts/plot_inference_speed.py','tests/test_cpu_wkv_inference.py']:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(folder/'verification.json',dict(status='complete',same_weights=True,same_inputs=True,repeated_generation_state_reset_verified=True,total_timed_tokens=2304,gpu_hybrid_parity=parity))
    return result


def publish(folder,r):
    os.environ['WANDB_BASE_URL']='http://dian.sea-group.org:9098';os.environ.pop('WANDB_RUN_ID',None);os.environ.pop('WANDB_RESUME',None)
    from rwkv_rosa.native.wandb_auth import configure_wandb_auth
    configure_wandb_auth()
    import wandb
    identity=(folder/'wandb_run_id.txt').read_text().strip() if (folder/'wandb_run_id.txt').exists() else None
    run=wandb.init(id=identity,resume='allow',project='ROSA-RWKV7',entity='mobius',group='needle-of-ea5e442b',job_type='inference-placement-benchmark',name='step1800-128K-GPU-hybrid-CPU-speed',dir=str(folder),config=r['manifest'],settings=wandb.Settings(disable_git=True))
    run.config.update(r['manifest'],allow_val_change=True)
    (folder/'wandb_run_id.txt').write_text(run.id+'\n');(folder/'wandb_url.txt').write_text(run.url+'\n')
    values={};table=[]
    for g in r['summary']:
        key=g['placement']+'/'+g['mode'];table.append([g['placement'],g['mode'],g['tokens_per_second'],g['mean_ms_per_token'],g['estimated_ttft_seconds']])
        for k,v in g.items():
            if isinstance(v,(float,int)):values[key+'/'+k]=v
    values['comparison']=wandb.Table(columns=['placement','task','tokens_per_second','ms_per_token','TTFT_estimate_seconds'],data=table);values['speed_plot']=wandb.Image(str(folder/'comparison.png'));run.log(values)
    run.summary.update(dict(status='complete',timed_decode_tokens=2304,training=False,placements=3,batch_size=1,document_tokens=131072))
    artifact=wandb.Artifact('128K-inference-placement-speed-'+run.id,type='benchmark')
    for p in folder.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.md','.png','.txt','.log') and p.name!='pipeline_status.json':artifact.add_file(str(p))
    for name in ['gpu','hybrid','cpu','evaluation_source']:artifact.add_dir(str(folder/name),name=name)
    run.log_artifact(artifact);identity,url=run.id,run.url;run.finish()
    remote=wandb.Api(timeout=60).run('mobius/ROSA-RWKV7/'+identity);assert remote.summary['timed_decode_tokens']==2304 and remote.summary['status']=='complete'
    v=json.loads((folder/'verification.json').read_text());v.update(wandb_verified=True,wandb_url=url);write(folder/'verification.json',v)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();folder=Path(a.output)
    r=report(folder)
    subprocess.run([os.environ.get('ROSA_PLOT_PYTHON', sys.executable),str(ROOT/'scripts/plot_inference_speed.py'),str(folder)],check=True)
    if a.publish:publish(folder,r)
    with zipfile.ZipFile(folder/'data_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob('*'):
            if path.is_file() and path.name!='pipeline_status.json' and 'wandb' not in path.relative_to(folder).parts and path.suffix in ('.json','.csv','.jsonl','.md','.png','.txt','.log','.py','.cpp'):z.write(path,path.relative_to(folder))
    print('COMPLETE',folder,flush=True)


if __name__=='__main__':main()
