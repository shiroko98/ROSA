#!/usr/bin/env python3
"""Publish only verified, completed long-answer experiments to W&B."""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);r=json.loads((out/'summary.json').read_text())
    assert r['status']=='complete' and r['generated_answers']==576
    assert json.loads((out/'verification.json').read_text())['status']=='complete'
    os.environ['WANDB_BASE_URL']='http://dian.sea-group.org:9098'
    os.environ.pop('WANDB_RUN_ID',None);os.environ.pop('WANDB_RESUME',None)
    from rwkv_rosa.native.wandb_auth import configure_wandb_auth
    configure_wandb_auth()
    import wandb
    identity=(out/'wandb_run_id.txt').read_text().strip() if (out/'wandb_run_id.txt').exists() else None
    manifest=json.loads((Path(r['main_suite'])/'manifest.json').read_text())
    config={k:v for k,v in manifest.items() if k!='cases'}
    config.update(context_lengths=[8192,32768,131072,262144],contexts=144,paired_prompts=288,generated_answers=576,extrapolation_length=262144)
    run=wandb.init(id=identity,resume='allow',project='ROSA-RWKV7',entity='mobius',
                   group='needle-of-ea5e442b',job_type='gpu-long-answer-needle',
                   name='step1800-long-answer-8K-to-256K-greedy4096',dir=str(out),
                   config=config,settings=wandb.Settings(disable_git=True))
    (out/'wandb_run_id.txt').write_text(run.id+'\n');(out/'wandb_url.txt').write_text(run.url+'\n')
    metrics={};table=[]
    for g in r['groups']:
        if g['group_fields']!=['length','mode']:continue
        prefix=f"{g['length']//1024}K/{g['mode']}"
        for arm in ['base','trained']:
            for k,v in g[arm].items():
                if isinstance(v,(int,float)):metrics[f'{prefix}/{arm}/{k}']=v
        metrics[prefix+'/gain_pp']=g['final_contains_full_answer_gain_pp']
        for k,v in g['rosa'].items():metrics[f'{prefix}/rosa/{k}']=v
        table.append([g['length'],g['mode'],g['paired_cases'],g['base']['final_contains_full_answer_correct'],g['trained']['final_contains_full_answer_correct'],g['final_contains_full_answer_gain_pp'],str(g['final_contains_full_answer_gain_cluster_bootstrap_95_pp'])])
    metrics['length_comparison']=wandb.Table(columns=['document_tokens','task','paired_cases','base_hits','ROSA_hits','gain_pp','paired_95CI_pp'],data=table)
    for mode,g in r['main_suite_by_mode'].items():
        metrics['8K_128K_combined/'+mode+'/gain_pp']=g['final_contains_full_answer_gain_pp']
        metrics['8K_128K_combined/'+mode+'/base_rate']=g['base']['final_contains_full_answer_rate']
        metrics['8K_128K_combined/'+mode+'/ROSA_rate']=g['trained']['final_contains_full_answer_rate']
    for name in ['full_answer_comparison','answer_length_matrix']:metrics[name]=wandb.Image(str(out/(name+'.png')))
    run.log(metrics)
    run.summary.update({'status':'complete','contexts':144,'paired_prompts':288,'generated_answers':576,
                        'checkpoint_step':1800,'training_stopped':True,'all_trace_threshold_checks_passed':True,
                        'input_truncation':False,'includes_256K_extrapolation':True})
    artifact=wandb.Artifact('step1800-long-answer-256K-'+run.id,type='evaluation',
       description='Frozen paired baseline/ROSA. Long questions,16/64/128-token answers;8K/32K/128K/256K. QA4096 and prefix1024 generation budgets. Raw outputs and free-generation gate traces.')
    for path in sorted(out.iterdir()):
        if path.is_file() and not path.name.startswith('pipeline') and path.suffix in ('.json','.jsonl','.csv','.md','.png','.txt','.log'):artifact.add_file(str(path))
    artifact.add_dir(str(out/'evaluation_source'),name='evaluation_source')
    for name,key in [('main_8K_128K','main_suite'),('extrapolation_256K','extrapolation_suite')]:
        folder=Path(r[key])
        for path in folder.iterdir():
            if path.is_file() and (path.name.startswith(('manifest','verification','runtime_audit','worker_complete','jobs','queue_status','input_manifest','PROTOCOL'))):artifact.add_file(str(path),name=name+'/'+path.name)
        artifact.add_dir(str(folder/'cases'),name=name+'/cases')
        artifact.add_dir(str(folder/'traces'),name=name+'/traces')
    run.log_artifact(artifact)
    identity,url=run.id,run.url;run.finish()
    remote=wandb.Api(timeout=60).run('mobius/ROSA-RWKV7/'+identity)
    assert remote.summary['generated_answers']==576 and remote.summary['status']=='complete'
    v=json.loads((out/'verification.json').read_text());v.update(wandb_verified=True,wandb_url=url)
    (out/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
    with zipfile.ZipFile(out/'data_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in out.iterdir():
            if path.is_file() and not path.name.startswith('pipeline') and path.suffix in ('.json','.jsonl','.csv','.md','.png','.txt','.log'):archive.write(path,path.name)
        for path in (out/'evaluation_source').rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(out))
        for name,key in [('main_8K_128K','main_suite'),('extrapolation_256K','extrapolation_suite')]:
            for path in Path(r[key]).iterdir():
                if path.is_file() and path.name.startswith(('manifest','verification','runtime_audit','worker_complete','jobs','queue_status','input_manifest','PROTOCOL')):archive.write(path,name+'/'+path.name)
            for sub in ['cases','traces']:
                for path in (Path(r[key])/sub).glob('*.json'):archive.write(path,name+'/'+sub+'/'+path.name)
    print('VERIFIED',url)


if __name__=='__main__':main()
