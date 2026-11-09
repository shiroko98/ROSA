"""Independent checkpoint-driven CPU NIAH worker; never called by training.

Run one checkpoint via --checkpoint, or watch native_checkpoints.jsonl with
--watch. Only immutable, atomically published checkpoints are consumed. One
worker per output (flock); slow/failing evaluation cannot stop the trainer.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from .cpu_compare import write_json
from .position import normalize_metadata, scale_options

ROOT=Path(__file__).resolve().parents[2]


def request_from_checkpoint(checkpoint, output, *, length=8192, depths=(.1,.5,.9),
                            max_new_tokens=24, threads=4, baseline_cache=None, source_run_id=None):
    import torch
    checkpoint=Path(checkpoint).resolve(strict=True);output=Path(output)
    c=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if c.get('native_rosa_checkpoint_kind')!='adapter_optimizer_loop_with_external_base_v1':
        raise ValueError('Standalone monitor requires a verified external-base native ROSA checkpoint.')
    meta=normalize_metadata(c['native_rosa']);base=c['native_rosa_frozen_base']
    if not base.get('parameter_names'):
        raise ValueError('External-base checkpoint must include native parameter order.')
    if not c['state_dict'] or any(not k.startswith('rosa_cpu.') for k in c['state_dict']):
        raise ValueError('Expected compact ROSA-only state_dict.')
    if length<256 or not depths or any(not 0<float(d)<1 for d in depths) or min(max_new_tokens,threads)<=0:
        raise ValueError('Invalid independent needle monitor settings.')
    return {'step':int(c['global_step']),'base':base['path'],'backbone_sha256':base['sha256'],
        'backbone_parameter_names':base['parameter_names'],'adapter':str(checkpoint),
        'adapter_format':'native_checkpoint','source_checkpoint':str(checkpoint),
        'source_training_run_id':source_run_id,'position':meta['position'],'output_norm':meta['output_norm'], **scale_options(meta),
        'layers':[int(i) for i in meta['layers'].split(',') if i],'scale':meta['scale'],
        'length':length,'depths':list(depths),'difficulties':['unrelated','similar_archives'],
        'seed':20261109,'max_new_tokens':max_new_tokens,'threads':threads,
        'baseline_cache':str(baseline_cache or output.parent/'base_cache.json'),
        'result':str(output/'report.json')}


def candidates(train_dir, minimum_step=0):
    p=Path(train_dir)/'native_checkpoints.jsonl'
    if not p.exists():return []
    # A writer may still be appending the final line. Never consume partial JSON.
    rows=[]
    for line in p.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if int(r['step'])>=minimum_step and Path(r['path']).is_file():rows.append(r)
    # initial/hourly/user-stop can refer to the same optimizer step. Evaluate once.
    return list({int(r['step']):r for r in rows}.values())


def upload_result(request_path):
    from .wandb_auth import configure_wandb_auth
    configure_wandb_auth()
    import wandb
    p=Path(request_path);r=json.loads(p.read_text());report=json.loads(Path(r['result']).read_text())
    if report['status']!='complete':raise ValueError('Refuse to publish incomplete NIAH as complete.')
    identity=p.parent/'wandb_run_id.txt'
    if not identity.exists():identity.write_text(uuid.uuid4().hex[:8]+'\n')
    # This worker has an independent timeline. Never reuse the training run id or
    # append old checkpoint metrics to its newer training history step.
    os.environ.pop('WANDB_RUN_ID',None);os.environ.pop('WANDB_RESUME',None)
    run=wandb.init(project='ROSA-RWKV7',entity=os.environ.get('WANDB_ENTITY','mobius'),
        id=identity.read_text().strip(),resume='allow',job_type='standalone-needle-evaluation',
        group='needle-of-'+str(r.get('source_training_run_id') or 'unknown'),dir=str(p.parent),
        config={k:r.get(k) for k in ('step','source_checkpoint','source_training_run_id','position','length','depths','max_new_tokens')})
    try:
        metrics={'source_training_step':r['step'],'evaluation_seconds':report.get('wall_seconds',0)}
        for row in report['summary']:
            prefix=f'needle/{row["difficulty"]}/{row["arm"]}/{row["mode"]}'
            for key,value in row.items():
                if isinstance(value,(int,float)):metrics[prefix+'/'+key]=value
        run.log(metrics,commit=True);run.save(str(Path(r['result'])),base_path=str(p.parent),policy='now')
    finally:run.finish()


def evaluate_one(checkpoint, output_root, args):
    import torch
    c=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);step=int(c['global_step']);del c
    folder=Path(output_root)/f'step{step:09d}';folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status')=='complete':return
    source_id=(Path(args.train_dir)/'wandb_run_id.txt') if args.train_dir else None
    request=request_from_checkpoint(checkpoint,folder,length=args.length,depths=args.depths,
        max_new_tokens=args.max_new_tokens,threads=args.threads,
        baseline_cache=Path(args.train_dir)/'needle/base_cache.json' if args.train_dir else None,
        source_run_id=source_id.read_text().strip() if source_id and source_id.exists() else None)
    path=folder/'request.json';write_json(path,request)
    status={'status':'running','step':step,'checkpoint':str(checkpoint),'started_at':datetime.now().astimezone().isoformat()}
    write_json(status_path,status);start=time.monotonic()
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
        for key in ('WANDB_RUN_ID','WANDB_RESUME'):env.pop(key,None)
        with (folder/'worker.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','rwkv_rosa.native.periodic_needle','--request',str(path)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=args.timeout)
        result=json.loads(Path(request['result']).read_text());result['wall_seconds']=time.monotonic()-start
        write_json(request['result'],result)
        status.update(status='evaluation_complete',wall_seconds=result['wall_seconds']);write_json(status_path,status)
        if args.wandb:upload_result(path)
        status['status']='complete';write_json(status_path,status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),wall_seconds=time.monotonic()-start)
        write_json(status_path,status)
        # No training-process error propagation. Retain raw logs/weights for retry.
        print(status['error'],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir');p.add_argument('--checkpoint');p.add_argument('--output')
    p.add_argument('--watch',action='store_true');p.add_argument('--min-step',type=int,default=0)
    p.add_argument('--length',type=int,default=8192);p.add_argument('--depths',default='0.1,0.5,0.9')
    p.add_argument('--max-new-tokens',type=int,default=24);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--timeout',type=int,default=14400);p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--wandb',action='store_true');args=p.parse_args(argv)
    if args.watch and not args.train_dir:p.error('--watch requires --train-dir')
    if not args.watch and not args.checkpoint:p.error('Use --checkpoint or --watch --train-dir')
    if args.poll_seconds<=0 or args.timeout<=0:p.error('Positive timeout/poll interval required')
    args.depths=[float(x) for x in args.depths.split(',')]
    out=Path(args.output or Path(args.train_dir or Path(args.checkpoint).parent)/'standalone_needle');out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another independent worker already owns this output queue.')
        (out/'worker.pid').write_text(str(os.getpid())+'\n')
        if not args.watch:evaluate_one(args.checkpoint,out,args);return
        while not (out/'STOP_WORKER').exists():
            for row in candidates(args.train_dir,args.min_step):
                status=out/f'step{int(row["step"]):09d}'/'status.json'
                if status.exists():continue # failed jobs are explicit; no infinite retry loop
                evaluate_one(row['path'],out,args)
            time.sleep(args.poll_seconds)


if __name__=='__main__':main()
