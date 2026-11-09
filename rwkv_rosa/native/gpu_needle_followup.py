"""Queue trained-only ChatML evaluation after a particular GPU NIAH worker.

Never starts training. Avoids concurrent 192K forwards competing for VRAM.
Inherits the existing RWKV matrix, checkpoint, decoding limit and seeds exactly.
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

import psutil

from rwkv_rosa.native.cpu_compare import write_json

ROOT = Path(__file__).resolve().parents[2]


def followup_command(request, output):
    if request.get('prompt_template') != 'rwkv' or request.get('training') is not False:
        raise ValueError('Follow-up requires an independent RWKV-template inference job.')
    return ['bash', str(ROOT/'scripts/run_rwkv_gpu_needle.sh'),
            '--train-dir', request['train_dir'], '--checkpoint', request['checkpoint'],
            '--output', str(output), '--prompt-template', 'chatml', '--arms', 'trained',
            '--lengths', ','.join(map(str,request['lengths'])),
            '--depths', ','.join(map(str,request['depths'])),
            '--seeds', ','.join(map(str,request['seeds'])),
            '--max-new-tokens', str(request['max_new_tokens']),
            *(['--wandb'] if request.get('wandb') else [])]


def three_way_report(rwkv, chatml):
    for key in ('checkpoint','lengths','depths','seeds','max_new_tokens'):
        if rwkv['request'][key] != chatml['request'][key]:
            raise ValueError(f'Cannot combine different protocols: {key}')
    if rwkv['request']['prompt_template'] != 'rwkv' or chatml['request']['prompt_template'] != 'chatml':
        raise ValueError('Unexpected prompt formats in comparison.')
    rows = []
    for template, report, arms in (('rwkv',rwkv,('base','trained')),('chatml',chatml,('trained',))):
        for scope, key in (('overall','generation_overall'),('by_length_difficulty_mode','generation_summary')):
            for row in report[key]:
                if row['arm'] in arms:
                    rows.append({**row,'prompt_template':template,'scope':scope})
    return {'status':'complete' if all(r['status']['status']=='complete' for r in (rwkv,chatml)) else 'partial',
            'source_step':rwkv['request']['source_step'], 'checkpoint':rwkv['request']['checkpoint'],
            'comparison':'Only trained-RWKV vs base-RWKV isolates the ROSA change; trained-ChatML vs trained-RWKV is prompt-format ablation.',
            'rwkv_status':rwkv['status'], 'chatml_status':chatml['status'], 'summary':rows}


def same_process_alive(pid, created):
    try:
        p = psutil.Process(pid)
        return p.create_time() == created and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-output', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-hours', type=float, default=48)
    args = parser.parse_args(argv)
    previous = Path(args.after_output).resolve(strict=True)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.timeout_hours <= 0: parser.error('Positive timeout required.')
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'request.json').exists(): raise SystemExit('Evaluation already exists; do not duplicate it.')
        request = json.loads((previous/'request.json').read_text())
        previous_status = json.loads((previous/'status.json').read_text())
        command = followup_command(request,output)
        status = {'status':'waiting_previous_gpu_evaluation', 'pid':os.getpid(),
                  'after_output':str(previous), 'previous_pid':previous_status['pid'],
                  'command':command, 'training':False, 'automatic_training_restart':False,
                  'requested_arms':['trained'], 'prompt_template':'chatml',
                  'queued_at':datetime.now().astimezone().isoformat()}
        def publish(): write_json(output/'queue_status.json',status)
        publish()
        try:
            pid = int(previous_status['pid'])
            try:
                process = psutil.Process(pid)
                # Reject an unrelated PID before waiting on it.
                if 'rwkv_rosa.native.gpu_needle' not in process.cmdline():
                    raise RuntimeError('Previous worker PID now belongs to another command.')
                created = process.create_time()
            except psutil.NoSuchProcess:
                created = None
            deadline = time.monotonic() + args.timeout_hours*3600
            while created is not None and same_process_alive(pid,created):
                if (output/'CANCEL_QUEUE').exists():
                    status.update(status='cancelled_by_user'); publish(); return
                if time.monotonic() > deadline: raise TimeoutError('Previous GPU evaluator did not exit.')
                time.sleep(10)
            if (output/'CANCEL_QUEUE').exists():
                status.update(status='cancelled_by_user'); publish(); return
            dependency = json.loads((previous/'status.json').read_text())
            status['previous_final_status'] = dependency['status']
            if dependency['status'] != 'complete':
                status['warning'] = 'RWKV evaluation incomplete; running independent ChatML test anyway, no success inferred.'
            status.update(status='launching_chatml_trained_only'); publish()
            with (output/'worker.log').open('a') as log:
                child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
            status.update(status='chatml_evaluation_running',evaluation_pid=child.pid); publish()
            code = child.wait()
            result = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
            if code == 0 and result.get('status') == 'complete':
                combined = three_way_report(json.loads((previous/'report.json').read_text()),
                                            json.loads((output/'report.json').read_text()))
                write_json(output/'three_way_report.json',combined)
            status.update(status='complete' if code == 0 and result.get('status') == 'complete' else 'failed',
                          returncode=code, finished_at=datetime.now().astimezone().isoformat())
            publish()
        except BaseException:
            status.update(status='failed',error=traceback.format_exc()); publish(); raise


if __name__ == '__main__': main()
