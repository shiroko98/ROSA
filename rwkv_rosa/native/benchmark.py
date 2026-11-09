"""Paired device-placement timings using the actual upstream native SFT Trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = '/hikscale/models/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.pth'


def write_json(path, payload):
    p = Path(path); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'); tmp.replace(p)


def run(args):
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh placement-benchmark output directory.')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'pid': os.getpid(), 'config': vars(args), 'runs': {}, 'comparisons': {},
               'protocol': 'same frozen pretrained BF16 backbone, native CUDA/SFT Trainer, FP32 ROSA, CPU SAM in both arms',
               'quality_note': 'bounded throughput/placement test, NOT finetuning quality or 50K model-quality evidence'}
    source = ROOT / 'third_party/rwkv_lm_v7_native'
    files = [source/'train.py', *sorted((source/'src').glob('*.py')), *sorted((source/'cuda').glob('*')),
             *sorted(Path(__file__).parent.glob('*.py'))]
    summary['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files if p.is_file()}
    write_json(out/'comparison.json', summary)
    contexts = [int(i) for i in args.contexts.split(',')]
    for index, context in enumerate(contexts):
        prefix = Path(args.data_root).resolve()/f'ctx{context}'/'sft'
        metadata = json.loads((prefix.parent/'manifest.json').read_text())
        order = ('cuda', 'cpu') if index % 2 == 0 else ('cpu', 'cuda')
        for device in order:
            name = f'ctx{context}_{device}'
            case = out/name
            command = ['bash', str(ROOT/'scripts/run_rwkv_native_rosa.sh'), '--load_model', args.checkpoint,
                       '--data_file', str(prefix), '--ctx_len', str(context), '--epoch_steps', str(args.steps),
                       '--max_steps', str(args.steps), '--rosa_save', '1' if context == 8192 else '0', '--rosa_device', device,
                       '--proj_dir', str(case), '--enable_progress_bar', 'False']
            env = dict(os.environ, WANDB_MODE=args.wandb_mode, WANDB_RUN_GROUP=out.name,
                       OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
            summary['current'] = name; write_json(out/'comparison.json', summary)
            print(f'START {name}', flush=True)
            start = time.monotonic()
            with (out/f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                summary['worker_pid'] = process.pid; write_json(out/'comparison.json', summary)
                try:
                    code = process.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    code = None
            result = {'returncode': code, 'wall_seconds_including_startup': time.monotonic()-start,
                      'input_data_manifest': metadata, 'log': str(out/f'{name}.log')}
            steps_file = case/'native_steps.jsonl'
            if code != 0 or not steps_file.exists():
                result['status'] = 'timeout' if code is None else 'failed'
                summary['runs'][name] = result; summary['status'] = f'failed_{name}'
                write_json(out/'comparison.json', summary)
                raise RuntimeError(f'Native run {name} failed, see {result["log"]}')
            steps = [json.loads(line) for line in steps_file.read_text().splitlines() if line.strip()]
            audit = json.loads((case/'native_audit.json').read_text())
            if len(steps) != args.steps or not audit.get('backbone_unchanged') or not audit.get('rosa_values_changed'):
                raise AssertionError('Native run did not complete verified ROSA-only updates.')
            measured = steps[args.warmup:]
            times = [r['seconds'] for r in measured]
            result.update(status='success', audit=audit, all_steps=steps, warmup_steps_excluded=args.warmup,
                          median_step_seconds=statistics.median(times), mean_step_seconds=statistics.mean(times),
                          min_step_seconds=min(times), max_step_seconds=max(times),
                          input_tokens_per_second=context/statistics.mean(times),
                          max_measured_peak_gpu_GiB=max(r['peak_gpu_allocated_GiB'] for r in measured))
            summary['runs'][name] = result; write_json(out/'comparison.json', summary)
            print(f'PASS {name}: median={result["median_step_seconds"]:.4f}s, '
                  f'peak={result["max_measured_peak_gpu_GiB"]:.3f}GiB', flush=True)
        a, b = (summary['runs'][f'ctx{context}_{d}'] for d in ('cuda','cpu'))
        if [r['batch_sha256'] for r in a['all_steps']] != [r['batch_sha256'] for r in b['all_steps']]:
            raise AssertionError('CPU/GPU arms saw different data or order.')
        summary['comparisons'][str(context)] = {
            'same_batch_order_verified': True,
            'gpu_rosa_speedup_over_cpu': b['median_step_seconds']/a['median_step_seconds'],
            'gpu_rosa_extra_peak_GiB': a['max_measured_peak_gpu_GiB']-b['max_measured_peak_gpu_GiB'],
            'initial_step_loss_abs_difference': abs(a['all_steps'][0]['loss']-b['all_steps'][0]['loss'])}
        write_json(out/'comparison.json', summary)
    summary['status'] = 'complete'; summary.pop('worker_pid', None); summary.pop('current', None)
    write_json(out/'comparison.json', summary)
    print(json.dumps(summary['comparisons'], indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=BASE); p.add_argument('--out-dir', required=True)
    p.add_argument('--data-root', default=str(ROOT/'outputs/native_rwkv_01b_probe_data_20261109'))
    p.add_argument('--contexts', default='1024,8192,51200'); p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2); p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--wandb-mode', choices=('offline','online'), default='offline')
    a = p.parse_args()
    if a.warmup < 0 or a.steps <= a.warmup: p.error('Require steps > warmup >= 0.')
    run(a)
