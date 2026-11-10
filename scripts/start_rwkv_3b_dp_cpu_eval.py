#!/usr/bin/env python3
"""Detach an eight-GPU trainer and independent CPU checkpoint evaluators."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--data', default='/mnt/data/Datasets/SFT_RWKV7_13B_rwkv_g1_ctx192512/results/SFT_RWKV7_13B')
    parser.add_argument('--pad-token-id', type=int, default=None)
    parser.add_argument('--scale-mode', choices=('fixed', 'logit_gap'), default='logit_gap')
    parser.add_argument('--scale', type=float, default=.05)
    parser.add_argument('--min-match', type=int, default=6)
    parser.add_argument('--dynamic-max', type=float, default=.2)
    args = parser.parse_args()
    if min(args.batch, args.cpu_threads) < 1:
        parser.error('Batch and CPU thread counts must be positive.')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Retain exact code used to launch, including local DP changes, without
    # copying experiment outputs, credentials, or the 114GB dataset.
    snapshot = out / 'source_snapshot'
    paths = list(ROOT.glob('rosa_*.py')) + [ROOT / 'train_rwkv_native_rosa.py', ROOT / 'rwkv_vocab_v20230424.txt']
    for directory in ('rwkv_rosa', 'third_party', 'cpp_extensions', 'scripts'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.cu', '.cpp', '.h', '.sh', '.jinja') and '__pycache__' not in p.parts)
    paths.append(ROOT / 'build_rosa_sam_cpu_extension.py')
    paths.append(ROOT / 'build_rwkv_plain_sft.py')
    for path in paths:
        if path.name.endswith('.local.sh'):
            continue
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    data_prefix = Path(args.data).resolve()
    data_manifest = data_prefix.parent.parent / 'build_manifest.json'
    data_meta = json.loads(data_manifest.read_text()) if data_manifest.exists() else {}
    pad_id = args.pad_token_id if args.pad_token_id is not None else int(data_meta.get('pad_token_id', 65532))
    if data_meta and data_meta.get('status') != 'complete':
        raise ValueError('Dataset build is incomplete')
    env = dict(os.environ, ROSA_OUTPUT_DIR=str(out), NPROC_PER_NODE='8', MICRO_BSZ=str(args.batch),
               ROSA_DATA_PREFIX=str(data_prefix), ROSA_SFT_PAD_ID=str(pad_id),
               ROSA_SCALE_MODE=args.scale_mode, ROSA_FIXED_SCALE=str(args.scale),
               ROSA_DYNAMIC_MIN_MATCH=str(args.min_match), ROSA_DYNAMIC_MAX=str(args.dynamic_max))
    if data_meta.get('format', '').startswith('rwkv_g1_plain'):
        env.setdefault('WANDB_RUN_GROUP', 'rwkv7_3b_dp8_192512_dynamic_plain')
    env.setdefault('ROSA_RWKV_PYTHON', sys.executable)
    command = ['bash', str(ROOT / 'scripts/run_rwkv_3b_dp_192k_training.sh')]
    with (out / 'train.log').open('a') as log:
        train = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (out / 'train.pid').write_text(str(train.pid) + '\n')
    jobs = {'training': {'pid': train.pid, 'command': command}}
    for length in (8192, 192512):
        folder = out / f'cpu_needle_{length}'
        folder.mkdir()
        command = ['bash', str(ROOT / 'scripts/run_rwkv_standalone_needle.sh'),
                   '--train-dir', str(out), '--output', str(folder), '--watch',
                   '--length', str(length), '--depths', '0.1,0.5,0.9',
                   '--prompt-template', 'rwkv',
                   '--max-new-tokens', '24', '--threads', str(args.cpu_threads),
                   '--timeout', '172800', '--wandb']
        with (folder / 'watcher.log').open('a') as log:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        jobs[f'cpu_needle_{length}'] = {'pid': worker.pid, 'command': command}
    manifest = {'started_at': datetime.now().astimezone().isoformat(), 'output': str(out),
                'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'git_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'micro_batch_per_gpu': args.batch, 'world_size': 8, 'effective_batch': 8 * args.batch,
                'context': 192512, 'scale_mode': args.scale_mode, 'rosa_scale': args.scale,
                'rosa_dynamic_min_match': args.min_match,
                'data_prefix': str(data_prefix), 'pad_token_id': pad_id, 'data_format': data_meta.get('format'),
                'rosa_dynamic_max': args.dynamic_max, 'checkpoint_seconds': 7200, 'initial_checkpoint': True,
                'cpu_evaluation_lengths': [8192, 192512], 'jobs': jobs}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
