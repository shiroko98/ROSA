"""Isolated full-BPTT length search: frozen CUDA RWKV + CPU improved ROSA.

Each case uses real contiguous tokens and the original mask from one packed row.
OOM, process failure and wall-time limits remain distinct outcomes. No TBPTT,
BF16 fallback, synthetic supervision or silent windowing is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def next_length(success, failure, ceiling, resolution):
    """Geometric growth then aligned bisection; None means requested bracket found."""
    if failure is None:
        return None if success >= ceiling else min(ceiling, max(resolution, success * 2))
    if failure - success <= resolution:
        return None
    midpoint = ((success + failure) // (2 * resolution)) * resolution
    return max(success + resolution, midpoint)


def worker(args):
    import torch
    from .backbone import load_base_state, infer_config
    from .binidx_data import resolve_binidx_prefix
    from .experiment import clip_all_gradients, seed_all
    from .freeze_audit import check_frozen_backbone, parameter_hash
    from .model import RWKVRosaLM, configure_train_scope
    from rosa_optim import build_training_optimizers
    from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset
    from third_party.rwkv_lm_v7.sft_loss import masked_head_cross_entropy

    out = Path(args.out_dir)
    result_path = out / f'length_{args.length}.json'
    stage_path = out / f'length_{args.length}.stage.json'
    result = {'input_tokens': args.length, 'started_at': now(), 'worker_pid': os.getpid(),
              'status': 'initializing', 'steps_requested': args.steps, 'updates': []}
    def stage(name):
        result['stage'] = name
        write_json(stage_path, {k: v for k, v in result.items() if k != 'updates'})
        print(f'{now()} T={args.length} {name}', flush=True)

    try:
        seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        prefix = resolve_binidx_prefix(args.data)
        data, masks = MMapIndexedDataset(prefix), MMapIndexedDataset(prefix + '.mask')
        if args.length + 1 > len(data[args.row]):
            raise ValueError('Requested length exceeds source physical row; no repetition/padding allowed.')
        ids_cpu = torch.tensor(data[args.row][-(args.length + 1):].astype('int64')).unsqueeze(0)
        mask_cpu = torch.tensor(masks[args.row][-args.length:].astype('float32')).unsqueeze(0)
        if not mask_cpu.sum():
            raise ValueError('Selected true-data span has no supervised targets.')
        result.update(source_row=args.row, supervised_tokens=int(mask_cpu.sum()),
                      batch_sha256=hashlib.sha256(ids_cpu.numpy().tobytes() + mask_cpu.numpy().tobytes()).hexdigest())
        stage('load_checkpoint')
        base = load_base_state(args.checkpoint)
        cfg = infer_config(base)
        model = RWKVRosaLM(cfg, variant='improved', inject_layers=(0,), scale=.05,
                           sparse_values=True, value_shards=4, activation_checkpointing=True,
                           special_ids=(0, 65530, 65531, 65532, 65533, 65534),
                           dtype=torch.float32, base_state=base)
        del base
        model.place_devices('cuda', 'cpu')
        result['parameters'] = configure_train_scope(model, 'rosa_only')
        optimizer = build_training_optimizers(model, lr=1e-4, weight_decay=.01)
        model.train()
        stage('freeze_hash_before')
        frozen = check_frozen_backbone(model)
        result['backbone_sha256_before'] = frozen
        value_before = parameter_hash(model.rosa_value_store)
        ids = ids_cpu.to('cuda')
        loss_mask = mask_cpu.to('cuda')
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            stage(f'step_{step}_forward')
            hidden, stats, _ = model.forward_features(ids[:, :-1])
            loss = masked_head_cross_entropy(hidden, model.backbone.head, ids[:, 1:], loss_mask,
                                             chunk_size=32)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss.')
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - started
            stage(f'step_{step}_backward')
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - started - forward_seconds
            if any(p.grad is not None or p.requires_grad for p in model.backbone.parameters()):
                raise AssertionError('Backbone freeze violated.')
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or any(g.device.type != 'cpu' for g in grads):
                raise AssertionError('Missing CPU ROSA gradients.')
            gradient_norm = clip_all_gradients(optimizer, 1.)
            if gradient_norm <= 0:
                raise AssertionError('Zero ROSA gradient cannot establish successful training.')
            stage(f'step_{step}_optimizer')
            optimizer.step()
            states = [v for opt in (optimizer.dense_optimizer, optimizer.sparse_optimizer) if opt is not None
                      for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
            if any(v.device.type != 'cpu' for v in states):
                raise AssertionError('Optimizer state not CPU.')
            torch.cuda.synchronize()
            update = {'step': step, 'loss': float(loss), 'gradient_norm': gradient_norm,
                      'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
                      'total_step_seconds': time.perf_counter() - started,
                      'peak_gpu_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                      'peak_gpu_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
                      'peak_process_host_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      'all_rosa_gradients_cpu': True, 'optimizer_states_cpu': True, **stats}
            result['updates'].append(update)
            print(json.dumps(update), flush=True)
            del hidden, loss, grads, states
        optimizer.zero_grad()
        stage('freeze_hash_after')
        result['backbone_sha256_after'] = check_frozen_backbone(model, frozen)
        result['backbone_unchanged'] = True
        result['rosa_value_table_changed'] = parameter_hash(model.rosa_value_store) != value_before
        if not result['rosa_value_table_changed']:
            raise AssertionError('ROSA values did not change after optimizer update.')
        result['status'] = 'success'
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='cuda_oom', error=str(exc))
        try:
            result['peak_gpu_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30
            result['peak_gpu_reserved_GiB'] = torch.cuda.max_memory_reserved() / 2**30
        except Exception:
            pass
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        result['finished_at'] = now()
        write_json(result_path, result)
        print(f'{now()} T={args.length} outcome={result["status"]}', flush=True)
    return result


def search(args):
    from .binidx_data import resolve_binidx_prefix
    from .tracking import ArmTracker
    from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset

    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory for a length search.')
    out.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out)
    data = MMapIndexedDataset(resolve_binidx_prefix(args.data))
    ceiling = min(args.ceiling, int(data.sizes[args.row]) - 1)
    del data
    summary = {'started_at': now(), 'pid': os.getpid(), 'status': 'running', 'config': vars(args),
               'effective_ceiling': ceiling, 'largest_successful_length': 0, 'smallest_oom_length': None,
               'cases': [], 'scope': 'sample-specific full BPTT + optimizer update, not a corpus-wide safe length',
               'precision': 'FP32', 'backbone': 'frozen CUDA', 'learned_rosa': 'CPU',
               'checkpointing': 'per-block activation checkpointing, no TBPTT/state detachment'}
    write_json(out / 'search.json', summary)
    tracker = ArmTracker(args, 'improved', out, 'length-search-v1')
    best, failed, length = 0, None, min(args.start, ceiling)
    try:
        while True:
            summary['current_length'] = length
            summary['current_started_at'] = now()
            write_json(out / 'search.json', summary)
            print(f'{now()} START T={length} (best={best}, oom={failed})', flush=True)
            command = [sys.executable, '-u', '-m', 'rwkv_rosa.length_probe', '--worker', '--length', str(length),
                       '--out-dir', str(out), '--checkpoint', args.checkpoint, '--data', args.data,
                       '--row', str(args.row), '--steps', str(args.steps), '--seed', str(args.seed)]
            started = time.monotonic()
            with (out / f'length_{length}.log').open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                summary['worker_pid'] = process.pid
                write_json(out / 'search.json', summary)
                try:
                    returncode = process.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    returncode = None
            result_path = out / f'length_{length}.json'
            if returncode is None:
                result = {'input_tokens': length, 'status': 'timeout', 'timeout_seconds': args.timeout_seconds}
            elif result_path.exists():
                result = json.loads(result_path.read_text())
                if returncode != 0:
                    result.update(status='process_error', returncode=returncode)
            else:
                result = {'input_tokens': length, 'status': 'process_error', 'returncode': returncode}
            result['wall_seconds_including_load_and_audit'] = time.monotonic() - started
            summary['cases'].append(result)
            print(f'{now()} RESULT T={length}: {result["status"]}', flush=True)
            metrics = {'input_tokens': length, 'success': result['status'] == 'success',
                       'cuda_oom': result['status'] == 'cuda_oom',
                       'wall_seconds': result['wall_seconds_including_load_and_audit']}
            if result.get('updates'):
                metrics.update(result['updates'][-1])
            tracker.log('length_probe', metrics, len(summary['cases']))
            if result['status'] == 'success':
                best = max(best, length)
            elif result['status'] == 'cuda_oom':
                failed = length if failed is None else min(failed, length)
            else:
                summary['status'] = f'stopped_{result["status"]}'
            summary.update(largest_successful_length=best, smallest_oom_length=failed)
            write_json(out / 'search.json', summary)
            if summary['status'] != 'running':
                break
            if getattr(args, 'single_case', False):
                summary['status'] = 'single_case_complete' if result['status'] == 'success' else 'single_case_cuda_oom'
                break
            candidate = next_length(best, failed, ceiling, args.resolution)
            if candidate is None:
                summary['status'] = 'bracket_complete' if failed is not None else 'ceiling_passed'
                break
            length = candidate
        summary['finished_at'] = now()
        summary['wandb_url'] = tracker.url
        write_json(out / 'search.json', summary)
        tracker.summary('length_search', {'largest_successful_length': best,
                                         'smallest_oom_length': failed or 0, 'resolution': args.resolution})
    except BaseException:
        summary['status'] = 'search_error'
        summary['finished_at'] = now()
        write_json(out / 'search.json', summary)
        tracker.finish(failed=True)
        raise
    else:
        tracker.finish(failed=summary['status'].startswith('stopped_'))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cases', 'config')}, indent=2), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth')
    p.add_argument('--data', default='/hikscale/H800backup/Datasets/SFT_RWKV7_13B_2026_07_30/')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--start', type=int, default=2048)
    p.add_argument('--ceiling', type=int, default=86016)
    p.add_argument('--resolution', type=int, default=1024)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timeout-seconds', type=int, default=3600)
    p.add_argument('--single-case', action='store_true', help='Test start length only; do not bisect after OOM.')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--length', type=int, default=0)
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='offline')
    p.add_argument('--wandb-project', default='ROSA-RWKV7')
    p.add_argument('--wandb-entity', default='')
    p.add_argument('--wandb-group', default='rwkv7-frozen-cpu-rosa-length-search-20261109')
    p.add_argument('--wandb-run-name', default='full-bptt-length-search')
    args = p.parse_args(argv)
    if min(args.start, args.ceiling, args.resolution, args.steps, args.timeout_seconds) < 1 or args.row < 0:
        p.error('Length, resolution, steps and timeout must be positive; row must be nonnegative.')
    if args.worker and args.length < 1:
        p.error('--worker requires a positive --length.')
    return worker(args) if args.worker else search(args)


if __name__ == '__main__':
    main()
