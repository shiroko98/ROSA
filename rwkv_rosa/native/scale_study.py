"""Matched full-context real-data screening and checkpoint confirmation on CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from rwkv_rosa.backbone import load_base_state
from rwkv_rosa.freeze_audit import parameter_hash
from rwkv_rosa.native.adapter import CPUImprovedROSA, configure_output_head
from rwkv_rosa.native.cpu_compare import write_json
from rwkv_rosa.native.position import scale_options
from rosa_runtime import RosaAddressBatch
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset
from third_party.rwkv_lm_v7_native.src.sft_split import compute_sft_shuffled_split_indices

BASE = '/mnt/data/Models/RWKV-7/rwkv7-g1k-2.9b-20260930-ctx25600.pth'
DATA = '/mnt/data/Datasets/SFT_RWKV7_13B_rwkv_g1_ctx192512/results/SFT_RWKV7_13B'
CONTEXT = 192512
ADDRESS_FIELDS = ('addr_ids', 'raw_match_lens', 'fired_match_lens', 'valid_mask', 'special_mask')


def configurations():
    values = [{'id': 'base', 'mode': 'off'}]
    values += [dict(id=f'fixed_{s:g}', mode='fixed', scale=s) for s in (.01, .02, .05, .1, .2)]
    values += [dict(id=f'dyn_m{m}_c{c:g}', mode='logit_gap', min_match=m, maximum=c)
               for m in (2, 4, 6, 8, 12) for c in (.02, .05, .1, .2, .4)]
    return values


def prepare(study):
    study = Path(study); study.mkdir(parents=True, exist_ok=False)
    train, heldout = compute_sft_shuffled_split_indices(len(MMapIndexedDataset(DATA)), 2117, seed=42)
    selected_train = np.random.default_rng(20261109).choice(train, size=64, replace=False)
    calibration = heldout[8:16]; confirmation = heldout[16:24]
    assert not (set(selected_train) & (set(calibration) | set(confirmation)))
    assert not (set(calibration) & set(confirmation))
    # Reverse the native seed-42 split so these exact rows become its held-out
    # subset, and every trial sees the same training order and loss masks.
    permutation = np.random.default_rng(42).permutation(72)
    physical_sources = np.zeros(72, dtype=np.int64)
    physical_sources[permutation[:64]] = selected_train
    physical_sources[permutation[64:]] = calibration
    pilot = study / 'pilot_data'; pilot.mkdir()
    prefix = pilot / 'sft'
    for suffix, dtype in (('', np.uint16), ('.mask', np.uint8)):
        dataset = MMapIndexedDataset(DATA + suffix)
        with Path(str(prefix) + suffix + '.bin').open('wb') as stream:
            for index in physical_sources:
                stream.write(np.asarray(dataset[int(index)], dtype=dtype).tobytes())
        with MMapIndexedDataset.Index.writer(str(prefix) + suffix + '.idx', dtype) as writer:
            writer.write([CONTEXT + 1] * len(physical_sources), list(range(len(physical_sources) + 1)))
    manifest = {'status': 'prepared', 'data': DATA, 'base': BASE, 'context': CONTEXT,
                'full_data_train_rows': len(train), 'full_data_heldout_rows': len(heldout),
                'training_source_rows': selected_train.tolist(), 'calibration_source_rows': calibration.tolist(),
                'confirmation_source_rows': confirmation.tolist(), 'pilot_physical_source_rows': physical_sources.tolist(),
                'pilot_data': str(prefix), 'configurations': configurations(),
                'screen_supervised_tokens_per_row': 8192, 'pilot_optimizer_steps': 28,
                'pilot_per_gpu_batch': 2, 'pilot_lr': 3e-5, 'pilot_warmup_steps': 5,
                'selection': 'Screen initial adapters; train eight selected configurations with identical data; choose by exact calibration CE and independent confirmation CE. No old checkpoint resumes.'}
    write_json(study / 'manifest.json', manifest)
    return manifest


def native_body():
    from src.model import RWKV
    args = SimpleNamespace(n_layer=32, n_embd=2560, dim_att=2560, dim_ffn=10240,
                           vocab_size=65536, head_size=64, d_decay_lora=96, d_aaa_lora=96,
                           d_mv_lora=64, d_gate_lora=320, my_testing='x070', grad_cp=0, ctx_len=CONTEXT)
    with torch.device('meta'):
        body = RWKV(args)
    body.load_state_dict(load_base_state(BASE), strict=True, assign=True)
    body.eval().requires_grad_(False).cuda()
    configure_output_head(body.head, 'rms')
    return body


def adapter_for(body, config, weights=None):
    branch = CPUImprovedROSA(body.emb, position='post_norm', output_norm='rms', device='cuda',
        address_impl='compiled_cpu', scale=config.get('scale', .05),
        scale_mode='fixed' if config['mode'] == 'off' else config['mode'],
        dynamic_min_match=config.get('min_match', 6), dynamic_max=config.get('maximum', .2))
    branch.bind_output_head(body.head)
    if weights is not None:
        branch.load_state_dict(weights, strict=True)
    branch.eval().requires_grad_(False)
    return branch


@torch.inference_mode()
def cache_row(body, branch, source_row, folder, seed):
    data = MMapIndexedDataset(DATA); masks = MMapIndexedDataset(DATA + '.mask')
    ids = torch.tensor(data[int(source_row)].astype(np.int64)).unsqueeze(0)
    mask = torch.tensor(masks[int(source_row)][1:].astype(bool))
    active = mask.nonzero().flatten().numpy()
    chosen = np.sort(np.random.default_rng(seed).choice(active, size=min(8192, len(active)), replace=False))
    positions = torch.from_numpy(chosen)
    torch.cuda.synchronize(); started = time.monotonic()
    hidden = body._forward_features(ids[:, :-1].cuda())[:, positions.cuda()].clone()
    branch.begin(ids[:, :-1])
    addresses = {key: getattr(branch.addresses, key)[:, positions].clone() for key in ADDRESS_FIELDS}
    torch.cuda.synchronize()
    cache = {'source_row': int(source_row), 'context': CONTEXT, 'selected_positions': positions,
             'supervised_population': len(active), 'targets': ids[:, 1:][:, positions],
             'hidden': hidden.cpu(), 'addresses': addresses,
             'input_mask_sha256': hashlib.sha256(ids.numpy().tobytes() + mask.numpy().tobytes()).hexdigest(),
             'prefill_seconds': time.monotonic() - started}
    torch.save(cache, folder)
    return cache


@torch.inference_mode()
def evaluate(body, branch, config, cache, base_losses=None):
    h = cache['hidden'].cuda(); target = cache['targets'].cuda().flatten()
    address = RosaAddressBatch(**cache['addresses'], source='full_context_native_cache')
    if config['mode'] == 'off':
        after = h.float(); stats = {}
    else:
        branch.scale_mode = config['mode']; branch.scale = config.get('scale', .05)
        branch.scale_options.update(scale_mode=config['mode'], dynamic_max=config.get('maximum', .2),
                                    dynamic_min_match=config.get('min_match', 6))
        branch.addresses = address; branch.stats = {}
        after = branch.inject_post_norm(h)
        stats = dict(branch.stats)
    losses = []; correct = 0
    for start in range(0, len(target), 256):
        scores = body.head(after[0, start:start + 256]).float()
        y = target[start:start + 256]
        losses.append(F.cross_entropy(scores, y, reduction='none'))
        correct += int((scores.argmax(-1) == y).sum())
    loss = torch.cat(losses).cpu()
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError('Nonfinite study loss')
    delta = after - h.float()
    row = {'config': config['id'], 'source_row': cache['source_row'], 'scored_tokens': len(target),
           'supervised_population': cache['supervised_population'],
           'loss': float(loss.mean()), 'accuracy': correct / len(target), 'nll_sum': float(loss.double().sum()),
           'correct': correct, 'residual_rms': float(delta.square().mean().sqrt()),
           'residual_relative_rms': float(delta.norm() / h.float().norm()),
           'nonzero_residual_fraction': float(delta.abs().sum(-1).ne(0).float().mean()), **stats}
    if base_losses is not None:
        paired = loss - base_losses
        row['loss_delta_vs_base'] = float(paired.mean())
        valid = address.valid_mask.flatten(); hit = valid & address.addr_ids.flatten().eq(target.cpu())
        for group, selected in (('candidate_correct', hit), ('candidate_wrong', valid & ~hit), ('no_candidate', ~valid)):
            row[group] = {'tokens': int(selected.sum()),
                          'loss_delta': float(paired[selected].mean()) if selected.any() else 0.,
                          'worsened_fraction': float((paired[selected] > 1e-6).float().mean()) if selected.any() else 0.}
    return row, loss


def worker(args):
    study = Path(args.study); manifest = json.loads((study / 'manifest.json').read_text())
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    body = native_body()
    initial = adapter_for(body, {'mode': 'fixed'})
    slot = args.slot; results = []
    if args.phase == 'extra':
        for slot in range(8):
            cache = torch.load(study / f'calibration_cache_{slot}.pth', map_location='cpu', weights_only=True)
            _, base_loss = evaluate(body, initial, {'id': 'base', 'mode': 'off'}, cache)
            for maximum in (.02, .05, .1, .2, .4):
                config = dict(id=f'dyn_m1_c{maximum:g}', mode='logit_gap', min_match=1, maximum=maximum)
                row, _ = evaluate(body, initial, config, cache, base_loss); results.append(row)
                write_json(study / 'extra_screen.json', {'status': 'running', 'results': results})
        write_json(study / 'extra_screen.json', {'status': 'complete', 'results': results})
    elif args.phase == 'screen':
        cache = cache_row(body, initial, manifest['calibration_source_rows'][slot], study / f'calibration_cache_{slot}.pth', 20261109 + slot)
        baseline, base_loss = evaluate(body, initial, configurations()[0], cache)
        results.append(baseline)
        for config in configurations()[1:]:
            row, _ = evaluate(body, initial, config, cache, base_loss)
            results.append(row)
            write_json(study / f'screen_{slot}.json', {'status': 'running', 'slot': slot, 'results': results})
            print(json.dumps(row), flush=True)
        # Hold back confirmation scores until after all pilot weights are fixed.
        cache_row(body, initial, manifest['confirmation_source_rows'][slot], study / f'confirmation_cache_{slot}.pth', 20262109 + slot)
        write_json(study / f'screen_{slot}.json', {'status': 'complete', 'slot': slot, 'results': results})
    else:
        cache = torch.load(study / f'confirmation_cache_{slot}.pth', map_location='cpu', weights_only=True)
        baseline, base_loss = evaluate(body, initial, configurations()[0], cache); results.append(baseline)
        selected = json.loads((study / 'pilot_configurations.json').read_text())
        for config in selected:
            checkpoint = torch.load(study / 'pilots' / config['id'] / 'native-last.ckpt', map_location='cpu', weights_only=False, mmap=True)
            if int(checkpoint['global_step']) != manifest['pilot_optimizer_steps']:
                raise ValueError('Pilot did not complete the declared optimizer steps')
            meta = checkpoint['native_rosa']
            if (meta['scale_mode'] != config['mode'] or
                    (config['mode'] == 'fixed' and meta['scale'] != config['scale']) or
                    (config['mode'] == 'logit_gap' and
                     (meta['dynamic_min_match'] != config['min_match'] or meta['dynamic_max'] != config['maximum']))):
                raise ValueError('Pilot checkpoint metadata differs from its assigned configuration')
            if checkpoint['native_rosa_frozen_base']['path'] != BASE:
                raise ValueError('Pilot checkpoint references a different frozen base')
            weights = {k[len('rosa_cpu.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('rosa_cpu.')}
            branch = adapter_for(body, config, weights)
            row, _ = evaluate(body, branch, config, cache, base_loss)
            row['pilot_step'] = int(checkpoint['global_step']); results.append(row)
            write_json(study / f'confirmation_{slot}.json', {'status': 'running', 'slot': slot, 'results': results})
            print(json.dumps(row), flush=True)
            del branch, weights, checkpoint
        write_json(study / f'confirmation_{slot}.json', {'status': 'complete', 'slot': slot, 'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--phase', choices=('prepare', 'screen', 'extra', 'confirmation'), required=True)
    parser.add_argument('--slot', type=int, default=0); args = parser.parse_args()
    if args.phase == 'prepare':
        print(json.dumps(prepare(args.study), indent=2))
    else:
        worker(args)


if __name__ == '__main__':
    main()
