#!/usr/bin/env python3
"""Verify matched pilots, collect exact W&B eval values, and rank confirmation."""
import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np


def collect(study, api=None):
    configs = json.loads((study / 'pilot_configurations.json').read_text())
    result = []; reference_batches = None
    for config in configs:
        folder = study / 'pilots' / config['id']
        steps = [json.loads(line) for line in (folder / 'native_steps.jsonl').read_text().splitlines()]
        if len(steps) != 28 or steps[-1]['step'] != 28 or not (folder / 'native-last.ckpt').exists():
            raise ValueError('Incomplete pilot: ' + config['id'])
        batches = [step['batch_sha256'] for step in steps]
        if reference_batches is None:
            reference_batches = batches
        if batches != reference_batches:
            raise ValueError('Pilot input/mask order differs: ' + config['id'])
        audit = json.loads((folder / 'native_audit.json').read_text())
        if not audit.get('backbone_unchanged') or not audit.get('rosa_values_changed'):
            raise ValueError('Pilot failed frozen-base/adapter-update audit')
        values = {int(step): float(loss) for step, loss in re.findall(
            r'eval step (\d+) loss ([\d.]+)', (folder / 'train_log.txt').read_text())}
        run_id = (folder / 'wandb_run_id.txt').read_text().strip()
        if api is not None:
            history = list(api.run('mobius/ROSA-RWKV7/' + run_id).scan_history(keys=['_step', 'eval/loss']))
            values.update({int(x['_step']): float(x['eval/loss']) for x in history})
        if 0 not in values or 28 not in values:
            raise ValueError('Missing initial/final held-out eval: ' + config['id'])
        result.append({**config, 'calibration_initial_loss': values[0], 'calibration_final_loss': values[28],
                       'calibration_training_delta': values[28] - values[0],
                       'median_train_step_seconds': statistics.median(x['seconds'] for x in steps),
                       'median_gradient_norm': statistics.median(x['grad_norm'] for x in steps),
                       'frozen_backbone_sha256': audit['backbone_sha256_final'],
                       'pilot_run_id': run_id, 'pilot_checkpoint': str(folder / 'native-last.ckpt')})
    if len({x['frozen_backbone_sha256'] for x in result}) != 1:
        raise ValueError('Pilot frozen bases differ')
    write(study / 'pilot_summary.json', {'status': 'complete', 'matched_input_masks': True,
                                       'optimizer_steps_per_pilot': 28, 'configurations': result})
    return result


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def rank(study, pilot):
    files = [json.loads((study / f'confirmation_{i}.json').read_text()) for i in range(8)]
    if not all(x['status'] == 'complete' for x in files):
        raise ValueError('Confirmation is incomplete')
    rows = []
    for config in [{'id': 'base', 'mode': 'off'}, *pilot]:
        values = [next(r for r in file['results'] if r['config'] == config['id']) for file in files]
        populations = np.array([x['supervised_population'] for x in values], dtype=float)
        losses = np.array([x['nll_sum'] / x['scored_tokens'] for x in values])
        accuracies = np.array([x['accuracy'] for x in values])
        rows.append({**config, 'confirmation_loss': float(np.average(losses, weights=populations)),
                     'confirmation_accuracy': float(np.average(accuracies, weights=populations)),
                     'confirmation_row_losses': losses.tolist(), 'confirmation_populations': populations.tolist(),
                     'residual_relative_rms': float(np.mean([x['residual_relative_rms'] for x in values])),
                     'scored_confirmation_tokens': sum(x['scored_tokens'] for x in values)})
    baseline = rows[0]
    rng = np.random.default_rng(42); indices = rng.integers(0, 8, (10000, 8))
    for row in rows[1:]:
        delta = np.array(row['confirmation_row_losses']) - np.array(baseline['confirmation_row_losses'])
        weights = np.array(row['confirmation_populations'])
        bootstrap = (delta[indices] * weights[indices]).sum(1) / weights[indices].sum(1)
        row['confirmation_delta_vs_base'] = row['confirmation_loss'] - baseline['confirmation_loss']
        row['confirmation_delta_bootstrap_95'] = np.quantile(bootstrap, [.025, .975]).tolist()
        row['selection_loss'] = (row['calibration_final_loss'] + row['confirmation_loss']) / 2
    ranked = sorted(rows[1:], key=lambda x: x['selection_loss'])
    result = {'status': 'complete', 'baseline_confirmation': baseline, 'ranked_configurations': ranked,
              'selected': ranked[0], 'selection_rule': 'Mean of exact calibration CE and population-weighted confirmation CE',
              'scope': 'Eight configurations, 28 updates, one shared training sample order, eight calibration rows, eight separate confirmation rows. Confirmation scores sample up to 8192 targets per row from full 192512-token histories. This is a short-run configuration choice, not a universal optimum or generation benchmark.'}
    write(study / 'final_selection.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--final', action='store_true'); args = parser.parse_args()
    study = Path(args.study); api = None
    if args.wandb:
        import os
        os.environ.setdefault('WANDB_BASE_URL', 'http://dian.sea-group.org:9098')
        from rwkv_rosa.native.wandb_auth import configure_wandb_auth
        configure_wandb_auth()
        import wandb
        api = wandb.Api(timeout=30)
    pilot = collect(study, api)
    result = rank(study, pilot) if args.final else pilot
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
