#!/usr/bin/env python3
"""Export paired independent NIAH scores, raw answers and uncertainty."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def wilson(success, total):
    if not total:
        return [0., 0.]
    z = 1.959963984540054
    p = success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def summarize(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['seed']].append(pair)
    families = list(groups.values())
    deltas = np.array([np.mean([int(x['trained']['passkey_match']) - int(x['base']['passkey_match']) for x in family]) for family in families])
    indices = np.random.default_rng(20261110).integers(0, len(families), (10000, len(families)))
    bootstrap = deltas[indices].mean(1)
    row = {'paired_cases': len(pairs), 'seed_depth_families': len(families)}
    for arm in ('base', 'trained'):
        values = [x[arm] for x in pairs]
        success = sum(x['passkey_match'] for x in values)
        row[arm] = {'correct': success, 'accuracy': success/len(values),
            'accuracy_wilson_95': wilson(success, len(values)),
            'strict_correct': sum(x['strict_digits_only'] for x in values),
            'mean_answer_nll': float(np.mean([x['answer_nll'] for x in values])),
            'mean_first_token_rank': float(np.mean([x['first_answer_token_rank'] for x in values])),
            'generation_limit_hits': sum(x['hit_generation_limit'] for x in values),
            'final_answer_correct': sum(x.get('final_answer_passkey_match', False) for x in values),
            'unclosed_thinking': sum(x.get('unclosed_thinking', False) for x in values),
            **{f'correct_at_{budget}': sum(x[f'passkey_match_at_{budget}'] for x in values) for budget in (24, 64, 128)}}
    row['gain_percentage_points'] = 100*(row['trained']['accuracy']-row['base']['accuracy'])
    row['gain_cluster_bootstrap_95_pp'] = (100*np.quantile(bootstrap, [.025, .975])).tolist()
    final_deltas = np.array([np.mean([int(x['trained'].get('final_answer_passkey_match', False)) -
        int(x['base'].get('final_answer_passkey_match', False)) for x in family]) for family in families])
    row['final_answer_gain_percentage_points'] = 100*(row['trained']['final_answer_correct']-row['base']['final_answer_correct'])/len(pairs)
    row['final_answer_gain_cluster_bootstrap_95_pp'] = (100*np.quantile(final_deltas[indices].mean(1), [.025, .975])).tolist()
    row['base_wrong_trained_right'] = sum(not x['base']['passkey_match'] and x['trained']['passkey_match'] for x in pairs)
    row['base_right_trained_wrong'] = sum(x['base']['passkey_match'] and not x['trained']['passkey_match'] for x in pairs)
    row['both_correct'] = sum(x['base']['passkey_match'] and x['trained']['passkey_match'] for x in pairs)
    row['both_wrong'] = sum(not x['base']['passkey_match'] and not x['trained']['passkey_match'] for x in pairs)
    row['answer_nll_delta_trained_minus_base'] = row['trained']['mean_answer_nll']-row['base']['mean_answer_nll']
    return row


def collect(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    documents = [json.loads(p.read_text()) for p in sorted((folder/'cases').glob('*.json'))]
    documents = [x for x in documents if x['status'] == 'complete']
    pairs = []; raw = []
    for document in documents:
        spec = document['spec']
        for mode in ('qa', 'exact_prefix'):
            values = [x for x in document['results'] if x['mode'] == mode]
            if len(values) != 2 or {x['arm'] for x in values} != {'base', 'trained'}:
                raise ValueError('Unpaired case')
            arms = {x['arm']: x for x in values}
            for value in arms.values():
                text = value['generated_text']
                unclosed = '<think>' in text and '</think>' not in text
                final_text = '' if unclosed else text.rsplit('</think>', 1)[-1]
                first = re.search(r'(?<!\d)\d{6}(?!\d)', final_text)
                value.update(unclosed_thinking=unclosed, final_answer_text=final_text,
                    final_answer_passkey_match=bool(first and first.group() == value['secret']))
            pair = {**spec, 'mode': mode, **arms}
            if arms['base']['prompt_tokens'] != arms['trained']['prompt_tokens']:
                raise ValueError('Unequal prompt lengths across arms')
            pairs.append(pair)
            raw.extend({**x, 'stem_sha256': document['stem_sha256'],
                'prompt_sha256': document['prompt_sha256'][mode],
                **document['address_diagnostics'][mode]} for x in values)
    complete = len(documents) == len(manifest['cases'])
    result = {'status': 'complete' if complete else 'running', 'manifest': manifest,
              'completed_contexts': len(documents), 'planned_contexts': len(manifest['cases']),
              'generated_answers': len(raw), 'by_mode': {}, 'by_length': [], 'by_depth': [], 'by_cell': []}
    for mode in ('qa', 'exact_prefix'):
        subset = [x for x in pairs if x['mode'] == mode]
        if not subset:
            continue
        result['by_mode'][mode] = summarize(subset)
        for key, destination in [('length', 'by_length'), ('depth', 'by_depth')]:
            for value in sorted({x[key] for x in subset}):
                result[destination].append({'mode': mode, key: value,
                    **summarize([x for x in subset if x[key] == value])})
        for length in sorted({x['length'] for x in subset}):
            for depth in sorted({x['depth'] for x in subset}):
                cell = [x for x in subset if x['length'] == length and x['depth'] == depth]
                if cell:
                    result['by_cell'].append({'mode': mode, 'length': length, 'depth': depth, **summarize(cell)})
    result['uncertainty_note'] = ('Repeated seed-depth families share a secret across lengths. Bootstrap resamples these families, retaining all lengths within each family. Only15 independent families and3 replicates/cell; intervals are exploratory. Per-arm Wilson intervals are descriptive and do not account for family correlation. A degenerate bootstrap interval with all failures does not establish equivalence.')
    write(folder/'summary.json', result)
    if raw:
        fields = sorted({key for row in raw for key in row})
        with (folder/'answers.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields);writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in raw)
        (folder/'answers.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in raw))
    if complete:
        lines = ['# Frozen RWKV-7 2.9B: base versus trained ROSA', '',
            f"Checkpoint: step {manifest['checkpoint_step']}; `{manifest['checkpoint']}`. The original frozen base hash was verified in every worker. Training stopped after publishing step1800.", '',
            f"75 synthetic contexts: document lengths8192/16384/32768/65536/131072, five needle depths10/30/50/70/90%, three seeded examples per cell, similar-archive distractors. Same tokens, sampling `{manifest.get('sampling_policy', {'temperature':0., 'top_p':1.})}`, token budgets `{manifest.get('max_new_tokens_by_mode', manifest['max_new_tokens'])}`, device `{manifest['device']}`, precision `{manifest['dtype']}` for both arms. The128K label is document length; actual complete prompt length includes a short query suffix and is recorded in answers.csv.", '',
            'Predeclared primary: natural QA, first standalone six-digit sequence equals the secret. This can include thinking text, so a stricter final-answer-only result is prominently reported below: exclude unclosed <think> output and take the first six-digit number after </think> (or the whole output when there is no thinking tag). Secondary: exact-prefix completion. Strict digits-only format,24/64-token budgets and teacher-forced NLL are separate metrics. These tasks must not be combined to claim a QA gain.', '',
            '| QA scoring | Base correct | ROSA correct | Gain (pp) | Paired family bootstrap95% (pp) |',
            '|---|---:|---:|---:|---:|',
            f"| Final-answer-only | {result['by_mode']['qa']['base']['final_answer_correct']}/75 | {result['by_mode']['qa']['trained']['final_answer_correct']}/75 | {result['by_mode']['qa']['final_answer_gain_percentage_points']:+.2f} | {result['by_mode']['qa']['final_answer_gain_cluster_bootstrap_95_pp']} |", '',
            '| Mode | Base | ROSA | Gain (pp) | Paired family bootstrap95% (pp) |',
            '|---|---:|---:|---:|---:|']
        for mode, row in result['by_mode'].items():
            lines.append(f"| {mode} | {row['base']['correct']}/{row['paired_cases']} ({100*row['base']['accuracy']:.2f}%) | {row['trained']['correct']}/{row['paired_cases']} ({100*row['trained']['accuracy']:.2f}%) | {row['gain_percentage_points']:+.2f} | {row['gain_cluster_bootstrap_95_pp']} |")
        lines += ['', '| Document tokens | Mode | Base content hit | ROSA content hit | Content gain (pp) | Base final-answer hit | ROSA final-answer hit | Final-answer gain (pp) |', '|---:|---|---:|---:|---:|---:|---:|---:|']
        for row in result['by_length']:
            lines.append(f"| {row['length']} | {row['mode']} | {row['base']['correct']}/{row['paired_cases']} | {row['trained']['correct']}/{row['paired_cases']} | {row['gain_percentage_points']:+.2f} | {row['base']['final_answer_correct']}/{row['paired_cases']} | {row['trained']['final_answer_correct']}/{row['paired_cases']} | {row['final_answer_gain_percentage_points']:+.2f} |")
        lines += ['', '| Needle depth | Mode | Base content hit | ROSA content hit | Base final-answer hit | ROSA final-answer hit |', '|---:|---|---:|---:|---:|---:|']
        for row in result['by_depth']:
            lines.append(f"| {100*row['depth']:.0f}% | {row['mode']} | {row['base']['correct']}/{row['paired_cases']} | {row['trained']['correct']}/{row['paired_cases']} | {row['base']['final_answer_correct']}/{row['paired_cases']} | {row['trained']['final_answer_correct']}/{row['paired_cases']} |")
        lines += ['', '| Metric | QA base | QA ROSA | Prefix base | Prefix ROSA |', '|---|---:|---:|---:|---:|']
        for key in ('strict_correct', 'generation_limit_hits', 'unclosed_thinking', 'correct_at_24', 'correct_at_64', 'correct_at_128', 'mean_answer_nll'):
            values=[result['by_mode'][mode][arm][key] for mode in ('qa','exact_prefix') for arm in ('base','trained')]
            lines.append('| '+key+' | '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in values)+' |')
        lines += ['', 'No target X was set; report measured gains without a post-hoc pass threshold.', '', result['uncertainty_note'], '',
            'QA may consume its token budget in thinking/repetition; generation-limit counts and complete text are retained. The primary first-six-digit hit can occur inside thinking text; final-answer-only hits after a closed </think> tag are also exported. A low token-level NLL or a literal suffix-address hit is not a successful generated answer. Both arms use the same runtime; stateful BF16 split shapes can differ from full replay as recorded in runtime audits. Only one checkpoint, one synthetic Chinese archive format and three replicates/cell were tested; this is not an official or comprehensive retrieval benchmark.', '',
            'Checkpoint selection is held-out CE only, before synthetic test generation. See checkpoint_selection.json for all saved-checkpoint scores and exact sampled-probe limitations. No test examples were used for training, hyperparameter choice or checkpoint choice.', '']
        (folder/'REPORT.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = collect(Path(args.output))
    print(json.dumps({k: result[k] for k in ('status', 'completed_contexts', 'generated_answers', 'by_mode')}, indent=2))


if __name__ == '__main__':
    main()
