#!/usr/bin/env python3
"""Compare decode budgets and sampling on the exact same paired NIAH cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from summarize_detailed_cpu_needle import collect,write


def run(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    previous=Path(manifest['reference_experiment']);original=json.loads((previous/'summary.json').read_text())
    original_cases={p.name:json.loads(p.read_text()) for p in (previous/'cases').glob('*.json')}
    policies=[{'id':'greedy128','temperature':0.,'top_p':1.,'qa_budget':128,'prefix_budget':128}]
    configurations=[{'policy':policies[0],'status':'complete','completed_contexts':75,
        'by_mode':original['by_mode'],'by_length':original['by_length'],'by_depth':original['by_depth']}]
    prefix_checks=0;budget128_final={};raw=[]
    for policy in manifest['sampling_configurations']:
        report=collect(folder/policy['id'])
        configurations.append({'policy':{**policy,'qa_budget':512,'prefix_budget':128},
            'status':report['status'],'completed_contexts':report['completed_contexts'],
            'by_mode':report['by_mode'],'by_length':report['by_length'],'by_depth':report['by_depth']})
        for p in (folder/policy['id']/'cases').glob('*.json'):
            doc=json.loads(p.read_text())
            if doc['status']!='complete':continue
            before=original_cases[p.name]
            if doc['prompt_sha256']!=before['prompt_sha256'] or doc['secret']!=before['secret']:
                raise ValueError('Sampling comparison input/secret differs from original')
            if policy['id']=='greedy512':
                for row in doc['results']:
                    ref=next(x for x in before['results'] if x['arm']==row['arm'] and x['mode']==row['mode'])
                    if row['generated_ids'][:len(ref['generated_ids'])]!=ref['generated_ids']:
                        raise ValueError('Greedy first128 tokens do not reproduce original run')
                    prefix_checks+=1
        p=folder/policy['id']/'answers.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():raw.append({'policy':policy['id'],**json.loads(line)})
    complete=all(x['status']=='complete' for x in configurations)
    result={'status':'complete' if complete else 'running','manifest':manifest,
        'configurations':configurations,'greedy_original_prefixes_verified':prefix_checks,
        'new_generated_answers':len(raw),
        'interpretation':'Greedy128→greedy512 isolates QA budget. Compare sampling policies against greedy512 to assess sampling. Prefix task cap is128 throughout. All prompts/checkpoint/runtime match; one seeded sample per case/policy, no best-of. This is a descriptive sweep on an already examined test set, not an untouched test after decoder selection.'}
    write(folder/'summary.json',result)
    if raw:
        fields=sorted({k for row in raw for k in row})
        with (folder/'answers.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader()
            w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in raw)
        (folder/'answers.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in raw))
    entries=[]
    for config in configurations:
        for mode,row in config['by_mode'].items():
            entry={'policy':config['policy']['id'],'temperature':config['policy']['temperature'],
                'top_p':config['policy']['top_p'],'qa_budget':config['policy']['qa_budget'],
                'prefix_budget':128,'mode':mode,'cases':row['paired_cases'],
                'base_content_correct':row['base']['correct'],'ROSA_content_correct':row['trained']['correct'],
                'base_final_correct':row['base']['final_answer_correct'],'ROSA_final_correct':row['trained']['final_answer_correct'],
                'content_gain_pp':row['gain_percentage_points'],'final_gain_pp':row['final_answer_gain_percentage_points'],
                'final_gain_ci_low':row['final_answer_gain_cluster_bootstrap_95_pp'][0],
                'final_gain_ci_high':row['final_answer_gain_cluster_bootstrap_95_pp'][1],
                'base_truncated':row['base']['generation_limit_hits'],'ROSA_truncated':row['trained']['generation_limit_hits'],
                'base_strict':row['base']['strict_correct'],'ROSA_strict':row['trained']['strict_correct']}
            entries.append(entry)
    with (folder/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(entries[0]));w.writeheader();w.writerows(entries)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    r=run(Path(args.output))
    print(json.dumps({'status':r['status'],'generated':r['new_generated_answers'],'verified_greedy_prefixes':r['greedy_original_prefixes_verified'],
        'counts':[(x['policy']['id'],x['completed_contexts']) for x in r['configurations']]},indent=2))
