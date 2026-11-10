#!/usr/bin/env python3
"""Summarize exact long-answer retrieval separately from partial copying."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def paired(rows):
 groups=defaultdict(dict)
 for r in rows:groups[r['case_index']][r['arm']]=r
 if any(set(x)!= {'base','trained'} for x in groups.values()):raise ValueError('Unpaired long-answer comparison')
 pairs=list(groups.values());families=defaultdict(list)
 for p in pairs:families[p['base']['seed']].append(p)
 families=list(families.values());idx=np.random.default_rng(20261111).integers(0,len(families),(10000,len(families)))
 result={'paired_cases':len(pairs),'seed_families':len(families)}
 for arm in ('base','trained'):
  values=[p[arm] for p in pairs]
  result[arm]={key+'_correct':sum(x[key] for x in values) for key in ('whole_final_exact_match','final_contains_full_answer','content_contains_full_answer')}
  result[arm].update({key+'_rate':sum(x[key] for x in values)/len(values) for key in ('whole_final_exact_match','final_contains_full_answer','content_contains_full_answer')})
  result[arm].update(mean_correct_prefix_fraction=float(np.mean([x['best_correct_prefix_fraction'] for x in values])),
    mean_answer_nll=float(np.mean([x['answer_nll'] for x in values])),truncated=sum(x['hit_generation_limit'] for x in values),
    unclosed_thinking=sum(x['unclosed_thinking'] for x in values))
 for metric in ('whole_final_exact_match','final_contains_full_answer'):
  delta=np.array([np.mean([int(p['trained'][metric])-int(p['base'][metric]) for p in family]) for family in families])
  boot=delta[idx].mean(1)
  result[metric+'_gain_pp']=100*(result['trained'][metric+'_rate']-result['base'][metric+'_rate'])
  result[metric+'_gain_cluster_bootstrap_95_pp']=(100*np.quantile(boot,[.025,.975])).tolist()
 trained=[p['trained'] for p in pairs]
 result['rosa']={'cases_with_nonzero_residual':sum(x['rosa_nonzero_steps']>0 for x in trained),
  'cases_with_prompt_nonzero':sum(x['rosa_prompt_nonzero'] for x in trained),
  'nonzero_steps':sum(x['rosa_nonzero_steps'] for x in trained),'prediction_steps':sum(x['rosa_free_prediction_steps'] for x in trained),
  'below_match_threshold_activations':0,'threshold_checks_passed':all(x['rosa_threshold_verified'] for x in trained)}
 return result


def collect(folder):
 m=json.loads((folder/'manifest.json').read_text());docs=[json.loads(p.read_text()) for p in sorted((folder/'cases').glob('*.json'))];docs=[x for x in docs if x['status']=='complete'];rows=[]
 for d in docs:
  for r in d['results']:
   r.update(prompt_sha256=d['prompt_sha256'][r['mode']],stem_sha256=d['stem_sha256'],complete_distractor_records=d['complete_distractor_records']);rows.append(r)
 report={'status':'complete' if len(docs)==len(m['cases']) else 'running','completed_contexts':len(docs),
  'planned_contexts':len(m['cases']),'generated_answers':len(rows),'manifest':m,'by_mode':{},'by_answer':[],'by_length':[],'by_length_answer':[]}
 for mode in ('qa','exact_prefix'):
  rr=[x for x in rows if x['mode']==mode]
  if not rr:continue
  report['by_mode'][mode]=paired(rr)
  for key in ('answer','length','length_answer'):
   if key=='answer':fields=('answer_kind','nominal_answer_tokens')
   elif key=='length':fields=('length',)
   else:fields=('length','answer_kind','nominal_answer_tokens')
   groups=defaultdict(list)
   for r in rr:groups[tuple(r[k] for k in fields)].append(r)
   for vals,group in sorted(groups.items()):
    entry={'mode':mode,**dict(zip(fields,vals)),**paired(group),
     'actual_answer_tokens_min':min(x['actual_answer_tokens'] for x in group),'actual_answer_tokens_max':max(x['actual_answer_tokens'] for x in group),
     'answer_chars_min':min(x['answer_chars'] for x in group),'answer_chars_max':max(x['answer_chars'] for x in group),
     'question_tokens_min':min(x['question_tokens'] for x in group),'question_tokens_max':max(x['question_tokens'] for x in group),
     'input_tokens_min':min(x['prompt_tokens'] for x in group),'input_tokens_max':max(x['prompt_tokens'] for x in group)}
    report['by_'+key].append(entry)
 write(folder/'summary.json',report)
 if rows:
  (folder/'answers.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  fields=sorted({k for x in rows for k in x})
  with (folder/'answers.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in rows)
  for key in ('answer','length','length_answer'):
   entries=[]
   for x in report['by_'+key]:
    e={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items() if k not in ('base','trained','rosa')}
    for arm in ('base','trained','rosa'):e.update({arm+'_'+k:v for k,v in x[arm].items()})
    entries.append(e)
   with (folder/('by_'+key+'.csv')).open('w',newline='') as f:
    w=csv.DictWriter(f,list(entries[0]));w.writeheader();w.writerows(entries)
 return report


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();r=collect(Path(a.output))
 print(json.dumps({k:r[k] for k in ('status','completed_contexts','generated_answers','by_mode')},indent=2))
