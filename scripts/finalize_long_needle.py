#!/usr/bin/env python3
"""Verify and report the completed 8K--256K paired long-answer experiment."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rwkv_rosa.native.long_answer_cases import build_long_case, long_answer_scores
from rwkv_rosa.native.needle_eval import clean_decode
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER
from summarize_long_needle import collect, paired, write


def wilson(correct, count):
    z = 1.959963984540054
    p = correct / count
    center = (p + z*z/(2*count))/(1 + z*z/count)
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/(1 + z*z/count)
    return [100*(center-radius), 100*(center+radius)]


def verify(folder, expected_workers, tokenizer, regenerate):
    report = collect(folder)
    assert report['status'] == 'complete', str(folder)
    assert len(list(folder.glob('worker_complete_*.json'))) == expected_workers
    assert not list(folder.glob('failure*'))
    audits = [json.loads(p.read_text()) for p in folder.glob('runtime_audit_*.json')]
    assert len(audits) == expected_workers
    assert all(x['checkpoint_step'] == 1800 and not x['training'] for x in audits)
    assert all(x['dynamic_min_match'] == 6 and x['dynamic_max'] == .3 for x in audits)
    assert len({x['base_sha256'] for x in audits}) == 1
    assert len({x['adapter_sha256'] for x in audits}) == 1
    totals = defaultdict(int)
    rows = []
    for path in sorted((folder/'cases').glob('*.json')):
        doc = json.loads(path.read_text())
        index = int(path.stem)
        spec = report['manifest']['cases'][index]
        assert doc['spec'] == spec and doc['status'] == 'complete'
        assert len(doc['results']) == 4
        traces = json.loads((folder/'traces'/path.name).read_text())
        if regenerate:
            case = build_long_case(tokenizer, spec['length'], spec['depth'], spec['seed'],
                                   spec['answer_kind'], spec['nominal_answer_tokens'])
            assert case['stem_sha256'] == doc['stem_sha256']
            assert case['secret'] == doc['secret']
            for mode, suffix in case['suffix_ids'].items():
                ids = case['stem_ids'] + suffix
                assert hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest() == doc['prompt_sha256'][mode]
                assert tokenizer.decode(suffix) == doc['suffix_text'][mode]
            totals['regenerated_documents'] += 1
        for mode in ('qa', 'exact_prefix'):
            arms = {r['arm']: r for r in doc['results'] if r['mode'] == mode}
            assert set(arms) == {'base', 'trained'}
            a, b = arms['base'], arms['trained']
            for key in ('prompt_tokens', 'question_tokens', 'target_text', 'answer_ids', 'sampling'):
                assert a[key] == b[key]
            assert a['prompt_tokens'] == spec['length'] + a['question_tokens']
            assert a['sampling']['temperature'] == 0
            assert a['max_new_tokens'] == report['manifest']['max_new_tokens_by_mode'][mode]
            events = traces[mode]
            assert len(events) == len(b['generated_ids']) == b['rosa_free_prediction_steps']
            assert [e['emitted_tokens'] for e in events] == list(range(len(events)))
            assert events[0]['phase'] == 'prompt'
            assert all(e['phase'] == 'generation' for e in events[1:])
            active = [e for e in events if e['residual_rms'] > 0]
            assert len(active) == b['rosa_nonzero_steps']
            assert (active[0]['emitted_tokens'] if active else None) == b['rosa_first_nonzero_step']
            assert (events[0]['residual_rms'] > 0) == b['rosa_prompt_nonzero']
            for e in events:
                assert math.isfinite(e['effective_scale']) and math.isfinite(e['residual_rms'])
                assert 0 <= e['effective_scale'] <= .300001 and e['residual_rms'] >= 0
                if not e['valid'] or e['match_tokens'] < 6:
                    assert e['effective_scale'] == 0 and e['residual_rms'] == 0
            totals['verified_trace_prediction_steps'] += len(events)
            totals['nonzero_residual_steps'] += len(active)
            totals['paired_prompts'] += 1
            totals['divergent_generated_sequences'] += a['generated_ids'] != b['generated_ids']
            for row in arms.values():
                assert clean_decode(tokenizer, row['generated_ids']) == row['generated_text']
                for key, value in long_answer_scores(tokenizer, row['generated_text'], row['target_text']).items():
                    assert row[key] == value, (path, mode, key)
                row.update(prompt_sha256=doc['prompt_sha256'][mode], experiment=str(folder))
                rows.append(row)
    proof = dict(totals, status='verified', worker_count=expected_workers,
                 base_sha256=audits[0]['base_sha256'], adapter_sha256=audits[0]['adapter_sha256'],
                 complete_contexts=report['completed_contexts'], generated_answers=len(rows),
                 input_truncation=False, below_threshold_nonzero_residuals=0,
                 scores_recomputed=True, exact_prompt_reconstruction=regenerate)
    write(folder/'verification.json', proof)
    manifest = report['manifest']
    manifest.update(status='complete', finished_at=datetime.now().astimezone().isoformat())
    write(folder/'manifest.json', manifest)
    return rows, proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main', required=True)
    p.add_argument('--extrapolation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--regenerate-prompts', action='store_true')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    tokenizer = TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    rows, proofs = [], []
    for name, workers in ((args.main, 16), (args.extrapolation, 8)):
        rr, proof = verify(Path(name), workers, tokenizer, args.regenerate_prompts)
        rows.extend(rr); proofs.append(proof)
    assert len({x['base_sha256'] for x in proofs}) == len({x['adapter_sha256'] for x in proofs}) == 1
    assert len(rows) == 576
    # Every answer/position family is identical across all four document lengths.
    families = defaultdict(list)
    for row in rows:
        families[(row['seed'], row['mode'], row['arm'])].append(row)
    assert len(families) == 144
    for group in families.values():
        assert {x['length'] for x in group} == {8192,32768,131072,262144}
        assert len({x['target_text'] for x in group}) == len({x['key'] for x in group}) == 1
        assert len({x['depth'] for x in group}) == 1
    examples=[]
    for mode in ('qa','exact_prefix'):
        for arm in ('base','trained'):
            rr=[x for x in rows if x['mode']==mode and x['arm']==arm]
            for success in (True,False):
                options=[x for x in rr if x['final_contains_full_answer']==success]
                if options:
                    e=sorted(options,key=lambda x:(-x['length'],-x['actual_answer_tokens'],x['case_index']))[0]
                    examples.append({k:e[k] for k in ('length','mode','arm','case_index','key','target_text',
                        'actual_answer_tokens','generated_text','final_answer_text','final_contains_full_answer',
                        'best_correct_prefix_fraction','hit_generation_limit')})
    write(out/'examples.json',examples)
    predecessor=Path(json.loads((Path(args.main)/'manifest.json').read_text())['superseded_lower_budget_experiment'])
    budget_groups=defaultdict(lambda:dict(count=0,old_full_hits=0,new_full_hits=0,identical_generated_prefixes=0))
    for path in sorted((predecessor/'cases').glob('*.json')):
        old=json.loads(path.read_text())
        if old['status']!='complete':continue
        new=json.loads((Path(args.main)/'cases'/path.name).read_text())
        assert old['prompt_sha256']==new['prompt_sha256']
        for a in old['results']:
            b=next(x for x in new['results'] if x['arm']==a['arm'] and x['mode']==a['mode'])
            g=budget_groups[(a['mode'],a['arm'])];g['count']+=1
            g['old_full_hits']+=int(a['final_contains_full_answer']);g['new_full_hits']+=int(b['final_contains_full_answer'])
            assert b['generated_ids'][:len(a['generated_ids'])]==a['generated_ids']
            g['identical_generated_prefixes']+=1
    budget_comparison=[dict(mode=k[0],arm=k[1],**v) for k,v in sorted(budget_groups.items())]
    write(out/'output_budget_comparison.json',{'scope':'Only the57 completed predecessor documents; incomplete and unbalanced across context lengths, not a separate full benchmark.',
      'old_budgets':{'qa':1024,'exact_prefix':256},'new_budgets':{'qa':4096,'exact_prefix':1024},
      'comparisons':budget_comparison})
    groups = []
    for fields in [('length','mode'), ('length','mode','answer_kind','nominal_answer_tokens'),
                   ('length','mode','depth')]:
        grouped = defaultdict(list)
        for row in rows: grouped[tuple(row[f] for f in fields)].append(row)
        for values, group in sorted(grouped.items()):
            result = dict(zip(fields,values)); result.update(paired(group))
            for arm in ('base','trained'):
                result[arm]['full_span_wilson_95_percent'] = wilson(result[arm]['final_contains_full_answer_correct'],result['paired_cases'])
            for name in ('prompt_tokens','question_tokens','actual_answer_tokens','answer_chars'):
                result[name+'_min'] = min(r[name] for r in group)
                result[name+'_max'] = max(r[name] for r in group)
            result['group_fields'] = list(fields); groups.append(result)
    main_metrics=json.loads((Path(args.main)/'summary.json').read_text())['by_mode']
    summary = {'status':'complete','checkpoint_step':1800, 'contexts':144,'paired_prompts':288,
               'generated_answers':576,'seed_families':36,'training_stopped':True,
               'groups':groups,'verification':proofs,'output_budget_comparison':budget_comparison,
               'main_suite_by_mode':main_metrics,
               'main_suite':args.main,'extrapolation_suite':args.extrapolation,
               'metric':'full target occurs contiguously in final output after excluding thinking; extra surrounding output allowed',
               'confidence_interval':'paired bootstrap,10000 resamples,seed20261111,clustered by answer/position seed; all-zero intervals can degenerate. Per-length Wilson intervals use36 independent synthetic seed families; groups with6 cases are exploratory.'}
    write(out/'summary.json',summary)
    (out/'answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    keys=sorted({k for r in rows for k in r})
    with (out/'answers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    flat=[]
    for g in groups:
        r={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in g.items() if k not in ('base','trained','rosa')}
        for arm in ('base','trained','rosa'):
            r.update({arm+'_'+k:json.dumps(v) if isinstance(v,list) else v for k,v in g[arm].items()})
        flat.append(r)
    keys=sorted({k for r in flat for k in r})
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(flat)
    text = ['# 冻结 RWKV-7 2.9B：长问题、长答案与 256K 外推检索对比', '',
      '本次训练已停止，评测固定使用 step1800 ROSA 检查点。两臂共享同一冻结 BF16 本体、FP32 输出头计算和相同输入；仅 ROSA 残差开关不同。未使用测试数据更新权重。', '',
      '共144份文档、288组成对提示、576次生成；8K/32K/128K/256K各36份。每档覆盖10%/50%/90%三个位置、每位置两次重复、两类答案（随机数字串和合成中文结构化条目）以及16/64/128三档目标token长度。36个答案/位置种子在四种文档长度下复用。256K为262,144个文档tokens，超过训练的192,512上下文，单列为外推。', '',
      r'问答采用 `User: …\n\nAssistant:` 格式，问题含32位档案编号和项目/地区/批次条件；前缀补全在Assistant后另给出源档案字段前缀，属于较容易的复制诊断。两者均为greedy（temperature=0，top_p=1，top_k=0，无重复惩罚），输出上限分别4096和1024 tokens，不强制关闭thinking。', '',
      '**主指标（完整答案检索率）**：去除已闭合thinking前缀后，最终输出中连续出现完整目标字符串；允许前后额外文字。不闭合thinking视为没有最终答案。内部字符、数字、顺序、标点必须一致。另报整段输出严格EM、含thinking的全文命中、正确前缀覆盖率，后两者不得冒充最终完整答案准确率。源记录末尾句号为分隔符，不属于目标字段。', '',
      '## 按输入长度的完整答案检索率', '',
      '|文档长度|任务|实际输入tokens|Base|ROSA|提升（百分点）|配对差值95%区间|',
      '|---|---|---|---|---|---|---|']
    for g in groups:
        if g['group_fields'] != ['length','mode']:continue
        n=g['paired_cases'];ci=g['final_contains_full_answer_gain_cluster_bootstrap_95_pp']
        text.append(f"|{g['length']//1024}K{'（外推）' if g['length']==262144 else ''}|{'自然问答' if g['mode']=='qa' else '前缀补全'}|{g['prompt_tokens_min']}–{g['prompt_tokens_max']}|{g['base']['final_contains_full_answer_correct']}/{n} ({100*g['base']['final_contains_full_answer_rate']:.2f}%)|{g['trained']['final_contains_full_answer_correct']}/{n} ({100*g['trained']['final_contains_full_answer_rate']:.2f}%)|{g['final_contains_full_answer_gain_pp']:+.2f}|[{ci[0]:.2f}, {ci[1]:.2f}]|")
    text += ['', '区间按答案/位置种子进行10,000次配对bootstrap；每长度36个独立合成种子。全零观测可能导致bootstrap退化为[0,0]，不能据此认定总体真实准确率恰为零。例如0/36的Wilson 95%区间约为0%–9.64%。各答案类型/长度子组仅6例，结论仅限探索性比较。', '',
      '## 按答案类型及长度拆分', '',
      '|文档|任务|答案类型|标称/实际答案tokens|答案字符数|Base完整命中|ROSA完整命中|ROSA平均正确前缀覆盖率|',
      '|---|---|---|---|---|---|---|---|']
    for g in groups:
        if g['group_fields'] != ['length','mode','answer_kind','nominal_answer_tokens']:continue
        text.append(f"|{g['length']//1024}K|{'问答' if g['mode']=='qa' else '前缀'}|{'数字' if g['answer_kind']=='code' else '中文条目'}|{g['nominal_answer_tokens']} / {g['actual_answer_tokens_min']}–{g['actual_answer_tokens_max']}|{g['answer_chars_min']}–{g['answer_chars_max']}|{g['base']['final_contains_full_answer_correct']}/6|{g['trained']['final_contains_full_answer_correct']}/6|{100*g['trained']['mean_correct_prefix_fraction']:.2f}%|")
    text += ['', '## 输出预算与ROSA是否实际生效', '',
      'ROSA使用训练保存的logit_gap动态增益，min_match=6、dynamic_max=0.3、margin=0.5。min_match约束历史匹配前缀长度，不约束答案长度。逐token跟踪仅记录自由生成预测，不把teacher forcing计入生效次数。', '',
      '|文档|任务|Base/ROSA触及输出上限|Base/ROSA未闭合thinking|Base/ROSA整段EM|ROSA生效样本|非零残差预测步/总预测步|',
      '|---|---|---|---|---|---|---|']
    for g in groups:
        if g['group_fields'] != ['length','mode']:continue
        b,t=g['base'],g['trained'];r=g['rosa'];n=g['paired_cases']
        text.append(f"|{g['length']//1024}K|{'问答' if g['mode']=='qa' else '前缀'}|{b['truncated']}/{t['truncated']}|{b['unclosed_thinking']}/{t['unclosed_thinking']}|{b['whole_final_exact_match_correct']}/{t['whole_final_exact_match_correct']}|{r['cases_with_nonzero_residual']}/{n}|{r['nonzero_steps']}/{r['prediction_steps']}|")
    text += ['', '完整逐位置数据见comparison.csv；逐题目标、输出token、模型文本、长度、NLL诊断与激活统计见answers.csv/answers.jsonl；原始逐token增益见各实验的traces目录。所有低于6-token匹配的预测步均核实为零残差。完整输入已按种子重建校验SHA256，无截断。', '',
      '## 结论与适用范围', '',
      '请严格区分自然问答的端到端定位能力与给定精确前缀后的复制能力。ROSA在部分生成步生效或答案NLL降低，并不等价于成功输出完整答案。长答案结果不得沿用此前3-token短口令的准确率；合成中文结构化条目也不代表通用长文语义问答能力。报告实测提升，不事后选择达标阈值或最好子组。', '',
      f"Base SHA256: `{proofs[0]['base_sha256']}`", '',
      f"ROSA SHA256: `{proofs[0]['adapter_sha256']}`", '',
      '指标表述模板：在冻结同一RWKV-7 2.9B本体的条件下，使用固定step1800 ROSA，在8K/32K/128K及256K外推、10%/50%/90%多位置、相似档案干扰的独立合成测试集上，分别衡量自然问答与精确前缀补全的最终完整答案检索率；以同一输入、相同greedy解码和输出预算的无ROSA模型作为成对基线，报告准确率、百分点差及配对95%区间。', '']
    for mode in ('qa','exact_prefix'):
        total=main_metrics[mode]
        extra=next(g for g in groups if g['group_fields']==['length','mode'] and g['length']==262144 and g['mode']==mode)
        text.append(f"实测{'自然问答' if mode=='qa' else '精确前缀补全'}：8K–128K整体Base {total['base']['final_contains_full_answer_correct']}/108，ROSA {total['trained']['final_contains_full_answer_correct']}/108（{total['final_contains_full_answer_gain_pp']:+.2f}个百分点）；256K外推Base {extra['base']['final_contains_full_answer_correct']}/36，ROSA {extra['trained']['final_contains_full_answer_correct']}/36（{extra['final_contains_full_answer_gain_pp']:+.2f}个百分点）。")
        ci=total['final_contains_full_answer_gain_cluster_bootstrap_95_pp']
        text.append(f"8K–128K整体提升的配对95%区间为[{ci[0]:.2f}, {ci[1]:.2f}]个百分点。")
        text.append('')
    text += ['输出上限对照（仅旧实验已完成的57份文档，非完整均衡测试）：', '',
             '|任务|模型|旧上限完整命中|新上限完整命中|原生成前缀复现|',
             '|---|---|---|---|---|']
    for g in budget_comparison:
        text.append(f"|{g['mode']}|{g['arm']}|{g['old_full_hits']}/{g['count']}|{g['new_full_hits']}/{g['count']}|{g['identical_generated_prefixes']}/{g['count']}|")
    text += ['', '旧上限为问答1024/前缀256，新上限为4096/1024。部分完整命中同时伴随后续重复或额外档案文字，不能将“包含完整目标”解释为严格遵循输出格式；整段EM已单列。', '']
    (out/'RESULTS_zh.md').write_text('\n'.join(text))
    source=out/'evaluation_source';source.mkdir(exist_ok=True)
    paths=['rwkv_rosa/native/long_answer_cases.py','rwkv_rosa/native/long_gpu_needle.py',
           'rwkv_rosa/native/fast_gpu_needle.py','rwkv_rosa/native/gpu_state.py',
           'rwkv_rosa/native/gpu_needle.py','rwkv_rosa/native/needle_eval.py',
           'scripts/summarize_long_needle.py','scripts/finalize_long_needle.py',
           'scripts/plot_long_needle.py','scripts/publish_long_needle.py',
           'scripts/run_detailed_gpu_needle.sh','tests/test_long_answer_cases.py',
           'rwkv_rosa/native/adapter.py','rwkv_rosa/native/position.py','rwkv_rosa/native/dynamic_scale.py',
           'rwkv_rosa/native/cpu_compare.py','rwkv_rosa/backbone.py','rwkv_rosa/model.py',
           'rosa_runtime.py','rosa_addressing.py','rosa_value_store.py','rosa_cpp_extension.py',
           'rwkv_rosa/native/cuda/stateful_wkv.cu','cpp_extensions/rosa_sam_cpu_extension.cpp',
           'third_party/rwkv_lm_v7/tokenizer.py','rwkv_vocab_v20230424.txt']
    for name in paths:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    for directory in ('third_party/rwkv_lm_v7_native/src','third_party/rwkv_lm_v7_native/cuda'):
        for original in (ROOT/directory).rglob('*'):
            if original.is_file() and original.suffix in ('.py','.cu','.cpp','.cuh','.h'):
                dest=source/original.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
    meta=Path(json.loads((Path(args.main)/'manifest.json').read_text())['adapter_snapshot']).with_suffix('.json')
    shutil.copy2(meta,out/'adapter_metadata.json')
    import torch
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__})
    (out/'git_revision.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
    (out/'source_notes.txt').write_text('Source checkout contains uncommitted work; evaluation_source is the reproducible snapshot. Main-suite workers used a fixed192512 capacity; queued256K workers use length+4096+2048. The capacity affects validation only; full input tensors enter the recurrent forward without cropping.\n')
    if Path('/tmp/rosa-long-answer-tests.log').exists():shutil.copy2('/tmp/rosa-long-answer-tests.log',out/'long_answer_tests.log')
    write(out/'verification.json',{'status':'complete','contexts':144,'paired_prompts':288,
          'generated_answers':576,'same_targets_across_4_lengths':True,'proofs':proofs})
    print(json.dumps({'status':'complete','output':str(out),'answers':len(rows)},ensure_ascii=False))


if __name__ == '__main__':
    main()
