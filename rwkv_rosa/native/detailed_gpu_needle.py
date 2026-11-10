"""Fixed-checkpoint paired GPU NIAH over a predeclared independent matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from .cpu_compare import write_json
from .fast_gpu_needle import FastEvaluator
from .gpu_needle import NativeGPU
from .needle_eval import build_case, clean_decode, answer_scores
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def worker(folder, slot, workers):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((folder/'manifest.json').read_text())
    model = NativeGPU(manifest['train_dir'], manifest['adapter_snapshot'],
                      max(x['length'] for x in manifest['cases']) + 1024)
    if model.meta['step'] != manifest['checkpoint_step']:
        raise ValueError('Unexpected checkpoint step')
    model.tokenizer = TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
    evaluator = FastEvaluator(model)
    audit = [evaluator.audit(model.tokenizer.encode('User: 你好。\n\nAssistant:'))]
    if slot == 0:
        case = build_case(model.tokenizer, 8192, .5, 2026111099, 'similar_archives', prompt_template='rwkv')
        audit.append(evaluator.audit(case['stem_ids']+case['suffix_ids']['qa']))
    write_json(folder/f'runtime_audit_{slot}.json', {
        'device': torch.cuda.get_device_name(), 'backbone_dtype': 'bfloat16',
        'head_and_adapter_dtype': 'float32', 'recurrence_state_dtype': 'float32',
        'base_sha256_native_order': model.meta['base_sha256_native_order'],
        'adapter_state_sha256': model.meta['adapter_state_sha256'],
        'checkpoint_step': model.meta['step'], 'native_stateful_audits': audit,
        'note': 'Same native prefill and stateful decoding in both arms. Split BF16 GEMM shapes can differ from full-history replay; audit records differences. No CPU-FP32 results mixed into this experiment.',
    })
    for index in range(slot, len(manifest['cases']), workers):
        spec = manifest['cases'][index]
        path = folder/'cases'/f'{index:04d}.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
            continue
        started = time.monotonic()
        write_json(path, {'status': 'running', 'spec': spec, 'pid': __import__('os').getpid()})
        case = build_case(model.tokenizer, spec['length'], spec['depth'], spec['seed'],
                          spec['difficulty'], prompt_template='rwkv')
        rows = []; diagnostics = {}
        for mode in ('qa', 'exact_prefix'):
            answers, seconds = evaluator.evaluate(case, mode, ('base', 'trained'), manifest['max_new_tokens'])
            for row in answers:
                row.update(case_index=index, key=case['key'], seconds=seconds)
                text = row['generated_text']
                unclosed = '<think>' in text and '</think>' not in text
                final_text = '' if unclosed else text.rsplit('</think>', 1)[-1]
                row.update(unclosed_thinking=unclosed, final_answer_text=final_text,
                    final_answer_passkey_match=answer_scores(final_text, case['secret'])['passkey_match'])
                for budget in (24, 64, 128):
                    text = clean_decode(model.tokenizer, row['generated_ids'][:budget])
                    row[f'passkey_match_at_{budget}'] = answer_scores(text, case['secret'])['passkey_match']
                rows.append(row)
                with (folder/f'answers_{slot}.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            example = answers[0]
            diagnostics[mode] = {'suffix_only_exact_match': example['suffix_only_exact_match'],
                'teacher_forced_address_token_hits': example['address_token_hits'],
                'answer_tokens': len(example['answer_ids']), 'candidate_ids': example['address_candidates'],
                'target_ids': example['answer_ids'], 'prompt_last_match_len': example['query_last_match_len']}
            print('PAIRED', json.dumps({'index':index, 'length':spec['length'], 'depth':spec['depth'], 'mode':mode,
                'seconds':seconds, 'answers':[{k:row[k] for k in ('arm','passkey_match','strict_digits_only','answer_nll','hit_generation_limit')} for row in answers]}), flush=True)
        write_json(path, {'status':'complete', 'spec':spec, 'case_id':case['id'],
            'stem_sha256':case['stem_sha256'], 'secret':case['secret'], 'key':case['key'],
            'needle_start':case['needle_start'], 'needle_tokens':case['needle_tokens'],
            'prompt_sha256':{mode:hashlib.sha256(torch.tensor(case['stem_ids']+suffix,dtype=torch.int64).numpy().tobytes()).hexdigest()
                             for mode,suffix in case['suffix_ids'].items()},
            'wall_seconds':time.monotonic()-started, 'results':rows, 'address_diagnostics':diagnostics})
    write_json(folder/f'worker_complete_{slot}.json', {'status':'complete',
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--workers',type=int,required=True);args=p.parse_args()
    folder=Path(args.output)
    try:
        worker(folder,args.slot,args.workers)
    except Exception:
        write_json(folder/f'failure_{args.slot}.json',{'error':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
