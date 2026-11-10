#!/usr/bin/env python3
"""Paired, fixed-checkpoint independent CPU NIAH; never invokes training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from rwkv_rosa.backbone import build_backbone, infer_config, load_base_state
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.cpu_compare import CPUInference, state_hash, write_json
from rwkv_rosa.native.cpu_wkv_inference import install_cpu_wkv
from rwkv_rosa.native.needle_eval import build_case, evaluate, address_diagnostic, clean_decode, answer_scores
from rwkv_rosa.native.position import normalize_metadata, scale_options
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT = Path(__file__).resolve().parents[1]


def flatten_state(state):
    for value in state:
        if isinstance(value, dict):
            yield from flatten_state(value.values())
        else:
            yield value


@torch.inference_mode()
def worker(folder, slot, workers, threads):
    if torch.cuda.is_available():
        raise RuntimeError('Independent evaluation must be CPU only')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    manifest = json.loads((folder / 'manifest.json').read_text())
    ck = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    meta = normalize_metadata(ck['native_rosa'])
    if meta['position'] != 'post_norm':
        raise ValueError('Shared base prefix is only valid for post_norm adapters')
    base = load_base_state(manifest['base'])
    if state_hash(base, ck['native_rosa_frozen_base']['parameter_names']) != ck['native_rosa_frozen_base']['sha256']:
        raise ValueError('Frozen backbone hash differs from training')
    body = build_backbone(infer_config(base), dtype=torch.float32, state_dict=base)
    del base
    model = CPUInference(body)
    # Check the real checkpoint, not only random toy recurrences, before using
    # the opt-in compiled CPU implementation. Both arms use this same runtime.
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8] * 8])
    expected, ref_state = model.features(ids, return_state=True)
    install_cpu_wkv(body)
    actual, compiled_state = model.features(ids, return_state=True)
    error = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    state_errors = [float((x-y).abs().max()) for x, y in zip(flatten_state(compiled_state), flatten_state(ref_state))]
    _, half = model.features(ids[:, :29], return_state=True)
    continued, _ = model.features(ids[:, 29:], state=half, return_state=True)
    torch.testing.assert_close(continued, actual[:, 29:], atol=3e-4, rtol=3e-4)
    write_json(folder / f'runtime_audit_{slot}.json', {
        'device': 'cpu', 'dtype': 'float32', 'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'checkpoint_step': ck['global_step'], 'same_frozen_base_hash': True,
        'real_64_token_reference_max_hidden_error': error,
        'real_64_token_reference_max_state_error': max(state_errors),
        'chunk_continuation_verified': True, 'shared_prefix_valid_for_post_norm': True,
        'threads': threads, 'affinity': sorted(os.sched_getaffinity(0)),
    })
    adapter = CPUImprovedROSA(body.emb, position='post_norm', output_norm=meta['output_norm'],
        address_impl='compiled_cpu', device='cpu', scale=meta['scale'], **scale_options(meta))
    adapter.load_state_dict({k[len('rosa_cpu.'):]: v for k, v in ck['state_dict'].items()})
    adapter.eval().requires_grad_(False)
    adapter.bind_output_head(body.head)
    del ck
    tokenizer = TRIE_TOKENIZER(str(ROOT / 'rwkv_vocab_v20230424.txt'))
    for index in range(slot, len(manifest['cases']), workers):
        spec = manifest['cases'][index]
        path = folder / 'cases' / f'{index:04d}.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
            continue
        started = time.monotonic()
        write_json(path, {'status': 'running', 'spec': spec, 'pid': os.getpid()})
        case = build_case(tokenizer, spec['length'], spec['depth'], spec['seed'],
                          difficulty=spec['difficulty'], prompt_template='rwkv')
        model.rosa = None
        state = None
        prefill_start = time.monotonic()
        # Bound RAM while retaining every token and the exact recurrent state.
        for start in range(0, len(case['stem_ids']), manifest['prefill_chunk']):
            _, state = model.features(torch.tensor([case['stem_ids'][start:start+manifest['prefill_chunk']]]),
                                      state=state, return_state=True)
        prefill_seconds = time.monotonic() - prefill_start
        rows = []
        for arm in ('base', 'trained'):
            model.rosa = None if arm == 'base' else adapter
            for mode in ('qa', 'exact_prefix'):
                t = time.monotonic()
                row = evaluate(model, tokenizer, case, mode, state, manifest['max_new_tokens'])
                row.update(arm=arm, mode=mode, length=spec['length'], depth=spec['depth'],
                           difficulty=spec['difficulty'], seed=spec['seed'], case_index=index,
                           key=case['key'], secret=case['secret'], seconds=time.monotonic()-t)
                for budget in (24, 64, 128):
                    text = clean_decode(tokenizer, row['generated_ids'][:budget])
                    row[f'passkey_match_at_{budget}'] = answer_scores(text, case['secret'])['passkey_match']
                rows.append(row)
                with (folder / f'answers_{slot}.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print('ANSWER', json.dumps({k: row[k] for k in ('case_index', 'arm', 'mode', 'length', 'depth', 'passkey_match', 'answer_nll', 'seconds')}), flush=True)
        diagnostics = {mode: address_diagnostic(case, mode, adapter.address_engine) for mode in ('qa', 'exact_prefix')}
        write_json(path, {'status': 'complete', 'spec': spec, 'case_id': case['id'],
            'stem_sha256': case['stem_sha256'], 'needle_start': case['needle_start'],
            'needle_tokens': case['needle_tokens'], 'secret': case['secret'], 'key': case['key'],
            'prompt_sha256': {mode: hashlib.sha256(torch.tensor(case['stem_ids']+suffix, dtype=torch.int64).numpy().tobytes()).hexdigest()
                              for mode, suffix in case['suffix_ids'].items()},
            'prefill_seconds': prefill_seconds, 'wall_seconds': time.monotonic()-started,
            'results': rows, 'address_diagnostics': diagnostics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    folder = Path(args.output)
    try:
        worker(folder, args.slot, args.workers, args.threads)
    except Exception:
        write_json(folder / f'failure_{args.slot}.json', {'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
