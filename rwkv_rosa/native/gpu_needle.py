"""Independent, inference-only NIAH using the exact native CUDA BF16 backbone.

No trainer/optimizer or reference GPU recurrence. Post-norm lets paired arms
share frozen features until their generated histories diverge. Native kernels
have no streaming API: generation deliberately replays the full history.
Teacher-forced probes and free-running answers are saved/scored separately.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from rosa_runtime import RosaAddressBatch
from rwkv_rosa.backbone import load_base_state
from rwkv_rosa.native.adapter import CPUImprovedROSA, configure_output_head
from rwkv_rosa.native.position import scale_options
from rwkv_rosa.native.cpu_compare import state_hash, write_json
from rwkv_rosa.native.needle_eval import answer_scores, build_case, clean_decode
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT = Path(__file__).resolve().parents[2]
ADDRESS_FIELDS = ('addr_ids', 'raw_match_lens', 'fired_match_lens', 'valid_mask', 'special_mask')
STOPS = (0, 65531, 65532)


def parse_arms(value):
    arms = tuple(value.split(','))
    if not arms or len(set(arms)) != len(arms) or any(a not in ('base','trained') for a in arms):
        raise argparse.ArgumentTypeError('Arms must be base, trained, or base,trained (no duplicates).')
    return tuple(a for a in ('base','trained') if a in arms)


def tail_pad(ids, multiple=16):
    if ids.ndim != 2 or not ids.shape[1]:
        raise ValueError('Expected nonempty [batch, time] IDs.')
    return F.pad(ids, (0, (-ids.shape[1]) % multiple), value=65532)


def slice_addresses(addresses, start, end):
    return RosaAddressBatch(**{k: getattr(addresses, k)[:, start:end] for k in ADDRESS_FIELDS},
                            source=addresses.source)


def rank(scores, token):
    return 1 + int((scores > scores[token]).sum())


def grouped_summary(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for keys, group in sorted(groups.items()):
        entry = dict(zip(fields, keys)); entry['cases'] = len(group)
        for key in ('passkey_match', 'strict_digits_only', 'suffix_only_exact_match',
                    'first_answer_token_probability', 'first_answer_token_rank', 'answer_nll',
                    'post_norm_target_logit_delta', 'post_norm_target_rank_before', 'post_norm_target_rank_after',
                    'post_norm_residual_zero_fraction'):
            vals = [r[key] for r in group if key in r]
            if vals:
                entry[key + ('_rate' if isinstance(vals[0], bool) else '_mean')] = sum(vals) / len(vals)
        for key in ('passkey_match', 'strict_digits_only'):
            if key in group[0]: entry[key + '_correct'] = sum(r[key] for r in group)
        result.append(entry)
    return result


class NativeGPU:
    def __init__(self, train_dir, checkpoint, max_context):
        meta = json.loads(Path(checkpoint).with_suffix('.json').read_text())
        if meta.get('format') != 'native_rosa_weights_only_v1' or meta.get('position') != 'post_norm' or meta.get('layers'):
            raise ValueError('This evaluator requires an explicit post_norm adapter snapshot and metadata.')
        args = SimpleNamespace(**ast.literal_eval((Path(train_dir) / 'train_log.txt').read_text().splitlines()[1]))
        args.grad_cp = 0
        args.ctx_len = max_context
        if Path(meta['base_model']).resolve() != Path(args.load_model).resolve():
            raise ValueError('Training arguments and checkpoint name different frozen bases.')
        from src.model import RWKV
        state = load_base_state(meta['base_model'])
        if state_hash(state, meta['base_parameter_names']) != meta['base_sha256_native_order']:
            raise ValueError('Frozen external base fingerprint mismatch.')
        adapter_state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if state_hash(adapter_state) != meta['adapter_state_sha256']:
            raise ValueError('Adapter fingerprint mismatch.')
        if not all(bool(torch.isfinite(v).all()) for v in adapter_state.values()):
            raise FloatingPointError('Nonfinite adapter weights.')
        with torch.device('meta'):
            self.body = RWKV(args)
        self.body.load_state_dict(state, strict=True, assign=True)
        self.body.eval().requires_grad_(False).cuda()
        embedding = torch.nn.Embedding.from_pretrained(state['emb.weight'].float(), freeze=True)
        self.adapter = CPUImprovedROSA(embedding, position='post_norm', scale=meta['scale'],
                                       device='cuda', address_impl='compiled_cpu', output_norm=meta.get('output_norm', 'none'), **scale_options(meta))
        self.adapter.load_state_dict(adapter_state, strict=True)
        self.adapter.bind_output_head(self.body.head)
        configure_output_head(self.body.head, self.adapter.output_norm)
        self.adapter.eval().requires_grad_(False)
        self.meta = meta
        if any(p.dtype != torch.bfloat16 or p.device.type != 'cuda' for p in self.body.parameters()):
            raise ValueError('Native body must be BF16 CUDA.')
        if any(p.dtype != torch.float32 or p.device.type != 'cuda' for p in self.adapter.parameters()):
            raise ValueError('Native adapter must be FP32 CUDA.')
        if any(p.requires_grad for p in (*self.body.parameters(), *self.adapter.parameters())):
            raise ValueError('Evaluation cannot allocate gradients.')

    @torch.inference_mode()
    def features(self, history, start=None, end=None):
        ids = torch.tensor([history], device='cuda', dtype=torch.long)
        features = self.body._forward_features(tail_pad(ids))
        # Clone so a handful of selected vectors do not retain the full context.
        if start is None: start = len(history) - 1
        if end is None: end = len(history)
        return features[:, start:end].clone()

    @torch.inference_mode()
    def paired_logits(self, history, hidden, start, end):
        self.adapter.begin(torch.tensor([history], dtype=torch.long))
        full_addresses = self.adapter.addresses
        self.adapter.addresses = slice_addresses(full_addresses, start, end)
        after = self.adapter.inject_post_norm(hidden)
        base = self.body.head(hidden).float()[0]
        trained = self.body.head(after).float()[0]
        return base, trained, full_addresses, after - hidden

    @torch.inference_mode()
    def probe(self, case, mode):
        prompt = case['stem_ids'] + case['suffix_ids'][mode]
        target = case['answer_ids']
        history = prompt + target[:-1]
        start, end = len(prompt) - 1, len(prompt) - 1 + len(target)
        torch.cuda.synchronize(); began = time.monotonic()
        hidden = self.features(history, start, end)
        base, trained, addresses, delta = self.paired_logits(history, hidden, start, end)
        candidates = addresses.addr_ids[0, start:end].tolist()
        valid = addresses.valid_mask[0, start:end].tolist()
        hits = [bool(v and a == t) for a, t, v in zip(candidates, target, valid)]
        candidate = candidates[0]; is_valid = valid[0]
        first = target[0]; common = {
            'case': case['id'], 'length': case['length'], 'depth': case['depth'],
            'seed': case['seed'], 'difficulty': case['difficulty'], 'mode': mode,
            'prompt_template': case['prompt_template'],
            'secret': case['secret'], 'prompt_tokens': len(prompt), 'answer_ids': target,
            'stem_sha256': case['stem_sha256'], 'needle_depth_actual': case['needle_start'] / len(prompt),
            'suffix_only_exact_match': all(hits), 'address_token_hits': sum(hits),
            'address_candidates': candidates, 'query_last_candidate_valid': bool(is_valid),
            'query_last_candidate_is_target': bool(is_valid and candidate == first),
            'query_last_match_len': int(addresses.fired_match_lens[0, start]),
        }
        rows = []
        for arm, logits in (('base', base), ('trained', trained)):
            nll = float(F.cross_entropy(logits, torch.tensor(target, device='cuda')))
            if not math.isfinite(nll) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite native logits/NLL.')
            row = {**common, 'arm': arm, 'answer_nll': nll,
                   'first_answer_token_rank': rank(logits[0], first),
                   'first_answer_token_probability': float(logits[0].softmax(-1)[first]),
                   'teacher_forced_argmax_ids': logits.argmax(-1).tolist()}
            if arm == 'trained':
                row.update(post_norm_target_logit_before=float(base[0, first]),
                           post_norm_target_logit_after=float(trained[0, first]),
                           post_norm_target_logit_delta=float(trained[0, first] - base[0, first]),
                           post_norm_target_rank_before=rank(base[0], first),
                           post_norm_target_rank_after=rank(trained[0], first),
                           post_norm_candidate_logit_delta=float(trained[0, candidate] - base[0, candidate]),
                           post_norm_candidate_rank_before=rank(base[0], candidate),
                           post_norm_candidate_rank_after=rank(trained[0], candidate),
                           post_norm_residual_zero_fraction=float((delta == 0).float().mean()),
                           post_norm_hidden_residual_norm=float(delta.float().norm()))
            rows.append(row)
        torch.cuda.synchronize()
        return rows, {'base': base[0].cpu(), 'trained': trained[0].cpu()}, time.monotonic() - began

    @torch.inference_mode()
    def generate_pair(self, case, mode, max_new_tokens, progress=None, arms=('base','trained')):
        arms = parse_arms(','.join(arms))
        prompt = case['stem_ids'] + case['suffix_ids'][mode]
        start = time.monotonic()
        hidden = self.features(prompt)
        before, after, _, _ = self.paired_logits(prompt, hidden, len(prompt)-1, len(prompt))
        generated = {arm: [] for arm in arms}; logits = {'base': before[0], 'trained': after[0]}
        done = set(); calls = 1
        for step in range(max_new_tokens):
            for arm in generated:
                if arm in done: continue
                token = int(logits[arm].argmax(-1)); generated[arm].append(token)
                if token in STOPS: done.add(arm)
            if len(done) == len(arms) or step + 1 == max_new_tokens: break
            active = [arm for arm in generated if arm not in done]
            unique_histories = {}
            for arm in active:
                unique_histories.setdefault(tuple(generated[arm]), []).append(arm)
            for continuation, matching_arms in unique_histories.items():
                history = prompt + list(continuation)
                hidden = self.features(history); calls += 1
                if 'base' in matching_arms: logits['base'] = self.body.head(hidden)[0, 0].float()
                if 'trained' in matching_arms:
                    # SAM is still causal, on the full generated history; no
                    # teacher-forced answer tokens are used for generation.
                    self.adapter.begin(torch.tensor([history], dtype=torch.long))
                    self.adapter.addresses = slice_addresses(self.adapter.addresses, len(history)-1, len(history))
                    logits['trained'] = self.body.head(self.adapter.inject_post_norm(hidden))[0, 0].float()
            if progress: progress(step + 1)
        torch.cuda.synchronize()
        return {arm: {**answer_scores(clean_decode(self.tokenizer, ids), case['secret']),
                      'generated_ids': ids, 'generated_text': clean_decode(self.tokenizer, ids),
                      'hit_generation_limit': len(ids) == max_new_tokens and ids[-1] not in STOPS,
                      'generation_seconds_paired': time.monotonic() - start,
                      'full_context_forward_calls_paired': calls}
                for arm, ids in generated.items()}


def append_rows(path, rows):
    with Path(path).open('a') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lengths', default='8192,32768,65536,131072,192512')
    p.add_argument('--depths', default='0.1,0.3,0.5,0.7,0.9')
    p.add_argument('--seeds', default='20261109')
    p.add_argument('--prompt-template', choices=('rwkv','chatml'), default='rwkv')
    p.add_argument('--arms', type=parse_arms, default=('base','trained'),
                   help='Generated/scored variants; base,trained by default. Use trained for ChatML-only follow-up.')
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(',')]
    depths = [float(x) for x in args.depths.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if min(lengths) < 256 or any(not 0 <= d <= 1 for d in depths) or args.max_new_tokens < 1:
        p.error('Invalid matrix dimensions/token limit.')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    # Refuse silent append/rerun mixing; a new output folder is cheap.
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'request.json').exists(): raise SystemExit('Output already exists; choose a new directory.')
        torch.set_num_threads(4); torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.94)
        request = {**vars(args), 'lengths': lengths, 'depths': depths, 'seeds': seeds,
                   'body_device': 'cuda', 'body_dtype': 'bf16', 'adapter_device': 'cuda', 'adapter_dtype': 'fp32',
                   'sam_device': 'cpu', 'sam_impl': 'compiled_cpu', 'position': 'post_norm',
                   'tf32': False, 'sampling': 'greedy', 'training': False,
                   'source_training_run_id': (Path(args.train_dir)/'wandb_run_id.txt').read_text().strip(),
                   'expected_case_mode_pairs': len(lengths)*len(depths)*len(seeds)*4,
                   'expected_generated_answers': len(lengths)*len(depths)*len(seeds)*4*len(args.arms),
                   'base_head_probe_only_when_unselected': 'Post-norm diagnostic, not base generation or a scored arm',
                   'generation_runtime': 'native full-context replay; identical histories share frozen features',
                   'length_definition': 'document tokens, excluding query and generated answer',
                   'teacher_forcing': 'diagnostics only; never scored as generated answers',
                   'started_at': datetime.now().astimezone().isoformat()}
        write_json(output/'request.json', request)
        status = {'status': 'loading', 'pid': os.getpid(), 'probed_pairs': 0, 'generated_pairs': 0}
        write_json(output/'status.json', status)
        run = None; probes = []; results = []; pilot = []; began = time.monotonic()
        def publish():
            report = {'request': request, 'status': dict(status), 'wall_seconds': time.monotonic()-began,
                      'probe_summary': grouped_summary(probes, ('arm','length','difficulty','mode')),
                      'generation_summary': grouped_summary(results, ('arm','length','difficulty','mode')),
                      'generation_overall': grouped_summary(results, ('arm',)),
                      'short_context_pilot': grouped_summary(pilot, ('arm','length','difficulty','mode')),
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                      'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30}
            write_json(output/'report.json', report); write_json(output/'status.json', status)
        try:
            model = NativeGPU(args.train_dir, args.checkpoint, max(lengths)+1024)
            model.tokenizer = TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
            request.update(source_step=model.meta['step'], base_sha256=model.meta['base_sha256_native_order'],
                           adapter_sha256=model.meta['adapter_state_sha256'], gpu=torch.cuda.get_device_name(),
                           train_context=model.meta['ctx_len'], generation_max_context=max(lengths)+1024,
                           output_norm=model.meta.get('output_norm', 'none'), **scale_options(model.meta),
                           output_head_math='float32' if model.meta.get('output_norm') in ('rms', 'rms_match') else 'bfloat16')
            write_json(output/'request.json', request)
            if args.wandb:
                from rwkv_rosa.native.wandb_auth import configure_wandb_auth
                configure_wandb_auth()
                import wandb
                try:
                    run = wandb.init(project='ROSA-RWKV7', entity=os.environ.get('WANDB_ENTITY','mobius'),
                        group='needle-of-'+request['source_training_run_id'], job_type='gpu-needle-evaluation',
                        name=f'3B-postnorm-step{model.meta["step"]}-GPU-NIAH-{args.prompt_template}-{"-".join(args.arms)}', dir=str(output), config=request,
                        settings=wandb.Settings(init_timeout=45))
                    (output/'wandb_run_id.txt').write_text(run.id+'\n')
                except Exception:
                    (output/'wandb_error.txt').write_text(traceback.format_exc())
            sanity = []
            for template, text in (
                ('plain_rwkv', 'User: 你好，请简单介绍一下自己。\n\nAssistant:'),
                ('chatml_rwkv', '<|im_start|>User: 你好，请简单介绍一下自己。<|im_end|>\n<|im_start|>Assistant: '),
                ('continuation', 'The capital of France is')):
                ids = model.tokenizer.encode(text)
                fake_case = {'stem_ids': ids, 'suffix_ids': {'sanity': []}, 'secret': '000000'}
                answers = model.generate_pair(fake_case, 'sanity', 16, arms=args.arms)
                sanity.append({'template': template, 'prompt': text, 'results': answers})
            write_json(output/'sanity.json', sanity)
            print('GPU SANITY', json.dumps(sanity,ensure_ascii=False),flush=True)
            # Small, clearly separated prompt-compatibility pilot, not part of
            # the 8K--192K matrix/statistics. Its own seed avoids duplicate cases.
            for difficulty in ('unrelated','similar_archives'):
                for mode in ('qa','exact_prefix'):
                    case = build_case(model.tokenizer, 1024, .5, seeds[0]+1, difficulty,
                                      prompt_template=args.prompt_template)
                    status.update(status='short_context_pilot', active_mode=mode, active_difficulty=difficulty)
                    publish()
                    rows, _, seconds = model.probe(case, mode)
                    answers = model.generate_pair(case, mode, args.max_new_tokens, arms=args.arms)
                    rows = [{**r, **answers[r['arm']], 'probe_seconds_paired': seconds}
                            for r in rows if r['arm'] in args.arms]
                    pilot.extend(rows); append_rows(output/'pilot_results.jsonl', rows); publish()
                    print('GPU PILOT', json.dumps(rows,ensure_ascii=False),flush=True)
            tasks = []
            # Round-robin over lengths: long-context results appear early, rather
            # than waiting for every short-context depth first. Same passkey per
            # seed/depth/difficulty across lengths enables controlled comparison.
            for seed in seeds:
                for depth_i, depth in enumerate(depths):
                    for difficulty in ('unrelated','similar_archives'):
                        for mode in ('qa','exact_prefix'):
                            for length in lengths:
                                case = build_case(model.tokenizer, length, depth, seed + depth_i*997, difficulty, prompt_template=args.prompt_template)
                                tasks.append((case, mode))
            # Broad probability/retrieval sweep first. This is not free generation.
            for i, (case, mode) in enumerate(tasks):
                status.update(status='probing', active_case=case['id'], active_mode=mode, active_difficulty=case['difficulty'])
                publish()
                rows, first, seconds = model.probe(case, mode)
                rows = [row for row in rows if row['arm'] in args.arms]
                for row in rows: row['probe_seconds_paired'] = seconds
                append_rows(output/'probes.jsonl', rows); probes.extend(rows)
                status['probed_pairs'] = i + 1; publish()
                print('GPU PROBE', json.dumps({'done':i+1,'total':len(tasks),'length':case['length'],'mode':mode,
                      'seconds':seconds,'suffix_hit':rows[0]['suffix_only_exact_match'],
                      'ranks':{r['arm']:r['first_answer_token_rank'] for r in rows}},ensure_ascii=False),flush=True)
                if run:
                    run.log({'source_training_step':model.meta['step'],'probe/completed_pairs':i+1,
                             'probe/length':case['length'],
                             **{f'probe/answer_nll_{r["arm"]}':r['answer_nll'] for r in rows}})
            for i, (case, mode) in enumerate(tasks):
                status.update(status='generating', active_case=case['id'], active_mode=mode,
                              active_difficulty=case['difficulty'], active_generated_tokens=0)
                publish()
                def heartbeat(n):
                    status['active_generated_tokens'] = n
                    write_json(output/'status.json', status)
                answers = model.generate_pair(case, mode, args.max_new_tokens, heartbeat, arms=args.arms)
                rows = [{**probes[len(args.arms)*i+j], **answers[arm]} for j,arm in enumerate(args.arms)]
                append_rows(output/'results.jsonl',rows); results.extend(rows)
                status['generated_pairs'] = i+1; publish()
                print('GPU ANSWERS', json.dumps(rows,ensure_ascii=False),flush=True)
                if run:
                    metrics={'source_training_step':model.meta['step'],'generation/completed_pairs':i+1,
                             'generation/length':case['length']}
                    for row in grouped_summary(results,('arm',)):
                        for k,v in row.items():
                            if isinstance(v,(float,int)): metrics[f'generation/{row["arm"]}/{k}']=v
                    run.log(metrics)
            status.update(status='complete', finished_at=datetime.now().astimezone().isoformat()); publish()
            if run:
                for name in ('request.json','report.json','probes.jsonl','results.jsonl','sanity.json','pilot_results.jsonl'):
                    run.save(str(output/name), base_path=str(output), policy='now')
        except BaseException:
            status.update(status='failed',error=traceback.format_exc()); publish(); raise
        finally:
            if run: run.finish(exit_code=0 if status['status']=='complete' else 1)


if __name__ == '__main__': main()
