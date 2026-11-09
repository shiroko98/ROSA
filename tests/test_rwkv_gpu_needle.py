"""CUDA evaluator orchestration is testable without a GPU/native extension."""
import json
import argparse
from pathlib import Path
import tempfile
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import torch

from rosa_runtime import RosaAddressBatch
from rwkv_rosa.native.gpu_needle import NativeGPU, grouped_summary, slice_addresses, tail_pad, parse_arms
from rwkv_rosa.native.needle_eval import build_case, address_diagnostic, make_engine, ROOT
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER


def addresses(n):
    return RosaAddressBatch(addr_ids=torch.arange(n).reshape(1,n),
        raw_match_lens=torch.ones(1,n,dtype=torch.long),
        fired_match_lens=torch.ones(1,n,dtype=torch.long),
        valid_mask=torch.ones(1,n,dtype=torch.bool),
        special_mask=torch.zeros(1,n,dtype=torch.bool),source='test')


class FakeAdapter:
    def begin(self, ids): self.addresses = addresses(ids.shape[1])
    def inject_post_norm(self, hidden): return hidden


class FakeNative(NativeGPU):
    def __init__(self, diverge=False):
        self.seen = []; self.diverge = diverge; self.adapter = FakeAdapter()
        self.body = SimpleNamespace(head=self.head)
        self.tokenizer = SimpleNamespace(idx2token={1:b'123',2:b'456',3:b'999',4:b'000'})

    @staticmethod
    def scores(token):
        out = torch.full((1,1,8), -20.); out[0,0,token] = 20.
        return out

    def features(self, history, start=None, end=None):
        self.seen.append(list(history))
        return torch.tensor([[[float(history[-1])]]])

    def head(self, hidden):
        token = {7:1,1:2,3:4,2:0,4:0}[int(hidden[0,0,0])]
        return self.scores(token)

    def paired_logits(self, history, hidden, start, end):
        base = self.head(hidden)[0]
        trained = self.scores(3)[0] if self.diverge else base
        return base, trained, None, hidden*0


class GPUHandoffTest(unittest.TestCase):
    def test_padding_is_only_at_tail_and_preserves_original_ids(self):
        for length in (1,15,16,17,192512):
            original = torch.arange(length).reshape(1,-1)
            padded = tail_pad(original)
            self.assertEqual(padded.shape[1] % 16, 0)
            self.assertTrue(torch.equal(padded[:,:length], original))
            self.assertTrue(bool((padded[:,length:]==65532).all()))
        with self.assertRaises(ValueError): tail_pad(torch.empty(1,0))

    def test_address_slice_keeps_fields_and_source(self):
        original = addresses(12); sliced = slice_addresses(original,8,11)
        self.assertEqual(sliced.addr_ids.tolist(), [[8,9,10]])
        self.assertEqual(sliced.valid_mask.shape, (1,3))
        self.assertEqual(sliced.source, original.source)

    @patch('torch.cuda.synchronize')
    def test_free_generation_uses_only_prompt_then_own_predictions(self, _):
        model = FakeNative()
        case = {'stem_ids':[6], 'suffix_ids':{'qa':[7]}, 'secret':'123456', 'answer_ids':[3,4]}
        answers = model.generate_pair(case,'qa',8)
        self.assertEqual(model.seen, [[6,7],[6,7,1],[6,7,1,2]])
        for row in answers.values():
            self.assertEqual(row['generated_ids'], [1,2,0])
            self.assertTrue(row['passkey_match']); self.assertTrue(row['strict_digits_only'])
            self.assertFalse(row['hit_generation_limit'])
            self.assertEqual(row['full_context_forward_calls_paired'],3)
        # In particular, answer_ids are NOT used to seed generation.

    @patch('torch.cuda.synchronize')
    def test_shared_prefix_then_diverging_arms_have_separate_histories(self, _):
        model = FakeNative(diverge=True)
        case = {'stem_ids':[6], 'suffix_ids':{'qa':[7]}, 'secret':'123456'}
        answers = model.generate_pair(case,'qa',8)
        self.assertTrue(answers['base']['passkey_match'])
        self.assertFalse(answers['trained']['passkey_match'])
        self.assertEqual(answers['trained']['generated_ids'],[3,4,0])
        self.assertEqual(model.seen,[[6,7],[6,7,1],[6,7,3],[6,7,1,2],[6,7,3,4]])

    @patch('torch.cuda.synchronize')
    def test_generation_limit(self, _):
        model = FakeNative()
        case = {'stem_ids':[6], 'suffix_ids':{'qa':[7]}, 'secret':'123456'}
        answers = model.generate_pair(case,'qa',1)
        self.assertEqual(model.seen,[[6,7]])
        self.assertTrue(answers['base']['hit_generation_limit'])
        self.assertFalse(answers['base']['passkey_match'])

    @patch('torch.cuda.synchronize')
    def test_trained_only_never_generates_or_scores_base(self, _):
        model = FakeNative(diverge=True)
        case = {'stem_ids':[6], 'suffix_ids':{'qa':[7]}, 'secret':'999000'}
        answers = model.generate_pair(case,'qa',8,arms=('trained',))
        self.assertEqual(set(answers),{'trained'})
        self.assertEqual(answers['trained']['generated_ids'],[3,4,0])
        self.assertEqual(model.seen,[[6,7],[6,7,3],[6,7,3,4]])
        self.assertTrue(answers['trained']['passkey_match'])
        self.assertFalse(answers['trained']['hit_generation_limit'])

    @patch('torch.cuda.synchronize')
    def test_base_only_never_generates_or_scores_trained(self, _):
        model = FakeNative(diverge=True)
        case = {'stem_ids':[6], 'suffix_ids':{'qa':[7]}, 'secret':'123456'}
        answers = model.generate_pair(case,'qa',8,arms=('base',))
        self.assertEqual(set(answers),{'base'})
        self.assertEqual(model.seen,[[6,7],[6,7,1],[6,7,1,2]])
        self.assertTrue(answers['base']['passkey_match'])

    def test_arm_parser_validates_and_canonicalizes_order(self):
        self.assertEqual(parse_arms('trained'),('trained',))
        self.assertEqual(parse_arms('trained,base'),('base','trained'))
        for invalid in ('','other','base,base','base,'):
            with self.assertRaises(argparse.ArgumentTypeError):parse_arms(invalid)

    def test_summary_does_not_conflate_retrieval_with_generation(self):
        rows=[{'arm':'trained','mode':'qa','length':8192,'passkey_match':False,
               'strict_digits_only':False,'suffix_only_exact_match':True}]
        row=grouped_summary(rows,('arm','mode','length'))[0]
        self.assertEqual(row['passkey_match_correct'],0)
        self.assertEqual(row['suffix_only_exact_match_rate'],1.)
        self.assertEqual(grouped_summary([],('arm',)),[])

    def test_rwkv_template_is_explicit_has_no_chatml_and_preserves_retrieval(self):
        tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
        case=build_case(tokenizer,1024,.5,20261109,'similar_archives',prompt_template='rwkv')
        self.assertEqual(case['prompt_template'],'rwkv')
        self.assertEqual(len(case['stem_ids']),1024)
        self.assertNotIn(65530,case['stem_ids'])
        self.assertNotIn(65531,case['suffix_ids']['qa'])
        text=tokenizer.decode(case['stem_ids']+case['suffix_ids']['qa'])
        self.assertTrue(text.startswith('User:'))
        self.assertTrue(text.endswith('\n\nAssistant:'))
        self.assertTrue(address_diagnostic(case,'exact_prefix',make_engine())['suffix_only_exact_match'])
        self.assertNotIn(case['secret'],tokenizer.decode(case['suffix_ids']['qa']))
        with self.assertRaises(ValueError):build_case(tokenizer,1024,.5,1,prompt_template='other')
        legacy=build_case(tokenizer,1024,.5,20261109)
        self.assertEqual(legacy['prompt_template'],'chatml')
        self.assertIn(65530,legacy['stem_ids'])

    def test_trained_only_cli_filters_probe_pilot_and_result_rows(self):
        from rwkv_rosa.native import gpu_needle as module
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            train=Path(folder)/'train';train.mkdir();(train/'wandb_run_id.txt').write_text('source')
            output=Path(folder)/'output';model=MagicMock()
            model.meta={'step':200,'base_sha256_native_order':'base','adapter_state_sha256':'rosa','ctx_len':192512}
            def probe(case, mode):
                common={'length':case['length'],'difficulty':case['difficulty'],'mode':mode,
                        'suffix_only_exact_match':True,'first_answer_token_rank':1,'answer_nll':.1}
                return [{**common,'arm':a} for a in ('base','trained')],{},.01
            model.probe.side_effect=probe
            model.generate_pair.side_effect=lambda *a,**kw:{arm:{'passkey_match':True,
                'strict_digits_only':True,'generated_ids':[1],'generated_text':'123456'} for arm in kw['arms']}
            stack.enter_context(patch.object(module,'NativeGPU',return_value=model))
            for fn in ('set_per_process_memory_fraction','get_device_name','max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda,fn,return_value=0))
            module.main(['--train-dir',str(train),'--checkpoint','/snapshot/step200.pth',
                         '--output',str(output),'--lengths','256','--depths','0.5',
                         '--prompt-template','chatml','--arms','trained','--max-new-tokens','2'])
            for name in ('probes.jsonl','pilot_results.jsonl','results.jsonl'):
                rows=[json.loads(line) for line in (output/name).read_text().splitlines()]
                self.assertEqual(len(rows),4)
                self.assertEqual({r['arm'] for r in rows},{'trained'})
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['request']['expected_generated_answers'],4)
            self.assertEqual(report['status']['status'],'complete')
            self.assertEqual(report['generation_overall'][0]['arm'],'trained')
            for call in model.generate_pair.call_args_list:
                self.assertEqual(call.kwargs['arms'],('trained',))

    def test_wrong_position_rejected_before_any_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'adapter.pth'
            checkpoint.with_suffix('.json').write_text(json.dumps({
                'format':'native_rosa_weights_only_v1','position':'block','layers':[31]}))
            with self.assertRaisesRegex(ValueError,'post_norm'):
                NativeGPU(folder,checkpoint,1024)


if __name__=='__main__':unittest.main()
