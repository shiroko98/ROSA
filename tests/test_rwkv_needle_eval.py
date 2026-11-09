"""Passkey construction, no-leak boundaries, suffix hits, and cached CPU scoring."""
import copy
import json
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F
from rwkv_rosa.backbone import build_backbone,RWKVConfig
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.cpu_compare import CPUInference
from rwkv_rosa.native import needle_eval as n


class NeedleEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer=n.TRIE_TOKENIZER(str(n.ROOT/'rwkv_vocab_v20230424.txt'))

    def test_natural_needle_boundaries_exact_suffix_and_no_answer_leak(self):
        engine=n.make_engine()
        for length in (1024,8192):
            for depth in (.1,.5,.9):
                c=n.build_case(self.tokenizer,length,depth,20261109+int(depth*10))
                self.assertEqual(len(c['stem_ids']),length)
                self.assertEqual(c,n.build_case(self.tokenizer,length,depth,20261109+int(depth*10)))
                raw=lambda ids:b''.join(self.tokenizer.idx2token[i] for i in ids).decode('utf-8',errors='replace')
                self.assertEqual(raw(c['stem_ids']).count(c['secret']),1)
                for suffix in c['suffix_ids'].values():self.assertNotIn(c['secret'],raw(suffix))
                diag=n.address_diagnostic(c,'exact_prefix',engine)
                self.assertTrue(diag['suffix_only_exact_match'])
                self.assertEqual(diag['candidate_ids'],c['answer_ids'])
        hard=n.build_case(self.tokenizer,1024,.5,20261109,difficulty='similar_archives')
        hardtext=b''.join(self.tokenizer.idx2token[i] for i in hard['stem_ids']).decode('utf-8',errors='replace')
        self.assertGreater(len(re.findall(r'\d{6}',hardtext)),10)
        self.assertEqual(hardtext.count(hard['secret']),1)
        self.assertTrue(n.address_diagnostic(hard,'exact_prefix',engine)['suffix_only_exact_match'])
        with self.assertRaises(ValueError):n.build_case(self.tokenizer,1024,.5,1,difficulty='invalid')
        with self.assertRaises(ValueError):n.build_case(self.tokenizer,8,.5,1)
        self.assertTrue(n.answer_scores('口令是123456。','123456')['passkey_match'])
        self.assertFalse(n.answer_scores('口令是123456。','123456')['strict_digits_only'])
        self.assertTrue(n.answer_scores(' 123456\n','123456')['strict_digits_only'])
        self.assertFalse(n.answer_scores('91234567','123456')['passkey_match'])
        self.assertFalse(n.answer_scores('999999 然后123456','123456')['passkey_match'])
        self.assertEqual(n.clean_decode(self.tokenizer,[649,0,633]),self.tokenizer.idx2token[649].decode())
        self.assertIn('<unk:999999>',n.clean_decode(self.tokenizer,[999999]))

    def model(self,rosa=True):
        torch.manual_seed(42)
        base=build_backbone(RWKVConfig(vocab_size=32,n_embd=16,n_layer=2,head_size=8))
        return CPUInference(base,CPUImprovedROSA(base.emb,layers=(0,),device='cpu') if rosa else None)

    def case(self,length=1024,depth=.5,seed=1):
        return {'id':f'{length}_{depth}','length':length,'depth':depth,'seed':seed,'key':'ABCDEFGH',
                'secret':'345678','answer_ids':[3,4,5,6,7,8],'stem_ids':[1,2,1,2],
                'needle_start':1,'suffix_ids':{'qa':[4,5,2],'exact_prefix':[1,2]}}

    def test_cached_queries_teacher_forced_nll_and_no_state_mutation(self):
        case=self.case();model=self.model();tok=SimpleNamespace(idx2token={i:str(i).encode() for i in range(32)})
        with torch.inference_mode():
            _,state=model.features(torch.tensor([case['stem_ids']]),return_state=True)
            original=copy.deepcopy(state)
            for mode in ('qa','exact_prefix'):
                result=n.evaluate(model,tok,case,mode,state,3)
                torch.testing.assert_close(state,original,atol=0,rtol=0)
                prompt=case['stem_ids']+case['suffix_ids'][mode]
                target=case['answer_ids'];full=torch.tensor([prompt+target[:-1]])
                logits=model(full)[0,len(prompt)-1:]
                expected=float(F.cross_entropy(logits,torch.tensor(target)))
                self.assertAlmostEqual(result['answer_nll'],expected,places=5)
                self.assertIn('query_last_injection_to_embedding_norm_ratio',result)
                self.assertEqual(result['prompt_tokens'],len(prompt))

    def test_end_to_end_report_arms_scope_and_remote_logging(self):
        tok=SimpleNamespace(idx2token={i:str(i).encode() for i in range(32)})
        wb=Mock();wb.init.return_value.id='test';wb.init.return_value.url='http://example.test/run'
        wb.init.return_value.entity='mobius';wb.init.return_value.project='ROSA-RWKV7'
        wb.Api.return_value.run.return_value.summary={'needle/trained/qa/8192/cases':3}
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=str(Path(tmp)/'out'),checkpoint='ckpt',base='base',seed=1,threads=2,max_new_tokens=2,wandb=True)
            with patch.object(n,'convert',return_value={'global_step':724}), \
                 patch.object(n,'TRIE_TOKENIZER',return_value=tok), \
                 patch.object(n,'build_case',side_effect=lambda _,l,d,s,**kw:self.case(l,d,s)), \
                 patch.object(n,'load_cpu',side_effect=lambda _,arm:self.model(arm!='base')), \
                 patch.object(n,'address_diagnostic',return_value={'suffix_only_exact_match':True}), \
                 patch('rwkv_rosa.native.wandb_auth.configure_wandb_auth'), \
                 patch.dict('sys.modules',{'wandb':wb}):
                report=n.run(args)
            self.assertEqual(report['status'],'complete');self.assertTrue(report['wandb']['verified'])
            self.assertEqual(sum(r['arm']=='base' for r in report['results']),14)
            self.assertEqual(sum(r['arm']=='trained' for r in report['results']),14)
            self.assertEqual(sum(r['arm']=='initial' for r in report['results']),2)
            self.assertEqual(n.summarize([]),[])
            existing=Path(tmp)/'existing';existing.mkdir()
            (existing/'manifest.json').write_text(json.dumps({'base_model':'base','source_checkpoint':'ckpt','global_step':724}))
            args.export=str(existing);args.difficulty='similar_archives';args.output=str(Path(tmp)/'hard');args.wandb=False
            with patch.object(n,'TRIE_TOKENIZER',return_value=tok), \
                 patch.object(n,'build_case',side_effect=lambda _,l,d,s,**kw:self.case(l,d,s)), \
                 patch.object(n,'load_cpu',side_effect=lambda _,arm:self.model(arm!='base')), \
                 patch.object(n,'address_diagnostic',return_value={'suffix_only_exact_match':True}):
                hard=n.run(args)
            self.assertEqual(sum(r['arm']=='trained' for r in hard['results']),8)
            self.assertTrue((Path(args.output)/'export').is_symlink())
            args.output=str(Path(tmp)/'mismatch');args.base='different'
            with self.assertRaises(ValueError):n.run(args)

        with patch('torch.cuda.is_available',return_value=True):
            with self.assertRaises(RuntimeError):n.run(args)


if __name__=='__main__':unittest.main()
