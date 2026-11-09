"""Persistent SAM policy and the deterministic ten-case evaluation protocol."""
import random
import unittest

import torch
from rosa_cpp_extension import load_rosa_sam_cpu_extension
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rwkv_rosa.native.fast_gpu_needle import ten_cases
from rwkv_rosa.native.gpu_state import IncrementalAddress,state_nbytes
from rwkv_rosa.native.needle_eval import make_engine,ROOT
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER


class IncrementalSAMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension=load_rosa_sam_cpu_extension()
        torch.set_num_threads(2)

    def test_raw_incremental_equals_static_with_clones_repetition_and_random(self):
        rng=random.Random(42)
        for ids in ([7]*1000,list(range(100)),[0,65530,11,7,7,0,11,65531]*20,
                    [rng.randrange(0,30) for _ in range(3000)]):
            expected=self.extension.sam_rosa_predict(ids,1)
            state=self.extension.IncrementalSAM(1)
            for cut in (1,len(ids)//2,len(ids)):
                p,m=state.prefill(ids[state.size():cut])
                e=self.extension.sam_rosa_predict(ids[:cut],1)
                self.assertEqual(list(p),e[0][cut-len(p):cut])
                self.assertEqual(list(m),e[1][cut-len(m):cut])
            for token in (7,0,65530,4):
                branch=state.clone();actual=branch.update_one(token)
                exp=self.extension.sam_rosa_predict(ids+[token],1)
                self.assertEqual(tuple(actual),(exp[0][-1],exp[1][-1]))
                self.assertEqual(state.size(),len(ids))

    def test_filtered_policy_prefill_and_decode_equals_training_engine(self):
        ids=[0,7,8,65530,7,8,0,7,8,65531,7,8]
        sam=IncrementalAddress();got=sam.prefill(ids[:6])
        expected=make_engine().forward_seq(torch.tensor([ids[:6]]))
        for name in ('addr_ids','raw_match_lens','fired_match_lens','valid_mask','special_mask'):
            self.assertTrue(torch.equal(getattr(got,name),getattr(expected,name)),name)
        for i in range(6,len(ids)):
            got=sam.update_one(ids[i]);expected=make_engine().forward_seq(torch.tensor([ids[:i+1]]))
            for name in ('addr_ids','raw_match_lens','fired_match_lens','valid_mask','special_mask'):
                self.assertTrue(torch.equal(getattr(got,name),getattr(expected,name)[:,-1:]),name)
        with self.assertRaises(ValueError):IncrementalAddress().prefill([])

    def test_state_memory_count(self):
        state=[{'x':torch.zeros(2,dtype=torch.bfloat16),'wkv':torch.zeros(3,dtype=torch.float32)}]
        self.assertEqual(state_nbytes(state),16)

    def test_ten_case_design_is_balanced_and_paired_before_results(self):
        tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
        left=list(ten_cases(tokenizer,1024,20261109,'rwkv'))
        right=list(ten_cases(tokenizer,1024,20261109,'chatml'))
        self.assertEqual(len(left),10)
        self.assertEqual(sum(c['difficulty']=='unrelated' for c,_ in left),5)
        self.assertEqual(sum(m=='qa' for _,m in left),5)
        for (a,m),(b,n) in zip(left,right):
            self.assertEqual(m,n)
            for key in ('id','secret','key','depth','seed','difficulty','answer_ids'):
                self.assertEqual(a[key],b[key])
            self.assertEqual(len(a['stem_ids']),1024);self.assertEqual(len(b['stem_ids']),1024)
            self.assertNotIn(a['secret'],tokenizer.decode(a['suffix_ids'][m]))
            self.assertNotIn(b['secret'],tokenizer.decode(b['suffix_ids'][n]))


if __name__=='__main__':unittest.main()
