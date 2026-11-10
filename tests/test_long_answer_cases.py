import unittest
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER
from rwkv_rosa.native.long_answer_cases import build_long_case,long_answer_scores
from rwkv_rosa.native.needle_eval import address_diagnostic,ROOT
from rwkv_rosa.native.adapter import CPUImprovedROSA
from rosa_addressing import RosaAddressEngine


class LongAnswerCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tok=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))

    def test_lengths_unique_answers_and_literal_retrieval(self):
        engine=RosaAddressEngine(min_match_len=1,pad_id=0,special_ids={0,65530,65531,65532,65533,65534},
            forbid_special_target=True,backend='sam',sequence_mode='online_sam',online_sam_impl='compiled_cpu')
        for kind in ('code','text'):
            for nominal in (16,64,128):
                c=build_long_case(self.tok,2048,.5,2026111010+nominal,kind,nominal)
                self.assertEqual(len(c['stem_ids']),2048)
                self.assertGreaterEqual(len(c['answer_ids']),nominal)
                self.assertGreater(len(c['suffix_ids']['qa']),80)
                self.assertNotIn(c['secret'],self.tok.decode(c['suffix_ids']['qa']))
                self.assertGreater(c['complete_distractor_records'],0)
                diag=address_diagnostic(c,'exact_prefix',engine)
                self.assertTrue(diag['suffix_only_exact_match'])
                self.assertGreaterEqual(diag['prompt_last_match_len'],6)

    def test_whole_answer_and_prefix_scores_separate_thinking(self):
        target='12345678901234567890123456789012'
        a=long_answer_scores(self.tok,' <think>'+target,target)
        self.assertTrue(a['content_contains_full_answer']);self.assertFalse(a['final_contains_full_answer'])
        b=long_answer_scores(self.tok,' <think>检查</think>\n'+target+'\n解释',target)
        self.assertTrue(b['final_contains_full_answer']);self.assertFalse(b['whole_final_exact_match'])
        self.assertEqual(b['best_correct_prefix_fraction'],1.)
        c=long_answer_scores(self.tok,target[:16],target)
        self.assertFalse(c['final_contains_full_answer']);self.assertEqual(c['best_correct_prefix_fraction'],.5)
        self.assertTrue(long_answer_scores(self.tok,target,target)['whole_final_exact_match'])


if __name__=='__main__':unittest.main()
