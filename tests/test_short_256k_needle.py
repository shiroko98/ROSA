import unittest
from pathlib import Path
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER
from rwkv_rosa.native.short_256k_needle import budget_scores


class ShortNeedleBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=TRIE_TOKENIZER(str(Path(__file__).resolve().parents[1]/'rwkv_vocab_v20230424.txt'))

    def test_open_thinking_and_later_closed_answer_use_actual_budget(self):
        prefix=self.tokenizer.encode('<think>目标是813507，需要检查。')
        ids=prefix+self.tokenizer.encode('</think>\n813507')
        short=budget_scores(self.tokenizer,ids,'813507',len(prefix))
        long=budget_scores(self.tokenizer,ids,'813507',len(ids)+1)
        self.assertTrue(short['content_hit']);self.assertFalse(short['final_hit'])
        self.assertTrue(short['hit_generation_limit'])
        self.assertTrue(long['final_hit']);self.assertTrue(long['strict_final'])
        self.assertFalse(long['hit_generation_limit'])

    def test_first_wrong_passkey_cannot_be_rescued_by_a_later_hit(self):
        ids=self.tokenizer.encode('123456，随后813507')
        result=budget_scores(self.tokenizer,ids,'813507',512)
        self.assertFalse(result['final_hit']);self.assertFalse(result['content_hit'])
        self.assertFalse(result['strict_final'])


if __name__=='__main__':unittest.main()
