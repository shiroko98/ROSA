import unittest
import torch

from rwkv_rosa.native.fast_gpu_needle import select_token


class SamplingTest(unittest.TestCase):
    def test_greedy_and_top_one_never_choose_weaker_token(self):
        logits=torch.tensor([0.,2.,1.])
        for seed in range(20):
            generator=torch.Generator().manual_seed(seed)
            self.assertEqual(select_token(logits),1)
            self.assertEqual(select_token(logits,temperature=1.,top_k=1,generator=generator),1)
            self.assertEqual(select_token(logits,temperature=1.,top_p=.1,generator=generator),1)
        torch.testing.assert_close(logits,torch.tensor([0.,2.,1.]),rtol=0,atol=0)

    def test_nucleus_keeps_crossing_token_and_seeded_reproducibility(self):
        logits=torch.log(torch.tensor([.5,.3,.2]))
        def draw(seed):
            gen=torch.Generator().manual_seed(seed)
            return [select_token(logits,temperature=1.,top_p=.6,generator=gen) for _ in range(100)]
        values=draw(77)
        self.assertEqual(values,draw(77));self.assertEqual(set(values),{0,1})
        self.assertNotEqual(values,draw(78))

    def test_invalid_parameters_fail(self):
        for options in ({'temperature':-1},{'temperature':float('nan')},{'top_p':0},{'top_p':1.1},{'top_k':-1}):
            with self.assertRaises(ValueError):select_token(torch.ones(3),**options)


if __name__=='__main__':unittest.main()
