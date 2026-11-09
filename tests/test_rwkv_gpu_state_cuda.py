"""Opt-in real CUDA tests; run with native launcher environment configured."""
import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('ROSA_RUN_NATIVE_STATE_TESTS')=='1' and torch.cuda.is_available(),
                     'Explicit native CUDA test environment required.')
class StatefulKernelCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rwkv_rosa.native.gpu_state import wkv_stateful
        cls.run_kernel=staticmethod(wkv_stateful)

    def inputs(self,t=64):
        torch.manual_seed(42)
        return [torch.randn(1,t,128,device='cuda',dtype=torch.bfloat16)*.1 for _ in range(6)]

    def test_native_full_and_arbitrary_split_match(self):
        from src.model import RWKV7_CLAMPW_CUDA
        x=self.inputs();original=RWKV7_CLAMPW_CUDA(*x)
        actual,final=self.run_kernel(*x)
        self.assertTrue(torch.equal(actual,original))
        state=None;pieces=[]
        for lo,hi in ((0,1),(1,15),(15,16),(16,33),(33,64)):
            y,state=self.run_kernel(*[a[:,lo:hi] for a in x],state);pieces.append(y)
        self.assertTrue(torch.equal(actual,torch.cat(pieces,1)))
        self.assertTrue(torch.equal(final,state))

    def test_snapshot_excludes_tail_padding(self):
        x=self.inputs(32)
        full,snapshot=self.run_kernel(*x,state_tokens=17)
        prefix,expected=self.run_kernel(*[a[:,:17] for a in x])
        self.assertTrue(torch.equal(full[:,:17],prefix))
        self.assertTrue(torch.equal(snapshot,expected))
        _,wrong=self.run_kernel(*x)
        self.assertFalse(torch.equal(snapshot,wrong))

    def test_initial_state_is_immutable_and_branches_independent(self):
        x=self.inputs(17);_,initial=self.run_kernel(*x)
        copy=initial.clone();a,sa=self.run_kernel(*[t[:,-1:] for t in x],initial)
        b,sb=self.run_kernel(*[t[:,-1:] for t in x],initial)
        self.assertTrue(torch.equal(initial,copy))
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(sa,sb))
        self.assertNotEqual(initial.data_ptr(),sa.data_ptr())


if __name__=='__main__':unittest.main()
