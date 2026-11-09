"""Do not confuse FP32 state storage with FP32 arithmetic under AMP."""
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rwkv_rosa.backbone import RWKVConfig, load_runtime
from rwkv_rosa.precision import wkv_policy


class MatmulTrace(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.bmm.default:
            self.calls.append((tuple(a.dtype for a in args if isinstance(a, torch.Tensor)), out.dtype))
        return out


class WKVPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.runtime = load_runtime(RWKVConfig(n_embd=16, head_size=8), torch.float32)

    def inputs(self, device="cpu", dtype=torch.float32, gradients=False):
        tensors = [torch.randn(1, 12, 16, device=device) * .1 for _ in range(6)]
        tensors[1] -= 2
        return [t.to(dtype).detach().requires_grad_(gradients) for t in tensors]

    def check_precision(self, device):
        inputs = self.inputs(device)
        with torch.no_grad():
            expected, expected_state = self.runtime.RWKV7_OP(*inputs, return_state=True)
            unfenced_trace = MatmulTrace()
            with wkv_policy(self.runtime, legacy=True), unfenced_trace, torch.autocast(device, dtype=torch.bfloat16):
                bad, bad_state = self.runtime.RWKV7_OP(*inputs, return_state=True)
            # Storage still says FP32, even though every recurrence bmm was BF16.
            self.assertEqual(bad_state.dtype, torch.float32)
            self.assertTrue(unfenced_trace.calls)
            self.assertTrue(all(out == torch.bfloat16 for _, out in unfenced_trace.calls))
            self.assertGreater(float((bad - expected).abs().max()), 0)
            trace = MatmulTrace()
            with trace, torch.autocast(device, dtype=torch.bfloat16):
                actual, state = self.runtime.RWKV7_OP(*inputs, return_state=True)
            self.assertTrue(all(dtypes == (torch.float32, torch.float32) and out == torch.float32
                                for dtypes, out in trace.calls))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(state, expected_state, atol=0, rtol=0)

    def test_cpu_autocast_recurrence_is_really_fp32(self):
        self.check_precision("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_autocast_recurrence_is_really_fp32(self):
        self.check_precision("cuda")

    def test_gradients_match_fp32_reference_inside_autocast(self):
        inputs = self.inputs(gradients=True)
        reference = [t.detach().clone().requires_grad_() for t in inputs]
        expected = self.runtime.RWKV7_OP(*reference)
        expected.square().sum().backward()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.runtime.RWKV7_OP(*inputs)
            loss = actual.square().sum()
        loss.backward()
        for a, b in zip(inputs, reference):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_bf16_inputs_and_streaming_state(self):
        inputs = self.inputs(dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            full, full_state = self.runtime.RWKV7_OP(*inputs, return_state=True)
            first, state = self.runtime.RWKV7_OP(*(t[:, :5] for t in inputs), return_state=True)
            last, state = self.runtime.RWKV7_OP(*(t[:, 5:] for t in inputs), state=state, return_state=True)
        torch.testing.assert_close(torch.cat((first, last), 1), full, atol=0, rtol=0)
        torch.testing.assert_close(state, full_state, atol=0, rtol=0)

    def test_policy_restored_on_error(self):
        operation = self.runtime.RWKV7_OP
        with self.assertRaises(RuntimeError):
            with wkv_policy(self.runtime, legacy=True):
                self.assertIs(self.runtime.RWKV7_OP, self.runtime._unfenced_wkv_op)
                raise RuntimeError("test")
        self.assertIs(self.runtime.RWKV7_OP, operation)


if __name__ == "__main__":
    unittest.main()
