import unittest
import torch

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.native.cpu_compare import CPUInference
from rwkv_rosa.native.cpu_wkv_inference import install_cpu_wkv


class CPUWKVInferenceTest(unittest.TestCase):
    def test_recurrence_continuation_and_nonmutation(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        body = build_backbone(RWKVConfig(vocab_size=32, n_embd=32, n_layer=2, head_size=8))
        reference = body._reference_runtime.RWKV7_OP
        compiled = install_cpu_wkv(body)
        values = [torch.randn(2, 37, 32) * .15 for _ in range(6)]
        values[1].sub_(2)
        state = torch.randn(2, 4, 8, 8) * .1
        before = state.clone()
        with torch.inference_mode():
            y, s = reference(*values, state=state, return_state=True)
            actual, after = compiled(*values, state=state, return_state=True)
            torch.testing.assert_close(actual, y, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(after, s, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(state, before, rtol=0, atol=0)
            p, middle = compiled(*(x[:, :13] for x in values), state=state, return_state=True)
            q, final = compiled(*(x[:, 13:] for x in values), state=middle, return_state=True)
            torch.testing.assert_close(torch.cat((p, q), 1), actual, rtol=0, atol=0)
            torch.testing.assert_close(final, after, rtol=0, atol=0)

    def test_model_features_and_chunked_prefix(self):
        torch.manual_seed(19)
        body = build_backbone(RWKVConfig(vocab_size=32, n_embd=32, n_layer=2, head_size=8))
        model = CPUInference(body)
        ids = torch.randint(0, 32, (1, 65))
        with torch.inference_mode():
            reference = model.features(ids)
            install_cpu_wkv(body)
            actual = model.features(ids)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            _, state = model.features(ids[:, :31], return_state=True)
            last, _ = model.features(ids[:, 31:], state=state, return_state=True)
            torch.testing.assert_close(last, actual[:, 31:], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
