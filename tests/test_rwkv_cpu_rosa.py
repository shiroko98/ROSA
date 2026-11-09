"""Frozen backbone + CPU ROSA, including actual CUDA cross-device autograd."""
from pathlib import Path
import tempfile
import contextlib
import io
import unittest

import torch

from rosa_optim import build_training_optimizers
from rwkv_rosa.backbone import RWKVConfig
from rwkv_rosa.experiment import clip_all_gradients, parse_args, restore_checkpoint, save_checkpoint
from rwkv_rosa.freeze_audit import check_frozen_backbone, parameter_hash
from rwkv_rosa.model import RWKVRosaLM, configure_train_scope


class CPUROSA(unittest.TestCase):
    def model(self, backbone_device='cpu', layers=(0,)):
        torch.manual_seed(42)
        cfg = RWKVConfig(vocab_size=24, n_embd=16, n_layer=2, head_size=8)
        model = RWKVRosaLM(cfg, variant='improved', inject_layers=layers,
                           sparse_values=True, value_shards=2, activation_checkpointing=True)
        model.place_devices(backbone_device, 'cpu')
        scope = configure_train_scope(model, 'rosa_only')
        self.assertEqual(scope['backbone_trainable_names'], [])
        self.assertEqual(scope['trainable_cuda_parameters'], 0)
        self.assertEqual(scope['trainable_parameters'], scope['trainable_cpu_parameters'])
        return model

    def exercise_update(self, device):
        model = self.model(device, layers=(0, 1))
        input_ids = torch.tensor([[1, 2, 1, 2, 1, 2]], device=device)
        frozen = check_frozen_backbone(model)
        learned = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = build_training_optimizers(model, lr=1e-3, weight_decay=0.01)
        devices = []
        handles = []
        for module in (model.rosa_value_store, model.key_projs[0], model.value_projs[0], model.hidden_norms[0]):
            # ValueStore.lookup calls its table directly, so hooks on actual
            # projections/norms certify their arithmetic devices, not .to flags.
            handles.append(module.register_forward_hook(lambda m, a, o: devices.append(o.device.type)))
        out = model(input_ids)['logits']
        self.assertEqual(out.device.type, torch.device(device).type)
        out.square().mean().backward()
        for n, p in model.named_parameters():
            if n.startswith('backbone.'):
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
            else:
                self.assertEqual(p.device.type, 'cpu')
                # Row shards untouched by this batch legitimately have no grad.
                if '.shards.' not in n:
                    self.assertIsNotNone(p.grad, n)
                if p.grad is not None:
                    self.assertEqual(p.grad.device.type, 'cpu')
        self.assertTrue(devices)
        self.assertEqual(set(devices), {'cpu'})
        self.assertGreater(clip_all_gradients(optimizer, 1), 0)
        optimizer.step()
        self.assertEqual(check_frozen_backbone(model, frozen), frozen)
        self.assertTrue(any(not torch.equal(p, learned[n]) for n, p in model.named_parameters() if p.requires_grad))
        for opt in (optimizer.dense_optimizer, optimizer.sparse_optimizer):
            for state in opt.state.values():
                self.assertTrue(all(v.device.type == 'cpu' for v in state.values() if isinstance(v, torch.Tensor)))
        for h in handles:
            h.remove()
        # CPU-only trainable deltas + optimizer survive a restore on a hybrid model.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            save_checkpoint(path, model, optimizer, signature='cpu', epoch=0, batch_offset=1, step=1, history=[])
            copy = self.model(device, layers=(0, 1))
            copy_opt = build_training_optimizers(copy, lr=1e-3, weight_decay=0.01)
            restore_checkpoint(path, copy, copy_opt, 'cpu')
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, dict(copy.named_parameters())[n], atol=0, rtol=0)
        return model

    def test_cpu_only_update_and_resume(self):
        self.exercise_update('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_backbone_cpu_rosa_update_and_parity(self):
        model = self.exercise_update('cuda')
        x = torch.tensor([[1, 2, 1, 2, 1, 2]], device='cuda')
        model.eval()
        full = model(x)['logits']
        first = model(x[:, :3], return_state=True)
        last = model(x[:, 3:], state=first['state'])
        torch.testing.assert_close(full, torch.cat([first['logits'], last['logits']], 1), atol=2e-6, rtol=2e-5)
        colocated = self.model('cuda')
        hybrid = self.model('cuda')
        colocated.place_devices('cuda')
        torch.testing.assert_close(colocated(x)['logits'], hybrid(x)['logits'], atol=2e-6, rtol=2e-5)

    def test_freeze_guard_and_parser(self):
        args = parse_args(['--variants', 'improved', '--freeze-backbone', '--rosa-device', 'cpu'])
        self.assertEqual(args.train_scope, 'rosa_only')
        for argv in [['--rosa-device', 'cpu'], ['--freeze-backbone']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        model = self.model()
        frozen = check_frozen_backbone(model)
        with torch.no_grad():
            next(model.backbone.parameters()).add_(1)
        with self.assertRaises(AssertionError):
            check_frozen_backbone(model, frozen)
        next(model.backbone.parameters()).requires_grad_(True)
        with self.assertRaises(AssertionError):
            check_frozen_backbone(model)
        model = RWKVRosaLM(RWKVConfig(vocab_size=24, n_embd=16, head_size=8), variant='original')
        with self.assertRaises(ValueError):
            model.place_devices('meta', 'cpu')
        with self.assertRaises(ValueError):
            configure_train_scope(model, 'rosa_only')


if __name__ == '__main__':
    unittest.main()
