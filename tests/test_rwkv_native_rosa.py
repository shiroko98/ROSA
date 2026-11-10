"""Native SFT extension: device pinning, optimizer/LR/resume and fusion equivalence."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import train_rwkv_native_rosa  # Safe import verifies the thin native-entry module without executing training.

from rwkv_rosa.backbone import RWKVConfig
from rwkv_rosa.model import RWKVRosaLM
from rwkv_rosa.native.adapter import CPUImprovedROSA, CPUROSAOptimizer, clip_cpu_gradients, gradient_norm
try:
    from rwkv_rosa.native.integration import add_arguments, install, normalize_completed_epoch_checkpoint, NativeROSACallback
except ModuleNotFoundError as exc:
    if exc.name not in ('pytorch_lightning', 'lightning_fabric', 'torchmetrics', 'pkg_resources'):
        raise
    add_arguments = install = None
from rwkv_rosa.native.prepare_probe_data import prepare
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


class NativeROSA(unittest.TestCase):
    def make(self, device='cpu'):
        torch.manual_seed(42)
        embedding = torch.nn.Embedding(32, 16).requires_grad_(False)
        branch = CPUImprovedROSA(embedding, layers=(0, 1), device=device)
        return embedding, branch

    def test_cpu_pin_update_schedule_and_restore(self):
        _, branch = self.make()
        identities = [id(p) for p in branch.parameters()]
        branch.to(dtype=torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in branch.parameters()))
        self.assertEqual(identities, [id(p) for p in branch.parameters()])
        optimizer = CPUROSAOptimizer(branch, lr=1e-3)
        optimizer.param_groups[0]['lr'] = 5e-4
        self.assertEqual(optimizer.dense.param_groups[0]['lr'], 5e-4)
        ids = torch.tensor([[1, 2, 1, 2, 1, 2]])
        x = torch.randn(1, 6, 16, requires_grad=True)
        before = branch.value_projs[0].weight.detach().clone()
        def closure():
            optimizer.zero_grad()
            branch.begin(ids)
            y = branch.inject(0, x)
            self.assertIs(branch.inject(5, y), y)
            loss = branch.inject(1, y).square().mean()
            loss.backward()
            self.assertGreater(clip_cpu_gradients(branch.parameters(), 1), 0)
            return loss
        with torch.no_grad():
            optimizer.step(closure)
        optimizer.assert_states_device('cpu')
        self.assertFalse(torch.equal(before, branch.value_projs[0].weight))
        state = optimizer.state_dict()
        _, other = self.make()
        other.load_state_dict(branch.state_dict())
        restored = CPUROSAOptimizer(other, lr=2e-3)
        restored.load_state_dict(state)
        self.assertEqual(restored.param_groups[0]['lr'], 5e-4)
        restored.zero_grad()
        with self.assertRaises(ValueError):
            restored.load_state_dict({})
        self.assertEqual(gradient_norm([]), 0.)
        p = torch.nn.Parameter(torch.ones(1)); p.grad = torch.tensor([float('nan')])
        with self.assertRaises(FloatingPointError):
            clip_cpu_gradients([p], 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_cpu_outputs_gradients_and_pinning(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        _, cpu = self.make('cpu'); _, gpu = self.make('cuda')
        cpu.cuda().bfloat16()
        gpu.cpu().bfloat16()
        self.assertTrue(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in cpu.parameters()))
        self.assertTrue(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in gpu.parameters()))
        ids = torch.tensor([[1, 2, 1, 2, 1, 2]], device='cuda')
        x = torch.randn(1, 6, 16, device='cuda')
        cpu.begin(ids); gpu.begin(ids)
        a = cpu.inject(0, x); b = gpu.inject(0, x)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        a.square().mean().backward(); b.square().mean().backward()
        for (n, p), (m, q) in zip(cpu.named_parameters(), gpu.named_parameters()):
            self.assertEqual(n, m)
            if p.grad is not None:
                left = p.grad.to_dense() if p.grad.is_sparse else p.grad
                right = q.grad.to_dense() if q.grad.is_sparse else q.grad
                torch.testing.assert_close(left, right.cpu(), atol=2e-6, rtol=2e-5)
        opt = CPUROSAOptimizer(gpu, lr=1e-3)
        clip_cpu_gradients(gpu.parameters(), 1); opt.step(); opt.assert_states_device('cuda')

    def test_reference_fusion_formula_equivalence(self):
        torch.manual_seed(42)
        ref = RWKVRosaLM(RWKVConfig(vocab_size=32, n_embd=16, head_size=8),
                         variant='improved', inject_layers=(0,), scale=.05, sparse_values=True, value_shards=4)
        branch = CPUImprovedROSA(ref.backbone.emb, layers=(0,), scale=.05)
        learned = {n: p for n, p in ref.state_dict().items() if not n.startswith('backbone.')}
        branch.load_state_dict(learned)
        ids = torch.tensor([[1, 2, 1, 2, 1, 2]])
        x = ref.backbone.emb(ids)
        branch.begin(ids)
        residual, _ = ref._improved_residual(0, x, branch.addresses)
        torch.testing.assert_close(branch.inject(0, x), x + residual, atol=0, rtol=0)

    @unittest.skipIf(install is None, 'Native Lightning dependency is optional in reference-only test environments')
    def test_install_guards_and_arguments(self):
        p = argparse.ArgumentParser(); add_arguments(p)
        args = p.parse_args(['--rosa', '1', '--rosa_device', 'cuda'])
        self.assertEqual(args.rosa_device, 'cuda')
        args.rosa_cpu = 0
        self.assertIsNone(install(None, args))
        args.rosa_cpu = 1; args.devices = 2; args.num_nodes = 1; args.strategy = 'auto'
        with self.assertRaises(ValueError): install(None, args)
        args.devices = 1; args.precision = 32
        with self.assertRaises(ValueError): install(None, args)

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_completed_epoch_checkpoint_normalization_preserves_totals(self):
        state = {'global_step': 10, 'loops': {'fit_loop': {
            'epoch_progress': {'current': {'ready': 1, 'completed': 1}},
            'epoch_loop.batch_progress': {'is_last_batch': True, 'current': {'ready': 10, 'completed': 10},
                                          'total': {'ready': 10, 'completed': 10}},
            'epoch_loop.batch_loop.optimizer_loop.optim_progress': {
                'optimizer_position': 1, 'optimizer': {'step': {'current': {'completed': 10}, 'total': {'completed': 10}}}}
        }}}
        self.assertTrue(normalize_completed_epoch_checkpoint(state))
        fit = state['loops']['fit_loop']
        self.assertEqual(state['global_step'], 10)
        self.assertEqual(fit['epoch_loop.batch_progress']['total']['completed'], 10)
        self.assertEqual(fit['epoch_loop.batch_progress']['current']['completed'], 0)
        self.assertEqual(fit['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer_position'], 0)
        self.assertFalse(normalize_completed_epoch_checkpoint(state))
        self.assertFalse(normalize_completed_epoch_checkpoint({}))

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_native_install_freezes_body_and_preserves_cpu_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Module(); model.emb = torch.nn.Embedding(32, 16)
            args = SimpleNamespace(rosa_cpu=1, devices=1, num_nodes=1, strategy='auto', precision='bf16',
                load_model='base', train_stage=0, load_partial=0, resume_ckpt_path=None, rosa_layers='0',
                n_layer=1, rosa_scale=.05, rosa_device='cpu', rosa_resume='', run_name='test', proj_dir=tmp,
                lr_init=1e-3, betas=(.9,.99), adam_eps=1e-8, weight_decay=0.)
            with patch.dict('os.environ', {}, clear=False):
                callback = install(model, args)
            self.assertEqual(args.rosa_no_checkpoint_layers, ())
            self.assertIsInstance(callback, NativeROSACallback)
            self.assertFalse(model.emb.weight.requires_grad)
            self.assertEqual(model.emb.weight.dtype, torch.bfloat16)
            self.assertTrue(all(p.requires_grad and p.device.type=='cpu' for p in model.rosa_cpu.parameters()))
            optimizer = model.configure_optimizers()
            model.configure_gradient_clipping(optimizer, gradient_clip_val=1)
            for layers in ('1', '0,0', '-1'):
                args.precision = 'bf16'; args.rosa_no_checkpoint_layers = layers
                with self.assertRaises(ValueError): install(model, args)

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_native_callback_cpu_lifecycle_without_cuda_dependencies(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            _, branch = self.make()
            args = SimpleNamespace(proj_dir=tmp, rosa_device='cpu', ctx_len=6, load_model='base',
                                   rosa_precision_policy='test', rosa_save=1, data_file='data',
                                   epoch_steps=2, rosa_layers='0,1', rosa_scale=.05)
            fake_body = SimpleNamespace(requires_grad=False, grad=None, device=torch.device('cuda'),
                                        dtype=torch.bfloat16, numel=lambda:16)
            named = [('emb.weight',fake_body), *[(f'rosa_cpu.{n}',p) for n,p in branch.named_parameters()]]
            model = SimpleNamespace(rosa_cpu=branch, emb=SimpleNamespace(weight=fake_body),
                                    named_parameters=lambda: iter(named), parameters=lambda: (p for _,p in named))
            optimizer = CPUROSAOptimizer(branch, lr=1e-3)
            trainer = SimpleNamespace(global_step=0, world_size=1, is_global_zero=True, optimizers=[optimizer], my_loss=1., my_wandb=Mock(),
                                      save_checkpoint=Mock())
            callback = NativeROSACallback(args)
            args.rosa_eval_initial=1; trainer.callbacks=[Mock()]
            (Path(tmp)/'STOP').touch()
            callback.save_full_checkpoint=Mock()
            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ', {'RWKV_KERNEL':'@rwkv3','RWKV_MY_TESTING':'x070'}))
                stack.enter_context(patch('rwkv_rosa.native.integration.frozen_hash', return_value='frozen'))
                for name in ('synchronize','reset_peak_memory_stats'):
                    stack.enter_context(patch.object(torch.cuda,name,return_value=None))
                for name in ('max_memory_allocated','max_memory_reserved'):
                    stack.enter_context(patch.object(torch.cuda,name,return_value=0))
                callback.on_fit_start(trainer,model)
                trainer.callbacks[0]._run_sft_eval.assert_called_once()
                ids=torch.tensor([[1,2,1,2,1,2]]);batch=(ids,ids,torch.ones_like(ids,dtype=torch.float32))
                callback.on_train_batch_start(trainer,model,batch,0)
                branch.begin(ids); loss=branch.inject(0,torch.randn(1,6,16)).square().mean()
                callback.on_before_backward(trainer,model,loss); loss.backward()
                callback.on_after_backward(trainer,model); optimizer.step()
                callback.on_train_batch_end(trainer,model,{},batch,0)
                trainer.global_step=1
                callback.on_train_batch_end(trainer,model,{},batch,0)
                callback.on_fit_end(trainer,model)
                self.assertTrue(trainer.should_stop)
                callback.save_full_checkpoint.assert_called_with(trainer,'user-stop')
                self.assertTrue(callback.audit['backbone_unchanged'])
                trainer.save_checkpoint.assert_called_once()
                checkpoint={};callback.on_save_checkpoint(trainer,model,checkpoint)
                callback.on_load_checkpoint(trainer,model,checkpoint)
                checkpoint['native_rosa']['ctx_len']=7
                with self.assertRaises(ValueError): callback.on_load_checkpoint(trainer,model,checkpoint)
                callback.on_exception(trainer,model,ValueError('test'))
                self.assertTrue((Path(tmp)/'native_failure.json').exists())

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_atomic_full_checkpoint_and_latest_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = NativeROSACallback(SimpleNamespace(proj_dir=tmp))
            current = SimpleNamespace(completed=2, processed=3)
            total = SimpleNamespace(completed=7, processed=8)
            progress = SimpleNamespace(current=current, total=total)
            trainer = SimpleNamespace(global_step=3, fit_loop=SimpleNamespace(
                epoch_loop=SimpleNamespace(batch_progress=progress)))
            def save(path):
                self.assertEqual((current.completed, total.completed), (3, 8))
                Path(path).write_bytes(b'full model and optimizer')
            trainer.save_checkpoint = Mock(side_effect=save)
            first = callback.save_full_checkpoint(trainer, 'initial')
            self.assertEqual((current.completed, total.completed), (2, 7))
            self.assertEqual((Path(tmp)/'native-last.ckpt').resolve(), first)
            trainer.global_step = 4
            second = callback.save_full_checkpoint(trainer, 'hourly')
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b'full model and optimizer')
            self.assertEqual((Path(tmp)/'native-last.ckpt').resolve(), second)
            self.assertEqual(len((Path(tmp)/'native_checkpoints.jsonl').read_text().splitlines()), 2)
            trainer.save_checkpoint.side_effect = RuntimeError('disk unavailable')
            with self.assertRaises(RuntimeError): callback.save_full_checkpoint(trainer, 'hourly')
            self.assertEqual((current.completed, total.completed), (2, 7))
            self.assertEqual((Path(tmp)/'native-last.ckpt').resolve(), second)

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_wall_clock_checkpoint_schedule(self):
        callback = NativeROSACallback(SimpleNamespace(proj_dir='unused', rosa_checkpoint_seconds=3600,
                                                     rosa_checkpoint_initial=1))
        callback.last_checkpoint_time = 100.
        trainer = SimpleNamespace(global_step=1)
        def save(trainer, reason): callback.last_checkpoint_time = 200.
        callback.save_full_checkpoint = Mock(side_effect=save)
        with patch('rwkv_rosa.native.integration.time.monotonic', return_value=200.):
            callback.maybe_save_checkpoint(trainer)
            callback.save_full_checkpoint.assert_called_once_with(trainer, 'initial')
            callback.maybe_save_checkpoint(trainer)
            self.assertEqual(callback.save_full_checkpoint.call_count, 1)
        with patch('rwkv_rosa.native.integration.time.monotonic', return_value=3800.):
            callback.maybe_save_checkpoint(trainer)
            callback.save_full_checkpoint.assert_called_with(trainer, 'hourly')
        callback.args.rosa_checkpoint_seconds = 0
        callback.maybe_save_checkpoint(trainer)
        self.assertEqual(callback.save_full_checkpoint.call_count, 2)
        parser = argparse.ArgumentParser(); add_arguments(parser)
        args = parser.parse_args(['--rosa_checkpoint_seconds','3600','--rosa_checkpoint_initial','1'])
        self.assertEqual(args.rosa_checkpoint_seconds, 3600)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                install(None, SimpleNamespace(rosa_cpu=1, rosa_checkpoint_seconds=value))
        with self.assertRaises(ValueError):
            install(None, SimpleNamespace(rosa_cpu=1, rosa_checkpoint_seconds=3600, rosa_save=0))

    def test_self_hosted_wandb_credentials_stay_on_selected_host(self):
        from rwkv_rosa.native.wandb_auth import configure_wandb_auth
        import os
        with patch.dict('os.environ', {'WANDB_BASE_URL':'http://example.test:9098'}, clear=True), \
             patch('rwkv_rosa.native.wandb_auth.netrc.netrc') as read:
            read.return_value.authenticators.return_value = ('api', None, 'secret-test')
            self.assertTrue(configure_wandb_auth())
            read.return_value.authenticators.assert_called_once_with('example.test:9098')
            self.assertEqual(os.environ['WANDB_API_KEY'], 'secret-test')
            self.assertFalse(configure_wandb_auth())
        with patch.dict('os.environ', {}, clear=True): self.assertFalse(configure_wandb_auth())
        with patch.dict('os.environ', {'WANDB_BASE_URL':'ftp://example.test'}, clear=True):
            with self.assertRaises(ValueError): configure_wandb_auth()
        with patch.dict('os.environ', {'WANDB_BASE_URL':'http://example.test:9098'}, clear=True), \
             patch('rwkv_rosa.native.wandb_auth.netrc.netrc', side_effect=FileNotFoundError):
            self.assertFalse(configure_wandb_auth())
        with patch.dict('os.environ', {'WANDB_BASE_URL':'http://example.test:9098'}, clear=True), \
             patch('rwkv_rosa.native.wandb_auth.netrc.netrc') as read:
            read.return_value.authenticators.return_value = None
            self.assertFalse(configure_wandb_auth())

    @unittest.skipIf(install is None, 'Native Lightning dependency unavailable')
    def test_native_install_rejects_unsupported_modes_and_needle_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=torch.nn.Module();model.emb=torch.nn.Embedding(32,16)
            defaults=dict(rosa_cpu=1,devices=1,num_nodes=1,strategy='auto',precision='bf16',load_model='base',
                train_stage=0,load_partial=0,resume_ckpt_path=None,rosa_layers='0',n_layer=2,rosa_scale=.05,
                rosa_device='cpu',rosa_resume='',run_name='test',proj_dir=tmp,lr_init=1e-3,betas=(.9,.99),
                adam_eps=1e-8,weight_decay=0.,rosa_needle_every_eval=0,ctx_len=8,data_file='toy',epoch_steps=1)
            self.assertIsNone(install(model,SimpleNamespace(rosa_cpu=0)))
            invalid=[{'devices':2},{'num_nodes':2},{'strategy':'deepspeed_stage_2'},{'precision':'32'},
                {'load_model':''},{'train_stage':1},{'load_partial':1},{'resume_ckpt_path':'old.ckpt'},
                {'rosa_layers':''},{'rosa_layers':'2'},{'rosa_layers':'0,0'}]
            needle=dict(rosa_needle_every_eval=1,rosa_needle_length=1024,rosa_needle_depths='0.5',
                rosa_needle_max_new_tokens=16,rosa_needle_cpu_threads=2,rosa_needle_timeout=120)
            for update in invalid:
                with self.assertRaises(ValueError):install(model,SimpleNamespace(**{**defaults,**update}))
            for update in ({'rosa_needle_depths':'nan'},{'rosa_needle_depths':'0'},{'rosa_needle_length':128},
                           {'rosa_needle_timeout':0}):
                with self.assertRaises(ValueError):install(model,SimpleNamespace(**{**defaults,**needle,**update}))
            args=SimpleNamespace(**{**defaults,**needle,'rosa_resume':'resume.ckpt'})
            callback=install(model,args)
            self.assertEqual(args.resume_ckpt_path,'resume.ckpt')
            args.rosa_checkpoint_adapter_only=1
            from rwkv_rosa.native.cpu_compare import state_hash
            body_state={'emb.weight':model.emb.weight}
            callback.audit={'backbone_sha256_before':state_hash(body_state)}
            compact={'state_dict':{'emb.weight':model.emb.weight,'rosa_cpu.gate_bias':model.rosa_cpu.gate_bias}}
            callback.on_save_checkpoint(None,model,compact)
            self.assertEqual(set(compact['state_dict']),{'rosa_cpu.gate_bias'})
            with patch('rwkv_rosa.backbone.load_base_state',return_value=body_state):
                callback.on_load_checkpoint(None,model,compact)
                self.assertIn('emb.weight',compact['state_dict'])
            compact['native_rosa_frozen_base']['sha256']='wrong'
            with patch('rwkv_rosa.backbone.load_base_state',return_value=body_state):
                with self.assertRaises(ValueError):callback.on_load_checkpoint(None,model,compact)
            args.rosa_checkpoint_adapter_only=0

            checkpoint={};callback.on_save_checkpoint(None,None,checkpoint)
            checkpoint['native_rosa_data_policy']['micro_bsz']=2
            with self.assertRaises(ValueError):callback.on_load_checkpoint(None,None,checkpoint)

    def test_paired_benchmark_controller_and_batch_identity(self):
        from rwkv_rosa.native.benchmark import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for context in (1024, 8192):
                folder = root / 'data' / f'ctx{context}'
                folder.mkdir(parents=True)
                (folder / 'manifest.json').write_text('{}')
            args = SimpleNamespace(out_dir=str(root/'out'), data_root=str(root/'data'), contexts='1024,8192',
                                   steps=3, warmup=1, checkpoint='test-base', cpu_threads=1,
                                   timeout_seconds=1, wandb_mode='offline')
            def popen(command, **kwargs):
                case = Path(command[command.index('--proj_dir') + 1]); case.mkdir()
                device = command[command.index('--rosa_device') + 1]
                seconds = 1. if device == 'cuda' else 2.
                rows = [dict(step=i, seconds=seconds, batch_sha256=f'batch{i}', loss=1., peak_gpu_allocated_GiB=1.)
                        for i in range(1,4)]
                (case/'native_steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                (case/'native_audit.json').write_text(json.dumps(dict(backbone_unchanged=True, rosa_values_changed=True)))
                return Mock(pid=1, wait=Mock(return_value=0))
            with patch('subprocess.Popen', side_effect=popen):
                result = run(args)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['comparisons']['1024']['gpu_rosa_speedup_over_cpu'], 2.)
            with self.assertRaises(FileExistsError): run(args)

    def test_probe_data_preserves_actual_suffix_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prefix = root / 'source'
            ids = [np.asarray([1, 2] * 16 + [3], dtype=np.uint16)] * 3
            masks = [np.asarray([0] * 17 + [1] * 16, dtype=np.uint8)] * 3
            for suffix, arrays in [('', ids), ('.mask', masks)]:
                np.concatenate(arrays).tofile(str(prefix) + suffix + '.bin')
                with MMapIndexedDataset.Index.writer(str(prefix) + suffix + '.idx', arrays[0].dtype.type) as w:
                    w.write([33] * 3, [0, 1, 2, 3])
            dest = prepare(str(prefix), root / 'derived', 16, rows=2)
            actual = MMapIndexedDataset(dest); mask = MMapIndexedDataset(dest + '.mask')
            self.assertEqual(actual[0].tolist(), ids[0][-17:].tolist())
            self.assertEqual(mask[0].tolist(), masks[0][-17:].tolist())
            with self.assertRaises(ValueError): prepare(str(prefix), root / 'bad', 64, rows=2)


if __name__ == '__main__': unittest.main()
