"""Length-search state machine and real CPU-only worker regression tests."""
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rwkv_rosa.backbone import RWKVConfig, build_backbone
from rwkv_rosa.length_probe import main, next_length, search, worker
from rwkv_rosa.model import RWKVRosaLM
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


class LengthSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prefix = self.root / 'corpus'
        for suffix, array in [('', np.asarray(([1, 2] * 32) + [3], dtype=np.uint16)),
                              ('.mask', np.ones(65, dtype=np.uint8))]:
            array.tofile(str(self.prefix) + suffix + '.bin')
            with MMapIndexedDataset.Index.writer(str(self.prefix) + suffix + '.idx', array.dtype.type) as writer:
                writer.write([65], [0, 1])
        self.checkpoint = self.root / 'base.pth'
        cfg = RWKVConfig(vocab_size=24, n_embd=16, n_layer=2, head_size=8)
        torch.manual_seed(42)
        torch.save(build_backbone(cfg).state_dict(), self.checkpoint)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name='out', **kwargs):
        return SimpleNamespace(out_dir=str(self.root / name), checkpoint=str(self.checkpoint), data=str(self.prefix),
                               row=0, seed=42, length=8, start=16, ceiling=64, resolution=8, steps=1,
                               timeout_seconds=1, wandb=False, **kwargs)

    def test_growth_and_bisection(self):
        self.assertEqual(next_length(2048, None, 86016, 1024), 4096)
        self.assertEqual(next_length(16384, 32768, 86016, 1024), 24576)
        self.assertEqual(next_length(0, 2048, 86016, 1024), 1024)
        self.assertIsNone(next_length(2048, 3072, 86016, 1024))
        self.assertIsNone(next_length(86016, None, 86016, 1024))
        self.assertEqual(next_length(65536, None, 86016, 1024), 86016)

    def test_worker_real_update_and_oom_and_error(self):
        original_to = torch.Tensor.to
        placement = RWKVRosaLM.place_devices
        def cpu_to(tensor, *args, **kwargs):
            if args and str(args[0]).startswith('cuda'):
                args = ('cpu', *args[1:])
            if str(kwargs.get('device', '')).startswith('cuda'):
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
            stack.enter_context(patch.object(RWKVRosaLM, 'place_devices',
                                            lambda model, *a, **k: placement(model, 'cpu', 'cpu')))
            for name in ['synchronize', 'empty_cache', 'reset_peak_memory_stats']:
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            for name in ['max_memory_allocated', 'max_memory_reserved']:
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            args = self.args('success')
            Path(args.out_dir).mkdir()
            result = worker(args)
            self.assertEqual(result['status'], 'success')
            self.assertTrue(result['backbone_unchanged'])
            self.assertTrue(result['rosa_value_table_changed'])
            self.assertTrue(result['updates'][0]['all_rosa_gradients_cpu'])
            with patch.object(RWKVRosaLM, 'forward_features', side_effect=torch.cuda.OutOfMemoryError('test')):
                args = self.args('oom')
                Path(args.out_dir).mkdir()
                result = worker(args)
                self.assertEqual(result['status'], 'cuda_oom')
            args = self.args('error')
            args.length = 66
            Path(args.out_dir).mkdir()
            with redirect_stderr(io.StringIO()):
                self.assertEqual(worker(args)['status'], 'error')

    def test_search_bracket(self):
        args = self.args('bracket')
        def popen(command, **kwargs):
            length = int(command[command.index('--length') + 1])
            out = Path(command[command.index('--out-dir') + 1])
            status = 'success' if length <= 40 else 'cuda_oom'
            process = Mock(pid=123)
            def wait(**kw):
                (out / f'length_{length}.json').write_text(json.dumps({'status': status,
                        'input_tokens': length, 'updates': [{'loss': 1.}]}))
                return 0
            process.wait.side_effect = wait
            return process
        with patch('subprocess.Popen', side_effect=popen), patch('rwkv_rosa.tracking.ArmTracker', return_value=Mock(url=None)):
            result = search(args)
        self.assertEqual(result['status'], 'bracket_complete')
        self.assertEqual(result['largest_successful_length'], 40)
        self.assertEqual(result['smallest_oom_length'], 48)
        self.assertEqual([r['input_tokens'] for r in result['cases']], [16, 32, 64, 48, 40])
        with self.assertRaises(FileExistsError):
            search(args)

    def test_timeout_process_error_and_ceiling(self):
        process = Mock(pid=1)
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), subprocess.TimeoutExpired('worker', 20), 0]
        with patch('subprocess.Popen', return_value=process), patch('rwkv_rosa.tracking.ArmTracker', return_value=Mock(url=None)):
            result = search(self.args('timeout'))
        self.assertEqual(result['status'], 'stopped_timeout')
        self.assertIsNone(result['smallest_oom_length'])
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        with patch('subprocess.Popen', return_value=Mock(pid=1, wait=Mock(return_value=17))), \
             patch('rwkv_rosa.tracking.ArmTracker', return_value=Mock(url=None)):
            self.assertEqual(search(self.args('process'))['status'], 'stopped_process_error')
        def popen(command, **kwargs):
            length = int(command[command.index('--length') + 1])
            out = Path(command[command.index('--out-dir') + 1])
            (out / f'length_{length}.json').write_text(json.dumps({'status': 'success', 'input_tokens': length}))
            return Mock(pid=1, wait=Mock(return_value=0))
        with patch('subprocess.Popen', side_effect=popen), patch('rwkv_rosa.tracking.ArmTracker', return_value=Mock(url=None)):
            args = self.args('ceiling')
            args.ceiling = 16
            self.assertEqual(search(args)['status'], 'ceiling_passed')
        with patch('subprocess.Popen', side_effect=ValueError('test')), patch('rwkv_rosa.tracking.ArmTracker', return_value=Mock(url=None)), \
             self.assertRaises(ValueError):
            search(self.args('exception'))

    def test_parser_validation(self):
        for argv in [['--out-dir', 'x', '--start', '0'], ['--out-dir', 'x', '--worker']]:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
        with patch('rwkv_rosa.length_probe.worker', return_value='ok'):
            self.assertEqual(main(['--out-dir', 'x', '--worker', '--length', '8']), 'ok')


if __name__ == '__main__':
    unittest.main()
