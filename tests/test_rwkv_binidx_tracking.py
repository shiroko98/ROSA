"""Packed-data integrity, equal history, and fail-closed tracking regressions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rwkv_rosa.binidx_data import load_binidx_documents, resolve_binidx_prefix
from rwkv_rosa.data import EqualHistoryDataset, document_hash, split_documents, validate_splits
from rwkv_rosa.experiment import parse_args, run
from rwkv_rosa.tracking import ArmTracker
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


def write_source(prefix, rows, masks=None):
    if masks is None:
        masks = [[1] * len(row) for row in rows]
    for suffix, values, dtype in [('', rows, np.uint16), ('.mask', masks, np.uint8)]:
        path = str(prefix) + suffix
        np.concatenate([np.asarray(row, dtype=dtype) for row in values]).tofile(path + '.bin')
        with MMapIndexedDataset.Index.writer(path + '.idx', dtype) as writer:
            writer.write([len(row) for row in values], list(range(len(values) + 1)))


class PackedDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'results').mkdir()
        self.prefix = self.root / 'results' / 'sft'
        # Duplicate example (with a packing LF), EOT padding, unmasked prompts,
        # an unsupervised example, plus an unterminated record tail.
        write_source(self.prefix, [[1, 2, 3, 23, 23, 11, 4, 5, 23],
                                   [11, 1, 2, 3, 23, 6, 7, 23],
                                   [8, 9, 23, 12, 13, 14]],
                                  [[0, 0, 1, 1, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 1, 0, 1, 1],
                                   [0, 0, 0, 0, 1, 1]])

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **kw):
        return load_binidx_documents(self.root, seed=42, max_rows=0, eot_id=23, **kw)

    def test_unpack_dedup_mask_history_split_and_fingerprint(self):
        docs, meta = self.load()
        self.assertEqual(len(docs), 4)
        self.assertEqual(meta['duplicates_removed'], 1)
        self.assertEqual(meta['padding_segments_removed'], 1)
        self.assertEqual(meta['unsupervised_segments_removed'], 1)
        self.assertEqual(meta['unterminated_rows'], 1)
        self.assertEqual(meta['source_stored_tokens'], 23)
        self.assertEqual(meta['selected_rows_sha256'], self.load()[1]['selected_rows_sha256'])
        self.assertEqual(resolve_binidx_prefix(str(self.prefix) + '.idx'), str(self.prefix))
        dataset = EqualHistoryDataset(docs, chunk_len=2, history_tokens=1)
        counted = 0
        for i in range(len(dataset)):
            row = dataset[i]
            counted += int(row['loss_mask'].sum())
            self.assertGreater(int(row['loss_mask'].sum()), 0)
            d, left, start, end = dataset.windows[i]
            self.assertEqual(row['input_ids'].tolist(), docs[d].input_ids[left:end].tolist())
            self.assertEqual(row['loss_mask'][:start-left].tolist(), [0] * (start-left))
        self.assertEqual(counted, sum(d.supervised_tokens for d in docs))
        a, b, c = split_documents(docs, 42)
        validate_splits(dict(train=a, val=b, test=c))
        self.assertEqual(len({document_hash(d) for d in docs}), len(docs))
        limited, info = self.load(max_docs=3)
        self.assertEqual(len(limited), 3)
        self.assertLessEqual(info['selected_rows_count'], 3)

    def test_invalid_files_mask_and_indices(self):
        with self.assertRaises(FileNotFoundError):
            resolve_binidx_prefix(self.root / 'missing')
        (self.root / 'emptydir').mkdir()
        with self.assertRaises(ValueError):
            resolve_binidx_prefix(self.root / 'emptydir')
        write_source(self.prefix, [[1, 2, 23]] * 3, [[0, 2, 1]] * 3)
        with self.assertRaises(ValueError):
            self.load()

    def test_toy_binidx_training_and_periodic_evaluation(self):
        args = parse_args(['--data', str(self.root), '--data-format', 'binidx', '--binidx-eot-id', '23',
                           '--binidx-max-rows', '0', '--out-dir', str(self.root / 'out'),
                           '--variants', 'original,improved', '--max-steps', '1', '--eval-windows', '1',
                           '--eval-every-steps', '1', '--chunk-len', '2', '--history-tokens', '1',
                           '--toy-dim', '16', '--toy-head-size', '8', '--no-save-checkpoints'])
        result = run(args)
        for variant in ['original', 'improved']:
            self.assertEqual(result['runs'][variant]['global_step'], 1)
            self.assertTrue(result['runs'][variant]['evaluation_is_truncated'])
            self.assertTrue((self.root / 'out' / variant / 'val.jsonl').exists())
        manifest = json.loads((self.root / 'out' / 'manifest.json').read_text())
        self.assertEqual(manifest['source_data']['format'], 'packed_sft_binidx')


class TrackingTest(unittest.TestCase):
    def test_online_log_resume_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(wandb=True, wandb_mode='online', wandb_project='test', wandb_entity='',
                                   wandb_group='g', wandb_run_name='n', out_dir=tmp)
            run_mock = Mock(offline=False, url='test-url', id='id1', project='test', entity='entity')
            run_mock.summary = {}
            sdk = Mock()
            sdk.init.return_value = run_mock
            with patch.dict('sys.modules', {'wandb': sdk}):
                tracker = ArmTracker(args, 'original', tmp, 'signature')
                tracker.log('train', {'loss': 1., 'ignored': 'string'}, 2)
                self.assertEqual(run_mock.log.call_args.args[0]['trainer/global_step'], 2)
                tracker.summary('val', {'loss': 2.})
                self.assertEqual(run_mock.summary['val/loss'], 2.)
                tracker.finish()
                ArmTracker(args, 'original', tmp, 'signature')
                self.assertEqual(sdk.init.call_args.kwargs['resume'], 'must')
                sdk.init.return_value = None
                with self.assertRaises(RuntimeError):
                    ArmTracker(args, 'original', tmp, 'signature')
            args.wandb = False
            tracker = ArmTracker(args, 'original', tmp, 'signature')
            tracker.log('train', {}, 0)
            tracker.summary('val', {})
            tracker.finish()


if __name__ == '__main__':
    unittest.main()
