"""Independent checkpoint monitor never blocks or propagates errors into training."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import torch
from rwkv_rosa.native.standalone_needle import request_from_checkpoint,candidates,evaluate_one,main
from rwkv_rosa.native.cpu_compare import write_json


class StandaloneNeedleTest(unittest.TestCase):
    def checkpoint(self,tmp,step=200,**overrides):
        p=tmp/f'step{step}.ckpt'
        c={'native_rosa_checkpoint_kind':'adapter_optimizer_loop_with_external_base_v1',
            'global_step':step,'state_dict':{'rosa_cpu.gate_bias':torch.tensor([0.])},
            'native_rosa':{'position':'post_norm','layers':'','scale':.05,'ctx_len':192512},
            'native_rosa_frozen_base':{'path':'base.pth','sha256':'basehash','parameter_names':['emb.weight']}}
        c.update(overrides);torch.save(c,p);return p

    def test_checkpoint_request_preserves_snapshot_identity_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);p=self.checkpoint(tmp)
            r=request_from_checkpoint(p,tmp/'out',source_run_id='training')
            self.assertEqual(r['step'],200);self.assertEqual(r['position'],'post_norm');self.assertEqual(r['layers'],[])
            self.assertEqual(r['adapter_format'],'native_checkpoint');self.assertEqual(r['source_training_run_id'],'training')
            self.assertEqual(r['adapter'],str(p.resolve()))
            with self.assertRaises(ValueError):request_from_checkpoint(p,tmp/'out',length=1)
            bad=self.checkpoint(tmp,2,native_rosa_checkpoint_kind='unknown')
            with self.assertRaises(ValueError):request_from_checkpoint(bad,tmp/'out')

    def test_completed_atomic_checkpoint_discovery_deduplicates_steps_and_ignores_partial_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);a=self.checkpoint(tmp,1);b=self.checkpoint(tmp,200)
            (tmp/'native_checkpoints.jsonl').write_text('\n'.join(json.dumps({'step':step,'path':str(p)}) for step,p in ((1,a),(200,b),(200,b),(201,tmp/'missing.ckpt')))+'\n{"partial":')
            self.assertEqual([x['step'] for x in candidates(tmp,2)],[200])

    def test_external_worker_success_and_failure_are_separate_from_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);p=self.checkpoint(tmp);(tmp/'wandb_run_id.txt').write_text('training')
            args=SimpleNamespace(train_dir=str(tmp),length=8192,depths=[.1,.5,.9],max_new_tokens=24,threads=4,timeout=120,wandb=True)
            def fake(command,**kw):
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');self.assertNotIn('WANDB_RUN_ID',kw['env'])
                r=json.loads(Path(command[-1]).read_text());write_json(r['result'],{'status':'complete','summary':[]});return Mock()
            with patch('rwkv_rosa.native.standalone_needle.subprocess.run',side_effect=fake) as run,patch('rwkv_rosa.native.standalone_needle.upload_result') as upload:
                evaluate_one(p,tmp/'out',args);evaluate_one(p,tmp/'out',args)
                run.assert_called_once();upload.assert_called_once()
            status=json.loads((tmp/'out/step000000200/status.json').read_text());self.assertEqual(status['status'],'complete')
            with patch('rwkv_rosa.native.standalone_needle.subprocess.run',side_effect=RuntimeError('worker failure')):
                # Failure recorded, NOT raised into a training process.
                evaluate_one(p,tmp/'failed',args)
            self.assertEqual(json.loads((tmp/'failed/step000000200/status.json').read_text())['status'],'failed')

    def test_training_launcher_disables_needle_while_keeping_sft_eval(self):
        script=Path('scripts/run_rwkv_3b_postnorm_training.sh').read_text()
        self.assertIn('--rosa_needle_every_eval 0',script)
        self.assertNotIn('standalone_needle',script) # separate launch, no subprocess wait in trainer
        with self.assertRaises(SystemExit):main(['--watch'])


if __name__=='__main__':unittest.main()
