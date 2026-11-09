"""Compact saves must not be followed by redundant frozen-body/final snapshots."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile
import unittest
from rwkv_rosa.native.integration import NativeROSACallback


class CompactSavePolicyTest(unittest.TestCase):
    def test_actual_native_save_helper_defers_only_compact_rosa_to_its_callback(self):
        source=Path('third_party/rwkv_lm_v7_native/src/trainer.py')
        method=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='save_train_checkpoint')
        scope={name:Mock() for name in ('mark_current_batch_completed_for_checkpoint','my_save','build_save_dict','prune_old_checkpoints','restore_progress_snapshot')}
        scope['is_deepspeed_strategy']=lambda _:False
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source.resolve()),'exec'),scope)
        trainer=SimpleNamespace(is_global_zero=False)
        args=SimpleNamespace(rosa_cpu=1,rosa_save=1,rosa_checkpoint_adapter_only=1,strategy='auto')
        scope['save_train_checkpoint'](args,trainer,None,'never-full-body.pth')
        scope['my_save'].assert_not_called()
        args.rosa_checkpoint_adapter_only=0
        scope['save_train_checkpoint'](args,trainer,None,'legacy-full.pth')
        scope['my_save'].assert_called_once()
        scope['my_save'].reset_mock();args.rosa_save=0
        scope['save_train_checkpoint'](args,trainer,None,'disabled.pth')
        scope['my_save'].assert_not_called()

    def test_same_step_user_stop_checkpoint_is_not_saved_again_at_fit_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback=NativeROSACallback(SimpleNamespace(proj_dir=tmp,rosa_save=1,rosa_checkpoint_seconds=3600,rosa_checkpoint_initial=1))
            callback.audit={'backbone_sha256_before':'frozen'};callback.last_saved_step=4
            trainer=SimpleNamespace(global_step=4,my_wandb=None)
            callback.save_full_checkpoint=Mock()
            with patch('rwkv_rosa.native.integration.frozen_hash',return_value='frozen'):
                callback.on_fit_end(trainer,None)
                callback.save_full_checkpoint.assert_not_called()
                trainer.global_step=5;callback.on_fit_end(trainer,None)
                callback.save_full_checkpoint.assert_called_once_with(trainer,'final')
