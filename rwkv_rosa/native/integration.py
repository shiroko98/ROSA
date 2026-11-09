"""Small extension installed by upstream train.py after strict base-weight loading."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import pytorch_lightning as pl
import torch

from rwkv_rosa.freeze_audit import parameter_hash
from .position import resolve_position, normalize_metadata, scale_options_from_args
from .adapter import CPUImprovedROSA, CPUROSAOptimizer, configure_output_head, clip_cpu_gradients

UPSTREAM_COMMIT = '1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1'


def add_arguments(parser):
    parser.add_argument('--rosa', '--rosa_cpu', dest='rosa_cpu', type=int, default=0)
    parser.add_argument('--rosa_device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--rosa_address_impl', choices=('fast','compiled_cpu'), default='fast')
    parser.add_argument('--rosa_position', choices=('block', 'post_norm'), default='block',
                        help='Inject before selected blocks, or after final norm immediately before head.')
    parser.add_argument('--rosa_layers', default=None, help='Block indices only; omit for post_norm.')
    parser.add_argument('--rosa_windowed_data', type=int, choices=(0,1), default=0,
                        help='Explicit loss-preserving windows after physical-row train/eval split.')
    parser.add_argument('--rosa_scale', type=float, default=.05)
    parser.add_argument('--rosa_output_norm', choices=('none', 'rms', 'rms_match'), default='none',
                        help='Opt-in post_norm value RMS normalization with FP32 output/head math. rms_match additionally matches post-projection RMS to z. Legacy checkpoints use none.')
    parser.add_argument('--rosa_scale_mode', choices=('fixed', 'logit_gap'), default='fixed')
    parser.add_argument('--rosa_dynamic_max', type=float, default=.2)
    parser.add_argument('--rosa_dynamic_margin', type=float, default=.5)
    parser.add_argument('--rosa_dynamic_chunk', type=int, default=128)
    parser.add_argument('--rosa_dynamic_min_match', type=int, default=6)
    parser.add_argument('--rosa_eval_initial', type=int, choices=(0,1), default=0)
    parser.add_argument('--rosa_needle_every_eval', type=int, choices=(0,1), default=0)
    parser.add_argument('--rosa_needle_length', type=int, default=8192)
    parser.add_argument('--rosa_needle_depths', default='0.1,0.5,0.9')
    parser.add_argument('--rosa_needle_max_new_tokens', type=int, default=16)
    parser.add_argument('--rosa_needle_cpu_threads', type=int, default=4)
    parser.add_argument('--rosa_needle_timeout', type=int, default=3600)
    parser.add_argument('--rosa_no_checkpoint_layers', default='',
                        help='Comma-separated block indices to execute without activation recomputation.')
    parser.add_argument('--rosa_save', type=int, default=1, help='0 disables checkpoint output for diagnostic profiles only.')
    parser.add_argument('--rosa_resume', default='', help='Full Lightning .ckpt including ROSA optimizer.')
    parser.add_argument('--rosa_checkpoint_seconds', type=float, default=0.,
                        help='Wall-clock interval between full resumable checkpoints; 0 disables periodic saves.')
    parser.add_argument('--rosa_checkpoint_adapter_only', type=int, choices=(0,1), default=0,
                        help='Save learned ROSA+optimizers+loops; reference immutable base by verified hash.')
    parser.add_argument('--rosa_checkpoint_initial', type=int, choices=(0, 1), default=0,
                        help='Also save a full checkpoint after the first optimizer update.')


def frozen_parameters(model):
    return ((n, p) for n, p in model.named_parameters() if not n.startswith('rosa_cpu.'))


def frozen_hash(model):
    if any(p.requires_grad or p.grad is not None for _, p in frozen_parameters(model)):
        raise AssertionError('Native RWKV backbone is not completely frozen.')
    return parameter_hash(SimpleNamespace(named_parameters=lambda: frozen_parameters(model)))


def install(model, args):
    if not args.rosa_cpu:
        return None
    interval = getattr(args, 'rosa_checkpoint_seconds', 0.)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError('rosa_checkpoint_seconds must be finite and nonnegative.')
    if (interval or getattr(args, 'rosa_checkpoint_initial', 0)) and not getattr(args, 'rosa_save', 1):
        raise ValueError('Periodic/initial checkpoints require rosa_save=1.')
    if int(args.devices) != 1 or int(args.num_nodes) != 1 or 'deepspeed' in str(args.strategy):
        raise ValueError('CPU ROSA currently supports native single-GPU strategy only, not DeepSpeed/DDP.')
    if args.precision != 'bf16':
        raise ValueError('Native fused x070 kernels require --precision bf16; no silent reference fallback.')
    if not args.load_model or args.train_stage != 0 or args.load_partial:
        raise ValueError('CPU ROSA requires strict pretrained loading with train_stage=0, load_partial=0.')
    if args.resume_ckpt_path:
        raise ValueError('Use --rosa_resume native-last.ckpt for CPU optimizer resume.')
    if getattr(args, 'rosa_needle_every_eval', 0):
        depths = [float(x) for x in args.rosa_needle_depths.split(',')]
        if not depths or any(not math.isfinite(x) or not 0 < x < 1 for x in depths):
            raise ValueError('Needle depths must be finite interior fractions.')
        if min(args.rosa_needle_length, args.rosa_needle_max_new_tokens, args.rosa_needle_cpu_threads, args.rosa_needle_timeout) <= 0:
            raise ValueError('Needle monitor sizes/timeout must be positive.')
        if args.rosa_needle_length < 256:
            raise ValueError('Needle monitor needs at least 256 document tokens.')
    args.rosa_position, layers = resolve_position(getattr(args, 'rosa_position', 'block'),
                                                   args.rosa_layers, n_layer=args.n_layer)
    args.rosa_layers = ','.join(map(str, layers))
    no_checkpoint = tuple(int(i) for i in getattr(args, 'rosa_no_checkpoint_layers', '').split(',') if i.strip())
    if any(i < 0 or i >= args.n_layer for i in no_checkpoint) or len(set(no_checkpoint)) != len(no_checkpoint):
        raise ValueError('Invalid/duplicate rosa_no_checkpoint_layers.')
    args.rosa_no_checkpoint_layers = no_checkpoint
    # The supplied base checkpoint stores BF16 weights; all native custom kernels
    # require BF16 inputs/weights. Trainable CPU memory remains FP32 independently.
    model.requires_grad_(False)
    model.bfloat16()
    # Native fused ops require BF16 tensors. Lightning's BF16 AMP promotes
    # LayerNorm outputs to FP32 on the single-device path, unlike DeepSpeed's
    # BF16 engine. Disable outer AMP; this is NOT FP32 backbone execution:
    # backbone storage/math stays BF16 and ROSA explicitly stays FP32.
    args.precision = 32
    args.rosa_precision_policy = 'bf16_backbone_fp32_rosa_no_outer_amp'
    args.run_name += f' ROSA-{args.rosa_device}'
    os.environ['WANDB_DIR'] = args.proj_dir
    # Both placements use FP32 ROSA arithmetic (no TF32), while native kernels
    # retain their required BF16 execution. This avoids a precision confound.
    torch.backends.cuda.matmul.allow_tf32 = False
    model.rosa_cpu = CPUImprovedROSA(model.emb, layers, args.rosa_scale, device=args.rosa_device,
                                      address_impl=getattr(args, 'rosa_address_impl', 'fast'), position=args.rosa_position,
                                      output_norm=getattr(args, 'rosa_output_norm', 'none'), **scale_options_from_args(args))
    if model.rosa_cpu.scale_mode == 'logit_gap':
        model.rosa_cpu.bind_output_head(model.head)
    if model.rosa_cpu.output_norm in ('rms', 'rms_match'):
        configure_output_head(model.head, model.rosa_cpu.output_norm)
    def configure_optimizers(self):
        return CPUROSAOptimizer(self.rosa_cpu, lr=args.lr_init, betas=args.betas,
                                eps=args.adam_eps, weight_decay=args.weight_decay)
    def configure_gradient_clipping(self, optimizer, optimizer_idx=0, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        return clip_cpu_gradients(self.rosa_cpu.parameters(), float(gradient_clip_val or 0.))
    model.configure_optimizers = MethodType(configure_optimizers, model)
    model.configure_gradient_clipping = MethodType(configure_gradient_clipping, model)
    if args.rosa_resume:
        args.resume_ckpt_path = args.rosa_resume
    callback = NativeROSACallback(args)
    # PL 1.9 restores model weights BEFORE callback checkpoint hooks. Expand and
    # validate external-base checkpoints from the LightningModule hook, before
    # strict load_state_dict. A callback hook alone is too late.
    def on_load_checkpoint(self, checkpoint):
        callback.on_load_checkpoint(None, self, checkpoint)
    model.on_load_checkpoint = MethodType(on_load_checkpoint, model)
    return callback


def normalize_completed_epoch_checkpoint(checkpoint):
    """PL 1.9 keeps exhausted per-epoch batch counters in fit-end checkpoints.

    At a COMPLETED epoch only, clear current counters so restoring the NEXT
    epoch does not immediately skip its dataloader. Never touch lifetime totals
    or global_step. Partial epochs are intentionally left unchanged.
    """
    fit = checkpoint.get('loops', {}).get('fit_loop', {})
    batches = fit.get('epoch_loop.batch_progress', {})
    epochs = fit.get('epoch_progress', {}).get('current', {})
    if not batches.get('is_last_batch') or epochs.get('completed', 0) != epochs.get('ready', -1):
        return False
    for key in batches.get('current', {}):
        batches['current'][key] = 0
    batches['is_last_batch'] = False
    progress = fit.get('epoch_loop.batch_loop.optimizer_loop.optim_progress', {})
    progress['optimizer_position'] = 0
    for tracker in progress.get('optimizer', {}).values():
        if isinstance(tracker, dict) and 'current' in tracker:
            for key in tracker['current']:
                tracker['current'][key] = 0
    return True


class NativeROSACallback(pl.Callback):
    def __init__(self, args):
        self.args = args
        self.out = Path(args.proj_dir)
        self.audit = {}
        self.last_step = 0
        self.started = None
        self.forward_seconds = self.backward_seconds = 0.
        self.last_checkpoint_time = None
        self.initial_checkpoint_saved = False
        self.last_saved_step = None

    def save_full_checkpoint(self, trainer, reason):
        """Save only at optimizer boundaries, atomically publish file and latest link.

        PL calls batch-end callbacks before incrementing its completed-batch
        counter. Temporarily mark the processed batch complete, as the upstream
        save helper does, without changing the running loop counters.
        """
        folder = self.out / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"native-{reason}-step{trainer.global_step:09d}-{stamp}.ckpt"
        temporary = path.with_suffix(".ckpt.tmp")
        progress = getattr(getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None), "batch_progress", None)
        snapshot = []
        for name in ("total", "current"):
            tracker = getattr(progress, name, None)
            if tracker is not None and hasattr(tracker, "completed"):
                snapshot.append((tracker, tracker.completed))
                tracker.completed = max(tracker.completed, getattr(tracker, "processed", tracker.completed))
        start = time.monotonic()
        try:
            trainer.save_checkpoint(str(temporary))
            temporary.replace(path)
        finally:
            for tracker, completed in snapshot:
                tracker.completed = completed
        # Relative symlink is valid if the entire run directory is moved. Never
        # overwrite an old hourly file through the latest symlink.
        link = self.out / "native-last.ckpt.tmp"
        link.unlink(missing_ok=True)
        link.symlink_to(path.relative_to(self.out))
        link.replace(self.out / "native-last.ckpt")
        record = {"step": trainer.global_step, "reason": reason, "path": str(path),
                  "utc": stamp, "bytes": path.stat().st_size,
                  "save_seconds": time.monotonic() - start}
        with (self.out / "native_checkpoints.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.last_checkpoint_time = time.monotonic()
        self.last_saved_step = trainer.global_step
        print("NATIVE ROSA CHECKPOINT:", json.dumps(record), flush=True)
        return path

    def write(self, name, value):
        p = self.out / name
        temp = p.with_suffix(p.suffix + '.tmp')
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(p)

    def on_save_checkpoint(self, trainer, model, checkpoint):
        checkpoint['native_rosa'] = {'ctx_len': self.args.ctx_len, 'data_file': self.args.data_file,
                                     'epoch_steps': self.args.epoch_steps, 'layers': self.args.rosa_layers,
                                     'scale': self.args.rosa_scale, 'position': getattr(self.args, 'rosa_position', 'block'),
                                     'output_norm': getattr(self.args, 'rosa_output_norm', 'none'), **scale_options_from_args(self.args)}
        checkpoint['native_rosa_data_policy'] = {'virtual_windows': int(getattr(self.args, 'rosa_windowed_data', 0)),
                                                  'micro_bsz': int(getattr(self.args, 'micro_bsz', 1)),
                                                  'accumulate_grad_batches': int(getattr(self.args, 'accumulate_grad_batches', 1))}
        if getattr(self.args, 'rosa_checkpoint_adapter_only', 0):
            checkpoint['native_rosa_checkpoint_kind'] = 'adapter_optimizer_loop_with_external_base_v1'
            checkpoint['native_rosa_frozen_base'] = {'path': self.args.load_model,
                'sha256': self.audit.get('backbone_sha256_before') or frozen_hash(model),
                'parameter_names': [n for n, _ in frozen_parameters(model)]}
            checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items() if k.startswith('rosa_cpu.')}
        normalize_completed_epoch_checkpoint(checkpoint)

    def on_load_checkpoint(self, trainer, model, checkpoint):
        expected = {'ctx_len': self.args.ctx_len, 'data_file': self.args.data_file,
                    'epoch_steps': self.args.epoch_steps, 'layers': self.args.rosa_layers, 'scale': self.args.rosa_scale, 'position': getattr(self.args, 'rosa_position', 'block'),
                                     'output_norm': getattr(self.args, 'rosa_output_norm', 'none'), **scale_options_from_args(self.args)}
        policy = {'virtual_windows': int(getattr(self.args, 'rosa_windowed_data', 0)),
                  'micro_bsz': int(getattr(self.args, 'micro_bsz', 1)),
                  'accumulate_grad_batches': int(getattr(self.args, 'accumulate_grad_batches', 1))}
        if 'native_rosa_data_policy' in checkpoint and checkpoint['native_rosa_data_policy'] != policy:
            raise ValueError('Native ROSA window/batch/accumulation policy changed on resume.')
        if 'native_rosa' not in checkpoint:
            raise ValueError('ROSA resume requires explicit adapter metadata; cannot infer injection position.')
        if normalize_metadata(checkpoint['native_rosa']) != normalize_metadata(expected):
            raise ValueError('Native ROSA resume data/context/epoch size/fusion semantics changed.')
        if checkpoint.get('native_rosa_checkpoint_kind') == 'adapter_optimizer_loop_with_external_base_v1':
            from rwkv_rosa.backbone import load_base_state
            base = load_base_state(self.args.load_model)
            from rwkv_rosa.native.cpu_compare import state_hash
            fingerprint = state_hash(base, checkpoint['native_rosa_frozen_base'].get('parameter_names'))
            if fingerprint != checkpoint['native_rosa_frozen_base']['sha256']:
                raise ValueError('External frozen base does not match adapter checkpoint hash.')
            checkpoint['state_dict'] = {**base, **checkpoint['state_dict']}
        normalize_completed_epoch_checkpoint(checkpoint)

    def on_fit_start(self, trainer, model):
        if not all(p.device.type == self.args.rosa_device and p.dtype == torch.float32 for p in model.rosa_cpu.parameters()):
            raise AssertionError('Lightning moved/cast the pinned ROSA module.')
        if not all(p.device.type == 'cuda' for _, p in frozen_parameters(model)):
            raise AssertionError('Frozen native backbone must be entirely CUDA.')
        self.last_step = trainer.global_step
        self.last_checkpoint_time = time.monotonic()
        self.audit = {'upstream_commit': UPSTREAM_COMMIT, 'entrypoint': 'native upstream train.py',
                      'model': 'native src.model.RWKV', 'dataset': 'native src.dataset.MyDataset',
                      'trainer': f'pytorch_lightning.Trainer {pl.__version__}',
                      'kernel': os.environ['RWKV_KERNEL'], 'my_testing': os.environ['RWKV_MY_TESTING'],
                      'backbone_parameters': sum(p.numel() for _, p in frozen_parameters(model)),
                      'trainable_cpu_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad and p.device.type == 'cpu'),
                      'trainable_cuda_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad and p.device.type == 'cuda'),
                      'rosa_position': model.rosa_cpu.position, 'rosa_layers': list(model.rosa_cpu.layers),
                      'rosa_output_norm': model.rosa_cpu.output_norm, 'rosa_scale_options': model.rosa_cpu.scale_options,
                      'output_head_math': 'float32' if model.rosa_cpu.output_norm in ('rms', 'rms_match') else 'backbone_dtype',
                      'rosa_device': self.args.rosa_device, 'sam_address_device': 'cpu', 'sam_address_impl': model.rosa_cpu.address_impl,
                      'sam_address_algorithm': model.rosa_cpu.address_algorithm,
                      'no_checkpoint_layers': list(getattr(self.args, 'rosa_no_checkpoint_layers', ())),
                      'backbone_dtype': str(model.emb.weight.dtype), 'rosa_dtype': str(model.rosa_cpu.gate_bias.dtype),
                      'backbone_sha256_before': frozen_hash(model),
                      'rosa_values_sha256_before': parameter_hash(model.rosa_cpu.rosa_value_store),
                      'base_checkpoint': self.args.load_model, 'ctx_len': self.args.ctx_len, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'precision_policy': self.args.rosa_precision_policy,
                      'precision_note': 'native BF16 CUDA execution without outer AMP; FP32 ROSA; no reference/streaming equivalence claim'}
        model._rosa_frozen_sha256 = self.audit['backbone_sha256_before']
        self.write('native_audit.json', self.audit)
        print('NATIVE ROSA AUDIT:', json.dumps(self.audit), flush=True)
        if getattr(self.args, 'rosa_eval_initial', 0):
            for callback in trainer.callbacks:
                if hasattr(callback, '_run_sft_eval'):
                    callback._run_sft_eval(trainer, model, int(trainer.global_step))

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        torch.cuda.synchronize()
        if self.started is None:
            self.started = time.perf_counter()
            self.forward_seconds = self.backward_seconds = 0.
            torch.cuda.reset_peak_memory_stats()
        self.batch_started = time.perf_counter()
        self.mask_tokens = float(batch[2].sum())
        self.batch_hash = hashlib.sha256(batch[0].detach().cpu().numpy().tobytes()
                                         + batch[2].detach().cpu().numpy().tobytes()).hexdigest()

    def on_before_backward(self, trainer, model, loss):
        torch.cuda.synchronize()
        self.forward_seconds += time.perf_counter() - self.batch_started
        self.backward_started = time.perf_counter()

    def on_after_backward(self, trainer, model):
        torch.cuda.synchronize()
        self.backward_seconds += time.perf_counter() - self.backward_started
        if any(p.grad is not None for _, p in frozen_parameters(model)):
            raise AssertionError('Frozen native backbone received gradients.')
        grads = [p.grad for p in model.rosa_cpu.parameters() if p.grad is not None]
        if not grads or any(g.device.type != self.args.rosa_device for g in grads):
            raise AssertionError('Native kernels did not deliver gradients to configured ROSA device.')

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.global_step <= self.last_step:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        effective_batch = getattr(self.args, 'effective_bsz', batch[0].shape[0])
        metrics = {'step': trainer.global_step, 'micro_batch_size': int(batch[0].shape[0]),
                   'effective_batch_size': effective_batch,
                   'input_tokens_per_second': effective_batch * self.args.ctx_len / elapsed,
                   'lr': float(getattr(trainer, 'my_lr', 0.)),
                   'grad_norm': float(getattr(trainer, 'my_grad_norm', 0.)), 'ctx_len': self.args.ctx_len,
                   'seconds': elapsed, 'rosa_device': self.args.rosa_device, 'forward_seconds': self.forward_seconds,
                   'backward_seconds': self.backward_seconds,
                   'peak_gpu_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                   'peak_gpu_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
                   'loss': float(getattr(trainer, 'my_loss', 0.)),
                   'batch_sha256': self.batch_hash, 'last_microbatch_supervised_tokens': self.mask_tokens, **model.rosa_cpu.stats}
        optimizer = trainer.optimizers[0]
        optimizer.assert_states_device(self.args.rosa_device)
        if 'backbone_sha256_after_first_update' not in self.audit:
            self.audit['backbone_sha256_after_first_update'] = frozen_hash(model)
            if self.audit['backbone_sha256_before'] != self.audit['backbone_sha256_after_first_update']:
                raise AssertionError('Frozen native backbone parameter bytes changed.')
            changed = parameter_hash(model.rosa_cpu.rosa_value_store) != self.audit['rosa_values_sha256_before']
            if not changed and model.rosa_cpu.scale_mode != 'logit_gap':
                raise AssertionError('CPU ROSA values did not update.')
            # A legitimately disabled dynamic branch can have a zero-gradient
            # first batch. Record that honestly instead of failing or claiming
            # its value table changed; observe later nonzero updates below.
            self.audit.update(rosa_values_changed=changed, optimizer_states_device=self.args.rosa_device,
                              backbone_gradients_absent=True)
            self.write('native_audit.json', self.audit)
        if not self.audit.get('rosa_values_changed') and metrics['grad_norm'] > 0:
            changed = parameter_hash(model.rosa_cpu.rosa_value_store) != self.audit['rosa_values_sha256_before']
            if changed:
                self.audit.update(rosa_values_changed=True, rosa_values_first_observed_update=trainer.global_step)
                self.write('native_audit.json', self.audit)
        with (self.out / 'native_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        if getattr(trainer, 'my_wandb', None) is not None:
            trainer.my_wandb.log({f'native/{k}': v for k, v in metrics.items() if isinstance(v, (float, int))}, step=trainer.global_step)
        print('NATIVE ROSA STEP:', json.dumps(metrics), flush=True)
        self.last_step = trainer.global_step
        self.started = None
        self.maybe_save_checkpoint(trainer)
        if (self.out / 'STOP').exists():
            if self.args.rosa_save and self.last_saved_step != trainer.global_step:
                self.save_full_checkpoint(trainer, 'user-stop')
            trainer.should_stop = True
            print('NATIVE ROSA STOP requested at completed optimizer boundary.',flush=True)

    def maybe_save_checkpoint(self, trainer):
        initial = getattr(self.args, 'rosa_checkpoint_initial', 0) and not self.initial_checkpoint_saved
        interval = getattr(self.args, 'rosa_checkpoint_seconds', 0.)
        due = interval > 0 and time.monotonic() - self.last_checkpoint_time >= interval
        if initial or due:
            self.save_full_checkpoint(trainer, 'initial' if initial else 'hourly')
            self.initial_checkpoint_saved = True

    def on_fit_end(self, trainer, model):
        after = frozen_hash(model)
        if after != self.audit['backbone_sha256_before']:
            raise AssertionError('Frozen backbone changed by end of native fit.')
        self.audit.update(backbone_sha256_final=after, backbone_unchanged=True, global_step=trainer.global_step)
        self.write('native_audit.json', self.audit)
        if self.args.rosa_save and self.last_saved_step != trainer.global_step:
            if getattr(self.args, 'rosa_checkpoint_seconds', 0.) or getattr(self.args, 'rosa_checkpoint_initial', 0):
                self.save_full_checkpoint(trainer, 'final')
            else:
                trainer.save_checkpoint(str(self.out / 'native-last.ckpt'))
        if getattr(trainer, 'my_wandb', None) is not None:
            trainer.my_wandb.finish()

    def on_exception(self, trainer, model, exception):
        self.write('native_failure.json', {'type': type(exception).__name__, 'message': str(exception),
                                         'global_step': trainer.global_step,
                                         'peak_gpu_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30})
        if getattr(trainer, 'my_wandb', None) is not None:
            trainer.my_wandb.finish(exit_code=1)
