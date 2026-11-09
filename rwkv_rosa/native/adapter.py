"""Device-pinned improved ROSA; native RWKV model/CUDA operators stay upstream."""
from __future__ import annotations

import math
import time
import weakref
import torch
from torch import nn

from rosa_addressing import RosaAddressEngine
from rosa_value_store import RosaValueStore
from rwkv_rosa.model import RMSNorm
from .position import resolve_position, validate_output_norm, validate_scale_options


class CPUImprovedROSA(nn.Module):
    def __init__(self, embedding, layers=None, scale=.05, device="cpu", address_impl="fast", position="block", output_norm="none",
                 scale_mode="fixed", dynamic_max=.2, dynamic_margin=.5, dynamic_chunk=128, dynamic_min_match=4):
        super().__init__()
        vocab, dim = embedding.weight.shape
        self.execution_device = torch.device(device)
        self.position, self.layers = resolve_position(position, layers)
        self.output_norm = validate_output_norm(self.position, output_norm)
        self.scale_options = validate_scale_options(self.position, self.output_norm, dict(
            scale_mode=scale_mode, dynamic_max=dynamic_max, dynamic_margin=dynamic_margin, dynamic_chunk=dynamic_chunk, dynamic_min_match=dynamic_min_match))
        self.scale_mode = scale_mode
        self._output_head_ref = None
        if scale_mode == 'logit_gap':
            self.register_buffer('_dynamic_scale_version', torch.tensor(1, dtype=torch.int64))
        if self.output_norm in ('rms', 'rms_match'):
            # A weights-only snapshot cannot silently load as legacy when a
            # publisher accidentally omits its normalization metadata.
            self.register_buffer('_output_norm_version', torch.tensor(1, dtype=torch.int64))
        if self.output_norm == 'rms_match':
            self.register_buffer('_output_alignment_version', torch.tensor(1, dtype=torch.int64))
        self.dim, self.scale = dim, scale
        count = len(self.layers) if self.position == "block" else 1
        self.slots = {layer: i for i, layer in enumerate(self.layers)}
        self.address_impl = address_impl
        self.address_algorithm = 'python_eager_suffix_updates'
        if address_impl == 'compiled_cpu':
            from rosa_cpp_extension import load_rosa_sam_cpu_extension
            self.address_algorithm = str(getattr(load_rosa_sam_cpu_extension(), 'algorithm_version', 'legacy_eager_suffix_updates'))
        self.address_engine = RosaAddressEngine(min_match_len=1, pad_id=0,
            special_ids={0, 65530, 65531, 65532, 65533, 65534}, forbid_special_target=True,
            backend='sam', sequence_mode='online_sam', online_sam_impl=address_impl)
        self.rosa_value_store = RosaValueStore(vocab_size=vocab, dim=dim, inject_layers=count,
                                             mode='per_layer', sparse_training=True, num_shards=4)
        self.rosa_value_store.copy_shared_weights_(embedding)
        self.hidden_norms = nn.ModuleList([RMSNorm(dim) for _ in range(count)])
        self.value_norms = nn.ModuleList([RMSNorm(dim) for _ in range(count)])
        self.key_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(count)])
        self.value_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(count)])
        for projection in [*self.key_projs, *self.value_projs]:
            nn.init.eye_(projection.weight)
        self.gate_bias = nn.Parameter(torch.zeros(count))
        self.length_scale = nn.Parameter(torch.ones(count))
        self.addresses = None
        self.stats = {}
        self._apply(lambda t: t)

    def _apply(self, fn, recurse=True):
        # Lightning strategy/precision conversion recursively visits child modules.
        # This explicit fixed-device module must NEVER follow the backbone to CUDA
        # or become BF16. Parameter identity is retained before/after optimizer setup.
        return super()._apply(lambda t: t.to(device=self.execution_device, dtype=torch.float32 if t.is_floating_point() else t.dtype),
                              recurse=recurse)

    def bind_output_head(self, head):
        # A weak reference avoids registering the frozen head under rosa_cpu,
        # accidentally including it in adapter checkpoints/optimizer parameters.
        self._output_head_ref = weakref.ref(head)

    def begin(self, idx):
        cpu_ids = idx.detach().cpu()
        started = time.perf_counter()
        self.addresses = self.address_engine.forward_seq(cpu_ids, device=torch.device('cpu'))
        address_seconds = time.perf_counter() - started
        a = self.addresses
        self.stats = {'rosa_address_seconds': address_seconds, 'rosa_coverage': float((a.valid_mask & cpu_ids.ne(0)).sum()) / max(1, int(cpu_ids.ne(0).sum())),
                      'rosa_active_addresses': int(torch.unique(a.addr_ids[a.valid_mask]).numel())}

    def inject(self, layer, x):
        if self.position != "block" or layer not in self.slots:
            return x
        return self._inject_slot(self.slots[layer], x)

    def inject_post_norm(self, x):
        if self.position != "post_norm":
            return x
        return self._inject_slot(0, x)

    def _inject_slot(self, slot, x):
        if self.addresses is None:
            raise RuntimeError('ROSA.begin must run once per native model forward.')
        a = self.addresses.to(self.execution_device)
        with torch.autocast(device_type=self.execution_device.type, enabled=False):
            # Both copies are differentiable; never detach the hidden or residual.
            local = x.to(device=self.execution_device, dtype=torch.float32)
            value = self.rosa_value_store.lookup(slot, a.addr_ids, shared_embedding=None)
            key = self.value_norms[slot](self.key_projs[slot](value))
            score = (self.hidden_norms[slot](local) * key).sum(-1) / math.sqrt(self.dim)
            score = score + self.length_scale[slot] * torch.log1p(a.fired_match_lens.float())
            gate = torch.sigmoid(score + self.gate_bias[slot])
            # The key norm above only calibrates the gate. Normalize the value
            # independently BEFORE its output projection, so projection amplitude
            # remains learnable. Legacy checkpoints retain the old raw-value path.
            output_value = value
            if self.output_norm in ('rms', 'rms_match'):
                output_value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            direction = self.value_projs[slot](output_value)
            if self.output_norm == 'rms_match':
                # Match the actual per-token head input scale, AFTER projection,
                # BEFORE gate/gain. Never normalize the sum or erase the gate.
                reference_rms = local.detach().square().mean(-1, keepdim=True).sqrt()
                direction = direction * torch.rsqrt(direction.square().mean(-1, keepdim=True) + 1e-6)
                direction = direction * reference_rms
            if self.scale_mode == 'logit_gap':
                from .dynamic_scale import logit_gap_scales
                head = None if self._output_head_ref is None else self._output_head_ref()
                if head is None:
                    raise RuntimeError('Dynamic ROSA requires bind_output_head(frozen_head).')
                eligible = a.valid_mask & (a.fired_match_lens >= self.scale_options['dynamic_min_match'])
                gain, useful = logit_gap_scales(x, direction, a.addr_ids, eligible, head,
                    maximum=self.scale_options['dynamic_max'], margin=self.scale_options['dynamic_margin'],
                    chunk_size=self.scale_options['dynamic_chunk'])
                # Controller is detached to avoid optimizing the denominator.
                # The gate and output direction retain their ordinary CE gradients.
                effective = gain * gate
                residual = direction * effective.unsqueeze(-1)
                active = a.valid_mask
                self.stats.update(rosa_dynamic_scale_mean=float(effective[active].detach().mean()) if active.any() else 0.,
                    rosa_dynamic_scale_max=float(effective.detach().max()),
                    rosa_dynamic_eligible_fraction=float(eligible[active].float().mean()) if active.any() else 0.,
                    rosa_dynamic_useful_fraction=float(useful[active].float().mean()) if active.any() else 0.)
            else:
                residual = self.scale * (direction * gate.unsqueeze(-1))
            residual = residual * a.valid_mask.unsqueeze(-1)
            if self.output_norm == 'rms_match':
                relative = residual.detach().norm(dim=-1) / math.sqrt(self.dim) / reference_rms.squeeze(-1).clamp_min(1e-12)
                self.stats['rosa_residual_relative_rms_mean'] = float(relative[a.valid_mask].mean()) if a.valid_mask.any() else 0.
                self.stats['rosa_residual_relative_rms_max'] = float(relative.max())
            self.stats['rosa_avg_gate'] = float(gate[a.valid_mask].detach().mean()) if a.valid_mask.any() else 0.
        if self.output_norm in ('rms', 'rms_match'):
            # Do not round the small output correction back to BF16. The native
            # head/loss path must also decode these features in FP32.
            self.stats['rosa_residual_rms'] = float(residual.detach().norm() / math.sqrt(residual.numel()))
            return x.float() + residual.to(device=x.device)
        return x + residual.to(device=x.device, dtype=x.dtype)


def configure_output_head(head, output_norm='none'):
    """FP32 output math without changing frozen parameter bytes/checkpoint hashes.

    Applied to BOTH paired evaluation arms. Casting only after a BF16 GEMM would
    not recover the lost correction. Native fused CE separately casts its weight.
    """
    if output_norm not in ('rms', 'rms_match'):
        return
    from types import MethodType
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.nn.functional.linear(x.float(), self.weight.float(),
                None if self.bias is None else self.bias.float())
    head.forward = MethodType(forward, head)


def gradient_norm(parameters):
    values = [p.grad.coalesce().values() if p.grad.is_sparse else p.grad
              for p in parameters if p.grad is not None]
    if not values:
        return 0.
    if len({v.device for v in values}) != 1:
        raise AssertionError('ROSA gradients must share the configured execution device.')
    return float(torch.linalg.vector_norm(torch.stack([v.detach().float().norm() for v in values])))


def clip_cpu_gradients(parameters, maximum):
    params = list(parameters)
    for p in params:
        if p.grad is not None and p.grad.is_sparse:
            p.grad = p.grad.coalesce()
    norm = gradient_norm(params)
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite CPU ROSA gradient norm.')
    scale = min(1., maximum / (norm + 1e-6)) if maximum > 0 else 1.
    for p in params:
        if p.grad is not None:
            (p.grad.values() if p.grad.is_sparse else p.grad).mul_(scale)
    return norm


class CPUROSAOptimizer(torch.optim.Optimizer):
    """One Lightning optimizer, CPU AdamW + SparseAdam sharing LR-schedule groups."""
    def __init__(self, branch, *, lr, betas=(.9, .99), eps=1e-8, weight_decay=0.):
        sparse = list(branch.rosa_value_store.sparse_parameters())
        sparse_ids = {id(p) for p in sparse}
        dense = [p for p in branch.parameters() if p.requires_grad and id(p) not in sparse_ids]
        super().__init__([*dense, *sparse], {'lr': lr})
        self.dense = torch.optim.AdamW(dense, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.sparse = torch.optim.SparseAdam(sparse, lr=lr, betas=betas, eps=eps)
        self.param_groups = self.dense.param_groups + self.sparse.param_groups
        for group in self.param_groups:
            group.setdefault('weight_decay', 0.)
            group['my_lr_scale'] = 1.

    def step(self, closure=None):
        # Lightning's closure owns native training_step, loss scaling and backward.
        # It MUST execute under enable_grad, before either child optimizer updates.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.dense.step()
        self.sparse.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.dense.zero_grad(set_to_none=set_to_none)
        self.sparse.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'format': 'cpu_rosa_adamw_sparseadam_v1', 'dense': self.dense.state_dict(),
                'sparse': self.sparse.state_dict()}

    def load_state_dict(self, state_dict):
        if state_dict.get('format') != 'cpu_rosa_adamw_sparseadam_v1':
            raise ValueError('Not a native CPU ROSA optimizer checkpoint.')
        self.dense.load_state_dict(state_dict['dense'])
        self.sparse.load_state_dict(state_dict['sparse'])
        self.param_groups = self.dense.param_groups + self.sparse.param_groups

    def assert_states_device(self, device):
        for optimizer in (self.dense, self.sparse):
            for state in optimizer.state.values():
                # Adam's scalar step counter intentionally stays on CPU even for CUDA parameters.
                if any(v.device.type != torch.device(device).type for k, v in state.items()
                       if isinstance(v, torch.Tensor) and k != 'step'):
                    raise AssertionError('ROSA optimizer state differs from configured device.')

# Historical names retained for checkpoint/type compatibility; placement is explicit.
NativeRosaAdapter = CPUImprovedROSA
NativeRosaOptimizer = CPUROSAOptimizer
