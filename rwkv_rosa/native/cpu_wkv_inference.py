"""Opt-in compiled FP32 CPU recurrence for independent inference workers."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[2]


def install_cpu_wkv(backbone):
    source = ROOT / 'cpp_extensions/rwkv_wkv_cpu_inference.cpp'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = ROOT / '.codex-build/cpu_wkv_inference'
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / f'wkv_{digest}.so'
    with (folder / 'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix('.tmp.so')
            subprocess.run(['g++', '-O3', '-march=native', '-ffp-contract=off',
                            '-fopenmp', '-shared', '-fPIC', str(source), '-o', str(temporary)], check=True)
            temporary.replace(library)
    module = ctypes.CDLL(str(library))
    operation = module.rwkv_wkv_cpu
    operation.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5
    operation.restype = None
    runtime = backbone._reference_runtime

    def forward(r, w, k, v, a, b, *, state=None, return_state=False):
        if torch.is_grad_enabled() or r.device.type != 'cpu':
            raise RuntimeError('Compiled CPU WKV is inference-only; disable autograd and hide CUDA.')
        batch, steps, channels = r.shape
        size = runtime.HEAD_SIZE
        heads = channels // size
        shape = (batch, heads, size, size)
        if channels % size or any(x.shape != r.shape for x in (w, k, v, a, b)):
            raise ValueError('Invalid WKV input dimensions')
        if state is not None and tuple(state.shape) != shape:
            raise ValueError('Invalid WKV state dimensions')
        arrays = [x.float().contiguous() for x in (r, torch.exp(-torch.exp(w.float())), k, v, a, b)]
        next_state = torch.zeros(shape) if state is None else state.float().contiguous().clone()
        output = torch.empty(r.shape, dtype=torch.float32)
        operation(*(x.data_ptr() for x in [*arrays, next_state, output]),
                  batch, steps, heads, size, torch.get_num_threads())
        output = output.to(runtime.DTYPE)
        return (output, next_state) if return_state else output

    runtime.RWKV7_OP = forward
    backbone._cpu_wkv_library = module
    return forward
