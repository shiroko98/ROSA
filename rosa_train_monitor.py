import time
from typing import Dict

import torch


def reset_cuda_peak_memory(device: torch.device, *, enabled: bool) -> None:
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def collect_step_system_metrics(
    *,
    device: torch.device,
    step_started_at: float,
    step_tokens: int,
    total_tokens_seen: int,
    epoch: int,
    batch_idx: int,
    num_batches: int,
    log_cuda_memory: bool,
) -> Dict[str, float]:
    elapsed_s = max(time.perf_counter() - step_started_at, 1e-12)
    metrics = {
        "epoch": float(epoch),
        "epoch_progress": float(batch_idx) / max(1.0, float(num_batches)),
        "step_wall_ms": elapsed_s * 1000.0,
        "tokens_per_s_wall": float(step_tokens) / elapsed_s,
        "tokens_seen_total": float(total_tokens_seen),
    }
    if log_cuda_memory and device.type == "cuda" and torch.cuda.is_available():
        metrics.update(
            {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        )
    return metrics
