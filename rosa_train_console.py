from typing import Any, Dict


def format_train_step_console_line(
    *,
    run_name: str,
    epoch: int,
    global_step: int,
    metrics: Dict[str, Any],
    grad_accum_steps: int,
) -> str:
    parts = [
        f"[{run_name}]",
        f"epoch {epoch:02d}",
        f"step {global_step}",
        f"loss {float(metrics.get('loss', 0.0)):.4f}",
        f"ppl {float(metrics.get('ppl', 0.0)):.4f}",
        f"acc {float(metrics.get('token_acc', 0.0)):.4f}",
        f"tokens {int(metrics.get('valid_tokens', 0.0))}",
        f"accum {grad_accum_steps}",
    ]
    if "tokens_per_s_wall" in metrics:
        parts.append(f"tok/s {float(metrics['tokens_per_s_wall']):.1f}")
    if "step_wall_ms" in metrics:
        parts.append(f"step_ms {float(metrics['step_wall_ms']):.1f}")
    if "optimizer_lr_dense" in metrics:
        parts.append(f"lr {float(metrics['optimizer_lr_dense']):.3e}")
    if "grad_norm_dense" in metrics:
        parts.append(f"grad {float(metrics['grad_norm_dense']):.3f}")
    if "cuda_max_memory_allocated_mb" in metrics:
        parts.append(f"max_mem {float(metrics['cuda_max_memory_allocated_mb']):.0f}MB")
    if "rosa_fire_coverage" in metrics:
        parts.append(f"fire {float(metrics['rosa_fire_coverage']):.3f}")
    if "rosa_avg_gate" in metrics:
        parts.append(f"gate {float(metrics['rosa_avg_gate']):.3f}")
    if "timing_model_rosa_address_ms" in metrics:
        parts.append(f"rosa_addr {float(metrics['timing_model_rosa_address_ms']):.2f}ms")
    if "timing_async_prefetch_prepare_ms" in metrics:
        parts.append(f"async_prep {float(metrics['timing_async_prefetch_prepare_ms']):.2f}ms")
    return " | ".join(parts)
