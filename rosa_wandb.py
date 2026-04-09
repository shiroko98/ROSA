import math
from typing import Any, Dict, Optional, Sequence


def _is_loggable_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and math.isfinite(float(value))


def prefixed_wandb_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key, value in metrics.items():
        if _is_loggable_scalar(value):
            payload[f"{prefix}/{key}" if prefix else key] = float(value)
    return payload


class WandbLogger:
    def __init__(self, run: Optional[Any] = None):
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_metrics(self, metrics: Dict[str, Any], *, step: Optional[int] = None, prefix: str = "") -> None:
        if not self.enabled:
            return
        payload = prefixed_wandb_metrics(metrics, prefix)
        if not payload:
            return
        if step is not None:
            payload.setdefault("trainer/global_step", float(step))
        self._run.log(payload, step=step)

    def log_raw(self, payload: Dict[str, Any], *, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        if step is not None:
            payload = dict(payload)
            payload.setdefault("trainer/global_step", float(step))
        self._run.log(payload, step=step)

    def update_summary(self, metrics: Dict[str, Any], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        for key, value in prefixed_wandb_metrics(metrics, prefix).items():
            self._run.summary[key] = value

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()


def init_wandb_logger(
    *,
    enabled: bool,
    is_main_process: bool,
    project: str,
    mode: str,
    out_dir: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    log_print=print,
) -> WandbLogger:
    if not enabled or not is_main_process or mode == "disabled":
        return WandbLogger()

    try:
        import wandb
    except Exception as exc:
        log_print(f"[wandb] 初始化失败，已降级为关闭：{exc}")
        return WandbLogger()

    run = wandb.init(
        project=project,
        entity=(entity or None),
        name=(name or None),
        group=(group or None),
        tags=list(tags or []),
        mode=mode,
        dir=out_dir,
        config=config,
    )
    try:
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")
    except Exception:
        pass
    return WandbLogger(run)
