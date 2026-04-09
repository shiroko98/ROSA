import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def atomic_torch_save(obj: Any, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_training_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    history: Dict[str, List[Dict[str, float]]]


class TrainingCheckpointManager:
    def __init__(
        self,
        *,
        out_dir: str,
        model_name: str,
        save_every_epochs: int,
        is_main_process: bool,
        keep_last_k: int = 2,
    ):
        self.model_name = model_name
        self.save_every_epochs = max(0, int(save_every_epochs))
        self.is_main_process = bool(is_main_process)
        self.keep_last_k = max(1, int(keep_last_k))
        self.checkpoint_dir = Path(out_dir) / "checkpoints" / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, epoch: int) -> bool:
        return self.is_main_process and self.save_every_epochs > 0 and epoch % self.save_every_epochs == 0

    def epoch_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"epoch_{epoch:03d}.pt"

    def last_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def save(
        self,
        *,
        epoch: int,
        global_step: int,
        history: Dict[str, List[Dict[str, float]]],
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        args: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.should_save(epoch):
            return
        payload = {
            "version": 1,
            "model_name": self.model_name,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "history": history,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "args": dict(args),
            "extra_state": extra_state or {},
        }
        epoch_path = self.epoch_path(epoch)
        atomic_torch_save(payload, str(epoch_path))
        atomic_torch_save(payload, str(self.last_path()))
        self._prune_old_epoch_checkpoints()

    def _prune_old_epoch_checkpoints(self) -> None:
        epoch_files = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if len(epoch_files) <= self.keep_last_k:
            return
        for path in epoch_files[:-self.keep_last_k]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
