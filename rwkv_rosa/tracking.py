"""Per-arm W&B metrics, fail-closed online initialization, persistent resume IDs.

Only numeric metrics/configuration are uploaded. No corpus samples, token arrays,
checkpoints, or full document-hash lists are attached to W&B.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid


class ArmTracker:
    def __init__(self, args, variant, out_dir, signature):
        self.run = None
        self.url = None
        if not args.wandb or args.wandb_mode == 'disabled':
            return
        import wandb  # Requested logging must never silently degrade to disabled.
        identity = Path(out_dir) / 'wandb_run.json'
        previous = json.loads(identity.read_text()) if identity.exists() else None
        run_id = previous['id'] if previous else uuid.uuid4().hex[:12]
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity or None,
            group=args.wandb_group or Path(args.out_dir).name,
            name=f'{args.wandb_run_name or Path(args.out_dir).name}-{variant}',
            id=run_id, resume='must' if previous and args.wandb_mode == 'online' else 'never',
            mode=args.wandb_mode, dir=str(out_dir),
            config={**vars(args), 'variant': variant, 'manifest_arm_signature': signature,
                    'wkv_precision_policy': 'fp32_autocast_disabled',
                    'quality_claim': 'bounded training comparison; not full-corpus/multi-seed evidence'},
            save_code=False, settings=wandb.Settings(disable_git=True),
        )
        if self.run is None or (args.wandb_mode == 'online' and self.run.offline):
            raise RuntimeError('Online W&B requested but no online run was established.')
        self.url = self.run.url if not self.run.offline else None
        identity.write_text(json.dumps({'id': self.run.id, 'url': self.url,
                                        'project': self.run.project, 'entity': self.run.entity}, indent=2) + '\n')
        self.run.define_metric('trainer/global_step')
        self.run.define_metric('*', step_metric='trainer/global_step')
        print(f'W&B {variant}: {self.url or "offline"}', flush=True)

    def log(self, prefix, metrics, step):
        if self.run is None:
            return
        payload = {f'{prefix}/{k}': v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
        self.run.log({**payload, 'trainer/global_step': step})

    def summary(self, prefix, metrics):
        if self.run is not None:
            self.run.summary.update({f'{prefix}/{k}': v for k, v in metrics.items()
                                     if isinstance(v, (int, float, bool))})

    def finish(self, failed=False):
        if self.run is not None:
            self.run.finish(exit_code=1 if failed else 0)
