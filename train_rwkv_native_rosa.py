#!/usr/bin/env python3
"""Execute the pinned upstream train.py, NOT the old reference experiment loop."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'third_party/rwkv_lm_v7_native'
if __name__ == '__main__':
    from rwkv_rosa.native.wandb_auth import configure_wandb_auth
    configure_wandb_auth()
    sys.path.insert(0, str(NATIVE))
    sys.path.insert(1, str(ROOT))
    os.chdir(NATIVE)  # Upstream CUDA extension source paths are intentionally relative.
    runpy.run_path(str(NATIVE / 'train.py'), run_name='__main__')
