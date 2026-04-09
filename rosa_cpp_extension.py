from __future__ import annotations

import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

import torch


EXTENSION_NAME = "rosa_sam_cpu_ext"
ROOT = Path(__file__).resolve().parent
SETUP_PATH = ROOT / "build_rosa_sam_cpu_extension.py"
BUILD_ROOT = ROOT / "outputs" / "cpp_extensions" / EXTENSION_NAME
BUILD_LIB = BUILD_ROOT / "lib"
BUILD_TEMP = BUILD_ROOT / "temp"


def _extension_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"


def _extension_binary_path() -> Path:
    return BUILD_LIB / f"{EXTENSION_NAME}{_extension_suffix()}"


def _ensure_torch_runtime_available() -> None:
    if os.name != "nt":
        return
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib_dir.exists():
        try:
            os.add_dll_directory(str(torch_lib_dir))
        except (AttributeError, FileNotFoundError, OSError):
            pass


def build_rosa_sam_cpu_extension(*, verbose: bool = False, force_rebuild: bool = False) -> Path:
    BUILD_LIB.mkdir(parents=True, exist_ok=True)
    BUILD_TEMP.mkdir(parents=True, exist_ok=True)
    binary_path = _extension_binary_path()
    if binary_path.exists() and not force_rebuild:
        return binary_path

    cmd = [
        sys.executable,
        str(SETUP_PATH),
        "build_ext",
        "--build-lib",
        str(BUILD_LIB),
        "--build-temp",
        str(BUILD_TEMP),
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )
    if not binary_path.exists():
        candidates = sorted(BUILD_LIB.glob(f"{EXTENSION_NAME}*{_extension_suffix()}"))
        if not candidates:
            raise FileNotFoundError(f"未找到已编译的扩展产物: {binary_path}")
        return candidates[0]
    return binary_path


def load_rosa_sam_cpu_extension(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    if EXTENSION_NAME in sys.modules and not force_rebuild:
        return sys.modules[EXTENSION_NAME]

    _ensure_torch_runtime_available()
    binary_path = build_rosa_sam_cpu_extension(verbose=verbose, force_rebuild=force_rebuild)

    if str(BUILD_LIB) not in sys.path:
        sys.path.insert(0, str(BUILD_LIB))
    try:
        if force_rebuild and EXTENSION_NAME in sys.modules:
            del sys.modules[EXTENSION_NAME]
        return importlib.import_module(EXTENSION_NAME)
    except ImportError as exc:
        raise ImportError(f"加载编译型 CPU 扩展失败: {binary_path}") from exc
