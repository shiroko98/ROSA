from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cpp_extensions" / "rosa_sam_cpu_extension.cpp"

if os.name == "nt":
    extra_compile_args = {"cxx": ["/O2"]}
else:
    extra_compile_args = {"cxx": ["-O3"]}


setup(
    name="rosa_sam_cpu_ext",
    ext_modules=[
        CppExtension(
            "rosa_sam_cpu_ext",
            [str(SOURCE)],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
