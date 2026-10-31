"""Optional ahead-of-time build of the AVX2 CPU kernel (dct/csrc/dct8_cpu.cpp) as ``dct._C``.

By default the package is pure Python. To build the kernel, install with torch already present
and build isolation off, so the extension is compiled against the torch you run with:

    DCT_BUILD_CPU_EXT=1 pip install --no-build-isolation .

The extension is tied to that torch version; after upgrading torch, reinstall. Without it, set
DCT_CPU_JIT=1 to compile the kernel on first use instead (see dct/_cpu.py).
"""

import os
import platform
import sys

from setuptools import setup


ext_modules, cmdclass = [], {}

if os.environ.get("DCT_BUILD_CPU_EXT", "").lower() in ("1", "true", "yes", "on"):
    if platform.machine().lower() not in ("x86_64", "amd64", "x64", "i686", "x86"):
        sys.exit("DCT_BUILD_CPU_EXT=1: the CPU kernel needs an x86-64 CPU with AVX2")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    # Keep in sync with dct._cpu.compile_flags().
    if sys.platform == "win32":
        cflags, ldflags = ["/O2", "/arch:AVX2", "/fp:precise", "/openmp"], []
    elif sys.platform == "darwin":
        cflags, ldflags = ["-O3", "-mavx2", "-mfma"], []
    else:
        cflags, ldflags = ["-O3", "-mavx2", "-mfma", "-fopenmp"], ["-fopenmp"]

    ext_modules = [CppExtension("dct._C", ["dct/csrc/dct8_cpu.cpp"],
                                extra_compile_args=cflags, extra_link_args=ldflags)]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
