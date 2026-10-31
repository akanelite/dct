"""Optional AVX2 CPU kernel for the 8x8 DCT-II (the JPEG case), with a pure-PyTorch fallback.

The kernel (``dct/csrc/dct8_cpu.cpp``) is used for float32 CPU tensors with ``dct=2`` and
``kernel_size=8`` when all of these hold:

- the extension is available: either built ahead of time as ``dct._C`` (``DCT_BUILD_CPU_EXT=1``
  at install time), or compiled on first use when ``DCT_CPU_JIT=1`` is set;
- the CPU supports AVX2 + FMA (as reported by ``torch.backends.cpu.get_cpu_capability()``);
- it is not disabled with ``DCT_DISABLE_CPU_EXT=1`` (or ``dct._cpu.enabled = False``).

Otherwise the PyTorch implementation in :mod:`dct.functional` is used; results agree to
float32 rounding.
"""

import importlib
import os
import sys
import warnings

import torch


Tensor = torch.Tensor

_TRUE = ("1", "true", "yes", "on")

# Runtime switch (also used by the tests to compare against the PyTorch path).
enabled = os.environ.get("DCT_DISABLE_CPU_EXT", "").lower() not in _TRUE

_ext = None
_tried = False
load_error: BaseException | None = None   # why the extension is unavailable, for diagnostics


def _cpu_has_avx2() -> bool:
    try:
        cap = torch.backends.cpu.get_cpu_capability()
    except AttributeError:      # torch < 2.1
        return False
    # PyTorch reports AVX2 only when FMA is present too.
    return cap in ("AVX2", "AVX512")


def compile_flags() -> tuple[list[str], list[str]]:
    # (cflags, ldflags). OpenMP matters: without it at::parallel_for runs single-threaded.
    if sys.platform == "win32":
        return ["/O2", "/arch:AVX2", "/fp:precise", "/openmp"], []
    if sys.platform == "darwin":
        return ["-O3", "-mavx2", "-mfma"], []
    return ["-O3", "-mavx2", "-mfma", "-fopenmp"], ["-fopenmp"]


def _jit_build():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "dct8_cpu.cpp")
    cflags, ldflags = compile_flags()
    return load("dct_cpu_ext", [src], extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)


def _load():
    global load_error
    try:
        return importlib.import_module("dct._C")
    except ModuleNotFoundError as e:      # not built: the normal case for a pure-Python install
        load_error = e
    except Exception as e:      # built, but cannot be loaded (e.g. against another torch version)
        load_error = e
        warnings.warn(f"dct: the prebuilt CPU kernel could not be loaded ({e!r}); "
                      "falling back to the PyTorch implementation", RuntimeWarning, stacklevel=3)
    if os.environ.get("DCT_CPU_JIT", "").lower() in _TRUE:
        try:
            ext = _jit_build()
            load_error = None
            return ext
        except Exception as e:
            load_error = e
            warnings.warn(f"dct: building the CPU kernel failed ({e!r}); "
                          "falling back to the PyTorch implementation", RuntimeWarning, stacklevel=3)
    return None


def get():
    """The loaded extension, or None when the kernel cannot or should not be used."""
    global _ext, _tried, load_error
    if not enabled:
        return None
    if not _tried:
        _tried = True
        if _cpu_has_avx2():
            _ext = _load()
        else:
            load_error = RuntimeError("the CPU does not support AVX2/FMA")
    return _ext


def is_available() -> bool:
    return get() is not None


def supports(x: Tensor, dct: int | None, kernel_size: int) -> bool:
    # x is already floating (integer inputs are converted to float32 before this check).
    return (dct == 2 and kernel_size == 8 and x.device.type == "cpu"
            and x.dtype == torch.float32 and x.layout == torch.strided and get() is not None)


# Autograd. Both transforms are linear, and forward/inverse with the same 1D matrix are each
# other's adjoint (see dct8_cpu.cpp), so each backward is the other op. Using .apply in the
# backward keeps higher-order gradients working. The JVP of a linear map is the map itself.

def _setup_context(ctx, inputs, output):
    ctx.args = inputs[1:]       # (mat, perm, sels)


def _vmap_rule(fn):    # fn: () -> the Function (a thunk, since the class is not defined yet)
    # torch.func.vmap support: fold the vmapped dimension into the batch dimension, run the
    # kernel once on (N * B, C, H, W), and split it back. Only the tensor input can be batched
    # (mat/perm are internal constants). Calling .apply keeps autograd working under vmap, so
    # vmap(grad(...)), jacrev and hessian work too.
    def vmap(info, in_dims, x, mat, perm, sels):
        if in_dims[0] is None:
            return fn().apply(x, mat, perm, sels), None
        x = x.movedim(in_dims[0], 0)
        n, b = x.shape[:2]
        out = fn().apply(x.reshape(n * b, *x.shape[2:]), mat, perm, sels)
        return out.reshape(n, b, *out.shape[1:]), 0
    return staticmethod(vmap)


class _Forward(torch.autograd.Function):

    @staticmethod
    def forward(x: Tensor, mat: Tensor, perm: Tensor, sels: Tensor) -> Tensor:
        return _ext.forward(x, mat, perm, sels)

    setup_context = staticmethod(_setup_context)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return _Inverse.apply(grad, *ctx.args), None, None, None

    @staticmethod
    def jvp(ctx, x_t: Tensor, *_):
        return _Forward.apply(x_t, *ctx.args)

    vmap = _vmap_rule(lambda: _Forward)


class _Inverse(torch.autograd.Function):

    @staticmethod
    def forward(y: Tensor, mat: Tensor, perm: Tensor, sels: Tensor) -> Tensor:
        return _ext.inverse(y, mat, perm, sels)

    setup_context = staticmethod(_setup_context)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return _Forward.apply(grad, *ctx.args), None, None, None

    @staticmethod
    def jvp(ctx, y_t: Tensor, *_):
        return _Inverse.apply(y_t, *ctx.args)

    vmap = _vmap_rule(lambda: _Inverse)


def dct2d(x: Tensor, mat: Tensor, perm: Tensor, sels: Tensor) -> Tensor:
    # (B, C, H, W) -> (B, sum(sels), H / 8, W / 8); mat is the forward 1D matrix (rows =
    # frequencies) and sels holds one coefficient count per channel (int64, length C).
    return _Forward.apply(x, mat, perm, sels)


def idct2d(y: Tensor, mat: Tensor, perm: Tensor, sels: Tensor) -> Tensor:
    # (B, sum(sels), h, w) -> (B, len(sels), 8h, 8w); computes mat^T y per block, so pass the
    # inverse 1D matrix (M^-1)^T to invert, or the forward matrix M to get the adjoint.
    return _Inverse.apply(y, mat, perm, sels)
