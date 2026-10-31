"""Functional interface: :func:`dct2d` and :func:`idct2d`.

These mirror :class:`dct.DCT2d` / :class:`dct.IDCT2d` (same arguments, plus the input first),
in the same way ``torch.nn.functional.max_pool2d`` mirrors ``torch.nn.MaxPool2d``. Kernels are
built on first use and cached per (type, kernel_size, selections, norm, device).
"""

import contextlib
import functools
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from . import _cpu


__all__ = ["dct2d", "idct2d"]


# Types

Tensor = torch.Tensor

_VALID_NORMS = ("ortho", "backward", "forward")

_VALID_TYPES = (1, 2, 3, 4)


# Checking

Selections = int | Sequence[int] | None


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_args(dct: int, kernel_size: int, selections: Selections,
                   norm: str) -> tuple[int, tuple[int, ...] | None]:
    # Validates the transform arguments. Returns (max_selections, per_channel): per_channel is
    # None for a single int (or None), else a tuple with one count per input channel.
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS}, got {norm!r}")
    if dct not in _VALID_TYPES:
        raise ValueError(f"dct type must be one of {_VALID_TYPES}, got {dct!r}")
    if not _is_int(kernel_size) or kernel_size < 1:
        raise ValueError(f"kernel_size must be a positive integer, got {kernel_size!r}")
    if dct == 1 and kernel_size < 2:
        raise ValueError("DCT-I requires kernel_size >= 2")

    k2 = kernel_size ** 2
    if selections is None:
        return k2, None
    if _is_int(selections):
        values, per_channel = (selections,), None
    elif isinstance(selections, Sequence) and not isinstance(selections, str) and len(selections) > 0:
        values = per_channel = tuple(selections)
    else:
        values = per_channel = None
    if values is None or not all(_is_int(s) and 1 <= s <= k2 for s in values):
        raise ValueError(f"selections must be None, an integer in [1, {k2}] (kernel_size ** 2), "
                         f"or a non-empty list of such integers (one per channel), got {selections!r}")
    return max(values), per_channel


def _channel_index(per_channel: tuple[int, ...], max_sel: int, device=None) -> Tensor | None:
    # Positions of the kept coefficients inside the (C * max_sel) layout, channel by channel.
    # None when every channel keeps max_sel coefficients (nothing to gather).
    if all(s == max_sel for s in per_channel):
        return None
    idx = [c * max_sel + j for c, s in enumerate(per_channel) for j in range(s)]
    return torch.tensor(idx, dtype=torch.int64, device=device)


def _check_batched(x: Tensor, name: str, what: str) -> None:
    if x.ndim != 4:
        raise RuntimeError(f"{name} only supports batched {what} tensors; "
                           f"expected a 4D tensor but received a {x.ndim}D tensor")


def _as_floating(x: Tensor, name: str) -> Tensor:
    # Integer inputs (e.g. uint8 images) are computed in float32, so dct2d(x) == dct2d(x.float()).
    # Without this, the kernel (values in [-1, 1]) would be cast to the integer dtype and
    # truncated to zeros.
    if x.is_floating_point():
        return x
    if x.dtype == torch.bool or x.is_complex():
        raise TypeError(f"{name} expects a real floating-point or integer tensor, got {x.dtype}")
    return x.to(torch.float32)


# Kernels

def _orthonormal_dct_matrix(d: int, k: int) -> Tensor:
    # Orthonormal DCT-I..IV matrix Q in float64 (y = Q @ x; rows = frequencies, columns = samples).
    # Same definition as scipy.fft.dct(..., norm="ortho", orthogonalize=True).
    n = torch.arange(k, dtype=torch.float64)
    u, i = n[:, None], n[None, :]
    if d == 1:
        w = torch.ones(k, dtype=torch.float64)
        w[0] = w[-1] = 1 / math.sqrt(2)
        return torch.cos(math.pi * u * i / (k - 1)) * math.sqrt(2 / (k - 1)) * w[:, None] * w[None, :]
    if d == 2:
        q = torch.cos(math.pi * u * (2 * i + 1) / (2 * k)) * math.sqrt(2 / k)
        q[0] /= math.sqrt(2)
        return q
    if d == 3:
        return _orthonormal_dct_matrix(2, k).T
    return torch.cos(math.pi * (2 * u + 1) * (2 * i + 1) / (4 * k)) * math.sqrt(2 / k)


def _norm_scale(d: int, k: int, norm: str) -> float:
    # In the orthogonalized variants, "backward"/"forward" are constant multiples of the orthonormal matrix.
    s = math.sqrt(2 * (k - 1) if d == 1 else 2 * k)
    return {"ortho": 1.0, "backward": s, "forward": 1 / s}[norm]


def _initialize_dct_kernel(d: int, k: int, norm: str, device: str | torch.device = None) -> Tensor:
    # Equals scipy.fft.dct(np.eye(k), d, norm=norm, orthogonalize=True), i.e. M^T.
    m = _orthonormal_dct_matrix(d, k) * _norm_scale(d, k, norm)
    return m.T.to(dtype=torch.float32, device=device)


def _initialize_inverse_dct_kernel(d: int, k: int, norm: str, device: str | torch.device = None) -> Tensor:
    # Equals scipy.fft.idct(np.eye(k), d, norm=norm, orthogonalize=True), i.e. (M^-1)^T.
    # Since M = sQ with Q orthogonal, M^-1 = Q^T / s and its transpose is Q / s.
    m = _orthonormal_dct_matrix(d, k) / _norm_scale(d, k, norm)
    return m.to(dtype=torch.float32, device=device)


def _expand_kernel(kernel: Tensor, dims: int) -> Tensor:
    for _ in range(dims - 1):
        kernel = kernel.kron(kernel)
    return kernel


def _zigzag_permutation(k: int, device: str | torch.device = None) -> Tensor:
    idx = torch.arange(0, k ** 2, device=device, dtype=torch.int64).reshape(k, k).flipud()
    return torch.cat([idx.diagonal(i) if (i + k) % 2 == 1 else idx.diagonal(i).flip(0) for i in range(1 - k, k)])


def _build_block_kernel(matrix_1d: Tensor, k: int) -> Tensor:
    # Build the separable 2D basis via a Kronecker product, unroll it to (k², 1, k, k), and
    # reorder it in zigzag (low -> high frequency) order. The zigzag index lives on the kernel's device.
    kernel = _expand_kernel(matrix_1d, 2).reshape(k ** 2, 1, k, k)
    return kernel[_zigzag_permutation(k, kernel.device)]


def _make_kernel(inverse: bool, dct: int, kernel_size: int, selections: int, norm: str,
                 device: str | torch.device = None) -> Tensor:
    # (selections, 1, k, k) float32 kernel. Analysis (DCT) uses the transposed 1D matrix as a
    # correlation (conv2d) filter; synthesis (IDCT) uses the untransposed matrix as a
    # transposed-convolution filter. This asymmetry is what makes the round trip exact.
    if inverse:
        matrix_1d = _initialize_inverse_dct_kernel(dct, kernel_size, norm, device)
    else:
        matrix_1d = _initialize_dct_kernel(dct, kernel_size, norm, device).T
    return _build_block_kernel(matrix_1d, kernel_size)[:selections]


@functools.lru_cache(maxsize=128)
def _cached_kernel(inverse: bool, dct: int, kernel_size: int, selections: int, norm: str,
                   device: torch.device) -> Tensor:
    # Built outside inference mode so that a kernel first created under torch.inference_mode()
    # can still be used later in autograd-recorded calls.
    with torch.inference_mode(False):
        return _make_kernel(inverse, dct, kernel_size, selections, norm, device)


@functools.lru_cache(maxsize=128)
def _cached_channel_index(per_channel: tuple[int, ...], max_sel: int, device: torch.device) -> Tensor | None:
    with torch.inference_mode(False):
        return _channel_index(per_channel, max_sel, device)


@functools.lru_cache(maxsize=8)
def _cpu_matrices(norm: str) -> tuple[Tensor, Tensor, Tensor]:
    # (forward 1D matrix M, inverse 1D matrix (M^-1)^T, zigzag order) for the 8x8 DCT-II CPU kernel.
    with torch.inference_mode(False):
        return (_initialize_dct_kernel(2, 8, norm).T.contiguous(),
                _initialize_inverse_dct_kernel(2, 8, norm).contiguous(),
                _zigzag_permutation(8))


@functools.lru_cache(maxsize=128)
def _cpu_selections(sels: tuple[int, ...]) -> Tensor:
    # Per-channel coefficient counts for the CPU kernel (all equal for a uniform selection).
    with torch.inference_mode(False):
        return torch.tensor(sels, dtype=torch.int64)


def _weight(kernel: Tensor, x: Tensor) -> Tensor:
    return kernel if kernel.dtype == x.dtype else kernel.to(x.dtype)


def _no_cudnn_tf32(x: Tensor):
    # cuDNN allows TF32 by default, and for some shapes it picks a TF32 algorithm for the
    # strided conv, costing ~1e-3 of float32 precision. Disable it for this call only,
    # keeping the caller's other cuDNN flags. Note: the flag is process-global while active.
    cudnn = torch.backends.cudnn
    if x.is_cuda and x.dtype == torch.float32 and cudnn.allow_tf32:
        return cudnn.flags(enabled=cudnn.enabled, benchmark=cudnn.benchmark,
                           benchmark_limit=cudnn.benchmark_limit,
                           deterministic=cudnn.deterministic, allow_tf32=False)
    return contextlib.nullcontext()


# Implementations (shared by the functional API and the modules)

def _dct2d(x: Tensor, kernel: Tensor, kernel_size: int, name: str,
           per_channel: tuple[int, ...] | None = None, index: Tensor | None = None,
           dct: int | None = None, norm: str | None = None) -> Tensor:
    # dct/norm describe the kernel; when given, the AVX2 CPU kernel may be used (see dct._cpu).
    _check_batched(x, name, "image")
    if x.size(2) % kernel_size != 0 or x.size(3) % kernel_size != 0:
        raise RuntimeError("the resolution must be divisible by the kernel size")
    if per_channel is not None and x.size(1) != len(per_channel):
        raise RuntimeError(f"{name}: the input has {x.size(1)} channels but selections has "
                           f"{len(per_channel)} entries (one per channel)")

    if x.dtype not in _cpu.DTYPES:
        x = _as_floating(x, name)       # bool/complex raise here; int64 becomes float32
    if _cpu.supports(x, dct, kernel_size):
        # The kernel writes each channel's selections in place: no index_select pass.
        m, _, perm = _cpu_matrices(norm)
        sels = per_channel or (kernel.size(0),) * x.size(1)
        return _cpu.dct2d(x, m, perm, _cpu_selections(sels))
    x = _as_floating(x, name)

    b, c, h, w = x.shape

    # x = F.conv2d(x, kernel.repeat(c, 1, 1, 1), None, kernel_size, groups=c)
    # Equivalent to the grouped convolution above, but much faster.
    x = x.reshape(b * c, 1, h, w)
    with _no_cudnn_tf32(x):
        x = F.conv2d(x, _weight(kernel, x), None, kernel_size)
    x = x.reshape(b, -1, h // kernel_size, w // kernel_size)
    if index is not None:
        # Per-channel selections: computed with max(selections) for every channel, then
        # each channel keeps its first selections[c] coefficients.
        x = x.index_select(1, index)
    return x


def _idct2d(x: Tensor, kernel: Tensor, kernel_size: int, selections: int, name: str,
            per_channel: tuple[int, ...] | None = None, index: Tensor | None = None,
            dct: int | None = None, norm: str | None = None) -> Tensor:
    # selections is max(selections) when per_channel is given; dct/norm as in _dct2d.
    _check_batched(x, name, "coefficient")
    b, n, h, w = x.shape
    if per_channel is not None and n != sum(per_channel):
        raise RuntimeError(f"{name}: the channel dimension ({n}) must equal "
                           f"sum(selections) ({sum(per_channel)}) for selections={list(per_channel)}")
    if per_channel is None and n % selections != 0:
        raise RuntimeError(f"the channel dimension ({n}) must be divisible by "
                           f"selections ({selections})")
    if x.dtype not in _cpu.DTYPES:
        x = _as_floating(x, name)       # bool/complex raise here; int64 becomes float32
    if _cpu.supports(x, dct, kernel_size):
        # The kernel reads each channel's selections in place: no zero-filled scatter pass.
        _, a, perm = _cpu_matrices(norm)
        sels = per_channel or (selections,) * (n // selections)
        return _cpu.idct2d(x, a, perm, _cpu_selections(sels))
    x = _as_floating(x, name)

    if per_channel is not None:
        if index is not None:
            # Scatter back into the (C * max_sel) layout; dropped coefficients are zero.
            x = x.new_zeros(b, len(per_channel) * selections, h, w).index_copy(1, index, x)
        n = x.size(1)
    c = n // selections

    # Equivalent to F.conv_transpose2d(x, kernel, None, k) on (B * C, selections, H, W), but
    # conv_transpose2d is several times slower on CPU and for float16 on CUDA. Each block's
    # k² pixels are a linear map of its coefficients: (k², sel) @ (sel, H * W), then
    # pixel_shuffle lays them out spatially. matmul also avoids cuDNN's TF32 path (on by
    # default), which a 1x1 conv would take, losing ~1e-3 of float32 precision; matmul only
    # uses TF32 if the caller opts in via torch.backends.cuda.matmul.allow_tf32.
    k = kernel_size
    weight = _weight(kernel, x).reshape(selections, k * k).T
    x = weight @ x.reshape(b * c, selections, h * w)
    x = F.pixel_shuffle(x.reshape(b * c, k * k, h, w), k)
    return x.reshape(b, c, h * k, w * k)


# Public API

def dct2d(input: Tensor, dct: int = 2, kernel_size: int = 8, selections: Selections = None,
          norm: str = "ortho") -> Tensor:
    """Block 2D DCT. Functional form of :class:`dct.DCT2d`.

    Args:
        input: (B, C, H, W) image tensor, floating-point or integer (e.g. uint8). Integer inputs
            are computed in float32, so a uint8 image gives exactly the same result as ``x.float()``.
        dct: DCT type, 1-4 (default 2, i.e. DCT-II); same definitions as scipy.fft.
        kernel_size: block side length; H and W must be divisible by it.
        selections: keep only the first N zigzag (low->high freq) coefficients per block.
            None keeps all kernel_size**2. A list gives one count per input channel
            (e.g. ``[32, 8, 8]`` for YCbCr); its length must equal C.
        norm: scipy-style normalization, one of "ortho" (default, orthonormal /
            energy-preserving), "backward", or "forward".

    Returns:
        (B, C * selections, H // kernel_size, W // kernel_size) coefficients, per block ordered
        low->high frequency; with a list, (B, sum(selections), ...) laid out channel by channel.
        Floating inputs keep their dtype; integer inputs produce float32.
    """
    sel, per_channel = _validate_args(dct, kernel_size, selections, norm)
    kernel = _cached_kernel(False, dct, kernel_size, sel, norm, input.device)
    index = None if per_channel is None else _cached_channel_index(per_channel, sel, input.device)
    return _dct2d(input, kernel, kernel_size, "dct2d", per_channel, index, dct, norm)


def idct2d(input: Tensor, dct: int = 2, kernel_size: int = 8, selections: Selections = None,
           norm: str = "ortho") -> Tensor:
    """Inverse of :func:`dct2d`. Functional form of :class:`dct.IDCT2d`.

    Use the same ``dct``, ``kernel_size``, ``selections`` and ``norm`` as the forward transform
    for exact reconstruction.

    Args:
        input: (B, C * selections, H, W) coefficient tensor — or (B, sum(selections), H, W) for a
            per-channel list — floating-point or integer (integer inputs, e.g. quantized
            coefficients, are computed in float32).
        dct, kernel_size, selections, norm: as in :func:`dct2d`.

    Returns:
        (B, C, H * kernel_size, W * kernel_size) reconstruction. Floating inputs keep their dtype;
        integer inputs produce float32 (rounding/clamping back to uint8 is left to the caller).
    """
    sel, per_channel = _validate_args(dct, kernel_size, selections, norm)
    kernel = _cached_kernel(True, dct, kernel_size, sel, norm, input.device)
    index = None if per_channel is None else _cached_channel_index(per_channel, sel, input.device)
    return _idct2d(input, kernel, kernel_size, sel, "idct2d", per_channel, index, dct, norm)
