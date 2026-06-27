import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DCT2d", "IDCT2d"]


# Types

Tensor = torch.Tensor

_VALID_NORMS = ("ortho", "backward", "forward")


# Checking

def _is_a_batched_tensor(x: Tensor) -> bool:
    return x.ndim == 4


def _is_resolution_divisible(x: Tensor, d: int) -> bool:
    return x.size(2) % d == 0 and x.size(3) % d == 0


# Helper Functions

def _initialize_dct_kernel(d: int, k: int, norm: str, device: str | torch.device = None) -> Tensor:
    return torch.as_tensor(scipy.fft.dct(np.eye(k), d, norm=norm, orthogonalize=True),
                           dtype=torch.float32, device=device)


def _initialize_inverse_dct_kernel(d: int, k: int, norm: str, device: str | torch.device = None) -> Tensor:
    return torch.as_tensor(scipy.fft.idct(np.eye(k), d, norm=norm, orthogonalize=True),
                           dtype=torch.float32, device=device)


def _expand_kernel(kernel: Tensor, dims: int) -> Tensor:
    for _ in range(dims - 1):
        kernel = kernel.kron(kernel)
    return kernel


def _zigzag_permutation(k: int, device: str | torch.device = None) -> Tensor:
    idx = torch.arange(0, k ** 2, device=device, dtype=torch.int64).reshape(k, k).flipud()
    return torch.cat([idx.diagonal(i) if (i + k) % 2 == 1 else idx.diagonal(i).flip(0) for i in range(1 - k, k)])


def _build_block_kernel(matrix_1d: Tensor, k: int) -> Tensor:
    # Kronecker product로 분리형 2D 기저를 만들고 (k², 1, k, k)로 펼친 뒤
    # zigzag(저→고주파) 순서로 재배열한다. zigzag 인덱스는 커널과 동일 디바이스.
    kernel = _expand_kernel(matrix_1d, 2).reshape(k ** 2, 1, k, k)
    return kernel[_zigzag_permutation(k, kernel.device)]


# Base Class

class _BlockTransform(nn.Module):

    def __init__(self, dct: int, kernel_size: int, selections: int | None, norm: str,
                 kernel_fn, transpose: bool, device: str | torch.device = None) -> None:
        super().__init__()
        if norm not in _VALID_NORMS:
            raise ValueError(f"norm must be one of {_VALID_NORMS}, got {norm!r}")
        self.dct = dct
        self.kernel_size = kernel_size
        self.norm = norm

        # 1D (역)DCT 행렬을 만든다. 분석(DCT)은 전치한 행렬을 상관(conv2d) 필터로,
        # 합성(IDCT)은 비전치 행렬을 전치합성곱 필터로 사용한다 — 이 비대칭이
        # 정/역 라운드트립을 성립시킨다.
        matrix_1d = kernel_fn(dct, kernel_size, norm, device)
        if transpose:
            matrix_1d = matrix_1d.T

        kernel = _build_block_kernel(matrix_1d, kernel_size)
        self.selections = kernel_size ** 2 if selections is None else selections
        if selections is not None:
            kernel = kernel[:selections]
        self.register_buffer("kernel", kernel)

    def _weight(self, x: Tensor) -> Tensor:
        return self.kernel if self.kernel.dtype == x.dtype else self.kernel.to(x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (f"dct={self.dct}, kernel_size={self.kernel_size}, "
                f"selections={self.selections}, norm={self.norm!r}")


# Modules

class DCT2d(_BlockTransform):

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: int | None = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, selections, norm,
                         _initialize_dct_kernel, transpose=True, device=device)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_batched_tensor(x):
            raise RuntimeError(f"{self.__class__.__name__} only supports batched image tensors; "
                               f"expected a 4D tensor but received a {x.ndim}D tensor")

        if not _is_resolution_divisible(x, self.kernel_size):
            raise RuntimeError("the resolution must be divisible by the kernel size")

        b, c, h, w = x.shape

        # x = F.conv2d(x, self.kernel.repeat(c, 1, 1, 1), None, self.kernel_size, groups=c)
        # Equivalent to the grouped convolution above, but much faster.
        x = x.reshape(b * c, 1, h, w)
        x = F.conv2d(x, self._weight(x), None, self.kernel_size)
        x = x.reshape(b, -1, h // self.kernel_size, w // self.kernel_size)

        return x


class IDCT2d(_BlockTransform):

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: int | None = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, selections, norm,
                         _initialize_inverse_dct_kernel, transpose=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_batched_tensor(x):
            raise RuntimeError(f"{self.__class__.__name__} only supports batched coefficient tensors; "
                               f"expected a 4D tensor but received a {x.ndim}D tensor")

        b, n, h, w = x.shape
        if n % self.selections != 0:
            raise RuntimeError(f"the channel dimension ({n}) must be divisible by "
                               f"selections ({self.selections})")
        c = n // self.selections

        # x = F.conv_transpose2d(x, self.kernel.repeat(c, 1, 1, 1), None, self.kernel_size, groups=c)
        # Equivalent to the grouped transposed convolution above, but much faster.
        x = x.reshape(b * c, -1, h, w)
        x = F.conv_transpose2d(x, self._weight(x), None, self.kernel_size)
        x = x.reshape(b, c, h * self.kernel_size, w * self.kernel_size)

        return x
