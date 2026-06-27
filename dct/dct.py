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
    return True if x.ndim == 4 else False


def _is_resolution_divisible(x: Tensor, d: int) -> bool:
    return True if x.size(2) % d == 0 and x.size(3) % d == 0 else False


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


# Abstract Classes

class _DCT(nn.Module):

    def __init__(self, dct: int, kernel_size: int, norm: str, device: str | torch.device = None) -> None:
        super().__init__()
        if norm not in _VALID_NORMS:
            raise ValueError(f"norm must be one of {_VALID_NORMS}, got {norm!r}")
        self.dct = dct
        self.kernel_size = kernel_size
        self.norm = norm
        self.register_buffer("kernel", _initialize_dct_kernel(dct, kernel_size, norm, device).T)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dct={self.dct}, kernel_size={self.kernel_size}, norm={self.norm!r}"


class _IDCT(nn.Module):

    def __init__(self, dct: int, kernel_size: int, norm: str, device: str | torch.device = None) -> None:
        super().__init__()
        if norm not in _VALID_NORMS:
            raise ValueError(f"norm must be one of {_VALID_NORMS}, got {norm!r}")
        self.dct = dct
        self.kernel_size = kernel_size
        self.norm = norm
        self.register_buffer("kernel", _initialize_inverse_dct_kernel(dct, kernel_size, norm, device))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dct={self.dct}, kernel_size={self.kernel_size}, norm={self.norm!r}"


# Modules

class DCT2d(_DCT):

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: int | None = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, norm, device)
        k = _expand_kernel(self.kernel, 2).reshape(self.kernel_size ** 2, 1, self.kernel_size, self.kernel_size)
        k = k[_zigzag_permutation(self.kernel_size, device)]

        if selections is not None:
            k = k[:selections]

        self.register_buffer("kernel", k)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_batched_tensor(x):
            raise RuntimeError(f"{self.__class__.__name__} only supports batched image tensors; "
                               f"expected a 4D tensor but received a {x.ndim}D tensor")

        if not _is_resolution_divisible(x, self.kernel_size):
            raise RuntimeError(f"the resolution must be divisible by the kernel size")

        b, c, h, w = x.shape

        # x = F.conv2d(x, self.kernel.repeat(c, 1, 1, 1), None, self.kernel_size, groups=c)
        # Equivalent to the grouped convolution above, but much faster.
        x = x.reshape(b * c, 1, h, w)
        x = F.conv2d(x, self.kernel, None, self.kernel_size)
        x = x.reshape(b, -1, h // self.kernel_size, w // self.kernel_size)

        return x

    def reparameterize(self) -> None:
        raise NotImplementedError


class IDCT2d(_IDCT):

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: int | None = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, norm, device)
        k = _expand_kernel(self.kernel, 2).reshape(self.kernel_size ** 2, 1, self.kernel_size, self.kernel_size)
        k = k[_zigzag_permutation(self.kernel_size, k.device)]

        self.selections = kernel_size ** 2
        if selections is not None:
            k = k[:selections]
            self.selections = selections

        self.register_buffer("kernel", k)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_batched_tensor(x):
            raise RuntimeError(f"{self.__class__.__name__} only supports batched image tensors; "
                               f"expected a 4D tensor but received a {x.ndim}D tensor")

        b, n, h, w = x.shape
        c = n // self.selections

        # x = F.conv_transpose2d(x, self.kernel.repeat(c, 1, 1, 1), None, self.kernel_size, groups=c)
        # Equivalent to the grouped transposed convolution above, but much faster.
        x = x.reshape(b * c, -1, h, w)
        x = F.conv_transpose2d(x, self.kernel, None, self.kernel_size)
        x = x.reshape(b, c, h * self.kernel_size, w * self.kernel_size)

        return x
