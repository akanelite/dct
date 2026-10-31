"""Module interface: :class:`DCT2d` and :class:`IDCT2d`.

Each module precomputes its kernel as a buffer (so it follows ``.to()`` and is saved in the
``state_dict``) and delegates the computation to :mod:`dct.functional`.
"""

import torch
import torch.nn as nn

from .functional import Selections, _channel_index, _dct2d, _idct2d, _make_kernel, _validate_args


__all__ = ["DCT2d", "IDCT2d"]


Tensor = torch.Tensor


class _BlockTransform(nn.Module):

    def __init__(self, inverse: bool, dct: int, kernel_size: int, selections: Selections,
                 norm: str, device: str | torch.device = None) -> None:
        super().__init__()
        max_sel, self._per_channel = _validate_args(dct, kernel_size, selections, norm)
        # int for a uniform count, list[int] for per-channel counts.
        self.selections = max_sel if self._per_channel is None else list(self._per_channel)
        self._max_selections = max_sel
        self.dct = dct
        self.kernel_size = kernel_size
        self.norm = norm
        self.register_buffer("kernel", _make_kernel(inverse, dct, kernel_size, max_sel, norm, device))
        index = None if self._per_channel is None else _channel_index(self._per_channel, max_sel, device)
        # Derived from the arguments, so it is not saved in the state_dict.
        self.register_buffer("index", index, persistent=False)

    def extra_repr(self) -> str:
        return (f"dct={self.dct}, kernel_size={self.kernel_size}, "
                f"selections={self.selections}, norm={self.norm!r}")


class DCT2d(_BlockTransform):
    """Block 2D DCT as a strided convolution. Module form of :func:`dct.dct2d`.

    Args:
        dct: DCT type, 1-4 (default 2, i.e. DCT-II); same definitions as scipy.fft.
        kernel_size: block side length; input H and W must be divisible by it.
        selections: keep only the first N zigzag (low->high freq) coefficients per block.
            None keeps all kernel_size**2. A list gives one count per input channel
            (e.g. ``[32, 8, 8]`` for YCbCr); its length must equal the number of input channels.
        norm: scipy-style normalization, one of "ortho" (default, orthonormal /
            energy-preserving), "backward", or "forward".
        device: device for the precomputed kernel buffer.

    Input:  (B, C, H, W) image tensor, floating-point or integer (e.g. uint8). Integer inputs
        are computed in float32, so a uint8 image gives exactly the same result as ``x.float()``.
    Output: (B, C * selections, H // kernel_size, W // kernel_size) coefficients,
        per block ordered low->high frequency; with a list, (B, sum(selections), ...) laid out
        channel by channel. Floating inputs keep their dtype; integer inputs produce float32.
    """

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: Selections = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(False, dct, kernel_size, selections, norm, device)

    def forward(self, x: Tensor) -> Tensor:
        return _dct2d(x, self.kernel, self.kernel_size, self.__class__.__name__,
                      self._per_channel, self.index)


class IDCT2d(_BlockTransform):
    """Inverse of :class:`DCT2d` via a per-block matmul followed by pixel shuffle.
    Module form of :func:`dct.idct2d`.

    Args mirror :class:`DCT2d`; use the same ``dct``, ``norm``, ``kernel_size`` and
    ``selections`` as the forward transform for exact reconstruction.

    Input:  (B, C * selections, H, W) coefficient tensor — or (B, sum(selections), H, W) for a
        per-channel list — floating-point or integer (integer inputs, e.g. quantized
        coefficients, are computed in float32).
    Output: (B, C, H * kernel_size, W * kernel_size) reconstruction. Floating inputs keep their
        dtype; integer inputs produce float32 (rounding/clamping back to uint8 is left to the caller).
    """

    def __init__(self, dct: int = 2, kernel_size: int = 8, selections: Selections = None,
                 norm: str = "ortho", device: str | torch.device = None) -> None:
        super().__init__(True, dct, kernel_size, selections, norm, device)

    def forward(self, x: Tensor) -> Tensor:
        return _idct2d(x, self.kernel, self.kernel_size, self._max_selections,
                       self.__class__.__name__, self._per_channel, self.index)
