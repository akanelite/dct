# Discrete Cosine Transform

This library implements the [Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf) for PyTorch, 
leveraging its optimized convolution operations to perform fast 2D DCT on both CPU and GPU.

While 2D DCT is typically implemented as two successive 1D DCTs, it is hard to achieve high performance without low-level optimizations or hardware acceleration. 
This implementation builds convolutional filters by unrolling a 2D basis derived from the Kronecker product of 1D bases, 
so the entire transform reduces to a single strided convolution backed by the highly optimized backends available in deep learning frameworks.

By default the transform is orthonormal (`norm="ortho"`, energy-preserving); pass `norm="backward"` or `"forward"` for the corresponding SciPy conventions.

## Installation

Requires Python 3.10+ and PyTorch 2.0+ (NumPy and SciPy are pulled in automatically).

Install the latest version from GitHub:

```bash
pip install git+https://github.com/akanelite/dct.git
```

Or clone the repository for local development:

```bash
git clone https://github.com/akanelite/dct.git
cd dct
pip install -e ".[dev]"   # editable install, includes pytest
```

> Using [uv](https://github.com/astral-sh/uv)? Swap `pip install` for `uv pip install`.

## Usage

The library exposes two `torch.nn.Module`s — `DCT2d` and its inverse `IDCT2d` — that
operate on batched 4D image tensors of shape `(B, C, H, W)`. Both `H` and `W` must be
divisible by `kernel_size`.

### Round-trip

```python
import torch
from dct import DCT2d, IDCT2d

dct = DCT2d(kernel_size=8)     # orthonormal block DCT-II
idct = IDCT2d(kernel_size=8)

x = torch.randn(2, 3, 64, 64)  # (B, C, H, W)
coeffs = dct(x)                # (2, 192, 8, 8) == (B, C * selections, H // k, W // k)
recon = idct(coeffs)           # (2, 3, 64, 64)

print((recon - x).abs().max())  # < 1e-4: near-exact reconstruction
```

With `selections=None` (the default) every block keeps all `kernel_size ** 2`
coefficients, so each input channel expands into `kernel_size ** 2` output channels,
ordered low-to-high frequency (zigzag).

### Running on GPU

`DCT2d`/`IDCT2d` are regular modules, so move them like any other:

```python
dct = DCT2d(kernel_size=8).to("cuda")
coeffs = dct(x.to("cuda"))
```

The kernel buffer is stored in `float32`; `float16` and `float64` inputs are cast
automatically inside `forward`, so mixed-precision pipelines work out of the box.

### Lossy compression with `selections`

Keep only the first `N` low-frequency coefficients per block to drop high-frequency
detail (JPEG-style):

```python
dct = DCT2d(kernel_size=8, selections=16)   # keep 16 lowest-frequency coeffs per block
idct = IDCT2d(kernel_size=8, selections=16)

coeffs = dct(x)        # (2, 48, 8, 8) == (B, C * 16, H // k, W // k)
recon = idct(coeffs)   # lossy reconstruction
```

Use the **same** `kernel_size`, `selections`, and `norm` for `DCT2d` and `IDCT2d`;
fewer selections means smaller tensors but larger reconstruction error.

### Normalization

`norm` follows SciPy's conventions:

| `norm` | Description |
| --- | --- |
| `"ortho"` | Orthonormal, energy-preserving (default) |
| `"backward"` | SciPy "backward" — forward transform left unnormalized |
| `"forward"` | SciPy "forward" — forward transform normalized by block size |

Both modules are fully differentiable, so gradients flow back to the input and they can
be dropped into any model as a fixed (non-learnable) transform layer.
