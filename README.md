# Discrete Cosine Transform

This library implements the [Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf) for PyTorch,
performing fast block-wise 2D DCT on both CPU and GPU with nothing but PyTorch's own optimized operators.
For the JPEG case (8x8 DCT-II) on CPU an optional hand-written AVX2 CPU kernel is also available
([see docs/cpu-kernel.md](docs/cpu-kernel.md)).

While 2D DCT is typically implemented as two successive 1D DCTs, it is hard to achieve high performance without low-level optimizations or hardware acceleration.
This implementation builds convolutional filters by unrolling a 2D basis derived from the Kronecker product of 1D bases,
so the forward transform reduces to a single strided convolution backed by the highly optimized backends available in deep learning frameworks.
The inverse is a per-block matrix multiply followed by a pixel shuffle, which is several times faster than a transposed convolution on CPU and for `float16` on GPU.

## Installation

Requires Python 3.10+ and PyTorch 2.0+, with no other runtime dependencies.

```bash
pip install git+https://github.com/akanelite/dct.git
```

For local development:

```bash
git clone https://github.com/akanelite/dct.git
cd dct
pip install -e ".[dev]"   # adds pytest plus numpy/scipy (used as test references)
pytest
```

To build the optional CPU kernel, see [docs/cpu-kernel.md](docs/cpu-kernel.md#building).

## Quick start

```python
import torch
from dct import DCT2d, IDCT2d, dct2d, idct2d

dct = DCT2d(kernel_size=8)      # orthonormal block DCT-II
idct = IDCT2d(kernel_size=8)

x = torch.randn(2, 3, 64, 64)   # (B, C, H, W); H and W divisible by kernel_size
coeffs = dct(x)                 # (2, 192, 8, 8) == (B, C * k * k, H // k, W // k)
recon = idct(coeffs)            # (2, 3, 64, 64), max error < 1e-4

# Functional form, same arguments:
coeffs = dct2d(x, kernel_size=8, selections=16)          # keep 16 coefficients per block
recon = idct2d(coeffs, kernel_size=8, selections=16)     # lossy reconstruction
```

## Documentation

- [Usage](docs/usage.md): modules vs. functional, input dtypes, `selections`, DCT types,
  normalization, GPU and precision, and gradients.
- [CPU kernel](docs/cpu-kernel.md): when it is used, benchmarks, and how to build it.
- [ONNX export and INT8 quantization](docs/export-and-quantization.md): exported ops, and how to
  quantize a model that uses the transform.
- [Changelog](CHANGELOG.md)
