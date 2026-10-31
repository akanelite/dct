# Discrete Cosine Transform

Fast block-wise 2D [Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf)
for PyTorch, on CPU and GPU.

- The forward transform is a single strided convolution, and the inverse is a matmul plus a pixel
  shuffle. Only PyTorch's own optimized operators are used.
- DCT types I–IV with SciPy's `"ortho"` (default), `"backward"` and `"forward"` normalizations.
- `selections` keep only the lowest-frequency coefficients, with an optional count per channel.
- Takes `uint8` images and other integer inputs directly, and is fully differentiable
  (`torch.func` included).
- Exports to ONNX as standard ops.
- An optional AVX2 CPU kernel for the JPEG case (8x8 DCT-II) is 3–6x faster than the PyTorch path (far more for `float16`).

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
