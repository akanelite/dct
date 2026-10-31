# Discrete Cosine Transform

This library implements the [Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf) for PyTorch,
performing fast block-wise 2D DCT on both CPU and GPU with nothing but PyTorch's own optimized operators.
For the JPEG case (8x8 DCT-II) on CPU an optional hand-written AVX2 CPU kernel is also available
([see below](#optional-avx2-cpu-kernel)).

While 2D DCT is typically implemented as two successive 1D DCTs, it is hard to achieve high performance without low-level optimizations or hardware acceleration.
This implementation builds convolutional filters by unrolling a 2D basis derived from the Kronecker product of 1D bases,
so the forward transform reduces to a single strided convolution backed by the highly optimized backends available in deep learning frameworks.
The inverse is a per-block matrix multiply followed by a pixel shuffle, which is several times faster than a transposed convolution on CPU and for `float16` on GPU.

By default the transform is an orthonormal DCT-II (`norm="ortho"`, energy-preserving); DCT types I–IV and the
`"backward"` / `"forward"` normalizations are also available.

## Installation

Requires Python 3.10+ and PyTorch 2.0+ — no other runtime dependencies.

Install the latest version from GitHub:

```bash
pip install git+https://github.com/akanelite/dct.git
```

Or clone the repository for local development:

```bash
git clone https://github.com/akanelite/dct.git
cd dct
pip install -e ".[dev]"   # editable install, adds pytest plus numpy/scipy (used as test references)
pytest                    # run the test suite (CUDA/MPS tests run when a device is available)
```

> Using [uv](https://github.com/astral-sh/uv)? Swap `pip install` for `uv pip install`.

## Usage

Like PyTorch itself, the library offers both a module and a functional interface:

| Module (`torch.nn.Module`) | Functional | |
| --- | --- | --- |
| `DCT2d(...)` | `dct2d(input, ...)` | forward block DCT |
| `IDCT2d(...)` | `idct2d(input, ...)` | inverse block DCT |

Both take the same arguments in the same order — `dct`, `kernel_size`, `selections`, `norm` —
and the functional form simply adds `input` first (as `F.max_pool2d` mirrors `nn.MaxPool2d`).
They operate on batched 4D image tensors of shape `(B, C, H, W)`; both `H` and `W` must be
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

### Functional interface

```python
from dct import dct2d, idct2d          # or: import dct.functional as DF

coeffs = dct2d(x, kernel_size=8)       # same result as DCT2d(kernel_size=8)(x)
recon = idct2d(coeffs, kernel_size=8)
```

Use the modules inside models — their kernel is a buffer, so it follows `.to(device)` and is
saved in the `state_dict`. The functional form builds each kernel on first use and caches it
per `(dct, kernel_size, selections, norm, device)`, so repeated calls cost nothing extra.

### Running on GPU

`DCT2d`/`IDCT2d` are regular modules, so move them like any other:

```python
dct = DCT2d(kernel_size=8).to("cuda")
coeffs = dct(x.to("cuda"))
```

The kernel buffer is stored in `float32`; `float16` and `float64` inputs are cast
automatically inside `forward`, so mixed-precision pipelines work out of the box.

`float32` results keep full precision on CUDA even though PyTorch lets cuDNN use TF32 by
default (which would otherwise cost ~1e-3 of accuracy): `DCT2d` disables TF32 for its own
convolution call only, and `IDCT2d` uses a matmul, which only uses TF32 if you opt in via
`torch.backends.cuda.matmul.allow_tf32`.

### Optional AVX2 CPU kernel

For CPU tensors with `dct=2` and `kernel_size=8`, a hand-written AVX2 + FMA kernel replaces the
PyTorch path when it is available. It processes 8 neighbouring blocks per SIMD vector with a
butterfly 1D DCT and always computes in `float32`. It reads `float32`, `float16`, `bfloat16`,
`uint8`, `int8`, `int16` and `int32` tensors directly, with no separate conversion pass;
floating inputs keep their dtype on output and integer inputs produce `float32`, as before. On an 8-thread AVX-512 machine with `(8, 3, 1024, 1024)` inputs:

| input | forward | inverse |
| --- | --- | --- |
| `float32` | 2.9x faster | 4.0x faster |
| `uint8` image / `int16` coefficients | 5.8x | 5.9x |
| `bfloat16` | 3.2x | 5.5x |
| `float16` | ~190x (PyTorch's CPU half convolution is very slow) | ~300x |

`float16`/`bfloat16` results are rounded once from the `float32` computation, so they are also
more accurate than computing in half precision. Every `norm`, `selections` (including per-channel
lists), autograd (backward, double backward) and `torch.func` transforms (`vmap`, `grad`, `jvp`,
`jacrev`, `hessian`) are supported. Other cases always use the PyTorch implementation, and
`float32` results agree with it to rounding.

The package stays pure Python by default. To build the kernel (a C++ compiler is needed — MSVC on
Windows, GCC/Clang elsewhere), install with your torch already present:

```bash
DCT_BUILD_CPU_EXT=1 pip install --no-build-isolation .    # builds dct._C against the installed torch
```

The build is tied to that torch version, so reinstall after upgrading torch. Alternatively, set
`DCT_CPU_JIT=1` to compile it on first use (needs `ninja`; cached by torch afterwards).
The kernel is only used on CPUs with AVX2/FMA, and `DCT_DISABLE_CPU_EXT=1` turns it off.

### Integer (uint8) input

Images can be passed as-is: integer tensors such as `uint8` are computed in `float32`, giving
exactly the same coefficients as `x.float()`. `IDCT2d` likewise accepts integer (e.g. quantized)
coefficients and returns `float32`; rounding and clamping back to `uint8` is up to you:

```python
img = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
coeffs = dct(img)                                            # float32, same as dct(img.float())
recon = idct(coeffs).round().clamp(0, 255).to(torch.uint8)   # lossless round trip
```

`bool` and complex tensors are rejected with a `TypeError`.

A level shift (e.g. `- 128`) only moves the DC coefficient, so normalization such as a
`BatchNorm2d(C * selections)` over the coefficient channels is left to the caller.

### Lossy compression with `selections`

Keep only the first `N` low-frequency coefficients per block to drop high-frequency
detail (JPEG-style):

```python
dct = DCT2d(kernel_size=8, selections=16)   # keep 16 lowest-frequency coeffs per block
idct = IDCT2d(kernel_size=8, selections=16)

coeffs = dct(x)        # (2, 48, 8, 8) == (B, C * 16, H // k, W // k)
recon = idct(coeffs)   # lossy reconstruction
```

Use the **same** `dct`, `kernel_size`, `selections`, and `norm` for `DCT2d` and `IDCT2d`;
fewer selections means smaller tensors but larger reconstruction error.

`selections` can also be a list with one count per input channel — e.g. keep more luma
than chroma detail for a YCbCr image. The output then has `sum(selections)` channels, laid out
channel by channel (the first 32 are Y, the next 8 Cb, the last 8 Cr):

```python
ycbcr = torch.randn(2, 3, 64, 64)                    # (B, 3, H, W), already converted to YCbCr
coeffs = dct2d(ycbcr, selections=[32, 8, 8])         # (2, 48, 8, 8)
recon = idct2d(coeffs, selections=[32, 8, 8])        # (2, 3, 64, 64)
```

The list length must equal the number of input channels, and `IDCT2d` / `idct2d` expect exactly
`sum(selections)` channels.

### DCT types

`dct` selects the DCT type (default `2`). Types 1–4 are supported, with the same
definitions as `scipy.fft.dct`; DCT-I requires `kernel_size >= 2`.

```python
dct = DCT2d(dct=4, kernel_size=8)
idct = IDCT2d(dct=4, kernel_size=8)
```

### Normalization

`norm` follows SciPy's conventions:

| `norm` | Description |
| --- | --- |
| `"ortho"` | Orthonormal, energy-preserving (default) |
| `"backward"` | SciPy "backward" — forward transform left unnormalized |
| `"forward"` | SciPy "forward" — forward transform normalized by block size |

All modes use SciPy's *orthogonalized* variant (`orthogonalize=True`), so `"backward"` and
`"forward"` are constant multiples of `"ortho"`. For DCT types I–III this differs from SciPy's
default for those two norms (`orthogonalize=False`) in how the first/last coefficients are scaled.

Both modules are fully differentiable, so gradients flow back to the input and they can
be dropped into any model as a fixed (non-learnable) transform layer.

### ONNX export and INT8 quantization

`DCT2d`/`IDCT2d` have no trainable parameters, so in a quantized model they are fixed
preprocessing. Under `torch.onnx.export` (both exporters), `torch.export`, `torch.compile` and
`torch.jit.trace` the PyTorch implementation is always used (the CPU kernel steps aside), so
the graph contains only standard ops: `Conv` for the forward transform, `MatMul` +
`DepthToSpace` for the inverse, and `Gather`/`Scatter` for per-channel `selections`. In ONNX the
forward transform is the node named after the module, e.g. `/dct/Conv`.

Quantized `torch.ao` tensors (e.g. after a `QuantStub`) are dequantized and transformed as
float, returning `float32`.

When the model downstream is quantized, keep in mind that DCT coefficients have very different
ranges: the DC term is up to `8 * 255` for `uint8` images while most AC terms are tens. A single
per-tensor int8 scale, which ONNX Runtime and TensorRT use for activations by default, rounds
most AC coefficients to zero. On natural-image-like data (all 64 coefficients):

| int8 quantization of the DCT output | AC SNR | AC coefficients zeroed |
| --- | --- | --- |
| per-tensor | 13 dB | 83% |
| per-tensor, after centering (`x - 128`) | 18 dB | 55% |
| one scale per coefficient channel | 40 dB | 1% |

So either quantize the coefficients per channel, or rescale each coefficient channel to a
similar range before the first quantized layer, and keep the DCT node itself in float (for
ONNX Runtime: `quantize_static(..., nodes_to_exclude=["/dct/Conv"])`).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
