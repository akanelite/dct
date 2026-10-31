# Usage

- [Modules and functional interface](#modules-and-functional-interface)
- [Input dtypes](#input-dtypes)
- [Keeping fewer coefficients: `selections`](#keeping-fewer-coefficients-selections)
- [DCT types](#dct-types)
- [Normalization](#normalization)
- [GPU and precision](#gpu-and-precision)
- [Gradients](#gradients)
- [How it works](#how-it-works)

## Modules and functional interface

Like PyTorch itself, the library offers both a module and a functional interface:

| Module (`torch.nn.Module`) | Functional | |
| --- | --- | --- |
| `DCT2d(...)` | `dct2d(input, ...)` | forward block DCT |
| `IDCT2d(...)` | `idct2d(input, ...)` | inverse block DCT |

Both take the same arguments in the same order (`dct`, `kernel_size`, `selections`, `norm`),
and the functional form simply adds `input` first, as `F.max_pool2d` mirrors `nn.MaxPool2d`.
They operate on batched 4D tensors of shape `(B, C, H, W)`, and both `H` and `W` must be
divisible by `kernel_size`.

```python
import torch
from dct import DCT2d, IDCT2d, dct2d, idct2d

dct = DCT2d(kernel_size=8)      # orthonormal block DCT-II
idct = IDCT2d(kernel_size=8)

x = torch.randn(2, 3, 64, 64)   # (B, C, H, W)
coeffs = dct(x)                 # (2, 192, 8, 8) == (B, C * selections, H // k, W // k)
recon = idct(coeffs)            # (2, 3, 64, 64), near-exact: max error < 1e-4

coeffs = dct2d(x, kernel_size=8)        # same result as DCT2d(kernel_size=8)(x)
recon = idct2d(coeffs, kernel_size=8)   # or: import dct.functional as DF
```

With `selections=None` (the default) every block keeps all `kernel_size ** 2` coefficients, so
each input channel expands into `kernel_size ** 2` output channels, ordered from low to high
frequency (zigzag).

Use the modules inside models. Their kernel is a buffer, so it follows `.to(device)` and is
saved in the `state_dict`. The functional form builds each kernel on first use and caches it
per `(dct, kernel_size, selections, norm, device)`, so repeated calls cost nothing extra.

## Input dtypes

| Input | Computed in | Output |
| --- | --- | --- |
| `float32`, `float64`, `float16`, `bfloat16` | the input dtype (see [GPU and precision](#gpu-and-precision)) | same dtype |
| integer (`uint8`, `int8`, `int16`, `int32`, `int64`) | `float32` | `float32` |
| `torch.ao` quantized (`quint8`, `qint8`, ...) | dequantized, then `float32` | `float32` |
| `bool`, complex | rejected with a `TypeError` | |

Images can therefore be passed as-is. A `uint8` image gives exactly the same coefficients as
`x.float()`. `IDCT2d` likewise accepts integer (e.g. quantized) coefficients and returns
`float32`; rounding and clamping back to `uint8` is up to you:

```python
img = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
coeffs = dct(img)                                            # float32, same as dct(img.float())
recon = idct(coeffs).round().clamp(0, 255).to(torch.uint8)   # lossless round trip
```

A level shift (e.g. `- 128`) only moves the DC coefficient, so normalization over the
coefficient channels, such as a `BatchNorm2d(C * selections)`, is left to the caller.

## Keeping fewer coefficients: `selections`

Keep only the first `N` low-frequency coefficients per block to drop high-frequency detail,
as JPEG does:

```python
dct = DCT2d(kernel_size=8, selections=16)   # keep the 16 lowest-frequency coefficients per block
idct = IDCT2d(kernel_size=8, selections=16)

coeffs = dct(x)        # (2, 48, 8, 8) == (B, C * 16, H // k, W // k)
recon = idct(coeffs)   # lossy reconstruction
```

Use the **same** `dct`, `kernel_size`, `selections` and `norm` for `DCT2d` and `IDCT2d`. Fewer
selections give smaller tensors but a larger reconstruction error.

`selections` can also be a list with one count per input channel, for example to keep more luma
than chroma detail in a YCbCr image. The output then has `sum(selections)` channels, laid out
channel by channel (here the first 32 are Y, the next 8 Cb and the last 8 Cr):

```python
ycbcr = torch.randn(2, 3, 64, 64)                # (B, 3, H, W), already converted to YCbCr
coeffs = dct2d(ycbcr, selections=[32, 8, 8])     # (2, 48, 8, 8)
recon = idct2d(coeffs, selections=[32, 8, 8])    # (2, 3, 64, 64)
```

The list length must equal the number of input channels, and `IDCT2d` / `idct2d` expect exactly
`sum(selections)` channels.

## DCT types

`dct` selects the DCT type (default `2`). Types 1–4 are supported, with the same definitions as
`scipy.fft.dct`. DCT-I requires `kernel_size >= 2`.

```python
dct = DCT2d(dct=4, kernel_size=8)
idct = IDCT2d(dct=4, kernel_size=8)
```

## Normalization

`norm` follows SciPy's conventions:

| `norm` | Description |
| --- | --- |
| `"ortho"` | Orthonormal, energy-preserving (default) |
| `"backward"` | SciPy "backward": forward transform left unnormalized |
| `"forward"` | SciPy "forward": forward transform normalized by block size |

All modes use SciPy's *orthogonalized* variant (`orthogonalize=True`), so `"backward"` and
`"forward"` are constant multiples of `"ortho"`. For DCT types I–III this differs from SciPy's
default for those two norms (`orthogonalize=False`) in how the first and last coefficients are
scaled.

## GPU and precision

`DCT2d`/`IDCT2d` are regular modules, so move them like any other:

```python
dct = DCT2d(kernel_size=8).to("cuda")
coeffs = dct(x.to("cuda"))
```

The kernel buffer is stored in `float32`, and `float16` / `float64` inputs are cast
automatically inside `forward`, so mixed-precision pipelines work out of the box. On CPU, the
8x8 DCT-II case is handled by the [AVX2 kernel](cpu-kernel.md) when it is available. It computes
half-precision inputs in `float32` and rounds once. Other half-precision cases on CPU use
PyTorch's CPU half convolution, which is very slow.

`float32` results keep full precision on CUDA even though PyTorch lets cuDNN use TF32 by default,
which would otherwise cost about 1e-3 of accuracy. `DCT2d` disables TF32 for its own convolution
call only, and `IDCT2d` uses a matmul, which only uses TF32 if you opt in via
`torch.backends.cuda.matmul.allow_tf32`.

## Gradients

Both transforms are fully differentiable, so gradients flow back to the input and they can be
dropped into any model as a fixed (non-learnable) layer. `torch.func` transforms (`vmap`,
`grad`, `jvp`, `jacrev`, `hessian`) work as well.

## How it works

While a 2D DCT is typically implemented as two successive 1D DCTs, that is hard to make fast
without low-level optimizations or hardware acceleration. This implementation instead builds
convolutional filters by unrolling a 2D basis derived from the Kronecker product of 1D bases.
The forward transform therefore reduces to a single strided convolution, backed by the highly
optimized backends that deep learning frameworks provide. The inverse is a per-block matrix
multiply followed by a pixel shuffle, which is several times faster than a transposed
convolution on CPU and for `float16` on GPU.

For the 8x8 DCT-II on CPU, an optional hand-written kernel replaces this path; see
[CPU kernel](cpu-kernel.md).
