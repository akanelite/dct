# Optional AVX2 CPU kernel

For the JPEG case (8x8 DCT-II) on CPU, a hand-written AVX2 + FMA kernel can replace the PyTorch
implementation. The package stays pure Python by default; the kernel has to be
[built](#building) first.

## When it is used

The kernel handles a call when all of the following hold. Otherwise, the PyTorch implementation
is used, and `float32` results agree with it to rounding.

- `dct=2` and `kernel_size=8`, on a CPU tensor;
- the input dtype is `float32`, `float16`, `bfloat16`, `uint8`, `int8`, `int16` or `int32`
  (`int64` and quantized inputs are converted to `float32` first);
- the extension is built and the CPU supports AVX2 + FMA (as reported by
  `torch.backends.cpu.get_cpu_capability()`);
- the call is not being traced or exported (`torch.compile`, `torch.export`, `torch.jit`,
  `torch.onnx.export`), so exported graphs contain only standard ops (see
  [export and quantization](export-and-quantization.md));
- it is not disabled with `DCT_DISABLE_CPU_EXT=1`.

Every `norm` and `selections` value is supported, including per-channel lists. Autograd
(backward, double backward) and `torch.func` transforms (`vmap`, `grad`, `jvp`, `jacrev`,
`hessian`) are supported too.

The kernel always computes in `float32`. Floating inputs keep their dtype on output
(`float16` / `bfloat16` are rounded once, so they are also more accurate than computing in half
precision), and integer inputs produce `float32`.

## Performance

`(8, 3, 1024, 1024)` input, 8 threads, AVX-512 machine, compared with the PyTorch implementation
on the same machine:

| input | forward | inverse |
| --- | --- | --- |
| `float32` | 2.9x faster | 4.0x faster |
| `uint8` image / `int16` coefficients | 5.8x | 5.9x |
| `bfloat16` | 3.2x | 5.5x |
| `float16` | ~190x (PyTorch's CPU half convolution is very slow) | ~300x |

The transform is memory-bound: at full `selections` a `float32` forward pass takes about as long
as copying the input once. That is why reading `uint8` directly (a quarter of the bytes) speeds
it up further.

## Building

A C++ compiler is needed: MSVC on Windows, GCC or Clang elsewhere.

**Ahead of time.** Install with your torch already present and build isolation off, so the
extension is compiled against the torch you run with:

```bash
DCT_BUILD_CPU_EXT=1 pip install --no-build-isolation .    # builds dct._C
```

The build is tied to that torch version, so reinstall after upgrading torch. If the prebuilt
module cannot be loaded, a warning is shown and the PyTorch implementation is used.

**On first use.** Set `DCT_CPU_JIT=1` to compile the kernel the first time it is needed. This
requires `ninja`, and torch caches the result afterwards.

**Disabling.** Set `DCT_DISABLE_CPU_EXT=1`, or `dct._cpu.enabled = False` at runtime.

To check whether the kernel is active, and why not:

```python
import dct._cpu
print(dct._cpu.is_available(), dct._cpu.load_error)
```

### Windows notes

- Build from a shell where MSVC is set up (e.g. "x64 Native Tools Command Prompt"). From other
  shells, set `DISTUTILS_USE_SDK=1` with `cl` on `PATH`.
- In Git Bash, make sure MSVC's `link.exe` comes before Git's `/usr/bin/link` on `PATH`.
- OpenMP (`/openmp`) is required for multithreading and is set by the build.

## How it works

The source is `dct/csrc/dct8_cpu.cpp`, and the Python wrapper is `dct/_cpu.py`.

- SIMD runs across blocks: one 256-bit vector holds the same pixel (or coefficient) of 8
  horizontally adjacent blocks, so every store of an output coefficient plane is a full vector
  store.
- Each 1D transform uses the even/odd (partial butterfly) decomposition instead of a full
  8x8 matrix multiply.
- Only the memory side depends on the dtype. Per-dtype load/store routines feed one `float32`
  compute core, and `bfloat16` is rounded to nearest even, as in `c10::BFloat16`.
- `selections` are applied in place: the forward pass writes only the kept coefficients of each
  channel, and the inverse pass reads only those and treats the rest as zero.
- Forward and inverse with the same 1D matrix are each other's adjoint, so the backward of each
  is the other op.

The kernel uses its own cached matrices, so edits to a module's `kernel` buffer are not seen by
it. The PyTorch and export paths still use the buffer.
