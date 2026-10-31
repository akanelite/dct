# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-31

### Added

- Optional AVX2 + FMA CPU kernel for the 8x8 DCT-II. It computes in `float32`, reads
  `float32`, `float16`, `bfloat16`, `uint8`, `int8`, `int16` and `int32` directly, and handles
  per-channel `selections` in place. Supports autograd and `torch.func` (`vmap`, `jvp`, ...).
- Build it with `DCT_BUILD_CPU_EXT=1 pip install --no-build-isolation .`, or compile it on first
  use with `DCT_CPU_JIT=1`; disable it with `DCT_DISABLE_CPU_EXT=1`. Without it, or for other
  cases, the pure-PyTorch implementation is used.
- `torch.ao` quantized tensors are accepted as input (dequantized, computed in `float32`).
- README section on ONNX export and INT8 quantization of models that use the transform.

### Notes

- Under `torch.onnx.export`, `torch.export`, `torch.compile` and `torch.jit.trace` the
  PyTorch implementation is always used, so exported graphs contain only standard ops.

## [1.0.0] - 2026-10-31

Initial release.

### Added

- `DCT2d` / `IDCT2d` modules and `dct2d` / `idct2d` functional API.
- DCT types I–IV with `"ortho"`, `"backward"` and `"forward"` normalization (SciPy definitions).
- Zigzag-ordered coefficients with uniform or per-channel `selections`.
- Floating-point and integer (e.g. `uint8`) inputs; full `float32` precision on CUDA regardless of TF32.
- PyTorch is the only runtime dependency.

[Unreleased]: https://github.com/akanelite/dct/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/akanelite/dct/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/akanelite/dct/releases/tag/v1.0.0
