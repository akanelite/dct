# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-31

Initial release.

### Added

- `DCT2d` / `IDCT2d` modules and `dct2d` / `idct2d` functional API.
- DCT types I–IV with `"ortho"`, `"backward"` and `"forward"` normalization (SciPy definitions).
- Zigzag-ordered coefficients with uniform or per-channel `selections`.
- Floating-point and integer (e.g. `uint8`) inputs; full `float32` precision on CUDA regardless of TF32.
- PyTorch is the only runtime dependency.

[1.0.0]: https://github.com/akanelite/dct/commits/main
