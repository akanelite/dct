import numpy as np
import pytest
import scipy.fft
import torch

import dct
from dct import DCT2d, IDCT2d, dct2d, idct2d
from dct.functional import _initialize_dct_kernel, _initialize_inverse_dct_kernel, _zigzag_permutation

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
                  + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", ["ortho", "backward"])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_roundtrip_is_exact(device, norm, k):
    fwd = DCT2d(kernel_size=k, norm=norm).to(device)
    inv = IDCT2d(kernel_size=k, norm=norm).to(device)
    x = torch.randn(2, 3, 2 * k, 2 * k, device=device)
    recon = inv(fwd(x))
    assert recon.shape == x.shape
    assert (recon - x).abs().max().item() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [4, 8])
def test_basis_is_orthonormal_for_ortho(device, k):
    fwd = DCT2d(kernel_size=k, norm="ortho").to(device)
    ker = fwd.kernel.reshape(k * k, k * k).cpu()
    gram = ker @ ker.T
    assert (gram - torch.eye(k * k)).abs().max().item() < 1e-4


def test_matches_scipy_dctn_for_ortho():
    k = 8
    fwd = DCT2d(kernel_size=k, norm="ortho")
    block = torch.randn(1, 1, k, k)
    mine = fwd(block).reshape(-1).numpy()           # zigzag order
    perm = _zigzag_permutation(k).numpy()
    grid = np.empty(k * k, np.float32)
    grid[perm] = mine
    grid = grid.reshape(k, k)
    ref = scipy.fft.dctn(block.numpy().reshape(k, k), type=2, norm="ortho",
                         orthogonalize=True)
    assert np.abs(grid - ref).max() < 1e-4


def test_invalid_norm_raises():
    with pytest.raises(ValueError):
        DCT2d(norm="nope")
    with pytest.raises(ValueError):
        IDCT2d(norm="nope")


def test_single_kernel_buffer_and_repr():
    fwd = DCT2d(kernel_size=8, norm="ortho")
    # Exactly one buffer, 'kernel', in 2D form (k², 1, k, k).
    buffers = dict(fwd.named_buffers())
    assert list(buffers.keys()) == ["kernel"]
    assert tuple(buffers["kernel"].shape) == (64, 1, 8, 8)
    assert "norm='ortho'" in repr(fwd)
    # The selections attribute is exposed.
    assert DCT2d(kernel_size=8).selections == 64
    assert DCT2d(kernel_size=8, selections=10).selections == 10


def test_double_precision_roundtrip():
    # The kernel buffer is initialized in float32, so .double() only raises the dtype to float64
    # (precision stays bounded by float32). This checks fp64 dtype propagation and behavior.
    fwd = DCT2d(kernel_size=8, norm="ortho").double()
    inv = IDCT2d(kernel_size=8, norm="ortho").double()
    x = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    out = fwd(x)
    recon = inv(out)
    assert out.dtype == torch.float64
    assert recon.dtype == torch.float64
    assert (recon - x).abs().max().item() < 1e-5


def test_half_precision_input():
    # Kernel buffer is float32 and input is float16; the cast in forward must handle it.
    devs = [d for d in DEVICES if d != "cpu"]  # CPU lacks half-precision conv
    if not devs:
        pytest.skip("no half-capable device")
    dev = devs[0]
    fwd = DCT2d(kernel_size=8, norm="ortho").to(dev)
    out = fwd(torch.randn(1, 3, 16, 16, device=dev, dtype=torch.float16))
    assert out.dtype == torch.float16


def test_dct_rejects_bad_input():
    fwd = DCT2d(kernel_size=8)
    with pytest.raises(RuntimeError):
        fwd(torch.randn(3, 32, 32))          # not 4D
    with pytest.raises(RuntimeError):
        fwd(torch.randn(1, 3, 30, 30))       # resolution not divisible


def test_idct_rejects_bad_input():
    inv = IDCT2d(kernel_size=8, selections=16)   # channels must be a multiple of 16
    with pytest.raises(RuntimeError, match="4D"):
        inv(torch.randn(3, 16, 4, 4)[0])         # 3D, not 4D
    # Non-divisible channels must be rejected with a clear message, not an obscure conv error.
    with pytest.raises(RuntimeError, match="divisible by selections"):
        inv(torch.randn(2, 24, 4, 4))            # 24 % 16 != 0 -> channels not divisible


@pytest.mark.parametrize("sel", [64, 32, 16, 8])
def test_selections_channel_count(sel):
    fwd = DCT2d(kernel_size=8, selections=sel)
    out = fwd(torch.randn(2, 3, 32, 32))
    assert out.shape[1] == 3 * sel


def test_selections_monotonic_rmse():
    x = torch.randn(2, 3, 32, 32)
    rmses = []
    for sel in (64, 32, 16, 8):
        f = DCT2d(kernel_size=8, selections=sel, norm="ortho")
        i = IDCT2d(kernel_size=8, selections=sel, norm="ortho")
        rmses.append((i(f(x)) - x).pow(2).mean().sqrt().item())
    assert rmses[0] < 1e-4                       # all coefficients -> lossless
    assert rmses[1] <= rmses[2] <= rmses[3]      # more truncation -> larger error


def test_gradient_flows_to_input():
    fwd = DCT2d(kernel_size=8)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    fwd(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("norm", ["ortho", "backward", "forward"])
@pytest.mark.parametrize("k", [2, 3, 8, 16])
def test_kernels_match_scipy(d, norm, k):
    # The torch-built 1D kernels must match scipy's definitions (scipy is a test-only dependency).
    eye = np.eye(k)
    ref_f = scipy.fft.dct(eye, d, norm=norm, orthogonalize=True)
    ref_i = scipy.fft.idct(eye, d, norm=norm, orthogonalize=True)
    assert np.abs(_initialize_dct_kernel(d, k, norm).numpy() - ref_f).max() < 1e-6
    assert np.abs(_initialize_inverse_dct_kernel(d, k, norm).numpy() - ref_i).max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("norm", ["ortho", "backward", "forward"])
def test_roundtrip_all_types(device, d, norm):
    fwd = DCT2d(dct=d, kernel_size=8, norm=norm).to(device)
    inv = IDCT2d(dct=d, kernel_size=8, norm=norm).to(device)
    x = torch.randn(2, 3, 32, 16, device=device)
    assert (inv(fwd(x)) - x).abs().max().item() < 1e-4


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        DCT2d(dct=5)
    with pytest.raises(ValueError):
        IDCT2d(dct=0)
    with pytest.raises(ValueError):
        DCT2d(dct=1, kernel_size=1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sel", [None, 10])
def test_idct_matches_conv_transpose(device, sel):
    # IDCT2d's matmul + pixel_shuffle path must equal the original conv_transpose2d definition.
    inv = IDCT2d(kernel_size=8, selections=sel).to(device)
    y = torch.randn(2, 3 * inv.selections, 4, 5, device=device)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):  # exact float32 reference
        ref = torch.nn.functional.conv_transpose2d(y.reshape(6, inv.selections, 4, 5), inv.kernel, None, 8)
    assert (inv(y) - ref.reshape(2, 3, 32, 40)).abs().max().item() < 1e-4


def test_idct_half_precision_input():
    devs = [d for d in DEVICES if d != "cpu"]
    if not devs:
        pytest.skip("no half-capable device")
    inv = IDCT2d(kernel_size=8, norm="ortho").to(devs[0])
    out = inv(torch.randn(1, 3 * 64, 2, 2, device=devs[0], dtype=torch.float16))
    assert out.dtype == torch.float16 and out.shape == (1, 3, 16, 16)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_integer_input_matches_float32(device, dtype):
    # Integer images (uint8 in particular) must give exactly the float32 result, not zeros.
    info = torch.iinfo(dtype)
    lo, hi = max(info.min, -1000), min(info.max, 1000)
    img = torch.randint(lo, hi + 1, (2, 3, 16, 24), dtype=dtype, device=device)
    fwd = DCT2d(kernel_size=8, selections=16).to(device)
    out = fwd(img)
    assert out.dtype == torch.float32
    assert torch.equal(out, fwd(img.float()))
    assert out.abs().max().item() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_uint8_roundtrip(device):
    img = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8, device=device)
    recon = IDCT2d(kernel_size=8).to(device)(DCT2d(kernel_size=8).to(device)(img))
    assert torch.equal(recon.round().clamp(0, 255).to(torch.uint8), img)


@pytest.mark.parametrize("device", DEVICES)
def test_idct_integer_coefficients(device):
    inv = IDCT2d(kernel_size=8, selections=16).to(device)
    q = torch.randint(-128, 128, (1, 3 * 16, 2, 2), dtype=torch.int8, device=device)
    out = inv(q)
    assert out.dtype == torch.float32
    assert torch.equal(out, inv(q.float()))


def test_bool_and_complex_inputs_raise():
    fwd, inv = DCT2d(kernel_size=8), IDCT2d(kernel_size=8)
    with pytest.raises(TypeError):
        fwd(torch.zeros(1, 1, 8, 8, dtype=torch.bool))
    with pytest.raises(TypeError):
        fwd(torch.zeros(1, 1, 8, 8, dtype=torch.complex64))
    with pytest.raises(TypeError):
        inv(torch.zeros(1, 64, 1, 1, dtype=torch.bool))


# ---- functional API ----

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [1, 2, 4])
@pytest.mark.parametrize("norm", ["ortho", "backward"])
@pytest.mark.parametrize("sel", [None, 10])
def test_functional_matches_module(device, d, norm, sel):
    kw = dict(dct=d, kernel_size=8, selections=sel, norm=norm)
    x = torch.randn(2, 3, 16, 24, device=device)
    y = DCT2d(**kw).to(device)(x)
    assert torch.equal(dct2d(x, **kw), y)
    assert torch.equal(idct2d(y, **kw), IDCT2d(**kw).to(device)(y))


def test_functional_positional_args_follow_module_order():
    x = torch.randn(1, 2, 16, 16)
    assert torch.equal(dct2d(x, 2, 8, 16, "ortho"), DCT2d(2, 8, 16, "ortho")(x))


def test_functional_namespace():
    from dct import functional as DF
    assert DF.dct2d is dct2d and DF.idct2d is idct2d
    assert dct.functional.dct2d is dct2d


def test_functional_roundtrip_and_uint8():
    img = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    coeffs = dct2d(img)
    assert torch.equal(coeffs, dct2d(img.float()))
    assert torch.equal(idct2d(coeffs).round().clamp(0, 255).to(torch.uint8), img)


def test_functional_gradient():
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    idct2d(dct2d(x)).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-5)   # orthonormal round trip = identity


def test_functional_kernel_survives_inference_mode():
    # A kernel first cached under inference_mode must still work for autograd later.
    with torch.inference_mode():
        dct2d(torch.randn(1, 1, 8, 8), kernel_size=4, selections=7)
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    dct2d(x, kernel_size=4, selections=7).sum().backward()
    assert x.grad is not None


def test_functional_errors_name_the_function():
    with pytest.raises(RuntimeError, match="dct2d only supports"):
        dct2d(torch.randn(3, 8, 8))
    with pytest.raises(RuntimeError, match="divisible by selections"):
        idct2d(torch.randn(1, 24, 2, 2), selections=16)
    with pytest.raises(ValueError):
        dct2d(torch.randn(1, 1, 8, 8), norm="nope")


@pytest.mark.parametrize("sel", [0, 65, 2.5])
def test_invalid_selections_raise(sel):
    with pytest.raises(ValueError, match="selections"):
        DCT2d(kernel_size=8, selections=sel)
    with pytest.raises(ValueError, match="selections"):
        dct2d(torch.randn(1, 1, 8, 8), selections=sel)


# ---- per-channel selections ----

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sels", [[32, 8, 8], [1, 64, 10], [16, 16, 16], [64, 64, 64]])
def test_per_channel_selections_match_per_channel_transforms(device, sels):
    x = torch.randn(2, 3, 16, 24, device=device)
    ref = torch.cat([dct2d(x[:, c:c + 1], selections=s) for c, s in enumerate(sels)], dim=1)
    y = dct2d(x, selections=sels)
    assert y.shape == (2, sum(sels), 2, 3)
    assert torch.equal(y, ref)
    assert torch.equal(DCT2d(selections=sels).to(device)(x), y)

    rec_ref = torch.cat([idct2d(ref[:, sum(sels[:c]):sum(sels[:c + 1])], selections=s)
                         for c, s in enumerate(sels)], dim=1)
    rec = idct2d(y, selections=sels)
    assert rec.shape == x.shape
    assert (rec - rec_ref).abs().max().item() < 1e-5
    assert torch.equal(IDCT2d(selections=sels).to(device)(y), rec)


def test_per_channel_full_selection_is_lossless_and_matches_int():
    x = torch.randn(1, 3, 16, 16)
    assert torch.equal(dct2d(x, selections=[64, 64, 64]), dct2d(x))
    assert (idct2d(dct2d(x, selections=[64, 64, 64]), selections=[64, 64, 64]) - x).abs().max() < 1e-4


def test_per_channel_uint8_and_gradient():
    img = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    assert torch.equal(dct2d(img, selections=[32, 8, 8]), dct2d(img.float(), selections=[32, 8, 8]))
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    idct2d(dct2d(x, selections=[32, 8, 8]), selections=[32, 8, 8]).pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    y = torch.randn(2, 48, 2, 2, requires_grad=True)
    idct2d(y, selections=[32, 8, 8]).sum().backward()
    assert y.grad.shape == y.shape


@pytest.mark.parametrize("device", DEVICES)
def test_per_channel_module_buffers(device):
    m = DCT2d(selections=(32, 8, 8)).to(device)
    assert m.selections == [32, 8, 8]
    assert "selections=[32, 8, 8]" in repr(m)
    assert list(m.state_dict().keys()) == ["kernel"]          # index is derived, not saved
    assert tuple(m.kernel.shape) == (32, 1, 8, 8)
    assert m.index.device.type == torch.device(device).type
    m.load_state_dict(DCT2d(selections=[32, 8, 8]).state_dict())


def test_per_channel_errors():
    with pytest.raises(RuntimeError, match="3 channels but selections has 2"):
        dct2d(torch.randn(1, 3, 8, 8), selections=[8, 8])
    with pytest.raises(RuntimeError, match=r"sum\(selections\)"):
        idct2d(torch.randn(1, 40, 1, 1), selections=[32, 4, 8])
    with pytest.raises(RuntimeError, match="3 channels but selections has 1"):
        DCT2d(selections=[8])(torch.randn(1, 3, 8, 8))


@pytest.mark.parametrize("sels", [[], [0, 8], [8, 65], [8, 2.5], [True, 8], "abc"])
def test_per_channel_invalid_values_raise(sels):
    with pytest.raises(ValueError, match="selections"):
        DCT2d(selections=sels)
    with pytest.raises(ValueError, match="selections"):
        dct2d(torch.randn(1, 2, 8, 8), selections=sels)


@pytest.mark.parametrize("selections", [None, [32, 8, 8]])
def test_quantized_tensor_input_is_dequantized(selections):
    # torch.ao quantized tensors (e.g. after a QuantStub) are transformed as their float values.
    x = torch.randn(1, 3, 16, 16)
    xq = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
    y = dct2d(xq, selections=selections)
    assert y.dtype == torch.float32
    assert torch.equal(y, dct2d(xq.dequantize(), selections=selections))
    yq = torch.quantize_per_tensor(y, 0.1, 0, torch.qint8)
    assert torch.equal(idct2d(yq, selections=selections), idct2d(yq.dequantize(), selections=selections))
