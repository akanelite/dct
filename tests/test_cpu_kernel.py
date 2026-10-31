"""The AVX2 CPU kernel (dct._cpu) against the PyTorch implementation.

Skipped when the kernel is unavailable: build it with ``DCT_BUILD_CPU_EXT=1 pip install
--no-build-isolation .``, or run the tests with ``DCT_CPU_JIT=1``.
"""

import pytest
import torch

from dct import DCT2d, IDCT2d, _cpu, dct2d, idct2d
from dct.functional import _cpu_matrices, _cpu_selections

pytestmark = pytest.mark.skipif(not _cpu.is_available(),
                                reason=f"CPU kernel unavailable: {_cpu.load_error!r}")

NORMS = ["ortho", "backward", "forward"]
# Includes widths that are not a multiple of 64 (the < 8 blocks tail) and a single block.
SHAPES = [(2, 3, 16, 64), (1, 2, 24, 40), (2, 1, 8, 8), (1, 3, 64, 136)]


def reference(fn, *args, **kw):
    # The same call through the PyTorch implementation, in float64.
    old, _cpu.enabled = _cpu.enabled, False
    try:
        args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
        return fn(*args, **kw)
    finally:
        _cpu.enabled = old


def rel_err(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-12)).item()


@pytest.fixture
def calls(monkeypatch):
    # Counts calls into the extension, to check that the kernel is really used.
    counts = {"forward": 0, "inverse": 0}
    ext = _cpu.get()

    class Spy:
        def forward(self, *a):
            counts["forward"] += 1
            return ext.forward(*a)

        def inverse(self, *a):
            counts["inverse"] += 1
            return ext.inverse(*a)

    monkeypatch.setattr(_cpu, "_ext", Spy())
    return counts


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sel", [None, 16, 1])
def test_matches_pytorch(norm, shape, sel, calls):
    x = torch.randn(*shape)
    y = dct2d(x, selections=sel, norm=norm)
    assert rel_err(y, reference(dct2d, x, selections=sel, norm=norm)) < 1e-5
    r = idct2d(y, selections=sel, norm=norm)
    assert rel_err(r, reference(idct2d, y, selections=sel, norm=norm)) < 1e-5
    assert calls == {"forward": 1, "inverse": 1}


@pytest.mark.parametrize("norm", NORMS)
def test_modules_use_kernel_and_roundtrip(norm, calls):
    x = torch.randn(2, 3, 32, 48)
    recon = IDCT2d(norm=norm)(DCT2d(norm=norm)(x))
    assert (recon - x).abs().max().item() < 1e-5
    assert calls == {"forward": 1, "inverse": 1}


def test_not_used_outside_its_case(calls):
    x = torch.randn(1, 1, 16, 16)
    dct2d(x, dct=3)
    dct2d(x, kernel_size=4)
    dct2d(x.double())
    _cpu.enabled = False
    try:
        dct2d(x)
    finally:
        _cpu.enabled = True
    assert calls["forward"] == 0


@pytest.mark.parametrize("sels", [[32, 8, 1], [1, 64, 10], [16, 16, 16], [64, 64, 64]])
@pytest.mark.parametrize("shape", [(2, 3, 16, 24), (1, 3, 8, 136)])
def test_per_channel_selections(sels, shape, calls):
    x = torch.randn(*shape)
    y = dct2d(x, selections=sels)
    assert y.shape == (shape[0], sum(sels), shape[2] // 8, shape[3] // 8)
    assert rel_err(y, reference(dct2d, x, selections=sels)) < 1e-5
    r = idct2d(y, selections=sels)
    assert rel_err(r, reference(idct2d, y, selections=sels)) < 1e-5
    assert calls == {"forward": 1, "inverse": 1}


def test_kernel_rejects_bad_selection_counts():
    ext = _cpu.get()
    m, a, perm = _cpu_matrices("ortho")
    x = torch.randn(1, 2, 8, 8)
    with pytest.raises(RuntimeError, match="one entry per input channel"):
        ext.forward(x, m, perm, _cpu_selections((8,)))
    with pytest.raises(RuntimeError, match=r"\[1, 64\]"):
        ext.forward(x, m, perm, _cpu_selections((8, 65)))
    with pytest.raises(RuntimeError, match="sum"):
        ext.inverse(torch.randn(1, 15, 1, 1), a, perm, _cpu_selections((8, 8)))


def test_uint8_and_noncontiguous_inputs():
    x = torch.randint(0, 256, (2, 3, 16, 32), dtype=torch.uint8)
    y = dct2d(x)
    assert y.dtype == torch.float32
    assert rel_err(y, reference(dct2d, x.float())) < 1e-5
    xc = torch.randn(2, 3, 16, 32).to(memory_format=torch.channels_last)
    assert rel_err(dct2d(xc), reference(dct2d, xc)) < 1e-5
    yt = torch.randn(2, 64 * 3, 4, 2).transpose(2, 3)          # non-contiguous coefficients
    assert rel_err(idct2d(yt), reference(idct2d, yt)) < 1e-5


def test_empty_batch():
    assert dct2d(torch.randn(0, 3, 16, 16)).shape == (0, 192, 2, 2)
    assert idct2d(torch.randn(0, 192, 2, 2)).shape == (0, 3, 16, 16)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("sel", [None, 10])
def test_gradients_match_pytorch(norm, sel):
    x = torch.randn(2, 3, 16, 24, requires_grad=True)
    y = dct2d(x, selections=sel, norm=norm)
    gy = torch.randn_like(y)
    (gx,) = torch.autograd.grad(y, x, gy)

    xr = x.detach().double().requires_grad_()
    (gx_ref,) = torch.autograd.grad(reference(dct2d, xr, selections=sel, norm=norm), xr, gy.double())
    assert rel_err(gx, gx_ref) < 1e-5

    c = torch.randn_like(y, requires_grad=True)
    r = idct2d(c, selections=sel, norm=norm)
    gr = torch.randn_like(r)
    (gc,) = torch.autograd.grad(r, c, gr)
    cr = c.detach().double().requires_grad_()
    (gc_ref,) = torch.autograd.grad(reference(idct2d, cr, selections=sel, norm=norm), cr, gr.double())
    assert rel_err(gc, gc_ref) < 1e-5


def test_backward_is_the_adjoint():
    # gradcheck needs float64, which the kernel does not take; instead check the adjoint
    # identity <F x, y> = <x, F^T y> that the backward relies on.
    x = torch.randn(1, 2, 16, 16)
    sels = _cpu_selections((20, 7))                     # per-channel, to cover both layouts
    y = torch.randn(1, 27, 2, 2)
    for norm in NORMS:
        m, _, perm = _cpu_matrices(norm)
        lhs = (_cpu.dct2d(x, m, perm, sels).double() * y.double()).sum()
        rhs = (x.double() * _cpu.idct2d(y, m, perm, sels).double()).sum()
        assert abs(lhs - rhs) / abs(lhs) < 1e-5


def test_double_backward_and_forward_ad():
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    y = dct2d(x, norm="backward")
    (g,) = torch.autograd.grad((y ** 2).sum(), x, create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), x)
    xr = x.detach().double().requires_grad_()
    yr = reference(dct2d, xr, norm="backward")
    (gr,) = torch.autograd.grad((yr ** 2).sum(), xr, create_graph=True)
    (ggr,) = torch.autograd.grad(gr.sum(), xr)
    assert rel_err(gg, ggr) < 1e-5

    t = torch.randn(1, 1, 16, 16)
    _, jvp = torch.func.jvp(lambda v: dct2d(v), (x.detach(),), (t,))
    assert rel_err(jvp, reference(dct2d, t)) < 1e-5


def test_inference_mode():
    with torch.inference_mode():
        x = torch.randn(1, 3, 16, 16)
        assert (idct2d(dct2d(x)) - x).abs().max().item() < 1e-5


@pytest.mark.parametrize("in_dim", [0, 2])
@pytest.mark.parametrize("sel", [None, [32, 8, 1]])
def test_vmap_matches_loop(in_dim, sel, calls):
    xs = torch.randn(4, 2, 3, 16, 24).movedim(0, in_dim)
    y = torch.func.vmap(lambda x: dct2d(x, selections=sel), in_dims=in_dim)(xs)
    loop = torch.stack([reference(dct2d, xs.select(in_dim, i), selections=sel) for i in range(4)])
    assert rel_err(y, loop) < 1e-5
    r = torch.func.vmap(lambda c: idct2d(c, selections=sel))(y)
    assert rel_err(r, torch.stack([reference(idct2d, c, selections=sel) for c in y])) < 1e-5
    assert calls == {"forward": 1, "inverse": 1}      # one kernel call for the whole vmapped batch


def test_per_sample_gradients_and_jacobians():
    w = torch.randn(1, 192, 2, 2)

    def loss(x):
        return (dct2d(x[None]) * w).pow(2).sum()

    xs = torch.randn(5, 3, 16, 16)
    g = torch.func.vmap(torch.func.grad(loss))(xs)
    g_ref = torch.stack([torch.func.grad(lambda x: reference(loss, x))(x.double()) for x in xs])
    assert rel_err(g, g_ref) < 1e-5

    x = torch.randn(1, 1, 8, 8)
    jac = torch.func.jacrev(lambda v: idct2d(dct2d(v, norm="backward"), norm="backward"))(x)
    assert (jac.reshape(64, 64) - torch.eye(64)).abs().max().item() < 1e-5
    hess = torch.func.hessian(lambda v: dct2d(v).pow(2).sum())(x)
    assert (hess.reshape(64, 64) - 2 * torch.eye(64)).abs().max().item() < 1e-5   # orthonormal


# ---- dtypes read and written by the kernel directly ----

HALF = [torch.float16, torch.bfloat16]
INTS = [torch.uint8, torch.int8, torch.int16, torch.int32]


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("sel", [None, [32, 8, 1]])
@pytest.mark.parametrize("shape", [(2, 3, 16, 72), (1, 3, 8, 24)])
def test_half_dtypes_round_once(dtype, sel, shape, calls):
    # Computed in float32 and rounded once on store, so the result equals the float32 kernel
    # rounded to the dtype bit for bit (this also checks round-to-nearest-even for bfloat16).
    x = torch.randn(*shape).to(dtype)
    y = dct2d(x, selections=sel)
    assert y.dtype == dtype
    assert torch.equal(y, dct2d(x.float(), selections=sel).to(dtype))
    r = idct2d(y, selections=sel)
    assert r.dtype == dtype
    assert torch.equal(r, idct2d(y.float(), selections=sel).to(dtype))
    assert calls == {"forward": 2, "inverse": 2}


@pytest.mark.parametrize("dtype", HALF)
def test_half_dtypes_accuracy_and_nan(dtype):
    x = torch.randn(1, 3, 16, 24).to(dtype)
    tol = 2.0 ** (-10 if dtype == torch.float16 else -7)
    assert rel_err(dct2d(x), reference(dct2d, x.double())) < tol
    x[0, 0, 0, 0] = float("nan")                  # only the first block becomes NaN
    y = dct2d(x)
    assert y[0, :64, 0, 0].isnan().all() and not y[0, :64, 0, 1:].isnan().any()
    assert not y[0, 64:].isnan().any()


@pytest.mark.parametrize("dtype", HALF)
def test_half_dtypes_gradients(dtype):
    x = torch.randn(1, 2, 16, 16).to(dtype).requires_grad_()
    y = dct2d(x, selections=10)
    gy = torch.randn_like(y)
    (gx,) = torch.autograd.grad(y, x, gy)
    assert gx.dtype == dtype
    assert torch.equal(gx, idct2d(gy.float(), selections=10, norm="ortho").to(dtype))   # ortho: adjoint = inverse


@pytest.mark.parametrize("dtype", INTS)
@pytest.mark.parametrize("sel", [None, 16, [32, 8, 1]])
def test_integer_dtypes_read_directly(dtype, sel, calls):
    info = torch.iinfo(dtype)
    lo, hi = max(info.min, -1000), min(info.max, 1000)
    x = torch.randint(lo, hi + 1, (2, 3, 16, 72), dtype=dtype)    # a full group of 8 blocks + a tail
    y = dct2d(x, selections=sel)
    assert y.dtype == torch.float32
    assert torch.equal(y, dct2d(x.float(), selections=sel))
    c = torch.randint(lo, hi + 1, tuple(y.shape), dtype=dtype)    # e.g. quantized coefficients
    r = idct2d(c, selections=sel)
    assert r.dtype == torch.float32
    assert torch.equal(r, idct2d(c.float(), selections=sel))
    assert calls == {"forward": 2, "inverse": 2}                   # no conversion pass + fallback


def test_int64_is_converted_then_uses_kernel(calls):
    x = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.int64)
    assert torch.equal(dct2d(x), dct2d(x.float()))
    assert calls["forward"] == 2
