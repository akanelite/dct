import numpy as np
import pytest
import scipy.fft
import torch

from dct import DCT2d, IDCT2d
from dct.dct import _zigzag_permutation

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
    mine = fwd(block).reshape(-1).numpy()           # zigzag 순서
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
    # 'kernel' 버퍼가 정확히 하나, 2D 형태(k², 1, k, k)
    buffers = dict(fwd.named_buffers())
    assert list(buffers.keys()) == ["kernel"]
    assert tuple(buffers["kernel"].shape) == (64, 1, 8, 8)
    assert "norm='ortho'" in repr(fwd)
    # selections 속성 노출
    assert DCT2d(kernel_size=8).selections == 64
    assert DCT2d(kernel_size=8, selections=10).selections == 10
