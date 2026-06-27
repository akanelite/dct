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


def test_double_precision_roundtrip():
    # 커널 buffer는 float32로 초기화되므로 .double()은 dtype만 float64로 올린다
    # (정밀도는 float32에 묶임). 여기서 검증하는 것은 fp64 dtype 전파와 동작이다.
    fwd = DCT2d(kernel_size=8, norm="ortho").double()
    inv = IDCT2d(kernel_size=8, norm="ortho").double()
    x = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    out = fwd(x)
    recon = inv(out)
    assert out.dtype == torch.float64
    assert recon.dtype == torch.float64
    assert (recon - x).abs().max().item() < 1e-5


def test_half_precision_input():
    # 커널 buffer는 float32, 입력은 float16 — forward 캐스팅으로 동작해야 함.
    devs = [d for d in DEVICES if d != "cpu"]  # CPU는 half conv 미지원
    if not devs:
        pytest.skip("no half-capable device")
    dev = devs[0]
    fwd = DCT2d(kernel_size=8, norm="ortho").to(dev)
    out = fwd(torch.randn(1, 3, 16, 16, device=dev, dtype=torch.float16))
    assert out.dtype == torch.float16


def test_dct_rejects_bad_input():
    fwd = DCT2d(kernel_size=8)
    with pytest.raises(RuntimeError):
        fwd(torch.randn(3, 32, 32))          # 4D 아님
    with pytest.raises(RuntimeError):
        fwd(torch.randn(1, 3, 30, 30))       # 해상도 비가분


def test_idct_rejects_bad_input():
    inv = IDCT2d(kernel_size=8, selections=16)   # 채널은 16의 배수여야 함
    with pytest.raises(RuntimeError, match="4D"):
        inv(torch.randn(3, 16, 4, 4)[0])         # 3D → 4D 아님
    # 비가분 채널은 명확한 메시지로 거부해야 한다(난해한 conv 에러가 아니라).
    with pytest.raises(RuntimeError, match="divisible by selections"):
        inv(torch.randn(2, 24, 4, 4))            # 24 % 16 != 0 → 채널 비가분
