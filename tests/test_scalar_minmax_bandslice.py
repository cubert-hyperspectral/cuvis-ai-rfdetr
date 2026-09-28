"""ScalarMinMaxBandSlice: golden reference (training-exporter recipe), port contract, band order, wavelength input forms."""

import numpy as np
import torch

from cuvis_ai_rfdetr.node.scalar_minmax_bandslice import ScalarMinMaxBandSlice

WL = (430 + 8 * np.arange(61)).astype(
    np.int32
)  # 430..910 nm -> nearest to 640/550/470 = [26, 15, 5]; (C,) np.int32 as CU3SDataNode emits


def _cube(batch=2):
    g = torch.Generator().manual_seed(0)
    return torch.rand((batch, 4, 5, 61), generator=g) * 3000.0 + 100.0  # raw-count-like values


def test_golden_reference_matches_training_exporter():
    cube = _cube()
    out = ScalarMinMaxBandSlice(bands_nm=(640.0, 550.0, 470.0)).forward(cube=cube, wavelengths=WL)[
        "rgb_image"
    ]
    c = cube.numpy()
    exp = np.empty((2, 4, 5, 3), np.float32)
    for b in range(2):
        lo, hi = c[b].min(), c[b].max()  # ONE scalar over all bands — the exporter recipe
        exp[b] = ((c[b] - lo) / (hi - lo))[..., [26, 15, 5]]
    assert torch.allclose(out, torch.from_numpy(exp), atol=1e-6)


def test_port_contract():
    out = ScalarMinMaxBandSlice().forward(cube=_cube(), wavelengths=WL)["rgb_image"]
    assert out.shape == (2, 4, 5, 3)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_band_order_follows_bands_nm():
    cube = _cube()
    a = ScalarMinMaxBandSlice(bands_nm=(640.0, 550.0, 470.0)).forward(cube=cube, wavelengths=WL)[
        "rgb_image"
    ]
    b = ScalarMinMaxBandSlice(bands_nm=(470.0, 550.0, 640.0)).forward(cube=cube, wavelengths=WL)[
        "rgb_image"
    ]
    assert torch.equal(a[..., 0], b[..., 2]) and torch.equal(a[..., 1], b[..., 1])


def test_accepts_batched_tensor_wavelengths():
    cube = _cube()
    a = ScalarMinMaxBandSlice().forward(cube=cube, wavelengths=WL)["rgb_image"]
    b = ScalarMinMaxBandSlice().forward(
        cube=cube, wavelengths=torch.from_numpy(WL)[None].repeat(2, 1)
    )["rgb_image"]
    assert torch.equal(a, b)
