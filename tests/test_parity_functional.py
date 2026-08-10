"""Parity-helper contract tests: exact top-fraction scoring, JPEG round-trip
determinism/equivalence, band-index resolution, and the PercentileComposite
node arithmetic. Everything runs without the rfdetr package installed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_rfdetr.functional import (
    jpeg_roundtrip,
    resolve_band_indices,
    top_frac_mean,
)
from cuvis_ai_rfdetr.node.percentile_composite import PercentileComposite


# ------------------------------------------------------------ top_frac_mean
def test_top_frac_mean_exact_top_k() -> None:
    """Integer-floor rule: a 987x405 map at 0.001 averages exactly 400 pixels."""
    rng = np.random.default_rng(7)
    a = rng.random((987, 405)).astype(np.float32)
    flat = np.sort(a.ravel())
    n = flat.size
    k = min(max(int(0.999 * n), 0), n - 1)
    assert n - k == 400  # the protocol constant for this geometry
    expected = float(flat[k:].mean())
    assert top_frac_mean(a, 0.001) == expected
    # torch input takes the identical path
    assert top_frac_mean(torch.from_numpy(a), 0.001) == expected


def test_top_frac_mean_brute_force_and_edges() -> None:
    rng = np.random.default_rng(3)
    a = rng.random((50, 40)).astype(np.float32)
    n = a.size
    for tf in (0.001, 0.01, 0.1, 0.5, 1.0):
        k = min(max(int((1.0 - tf) * n), 0), n - 1)
        expected = float(np.sort(a.ravel())[k:].mean())
        assert top_frac_mean(a, tf) == expected
    # tf=1.0 -> k=0 -> mean of everything
    assert top_frac_mean(a, 1.0) == pytest.approx(float(np.sort(a.ravel()).mean()))
    # single element and empty
    assert top_frac_mean(np.array([[0.25]], dtype=np.float32), 0.001) == pytest.approx(0.25)
    assert top_frac_mean(np.zeros((0,), dtype=np.float32), 0.001) == 0.0


def test_top_frac_mean_sparse_map_dilution() -> None:
    """A single confident blob is diluted by the zeros in the top fraction."""
    a = np.zeros((987, 405), dtype=np.float32)
    a[:10, :10] = 0.9  # 100 hot pixels, top-400 window holds 300 zeros
    got = top_frac_mean(a, 0.001)
    assert got == pytest.approx(0.9 * 100 / 400)


# ----------------------------------------------------------- jpeg_roundtrip
def test_jpeg_roundtrip_deterministic_and_lossy() -> None:
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    a = jpeg_roundtrip(frame, 95)
    b = jpeg_roundtrip(frame, 95)
    assert a.shape == frame.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)  # deterministic under a fixed Pillow
    assert not np.array_equal(a, frame)  # and genuinely lossy on noise


def test_jpeg_roundtrip_equals_disk_roundtrip(tmp_path) -> None:
    """In-memory must be byte-identical to save-as-.jpg + reopen."""
    from PIL import Image

    rng = np.random.default_rng(13)
    frame = rng.integers(0, 256, size=(405, 405, 3), dtype=np.uint8)
    p = tmp_path / "tile.jpg"
    Image.fromarray(frame).save(p, quality=95)
    with Image.open(p) as im:
        disk = np.asarray(im.convert("RGB"))
    assert np.array_equal(jpeg_roundtrip(frame, 95), disk)


def test_jpeg_roundtrip_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        jpeg_roundtrip(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        jpeg_roundtrip(np.zeros((4, 4), dtype=np.uint8))


# ---------------------------------------------------- resolve_band_indices
def test_resolve_band_indices_nearest() -> None:
    wl = np.arange(350, 1004, 4)  # 4 nm grid
    idx = resolve_band_indices(wl, (574.0, 500.0, 470.0))
    # 574 and 470 sit exactly on the grid; 500 is equidistant to 498/502 and
    # argmin tie-breaks to the FIRST (lower) index — the exporter rule.
    assert [int(wl[i]) for i in idx] == [574, 498, 470]
    with pytest.raises(ValueError, match="empty"):
        resolve_band_indices(np.array([]), (500.0,))


# ------------------------------------------------------ PercentileComposite
def _reference_composite(cube: np.ndarray, wl: np.ndarray, bands, p_low, p_high) -> np.ndarray:
    """Independent reimplementation of the exporter arithmetic (uint8)."""
    idxs = [int(np.argmin(np.abs(wl.astype(np.int64) - nm))) for nm in bands]
    chans = []
    for i in idxs:
        ch = cube[:, :, i].astype(np.float32)
        lo, hi = np.percentile(ch, (p_low, p_high))
        chans.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    return (np.stack(chans, -1) * 255.0 + 0.5).astype(np.uint8)


def test_percentile_composite_matches_reference() -> None:
    rng = np.random.default_rng(5)
    cube = (rng.random((2, 60, 40, 8)) * 4000.0).astype(np.float32)
    wl = np.linspace(450, 900, 8).astype(np.int32)
    node = PercentileComposite(bands_nm=(574.0, 500.0, 470.0))
    out = node.forward(
        cube=torch.from_numpy(cube),
        wavelengths=torch.from_numpy(np.stack([wl, wl])),
    )["rgb_image"]
    assert out.shape == (2, 60, 40, 3)
    assert out.dtype == torch.float32
    for b in range(2):
        ref = _reference_composite(cube[b], wl, (574.0, 500.0, 470.0), 1.0, 99.0)
        got = out[b].numpy()
        assert np.array_equal(got, ref.astype(np.float32))
        # integer-valued output in [0, 255] -> uint8 conversion is byte-exact
        assert np.array_equal(got.astype(np.uint8), ref)
        assert float(got.min()) >= 0.0 and float(got.max()) <= 255.0


def test_percentile_composite_uint8_recovery_through_node_input_path() -> None:
    """The composite output survives the detectors' uint8 ingestion exactly."""
    from cuvis_ai_rfdetr.functional import to_uint8_frames

    rng = np.random.default_rng(9)
    cube = (rng.random((1, 32, 24, 5)) * 900.0).astype(np.float32)
    wl = np.array([470, 500, 574, 650, 800], dtype=np.int32)
    out = PercentileComposite(bands_nm=(574.0, 500.0, 470.0)).forward(
        cube=torch.from_numpy(cube), wavelengths=torch.from_numpy(wl[None, :])
    )["rgb_image"]
    ref = _reference_composite(cube[0], wl, (574.0, 500.0, 470.0), 1.0, 99.0)
    assert np.array_equal(to_uint8_frames(out)[0], ref)


def test_percentile_composite_degenerate_band_maps_to_zero() -> None:
    cube = np.full((1, 16, 16, 3), 7.0, dtype=np.float32)  # constant everywhere
    wl = np.array([470, 500, 574], dtype=np.int32)
    out = PercentileComposite(bands_nm=(574.0, 500.0, 470.0)).forward(
        cube=torch.from_numpy(cube), wavelengths=torch.from_numpy(wl[None, :])
    )["rgb_image"]
    assert float(out.abs().max()) == 0.0


def test_percentile_composite_validates_hparams() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        PercentileComposite(bands_nm=(574.0, 500.0))
    with pytest.raises(ValueError, match="p_low"):
        PercentileComposite(p_low=99.0, p_high=1.0)
    node = PercentileComposite(bands_nm=(650, 550, 450), p_low=2, p_high=98)
    assert node.hparams["bands_nm"] == (650.0, 550.0, 450.0)
    assert node.hparams["p_low"] == 2.0


def test_percentile_composite_rejects_bad_cube() -> None:
    node = PercentileComposite()
    with pytest.raises(ValueError, match="B, H, W, C"):
        node.forward(
            cube=torch.zeros(4, 4, 3), wavelengths=torch.tensor([[470, 500, 574]])
        )
