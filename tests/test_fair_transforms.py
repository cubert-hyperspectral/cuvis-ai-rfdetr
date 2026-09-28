"""Tests for the fair-robust training transforms: RandomZoom, RandomShading, RandomGammaContrast,
RandomGaussianBlur (``cuvis_ai_rfdetr.transforms``).

Pure-tensor checks: registry build, prob=0 passthrough, shape/dtype contract, determinism under a
seeded generator, cube/mask alignment, and each transform's defining property. Skipped where
cuvis-ai-augment (the registry/base) is absent — same convention as ``test_transforms.py``.
"""

from __future__ import annotations

import pytest
import torch

try:
    import cuvis_ai_augment  # noqa: F401

    AUGMENT = True
except ImportError:  # pragma: no cover
    AUGMENT = False

pytestmark = [pytest.mark.skipif(not AUGMENT, reason="needs cuvis-ai-augment")]

NAMES = ("RandomZoom", "RandomShading", "RandomGammaContrast", "RandomGaussianBlur")


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _square_scene(
    b: int = 2, h: int = 64, w: int = 80, c: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """A bright labelled square (label 1) on a dark background (label 0)."""
    cube = torch.full((b, h, w, c), 0.1)
    mask = torch.zeros((b, h, w), dtype=torch.int32)
    cube[:, 16:48, 20:60] = 0.9
    mask[:, 16:48, 20:60] = 1
    return cube, mask


def _build(name: str, **kw):
    from cuvis_ai_augment.transforms.base import build_transform

    import cuvis_ai_rfdetr.transforms  # noqa: F401 — registers on import

    return build_transform({"type": name, **kw})


@pytest.mark.parametrize("name", NAMES)
def test_registered_and_prob_zero_is_passthrough(name: str) -> None:
    t = _build(name, prob=0.0)
    cube, mask = _square_scene()
    out, m = t(cube, mask, _gen())
    assert torch.equal(out, cube) and torch.equal(m, mask)


@pytest.mark.parametrize("name", NAMES)
def test_shape_dtype_contract_and_determinism(name: str) -> None:
    t = _build(name, prob=1.0)
    cube, mask = _square_scene()
    a, ma = t(cube, mask, _gen(3))
    b, mb = t(cube, mask, _gen(3))
    assert a.shape == cube.shape and a.dtype == cube.dtype
    assert ma.shape == mask.shape and ma.dtype == mask.dtype
    assert torch.equal(a, b) and torch.equal(ma, mb)
    out_none, m_none = t(cube, None, _gen(3))
    assert m_none is None and out_none.shape == cube.shape


@pytest.mark.parametrize("name", NAMES)
def test_rejects_bad_mask_shape(name: str) -> None:
    t = _build(name, prob=1.0)
    cube, _ = _square_scene()
    with pytest.raises(ValueError):
        t(cube, torch.zeros((2, 10, 10), dtype=torch.int32), _gen())


def test_zoom_out_shrinks_object_and_keeps_alignment() -> None:
    t = _build("RandomZoom", scale_range=(0.5, 0.5), prob=1.0)
    cube, mask = _square_scene()
    out, m = t(cube, mask, _gen(1))
    area_in, area_out = int(mask[0].sum()), int(m[0].sum())
    assert abs(area_out - 0.25 * area_in) <= 0.08 * area_in  # 0.5 x 0.5 of the square
    bright = out[0, ..., 0] > 0.5
    assert (bright & (m[0] > 0)).sum() / (bright | (m[0] > 0)).sum() > 0.85  # cube/mask aligned
    assert set(m.unique().tolist()) <= {0, 1}
    # every background pixel (pasted-window background or median-filled canvas) is the dark 0.1
    bg = (m[0] == 0) & ~bright
    assert (torch.abs(out[0][bg] - 0.1) < 0.05).float().mean() > 0.97


def test_zoom_in_enlarges_labelled_fraction() -> None:
    t = _build("RandomZoom", scale_range=(2.0, 2.0), prob=1.0)
    cube = torch.rand((1, 64, 64, 3), generator=_gen(5))
    mask = torch.ones((1, 64, 64), dtype=torch.int32)
    out, m = t(cube, mask, _gen(2))
    assert torch.equal(m, mask)  # all-foreground stays all-foreground
    assert out.shape == cube.shape and not torch.equal(out, cube)


def test_zoom_identity_band_is_noop() -> None:
    t = _build("RandomZoom", scale_range=(1.0, 1.0), prob=1.0)
    cube, mask = _square_scene()
    out, m = t(cube, mask, _gen())
    assert torch.equal(out, cube) and torch.equal(m, mask)


def test_zoom_rejects_bad_range() -> None:
    from cuvis_ai_rfdetr.transforms import RandomZoom

    with pytest.raises(ValueError):
        RandomZoom(scale_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        RandomZoom(scale_range=(2.0, 1.0))


def test_shading_only_darkens_within_strength_and_varies_spatially() -> None:
    t = _build("RandomShading", strength_range=(0.5, 0.5), grid=3, prob=1.0)
    cube = torch.ones((2, 40, 50, 4))
    mask = torch.randint(0, 3, (2, 40, 50), generator=_gen(), dtype=torch.int32)
    out, m = t(cube, mask, _gen(7))
    assert torch.equal(m, mask)
    assert (out <= 1.0 + 1e-6).all() and (out >= 0.5 - 1e-6).all()
    assert out[0, ..., 0].std() > 0.01  # a field, not a global gain
    assert torch.allclose(out[..., 0], out[..., 3])  # same field for every band


def test_shading_rejects_strength_above_one() -> None:
    from cuvis_ai_rfdetr.transforms import RandomShading

    with pytest.raises(ValueError):
        RandomShading(strength_range=(0.5, 1.5))


def test_gamma_contrast_identity_and_known_gamma() -> None:
    cube = torch.rand((2, 16, 16, 3), generator=_gen(9))
    ident = _build(
        "RandomGammaContrast", gamma_range=(1.0, 1.0), contrast_range=(1.0, 1.0), prob=1.0
    )
    out, _ = ident(cube, None, _gen())
    assert torch.allclose(out, cube, atol=1e-6)
    sq = _build("RandomGammaContrast", gamma_range=(2.0, 2.0), contrast_range=(1.0, 1.0), prob=1.0)
    out2, _ = sq(torch.full((1, 4, 4, 1), 0.5), None, _gen())
    assert torch.allclose(out2, torch.full((1, 4, 4, 1), 0.25), atol=1e-6)


def test_contrast_preserves_channel_mean_and_floors_at_zero() -> None:
    cube = torch.rand((1, 32, 32, 2), generator=_gen(4))
    t = _build("RandomGammaContrast", gamma_range=(1.0, 1.0), contrast_range=(1.2, 1.2), prob=1.0)
    out, _ = t(cube, None, _gen())
    assert (out >= 0).all()
    assert torch.allclose(out.mean(dim=(1, 2)), cube.mean(dim=(1, 2)), atol=0.02)


def test_blur_smooths_noise_keeps_constant_and_mask() -> None:
    t = _build("RandomGaussianBlur", sigma_range=(1.5, 1.5), prob=1.0)
    noise = torch.rand((1, 48, 48, 2), generator=_gen(11))
    mask = torch.randint(0, 2, (1, 48, 48), generator=_gen(), dtype=torch.int32)
    out, m = t(noise, mask, _gen())
    assert torch.equal(m, mask)
    assert out.std() < 0.6 * noise.std()
    assert abs(float(out.mean() - noise.mean())) < 0.01
    const = torch.full((1, 20, 20, 3), 0.4)
    out_c, _ = t(const, None, _gen())
    assert torch.allclose(out_c, const, atol=1e-6)


def test_per_sample_decisions_are_independent() -> None:
    t = _build("RandomGaussianBlur", sigma_range=(1.0, 1.0), prob=0.5)
    cube = torch.rand((8, 24, 24, 1), generator=_gen(13))
    out, _ = t(cube, None, _gen(21))
    changed = [not torch.equal(out[i], cube[i]) for i in range(8)]
    assert any(changed) and not all(changed)
