"""RF-DETR training transforms for cuvis-ai-augment's ``AugmentationCompose``.

Contributed through augment's official extension mechanism — list this module in
the compose node's ``extra_transform_modules`` and the transforms register under
the same ``@register`` decorator augment's own transforms use, with no change to
the augment plugin::

    AugmentationCompose(
        transforms=[{"type": "RandomMultiScaleResize", "scales": scales}],
        extra_transform_modules=["cuvis_ai_rfdetr.transforms"],
    )

``cuvis-ai-augment`` is deliberately not a pip dependency of this plugin (it is
git-tag-released); importing this module without it installed raises a clear
error. Any environment that composes transforms has augment installed anyway.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from cuvis_ai_augment.transforms.base import Transform, register
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "cuvis_ai_rfdetr.transforms requires cuvis-ai-augment (AugmentationCompose "
        "and the transform registry): pip install "
        '"cuvis-ai-augment @ git+https://github.com/cubert-hyperspectral/'
        'cuvis-ai-augment.git@v0.4.0"'
    ) from exc

from cuvis_ai_rfdetr.functional import compute_multi_scale_scales

__all__ = [
    "RandomGammaContrast",
    "RandomGaussianBlur",
    "RandomMultiScaleResize",
    "RandomShading",
    "RandomZoom",
]


def _check_range(name: str, value: Any, *, low: float = 0.0) -> tuple[float, float]:
    """Validate a ``(lo, hi)`` parameter range with ``low < lo <= hi``."""
    rng = tuple(float(v) for v in value)
    if len(rng) != 2 or rng[0] <= low or rng[1] < rng[0]:
        raise ValueError(f"{name} must be (lo, hi) with {low} < lo <= hi, got {value!r}")
    return rng[0], rng[1]


def _uniform(lo: float, hi: float, rng: torch.Generator) -> float:
    """One uniform draw in ``[lo, hi]`` from the shared generator."""
    return lo + (hi - lo) * float(torch.rand((), generator=rng))


@register("RandomMultiScaleResize")
class RandomMultiScaleResize(Transform):
    """Square multi-scale resize — the native RF-DETR training augmentation.

    Each call resizes the whole batch to one randomly drawn square size
    ``s x s`` from the native scale set (the sizes rfdetr's own dataloader
    draws from when ``multi_scale=True``): multiples of
    ``patch_size * num_windows`` centred on ``resolution``, extended by
    ``expanded_scales``. The cube is resized bilinearly, a connected mask with
    nearest-neighbour (labels preserved) — same rectangle, alignment kept.

    Pass either an explicit ``scales`` list (e.g. from
    ``RFDETRTrainable.multi_scale_scales()``, which reads the model's actual
    ``resolution`` / ``patch_size`` / ``num_windows``) or the parameters to
    compute it.

    Fidelity notes
    --------------
    * The native loop draws a scale **per image** (variable sizes are batched
      via padding masks); inside a compose the batch is already stacked, so the
      draw here is **per batch** — coarser randomization, same size
      distribution across steps.
    * The native ``scale_jitter`` alternative branch (downscale -> random-sized
      crop -> resize, 50/50 against the direct resize) is not replicated here.

    Parameters
    ----------
    scales : list[int] or None
        Explicit target sizes. When ``None``, computed from the parameters
        below via the native formula.
    resolution : int or None
        Base training resolution (required when ``scales`` is ``None``).
    expanded_scales : bool
        Native ``expanded_scales`` flag (11 offsets instead of 8).
    patch_size, num_windows : int
        The model's spatial divisibility unit (e.g. the SegMedium backbone uses
        ``12 x 2 = 24``; several detection variants use ``16 x 4 = 64``).
    prob : float
        Probability of applying the resize per batch; skipped batches pass
        through at their incoming size (the native loop always resizes).
    """

    def __init__(
        self,
        scales: list[int] | None = None,
        resolution: int | None = None,
        expanded_scales: bool = True,
        patch_size: int = 16,
        num_windows: int = 4,
        prob: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        if scales is None:
            if resolution is None:
                raise ValueError(
                    "RandomMultiScaleResize: pass either scales=[...] or resolution=..."
                )
            scales = compute_multi_scale_scales(
                int(resolution),
                expanded_scales=bool(expanded_scales),
                patch_size=int(patch_size),
                num_windows=int(num_windows),
            )
        scales = [int(s) for s in scales]
        if not scales or any(s <= 0 for s in scales):
            raise ValueError(f"RandomMultiScaleResize: invalid scales {scales!r}")
        self.scales: list[int] = scales

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # spatial-only
        self._validate_shapes(cube, mask)

        # One decision per batch: samples must share an output shape.
        if self.prob < 1.0 and bool(torch.rand((), generator=rng) >= self.prob):
            return cube, mask
        size = self.scales[int(torch.randint(len(self.scales), (1,), generator=rng))]

        resized = F.interpolate(
            cube.permute(0, 3, 1, 2),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        resized = resized.contiguous()

        mask_resized: Tensor | None = None
        if mask is not None:
            mask_resized = (
                F.interpolate(mask.unsqueeze(1).float(), size=(size, size), mode="nearest")
                .squeeze(1)
                .to(mask.dtype)
            )
        return resized, mask_resized


@register("RandomZoom")
class RandomZoom(Transform):
    """Per-sample zoom in or out at a fixed output size — simulates a camera moved up or down.

    A factor ``z`` is drawn log-uniformly from ``scale_range`` per sample (apparent object
    size relative to the input). ``z < 1`` zooms out: the frame is shrunk to ``z * (H, W)``
    and pasted at a random position on a canvas filled with the sample's per-channel median
    (the dominant background, e.g. the mat), with the mask canvas set to background 0.
    ``z > 1`` zooms in: a random ``(H, W) / z`` window is cropped and resized back to
    ``(H, W)``. The cube is resampled bilinearly (antialiased when shrinking), the mask with
    nearest-neighbour, so labels stay aligned. Output shape equals input shape.

    Parameters
    ----------
    scale_range : tuple[float, float]
        ``(lo, hi)`` zoom factors, ``0 < lo <= hi``; e.g. ``(0.5, 2.0)`` covers half to
        double apparent size. Values within 1 % of 1 leave the sample unchanged.
    prob : float
        Probability of applying the zoom per sample.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] | list[float] = (0.5, 2.0),
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        self.scale_range = _check_range("scale_range", scale_range)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # spatial-only
        self._validate_shapes(cube, mask)
        B, H, W, C = cube.shape
        apply = self._draw_apply_mask(B, rng, cube.device)
        if not apply.any():
            return cube, mask
        lo, hi = self.scale_range
        cube_out = cube.clone()
        mask_out = mask.clone() if mask is not None else None
        for b in range(B):
            if not bool(apply[b]):
                continue
            z = math.exp(_uniform(math.log(lo), math.log(hi), rng))
            if abs(z - 1.0) < 0.01:
                continue
            sample = cube[b].permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            m = mask[b].unsqueeze(0).unsqueeze(0).float() if mask is not None else None
            if z < 1.0:
                h2, w2 = max(1, round(H * z)), max(1, round(W * z))
                small = F.interpolate(
                    sample, size=(h2, w2), mode="bilinear", align_corners=False, antialias=True
                )
                fill = sample.reshape(C, -1).median(dim=1).values.view(1, C, 1, 1)
                canvas = fill.expand(1, C, H, W).clone()
                y0 = int(torch.randint(0, H - h2 + 1, (1,), generator=rng))
                x0 = int(torch.randint(0, W - w2 + 1, (1,), generator=rng))
                canvas[:, :, y0 : y0 + h2, x0 : x0 + w2] = small
                cube_out[b] = canvas[0].permute(1, 2, 0)
                if m is not None:
                    m_canvas = torch.zeros((H, W), dtype=mask.dtype, device=mask.device)
                    m_small = F.interpolate(m, size=(h2, w2), mode="nearest")[0, 0]
                    m_canvas[y0 : y0 + h2, x0 : x0 + w2] = m_small.to(mask.dtype)
                    mask_out[b] = m_canvas
            else:
                ch, cw = max(1, round(H / z)), max(1, round(W / z))
                y0 = int(torch.randint(0, H - ch + 1, (1,), generator=rng))
                x0 = int(torch.randint(0, W - cw + 1, (1,), generator=rng))
                crop = sample[:, :, y0 : y0 + ch, x0 : x0 + cw]
                up = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
                cube_out[b] = up[0].permute(1, 2, 0)
                if m is not None:
                    m_up = F.interpolate(
                        m[:, :, y0 : y0 + ch, x0 : x0 + cw], size=(H, W), mode="nearest"
                    )[0, 0]
                    mask_out[b] = m_up.to(mask.dtype)
        return cube_out, mask_out


@register("RandomShading")
class RandomShading(Transform):
    """Multiply each sample by a smooth spatial darkening field — uneven or missing light.

    A ``grid x grid`` array of gains ``1 - s * u`` (``u ~ U(0, 1)`` per cell, one strength
    ``s ~ U(strength_range)`` per sample) is upsampled bilinearly to ``(H, W)`` and applied to
    every band alike. Unlike a global gain, which a per-frame / per-channel min-max
    normalisation cancels exactly, a spatial field survives it — this is what a light going
    out or a shadowed corner looks like after normalisation. The mask is returned untouched.

    Parameters
    ----------
    strength_range : tuple[float, float]
        ``(lo, hi)`` maximum darkening, ``0 < lo <= hi <= 1`` (0.6 = a cell can drop to 40 %).
    grid : int
        Field resolution before upsampling (``>= 2``; 2 = a smooth ramp, larger = patchier).
    prob : float
        Probability of applying the shading per sample.
    """

    def __init__(
        self,
        strength_range: tuple[float, float] | list[float] = (0.2, 0.6),
        grid: int = 3,
        prob: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        self.strength_range = _check_range("strength_range", strength_range)
        if self.strength_range[1] > 1.0:
            raise ValueError(f"strength_range must stay <= 1, got {strength_range!r}")
        if int(grid) < 2:
            raise ValueError(f"grid must be >= 2, got {grid!r}")
        self.grid = int(grid)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # band-independent field
        self._validate_shapes(cube, mask)
        B, H, W, _ = cube.shape
        apply = self._draw_apply_mask(B, rng, cube.device)
        if not apply.any():
            return cube, mask
        lo, hi = self.strength_range
        strength = torch.rand((B, 1, 1, 1), generator=rng) * (hi - lo) + lo
        cells = torch.rand((B, 1, self.grid, self.grid), generator=rng)
        field = F.interpolate(
            1.0 - strength * cells, size=(H, W), mode="bilinear", align_corners=True
        )  # (B, 1, H, W)
        field = torch.where(apply.view(B, 1, 1, 1).cpu(), field, torch.ones_like(field))
        return cube * field.permute(0, 2, 3, 1).to(device=cube.device, dtype=cube.dtype), mask


@register("RandomGammaContrast")
class RandomGammaContrast(Transform):
    """Per-sample gamma and contrast jitter on non-negative (normalised) data.

    ``x' = max(x, 0) ** gamma`` with ``gamma`` drawn log-uniformly from ``gamma_range``, then
    contrast around the per-sample, per-channel mean ``x'' = (x' - mu) * c + mu`` with ``c``
    from ``contrast_range``, floored at 0 (no upper clamp, so it composes with gain
    augmentations). Mask untouched. Intended after a [0, 1] selector/normaliser, where it
    reshapes the tone curve — a change a per-frame min-max normalisation does not undo.

    Parameters
    ----------
    gamma_range : tuple[float, float]
        ``(lo, hi)`` gamma, ``0 < lo <= hi``; ``(1, 1)`` disables the gamma step.
    contrast_range : tuple[float, float]
        ``(lo, hi)`` contrast factor, ``0 < lo <= hi``; ``(1, 1)`` disables the contrast step.
    prob : float
        Probability of applying the jitter per sample.
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] | list[float] = (0.7, 1.4),
        contrast_range: tuple[float, float] | list[float] = (0.8, 1.2),
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        self.gamma_range = _check_range("gamma_range", gamma_range)
        self.contrast_range = _check_range("contrast_range", contrast_range)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths
        self._validate_shapes(cube, mask)
        B = cube.shape[0]
        apply = self._draw_apply_mask(B, rng, cube.device)
        if not apply.any():
            return cube, mask
        glo, ghi = self.gamma_range
        clo, chi = self.contrast_range
        log_g = torch.rand((B,), generator=rng) * (math.log(ghi) - math.log(glo)) + math.log(glo)
        gamma = torch.exp(log_g).view(B, 1, 1, 1).to(device=cube.device, dtype=cube.dtype)
        contrast = (
            (torch.rand((B,), generator=rng) * (chi - clo) + clo)
            .view(B, 1, 1, 1)
            .to(device=cube.device, dtype=cube.dtype)
        )
        x = cube.clamp_min(0.0) ** gamma
        mu = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - mu) * contrast + mu).clamp_min(0.0)
        return torch.where(apply.view(B, 1, 1, 1), x, cube), mask


@register("RandomGaussianBlur")
class RandomGaussianBlur(Transform):
    """Per-sample Gaussian blur — the focus shift of a camera height change.

    ``sigma`` (pixels, at the incoming resolution) is drawn uniformly from ``sigma_range``
    per sample; a separable kernel of radius ``ceil(3 * sigma)`` with reflect padding is applied
    to every band. Mask untouched (defocus does not move object boundaries' labels).

    Parameters
    ----------
    sigma_range : tuple[float, float]
        ``(lo, hi)`` blur sigma in pixels, ``0 < lo <= hi``.
    prob : float
        Probability of applying the blur per sample.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float] | list[float] = (0.5, 2.0),
        prob: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        self.sigma_range = _check_range("sigma_range", sigma_range)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths
        self._validate_shapes(cube, mask)
        B, H, W, C = cube.shape
        apply = self._draw_apply_mask(B, rng, cube.device)
        if not apply.any():
            return cube, mask
        out = cube.clone()
        lo, hi = self.sigma_range
        for b in range(B):
            if not bool(apply[b]):
                continue
            sigma = _uniform(lo, hi, rng)
            r = max(1, math.ceil(3.0 * sigma))
            r = min(r, (min(H, W) - 1) // 2) if min(H, W) > 2 else 0
            if r < 1:
                continue
            x = torch.arange(-r, r + 1, dtype=cube.dtype, device=cube.device)
            k = torch.exp(-(x**2) / (2.0 * sigma**2))
            k = (k / k.sum()).view(1, 1, 1, -1).expand(C, 1, 1, -1)
            img = cube[b].permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="reflect"), k, groups=C)
            img = F.conv2d(F.pad(img, (0, 0, r, r), mode="reflect"), k.transpose(2, 3), groups=C)
            out[b] = img[0].permute(1, 2, 0)
        return out, mask
