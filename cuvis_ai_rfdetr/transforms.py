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

__all__ = ["RandomMultiScaleResize"]


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
