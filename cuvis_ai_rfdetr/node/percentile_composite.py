"""Percentile-stretched false-color composite from a hyperspectral cube."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr.functional import resolve_band_indices


class PercentileComposite(Node):
    """Three-band false-color composite with per-band percentile stretch.

    Selects the cube channels nearest to ``bands_nm`` (integer-cast nearest
    wavelength), stretches each selected band independently between its
    ``p_low``/``p_high`` percentiles **computed over the full frame**, clips
    to ``[0, 1]``, and quantizes with ``*255 + 0.5`` to integer-valued
    float32 in ``[0, 255]``.

    This reproduces, arithmetic-for-arithmetic, the composite used by JPEG
    tile exporters for 3-channel detector training/evaluation: quantizing
    here (rather than downstream) makes the output byte-identical to a
    uint8 image, so a consumer that converts to uint8 (e.g.
    :class:`~cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector` /
    :class:`~cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter`)
    recovers the exact bytes the exporter would have written.

    Notes
    -----
    - Percentiles are per frame and per band, over the **entire** input frame
      — crop the frame (e.g. to a lane) *before* this node, and tile *after*
      it, exactly like the exporters do.
    - The adaptive stretch is deliberately not colorimetric: it maximizes
      per-band contrast rather than color fidelity.
    - Degenerate bands (``p_high - p_low <= 1e-6``) map to 0.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.IMAGE,
            NodeTag.RGB,
            NodeTag.HYPERSPECTRAL,
        }
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] (float32; raw counts "
            "or reflectance — the percentile stretch is scale-invariant).",
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1),
            description="Per-frame channel wavelengths in nm [B, C].",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Integer-valued float32 composite [B, H, W, 3] in "
            "[0, 255], band order = bands_nm order.",
        ),
    }

    def __init__(
        self,
        bands_nm: tuple[float, float, float] = (650.0, 550.0, 450.0),
        p_low: float = 1.0,
        p_high: float = 99.0,
        **kwargs: Any,
    ) -> None:
        """Configure the composite.

        Parameters
        ----------
        bands_nm : tuple[float, float, float]
            The three target wavelengths (nm), in output channel order
            (R, G, B of the false-color image). Each is resolved to the
            nearest cube channel per frame.
        p_low, p_high : float
            Percentiles of the per-band stretch window (defaults 1 / 99),
            with ``0 <= p_low < p_high <= 100``.
        """
        bands = tuple(float(b) for b in bands_nm)
        if len(bands) != 3:
            raise ValueError(
                f"PercentileComposite: bands_nm must have exactly 3 entries, got {len(bands)}."
            )
        p_low = float(p_low)
        p_high = float(p_high)
        if not 0.0 <= p_low < p_high <= 100.0:
            raise ValueError(
                f"PercentileComposite: need 0 <= p_low < p_high <= 100, "
                f"got p_low={p_low}, p_high={p_high}."
            )

        self.bands_nm = bands
        self.p_low = p_low
        self.p_high = p_high

        name, execution_stages = Node.consume_base_kwargs(kwargs)
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            bands_nm=self.bands_nm,
            p_low=self.p_low,
            p_high=self.p_high,
            **kwargs,
        )

    def forward(
        self,
        cube: Tensor,
        wavelengths: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Compose one stretched false-color image per frame."""
        if cube.dim() != 4:
            raise ValueError(
                f"PercentileComposite expects cube of shape [B, H, W, C], got {tuple(cube.shape)}."
            )
        batch = cube.shape[0]
        device = cube.device
        cube_np = cube.detach().to("cpu", dtype=torch.float32).numpy()
        wl_np = wavelengths.detach().to("cpu").numpy()

        out = torch.empty((batch, cube.shape[1], cube.shape[2], 3), dtype=torch.float32)
        for b in range(batch):
            idxs = resolve_band_indices(wl_np[b] if wl_np.ndim == 2 else wl_np, self.bands_nm)
            chans = []
            for i in idxs:
                ch = cube_np[b, :, :, i].astype(np.float32)
                lo, hi = np.percentile(ch, (self.p_low, self.p_high))
                chans.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
            u8 = (np.stack(chans, -1) * 255.0 + 0.5).astype(np.uint8)
            out[b] = torch.from_numpy(u8.astype(np.float32))

        return {"rgb_image": out.to(device)}
