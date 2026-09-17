"""ScalarMinMaxBandSlice — cube -> 3-band false-color EXACTLY as the walnut RF-DETR models were trained.

Recipe (export_walnut_rgb_baselines.py): a SINGLE scalar min-max over the WHOLE cube (all bands together, preserving the
relative band intensities), then select the channels nearest ``bands_nm`` (argmin |wavelength - nm|), output float32 in
[0, 1] with band order = bands_nm order. This is deliberately NOT the plugin's PercentileComposite (per-band percentile
stretch) — that changes the colour balance and the model was not trained on it. Feeds RFDETRSegmenter's ``rgb_image`` input
(the segmenter auto-scales <=1.5 inputs by 255). Stateless; default {ALWAYS} stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

try:  # palette metadata enums exist only in newer cuvis-ai; older training envs lack them
    from cuvis_ai_schemas.enums import NodeCategory, NodeTag
except Exception:  # pragma: no cover
    NodeCategory = NodeTag = None


class ScalarMinMaxBandSlice(Node):
    """Scalar (whole-cube) min-max normalize, then slice the 3 bands nearest ``bands_nm`` -> [B,H,W,3] float32 in [0,1]."""

    if NodeCategory is not None:
        _category = NodeCategory.TRANSFORM
        _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Hyperspectral cube [B,H,W,C]."
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Channel wavelengths in nm, shape (C,) as emitted by CU3SDataNode (a [B,C] tensor is also accepted).",
        ),
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="3-band scalar-min-max composite [B,H,W,3] in [0,1], band order = bands_nm.",
        ),
    }

    def __init__(
        self,
        bands_nm: tuple[float, float, float] = (640.0, 550.0, 470.0),
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(bands_nm=list(bands_nm), eps=eps, **kwargs)
        self.bands_nm = [float(b) for b in bands_nm]
        self.eps = float(eps)

    def forward(self, cube: torch.Tensor, wavelengths: Any, **_: Any) -> dict[str, torch.Tensor]:
        wl = (
            wavelengths[0]
            if (hasattr(wavelengths, "ndim") and wavelengths.ndim == 2)
            else wavelengths
        )
        wl = np.asarray(wl.detach().cpu() if torch.is_tensor(wl) else wl, dtype=np.float64).ravel()
        idx = [
            int(np.argmin(np.abs(wl - nm))) for nm in self.bands_nm
        ]  # nearest-band rule (== training exporter)
        lo = cube.amin(dim=(1, 2, 3), keepdim=True)  # scalar per-cube min over ALL bands
        hi = cube.amax(dim=(1, 2, 3), keepdim=True)
        norm = (cube - lo) / (hi - lo).clamp_min(self.eps)  # whole-cube scale -> keeps band ratios
        return {"rgb_image": norm[..., idx].to(torch.float32)}
