"""ScalarMinMaxBandSlice — hyperspectral cube -> 3-band false-colour image, one scalar min-max per cube.

The cube is normalised to [0, 1] with a SINGLE min / max over all pixels and bands, which keeps the
relative band intensities, then the three channels nearest ``bands_nm`` (argmin |wavelength - nm|) are
selected in ``bands_nm`` order. This is the input recipe some 3-band RF-DETR models are trained on. It
differs from ``PercentileComposite`` (per-band percentile stretch, which changes the colour balance) and
from ``FixedWavelengthSelector(norm_mode="per_frame")`` (per-channel min-max). The float32 [B, H, W, 3]
output in [0, 1] feeds ``RFDETRSegmenter.rgb_image`` directly. Stateless; default stage set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_rfdetr._compat import base_kwargs


class ScalarMinMaxBandSlice(Node):
    """Whole-cube min-max normalise, then slice the 3 bands nearest ``bands_nm`` -> [B, H, W, 3]."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Hyperspectral cube [B,H,W,C]."
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Channel wavelengths in nm, shape (C,) as emitted by CU3SDataNode "
            "(a [B, C] tensor is also accepted).",
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
        self.bands_nm = [float(b) for b in bands_nm]
        self.eps = float(eps)
        super().__init__(
            **base_kwargs(kwargs), bands_nm=list(self.bands_nm), eps=self.eps, **kwargs
        )

    def forward(self, cube: torch.Tensor, wavelengths: Any, **_: Any) -> dict[str, torch.Tensor]:
        """Normalise each cube by its global min / max, then pick the bands nearest ``bands_nm``."""
        wl = (
            wavelengths[0]
            if (hasattr(wavelengths, "ndim") and wavelengths.ndim == 2)
            else wavelengths
        )
        wl = np.asarray(wl.detach().cpu() if torch.is_tensor(wl) else wl, dtype=np.float64).ravel()
        idx = [int(np.argmin(np.abs(wl - nm))) for nm in self.bands_nm]  # nearest-band rule
        lo = cube.amin(dim=(1, 2, 3), keepdim=True)  # one min over all pixels and bands
        hi = cube.amax(dim=(1, 2, 3), keepdim=True)
        norm = (cube - lo) / (hi - lo).clamp_min(self.eps)  # a single scale keeps the band ratios
        return {"rgb_image": norm[..., idx].to(torch.float32)}
