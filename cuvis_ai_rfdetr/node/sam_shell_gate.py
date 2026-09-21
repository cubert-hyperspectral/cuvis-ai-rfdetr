"""SamShellGate — gate a shell score map by full-spectrum spectral angle to a reference spectrum.

Zeros out score-map pixels whose raw-cosine spectral angle (degrees) to a fixed reference exceeds
``threshold_deg`` — i.e. keeps a pixel only where BOTH the segmenter says "shell" AND the pixel's spectrum matches
the shell reference. Removes fake / off-spectrum pixels from a shell mask while keeping real-shell pixels (the
validated per-pixel SAM filter). Raw cosine is illumination-scale-invariant (|pixel| cancels), so the calibrated
threshold transfers across capture sessions. Reference is a fixed 61-band shell spectrum (mean over TRAIN shell
pixels); magnitude is irrelevant (cosine), only the spectral shape. Stateless apart from the reference buffer,
torch-native, default {ALWAYS} stage.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr._compat import base_kwargs


class SamShellGate(Node):
    """Keep score-map pixels only where the spectral angle to a shell reference is <= threshold_deg."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.SEGMENTATION, NodeTag.HYPERSPECTRAL, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1),
                         description="Hyperspectral cube [B, H, W, C]."),
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1),
                           description="Shell score map [B, H, W, 1] to be gated."),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1),
                           description="Gated score map [B, H, W, 1]: input where angle<=T, else 0."),
    }

    def __init__(self, reference: list[float], threshold_deg: float, eps: float = 1e-12, **kwargs: Any) -> None:
        ref = torch.as_tensor(reference, dtype=torch.float32).flatten()
        if ref.numel() < 2:
            raise ValueError(f"reference must be a >=2-band spectrum, got {ref.numel()} values")
        self.threshold_deg = float(threshold_deg)
        self.eps = float(eps)
        super().__init__(**base_kwargs(kwargs), reference=[float(x) for x in ref.tolist()],
                         threshold_deg=self.threshold_deg, eps=self.eps, **kwargs)
        self.register_buffer("_ref", ref)

    def forward(self, cube: Tensor, scores: Tensor, **_: Any) -> dict[str, Tensor]:
        """Raw-cosine spectral angle per pixel to the reference; keep scores where angle<=threshold, else zero."""
        ref = self._ref.to(cube.dtype)  # [C]
        num = (cube * ref).sum(dim=-1)  # [B, H, W]
        den = cube.norm(dim=-1) * ref.norm() + self.eps
        angle_deg = torch.rad2deg(torch.arccos((num / den).clamp(-1.0, 1.0)))
        keep = (angle_deg <= self.threshold_deg).to(scores.dtype).unsqueeze(-1)  # [B, H, W, 1]
        return {"scores": scores * keep}
