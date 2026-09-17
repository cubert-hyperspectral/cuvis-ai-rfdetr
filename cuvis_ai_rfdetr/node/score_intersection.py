"""ScoreIntersection — soft AND of two per-pixel score maps (elementwise minimum).

Two segmenters vote: a pixel keeps a high score only if BOTH gave it a high score, so thresholding the output at t is
exactly "both maps >= t" (the ensemble intersection). Shapes must match ([B, H, W, 1] at the shared input resolution).
Stateless, torch-native, differentiable; default {ALWAYS} stage.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr._compat import base_kwargs


class ScoreIntersection(Node):
    """Elementwise minimum of two score maps: the intersection (AND) of two segmenters."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "a": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, 1), description="Score map A [B, H, W, 1]."
        ),
        "b": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, 1), description="Score map B [B, H, W, 1]."
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="min(A, B) [B, H, W, 1]: >= t exactly where both inputs are >= t.",
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**base_kwargs(kwargs), **kwargs)

    def forward(self, a: Tensor, b: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the elementwise minimum of the two score maps."""
        return {"scores": torch.minimum(a, b)}
