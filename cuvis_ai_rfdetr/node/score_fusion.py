"""ScoreFusion — combine two per-pixel score maps with a selectable rule.

Generalizes ScoreIntersection (elementwise ``min``, the hard AND) to soft rules that recover recall the AND throws away:
``mean`` (arithmetic), ``gmean`` (geometric, penalizes disagreement more than mean), ``wmean`` (weighted toward ``a``), and
``max`` (OR). On the walnut deploy, fusing RF-DETR (RGB shell) with CARL (61-band, kernel-robust) at ``gmean``/``mean`` beats
the ``min`` ensemble: live recall 0.850 -> 0.90-0.92 and 18-Aug IoU 0.947 -> 0.97 while kernel false positives stay ~45x below
the single RGB model (see FUSION_170). Feed it un-thresholded score maps (RFDETRSegmenter ``threshold`` low, e.g. 0.05) so weak
instances still contribute to the average. Shapes must match ([B, H, W, 1]); stateless, torch-native, differentiable.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr._compat import base_kwargs

_MODES = ("min", "max", "mean", "gmean", "wmean")


class ScoreFusion(Node):
    """Fuse two score maps by ``mode`` (min | max | mean | gmean | wmean)."""

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
            description="Fused score map [B, H, W, 1] per ``mode``.",
        ),
    }

    def __init__(self, mode: str = "gmean", weight: float = 0.5, **kwargs: Any) -> None:
        if mode not in _MODES:
            raise ValueError(f"ScoreFusion: mode must be one of {_MODES}, got {mode!r}.")
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError(f"ScoreFusion: weight must be within [0, 1], got {weight}.")
        self.mode = mode
        self.weight = float(weight)
        super().__init__(**base_kwargs(kwargs), mode=self.mode, weight=self.weight, **kwargs)

    def forward(self, a: Tensor, b: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the fused score map per ``mode`` (``gmean`` clamps inputs to [0, 1] before the sqrt)."""
        if self.mode == "min":
            out = torch.minimum(a, b)
        elif self.mode == "max":
            out = torch.maximum(a, b)
        elif self.mode == "mean":
            out = 0.5 * (a + b)
        elif self.mode == "wmean":
            out = self.weight * a + (1.0 - self.weight) * b
        else:  # gmean
            out = torch.sqrt(a.clamp(0.0, 1.0) * b.clamp(0.0, 1.0))
        return {"scores": out}
