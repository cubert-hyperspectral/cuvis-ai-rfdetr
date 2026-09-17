"""FixedPCAProjection — cuvis-ai's TrainablePCA with a frozen, file-loaded projection and fixed unit scaling.

Subclasses ``cuvis_ai.node.dimensionality_reduction.TrainablePCA`` (so the projection math is the library's, not a
reimplementation) but loads mean / components / percentile range from an ``.npz`` given as an HPARAM instead of running
``statistical_initialization``: the projection a downstream model was TRAINED on must never be refit at inference (the
stateless ``PCA`` node refits per frame with arbitrary eigenvector signs — unusable in front of a trained model).
It first min-maxes each input cube globally to [0, 1] (``input_global_minmax``, on by default): the exporter did this
before fitting, so the fixed mean/comps assume that scale, and reproducing it makes the projection invariant to the
caller's absolute reflectance scale — cuvis.next's ``CU3SDataNode`` delivers raw-scale reflectance, not [0, 1], which
otherwise collapses the fixed projection (everything clamps to 1) and the downstream model sees a flat image. The
min-max is global (one min/max over all H*W*C), matching the export; per-channel scaling would change the relative band
magnitudes the fixed projection depends on, and it is idempotent on an already-[0, 1] cube. On top of the parent's
``(x - mean) @ components.T`` it applies the exporter's fixed scaling ``(p - lo) / (hi - lo)``
and clamps to [0, 1] — the clamp also protects ``to_uint8_frames``'s ``max <= 1.5`` range auto-detection from specular
outliers. npz keys: ``mean`` [C], ``comps`` [K, C], ``lo`` [K], ``hi`` [K] (as written by the walnut PCA dataset export).

TEMPORARY HOME: candidate for upstreaming into cuvis-ai next to TrainablePCA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr._compat import base_kwargs

try:  # cuvis-ai provides the parent; plugin-only envs can still import (and register) this module
    from cuvis_ai.node.dimensionality_reduction import TrainablePCA as _Base

    _HAVE_CUVIS_AI = True
except ModuleNotFoundError:  # pragma: no cover - exercised only in plugin-only envs
    from cuvis_ai_core.node.node import Node as _Base

    _HAVE_CUVIS_AI = False


class FixedPCAProjection(_Base):
    """Project cubes with a fixed, file-loaded PCA (mean/components/percentile scale) — never refit at inference."""

    # Only expose the projected image. The parent (TrainablePCA) also emits ``components`` [K, C] and
    # ``explained_variance_ratio`` [K]; those non-image ports break cuvis.next's per-output-port display/mask
    # handling and nothing downstream consumes them, so they are dropped here.
    OUTPUT_SPECS = {
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Fixed-PCA projection, scaled to [0, 1] [B, H, W, K].",
        ),
    }

    def __init__(
        self,
        projection_path: str,
        scale_to_unit: bool = True,
        clamp01: bool = True,
        input_global_minmax: bool = True,
        **kwargs: Any,
    ) -> None:
        if not _HAVE_CUVIS_AI:
            raise ImportError(
                "FixedPCAProjection requires cuvis-ai (parent class TrainablePCA). Install cuvis-ai in this env."
            )
        # a restored pipeline passes back the parent-recorded num_channels / n_components hparams;
        # both are re-derived from the npz, so drop them before forwarding kwargs.
        kwargs.pop("num_channels", None)
        kwargs.pop("n_components", None)
        proj = np.load(projection_path)
        mean = torch.from_numpy(np.asarray(proj["mean"], dtype=np.float32))
        comps = torch.from_numpy(np.asarray(proj["comps"], dtype=np.float32))
        lo = torch.from_numpy(np.asarray(proj["lo"], dtype=np.float32))
        hi = torch.from_numpy(np.asarray(proj["hi"], dtype=np.float32))
        if comps.ndim != 2 or mean.ndim != 1 or comps.shape[1] != mean.shape[0]:
            raise ValueError(
                f"FixedPCAProjection: expected comps [K, C] and mean [C], got {tuple(comps.shape)} / {tuple(mean.shape)}."
            )
        self.projection_path = str(projection_path)
        self.scale_to_unit = bool(scale_to_unit)
        self.clamp01 = bool(clamp01)
        self.input_global_minmax = bool(input_global_minmax)
        super().__init__(
            **base_kwargs(kwargs),
            num_channels=int(comps.shape[1]),
            n_components=int(comps.shape[0]),
            projection_path=self.projection_path,
            scale_to_unit=self.scale_to_unit,
            clamp01=self.clamp01,
            input_global_minmax=self.input_global_minmax,
            **kwargs,
        )
        self._mean.copy_(mean)
        self._components.copy_(comps)
        # parent buffer holds eigenvalues; the export stores explained-variance ratios — only the optional
        # ratio output port reads this, the projection itself never does.
        if "explained" in proj.files and np.asarray(proj["explained"]).shape[0] == comps.shape[0]:
            self._explained_variance.copy_(
                torch.from_numpy(np.asarray(proj["explained"], dtype=np.float32))
            )
        else:
            self._explained_variance.zero_()
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)
        self._statistically_initialized = True

    @staticmethod
    def _global_minmax(data: Tensor, eps: float = 1e-6) -> Tensor:
        """Per-frame whole-cube min-max to [0, 1] — one min/max over all H*W*C per batch item.

        Global (not per-channel), matching the exporter: this preserves relative band magnitudes, so the fixed
        mean/comps stay valid. Invariant to the caller's absolute reflectance scale and idempotent on a [0, 1] cube.
        """
        b = data.shape[0]
        flat = data.reshape(b, -1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        return ((flat - mn) / (mx - mn).clamp(min=eps)).reshape(data.shape)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Optional global min-max of the input, then parent projection, the fixed ``(p - lo)/(hi - lo)`` and [0, 1] clamp."""
        if self.input_global_minmax:
            data = self._global_minmax(data)
        out = super().forward(data=data)
        p = out["projected"]
        if self.scale_to_unit:
            p = (p - self._lo) / (self._hi - self._lo)
        if self.clamp01:
            p = p.clamp(0.0, 1.0)
        return {"projected": p}
