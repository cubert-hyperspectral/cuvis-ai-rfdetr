"""CarlSegmenter — CARL (IMSY-DKFZ) hyperspectral semantic segmenter as a cuvis-ai node.

Wraps the CARL ViT-Adapter/UperNet model trained on the 61-band walnut cubes (classes 0 bg / 1 shell / 2 fo /
3 fake_shell) and emits a per-pixel score map for one class (default: shell). Preprocessing reproduces the training
exporter + inference script exactly: per-cube scalar min-max to [0, 1], per-cube z-score, bilinear resize to
``image_size``, forward ``model(x, wavelengths_nm / 1000, mean-band image)``, bilinear upsample of the logits back to
the input size, softmax.

TEMPORARY HOME: this node belongs in its own ``cuvis-ai-carl`` plugin; it lives here so the walnut live pipelines
can be deployed together. The CARL repo is imported lazily from ``carl_repo`` on the first forward, so building /
validating a pipeline does not require CARL to be installed. The compiled CUDA deform-attn op is optional (the
pure-torch fallback is used when it is absent, which is fine for inference).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr._compat import base_kwargs


class CarlSegmenter(Node):
    """CARL 61-band semantic segmentation -> per-pixel score map of one class (default shell)."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.SEGMENTATION, NodeTag.INFERENCE, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Channel wavelengths in nm, shape (C,) as emitted by CU3SDataNode ([B, C] also accepted).",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Softmax probability of ``score_class`` per pixel [B, H, W, 1] at input resolution.",
        ),
        "labels": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Argmax class map [B, H, W] (0 bg / 1 shell / 2 fo / 3 fake_shell).",
        ),
    }

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        carl_repo: str,
        image_size: int = 384,
        score_class: int = 1,
        precision: str = "bf16",
        band_step: int = 1,
        compile: bool = False,
        compile_cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(
                f"CarlSegmenter: precision must be fp32 | bf16 | fp16, got {precision!r}."
            )
        self.checkpoint_path = str(checkpoint_path)
        self.config_path = str(config_path)
        self.carl_repo = str(carl_repo)
        self.image_size = int(image_size)
        self.score_class = int(score_class)
        self.precision = precision
        if int(band_step) < 1:
            raise ValueError("CarlSegmenter: band_step must be >= 1.")
        self.band_step = int(band_step)
        self.compile = bool(compile)
        self.compile_cache_dir = None if compile_cache_dir is None else str(compile_cache_dir)
        self._model: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            **base_kwargs(kwargs),
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            carl_repo=self.carl_repo,
            image_size=self.image_size,
            score_class=self.score_class,
            precision=self.precision,
            band_step=self.band_step,
            compile=self.compile,
            compile_cache_dir=self.compile_cache_dir,
            **kwargs,
        )

    def _ensure_model(self) -> Any:
        """Import CARL from ``carl_repo`` and load the checkpoint on first use."""
        if self._model is None:
            import yaml

            if self.carl_repo not in sys.path:
                sys.path.insert(0, self.carl_repo)
            from segmentation_heads.upernet.trainer import ViTAdapterTrainer  # type: ignore

            with open(self.config_path) as fh:
                cfg = yaml.safe_load(fh)
            m = ViTAdapterTrainer(cfg)
            sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            m.load_state_dict(sd.get("state_dict", sd), strict=False)
            self._model = m.to(self._device).eval()
            if self.compile and self._device.type == "cuda":
                if self.compile_cache_dir:
                    # Persistent inductor cache: a warm restart compiles in ~30 s instead of ~10 min.
                    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", self.compile_cache_dir)
                    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
                # torch.compile (inductor) fuses the memory-bound LayerNorm/elementwise chain of the spectral
                # transformer: ~1.6x on a 4070 with bit-identical masks. Needs triton (triton-windows on Windows);
                # first call compiles for minutes unless TORCHINDUCTOR_CACHE_DIR holds a warm cache.
                self._model.model = torch.compile(
                    self._model.model,
                    backend="inductor",
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                )
        return self._model

    @staticmethod
    def preprocess(cube: Tensor, image_size: int) -> Tensor:
        """[B,H,W,C] raw cube -> [B,C,S,S]: per-cube min-max to [0,1], per-cube z-score, bilinear resize."""
        lo = cube.amin(dim=(1, 2, 3), keepdim=True)
        hi = cube.amax(dim=(1, 2, 3), keepdim=True)
        x = (cube - lo) / (hi - lo).clamp_min(1e-6)
        x = x.permute(0, 3, 1, 2)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-6)
        return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)

    def forward(self, cube: Tensor, wavelengths: Any, **_: Any) -> dict[str, Tensor]:
        """Segment the cube; return the score map of ``score_class`` and the argmax label map."""
        model = self._ensure_model()
        _, h, w, _ = cube.shape
        wl = (
            wavelengths[0]
            if (hasattr(wavelengths, "ndim") and wavelengths.ndim == 2)
            else wavelengths
        )
        wl_np = np.asarray(wl.detach().cpu() if torch.is_tensor(wl) else wl, dtype=np.float32)
        wl_t = torch.as_tensor(wl_np).view(1, -1) / 1000.0
        # Move the raw cube to the device FIRST: preprocessing a 61-band cube on the CPU costs ~180 ms,
        # on the GPU ~14 ms (+ one ~40 ms host->device copy of the raw cube).
        x = self.preprocess(cube.to(self._device, torch.float32), self.image_size)
        if (
            self.band_step > 1
        ):  # every k-th band; the spectral encoder is wavelength-conditioned so this is valid input
            x = x[:, :: self.band_step]
            wl_t = wl_t[:, :: self.band_step]
        use_amp = self._device.type == "cuda" and self.precision != "fp32"
        amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model.model(x, wl_t.to(self._device), x.mean(1, keepdim=True))
        with torch.no_grad():
            logits = F.interpolate(
                logits.float(), size=(h, w), mode="bilinear", align_corners=False
            )
            prob = logits.softmax(1)
        scores = prob[:, self.score_class].unsqueeze(-1).to(cube.device)
        labels = logits.argmax(1).to(torch.int32).to(cube.device)
        return {"scores": scores, "labels": labels}
