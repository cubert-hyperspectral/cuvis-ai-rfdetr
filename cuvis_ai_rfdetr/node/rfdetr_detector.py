"""RF-DETR object detection node wrapping the official Roboflow ``rfdetr`` package."""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

#: Maps the ``variant`` hparam to the model class exported by :mod:`rfdetr`.
#: Only the Apache-2.0 model tier is exposed; the XL / 2XL checkpoints ship
#: under the non-open Roboflow Platform Model License and are deliberately
#: not wrapped by this node.
_VARIANT_CLASS_NAMES: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class RFDETRDetector(Node):
    """Single-frame object detection using the official Roboflow RF-DETR.

    Wraps the Apache-2.0 tier of RF-DETR (nano / small / medium / large) as a
    cuvis.ai inference node, e.g. for foreign-object or anomaly screening on
    RGB / false-color renderings of hyperspectral cubes.

    The ``rfdetr`` package is imported lazily on the first ``forward`` call:
    constructing the node (building, validating, or visualizing a pipeline)
    does not require RF-DETR to be installed.

    Fine-tuning happens OUTSIDE the pipeline graph: train with RF-DETR's own
    trainer (``pip install "rfdetr[train]"``) and point ``checkpoint_path`` at
    the resulting ``checkpoint_best_total.pth``. Without a checkpoint the node
    loads the official COCO-pretrained weights for the selected variant
    (fetched by ``rfdetr`` itself on first use).

    The wrapped model manages its own device placement, so it is kept as a
    plain attribute (not a registered submodule): moving the node with
    ``.to()`` intentionally does not move the detector.

    Inference
    ---------
    Emits per timestep:

    - ``scores`` ``[B, H, W, 1]`` — rasterized detection map: zeros, with each
      detection's box region filled with ``max(existing, confidence)``.
    - ``detections`` — per-image list of ``{"xyxy", "confidence", "class_id"}``
      dicts in input-pixel coordinates.
    - ``anomaly_score`` ``[B]`` — max box confidence per image (0.0 when the
      image has no detections).
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.IMAGE,
            NodeTag.RGB,
            NodeTag.BBOX,
            NodeTag.DETECTION,
            NodeTag.ANOMALY,
            NodeTag.INFERENCE,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB / false-color image [B, H, W, 3] in float32 "
            "(value range 0–1 or 0–255; auto-detected).",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Rasterized detection map [B, H, W, 1]: box regions "
            "filled with max(existing, confidence), background 0.",
        ),
        "detections": PortSpec(
            dtype=list,
            shape=(),
            description="Per-image list of detection dicts "
            "{'xyxy': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}.",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level score [B]: max box confidence "
            "(0.0 when no detections).",
        ),
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        variant: str = "medium",
        threshold: float = 0.5,
        resolution: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the detector without loading any weights yet.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to fine-tuned RF-DETR weights (e.g. the
            ``checkpoint_best_total.pth`` written by RF-DETR's own trainer).
            Passed to the model's ``pretrain_weights``. ``None`` loads the
            official COCO-pretrained weights for the selected variant.
        variant : str
            RF-DETR size variant: ``"nano"``, ``"small"``, ``"medium"``, or
            ``"large"`` (the Apache-2.0 tier).
        threshold : float
            Confidence threshold in ``[0, 1]`` passed to ``model.predict``.
        resolution : int, optional
            Optional input resolution forwarded to the RF-DETR constructor.
            ``None`` keeps the variant's default (RF-DETR enforces its own
            divisibility constraints).
        """
        variant_key = str(variant).lower()
        if variant_key not in _VARIANT_CLASS_NAMES:
            raise ValueError(
                f"RFDETRDetector: variant must be one of "
                f"{sorted(_VARIANT_CLASS_NAMES)} (Apache-2.0 tier), got {variant!r}."
            )
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"RFDETRDetector: threshold must be within [0, 1], got {threshold}."
            )
        if resolution is not None:
            resolution = int(resolution)
            if resolution <= 0:
                raise ValueError(
                    f"RFDETRDetector: resolution must be a positive int, got {resolution}."
                )

        self.checkpoint_path = checkpoint_path
        self.variant = variant_key
        self.threshold = threshold
        self.resolution = resolution

        name, execution_stages = Node.consume_base_kwargs(kwargs)
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            threshold=self.threshold,
            resolution=self.resolution,
            **kwargs,
        )

        # Lazily constructed on the first forward. Plain attribute on purpose:
        # rfdetr manages its own device, so the model must NOT become a
        # registered submodule (state_dict stays empty, .to() is a no-op for it).
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Construct the wrapped RF-DETR model on first use."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self) -> Any:
        """Import ``rfdetr`` lazily and instantiate the configured variant."""
        try:
            import rfdetr
        except ImportError as exc:
            raise ImportError(
                "RFDETRDetector requires the 'rfdetr' package at inference time. "
                "Install the Apache-2.0 core tier with: pip install 'rfdetr>=1.8,<2'"
            ) from exc

        class_name = _VARIANT_CLASS_NAMES[self.variant]
        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            raise RuntimeError(
                f"RFDETRDetector: the installed 'rfdetr' package does not export "
                f"{class_name}; rfdetr>=1.8 is required."
            )

        model_kwargs: dict[str, Any] = {}
        if self.checkpoint_path is not None:
            model_kwargs["pretrain_weights"] = str(self.checkpoint_path)
        if self.resolution is not None:
            model_kwargs["resolution"] = int(self.resolution)
        return model_cls(**model_kwargs)

    @staticmethod
    def _to_uint8_frames(rgb_image: Tensor) -> Any:
        """Convert ``[B, H, W, 3]`` float input to a uint8 HWC numpy batch.

        Auto-detects the value range: inputs whose maximum is <= 1.5 are
        treated as 0–1 and scaled by 255; everything else is assumed to
        already be 0–255.
        """
        x = rgb_image.detach().to(device="cpu", dtype=torch.float32)
        if x.numel() > 0 and float(x.max()) <= 1.5:
            x = x * 255.0
        return x.clamp(0.0, 255.0).round().to(torch.uint8).contiguous().numpy()

    def _predict_frame(
        self, model: Any, frame: Any
    ) -> list[tuple[float, float, float, float, float, int]]:
        """Run one uint8 HWC frame through ``model.predict``, flattened to tuples."""
        result = model.predict(frame, threshold=self.threshold)
        # rfdetr returns a single supervision.Detections for a single image;
        # tolerate list-wrapped results for robustness across versions.
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        if result is None:
            return []
        xyxy = getattr(result, "xyxy", None)
        if xyxy is None:
            return []
        confidence = getattr(result, "confidence", None)
        class_id = getattr(result, "class_id", None)

        rows: list[tuple[float, float, float, float, float, int]] = []
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[j])
            conf = float(confidence[j]) if confidence is not None else 0.0
            cid = int(class_id[j]) if class_id is not None else -1
            rows.append((x1, y1, x2, y2, conf, cid))
        return rows

    def forward(
        self,
        rgb_image: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Detect objects per frame and rasterize the boxes into a score map."""
        if rgb_image.dim() != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"RFDETRDetector expects rgb_image of shape [B, H, W, 3], "
                f"got {tuple(rgb_image.shape)}."
            )

        model = self._ensure_model()

        batch, height, width = rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]
        device = rgb_image.device
        frames = self._to_uint8_frames(rgb_image)

        scores = torch.zeros((batch, height, width, 1), dtype=torch.float32)
        anomaly_score = torch.zeros((batch,), dtype=torch.float32)
        detections: list[list[dict[str, Any]]] = []

        for idx in range(batch):
            items: list[dict[str, Any]] = []
            best_confidence = 0.0
            for x1, y1, x2, y2, conf, cid in self._predict_frame(model, frames[idx]):
                items.append(
                    {
                        "xyxy": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cid,
                    }
                )
                col0 = min(max(int(math.floor(x1)), 0), width)
                col1 = min(max(int(math.ceil(x2)), 0), width)
                row0 = min(max(int(math.floor(y1)), 0), height)
                row1 = min(max(int(math.ceil(y2)), 0), height)
                if col1 > col0 and row1 > row0:
                    # Fill the box region with max(existing, confidence).
                    scores[idx, row0:row1, col0:col1, 0].clamp_(min=conf)
                best_confidence = max(best_confidence, conf)
            detections.append(items)
            anomaly_score[idx] = best_confidence

        return {
            "scores": scores.to(device),
            "detections": detections,
            "anomaly_score": anomaly_score.to(device),
        }
