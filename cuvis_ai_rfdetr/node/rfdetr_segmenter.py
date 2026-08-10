"""RF-DETR instance-segmentation node wrapping the official Roboflow ``rfdetr`` package."""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr.functional import (
    jpeg_roundtrip as _jpeg_roundtrip,
)
from cuvis_ai_rfdetr.functional import (
    nms_rows,
    to_uint8_frames,
    top_frac_mean,
)

#: Maps the ``variant`` hparam to the segmentation model class exported by
#: :mod:`rfdetr`. The whole segmentation tier ships under Apache-2.0 (unlike
#: detection, where XL / 2XL are platform-licensed), so all sizes are exposed.
_SEG_VARIANT_CLASS_NAMES: dict[str, str] = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}


class RFDETRSegmenter(Node):
    """Single-frame instance segmentation using the official Roboflow RF-DETR-Seg.

    Same design as :class:`~cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector`
    (lazy ``rfdetr`` import, fine-tuned weights via ``checkpoint_path``,
    tiled or whole-frame inference), but wraps the **segmentation** tier and
    rasterizes per-instance masks instead of boxes:

    - ``scores`` ``[B, H, W, 1]`` — per-pixel max of ``instance_mask x confidence``
      over all detected instances (directly comparable to a segmentation
      model's probability map). Falls back to box fill if the underlying
      result carries no masks.
    - ``detections`` — per-image list of ``{"xyxy", "confidence", "class_id"}``.
    - ``anomaly_score`` ``[B]`` — max instance confidence per image.

    In ``"tiled"`` mode each full-width row-strip is segmented independently;
    strip mask-scores are pasted back at their row offset (pixelwise max) and
    boxes are NMS-merged, mirroring the tiled evaluation protocol.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.IMAGE,
            NodeTag.RGB,
            NodeTag.SEGMENTATION,
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
            "(value range 0-1 or 0-255; auto-detected).",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel instance-mask score map [B, H, W, 1]: "
            "max(mask_i * confidence_i) over instances, background 0.",
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
            description="Image-level score [B]: max instance confidence "
            "(default), or the mean of the top-fraction pixels of the score "
            "map when score_reduction='top_frac_mean' (0.0 when empty).",
        ),
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        variant: str = "medium",
        threshold: float = 0.5,
        resolution: int | None = None,
        tiling: str = "tiled",
        tile_rows: int = 405,
        row_starts: tuple[int, ...] = (0, 291, 582),
        nms_iou: float = 0.5,
        jpeg_roundtrip: bool = False,
        jpeg_quality: int = 95,
        class_filter: int | None = None,
        score_reduction: str = "max_conf",
        top_frac: float = 0.001,
        checkpoint_loader: str = "constructor",
        **kwargs: Any,
    ) -> None:
        """Configure the segmenter without loading any weights yet.

        Parameters mirror ``RFDETRDetector``; ``variant`` selects the
        segmentation tier size (``nano`` … ``2xlarge``, all Apache-2.0), and
        ``checkpoint_path`` points at fine-tuned RF-DETR-Seg weights (the
        ``.pth`` written by RF-DETR's own trainer) for weight transfer.

        Reproduction note — ``checkpoint_loader`` decides where the model
        *configuration* comes from, and the same checkpoint can yield
        materially different scores between the two loaders:

        - ``"constructor"`` (default): build the configured ``variant`` at
          the given ``resolution`` (or the class default when ``None``) and
          load the checkpoint's weights into it.
        - ``"from_checkpoint"``: delegate to ``rfdetr.RFDETR.from_checkpoint``
          — model class and configuration come from the checkpoint itself,
          falling back to **class defaults for fields the checkpoint does not
          carry, including resolution** (a fine-tuned checkpoint does not
          necessarily record its training resolution). Use this to reproduce
          harnesses that load checkpoints the same way; ``resolution`` (if
          set) is forwarded as an explicit override, and ``variant`` must
          match the class resolved from the checkpoint.

        Parity parameters (for byte-faithful reproduction of evaluation
        harnesses built around per-tile JPEG files and top-fraction image
        scores):

        - ``jpeg_roundtrip`` / ``jpeg_quality``: route each model input
          (every tile in ``"tiled"`` mode, the frame in ``"whole"`` mode)
          through an in-memory JPEG encode/decode before prediction. Lossy
          on purpose — harnesses that persist tiles as ``.jpg`` make the
          compression part of the score definition.
        - ``class_filter``: keep only instances of this ``class_id``
          (drop the rest before pasting/merging), matching harnesses that
          score a single foreground class.
        - ``score_reduction``: ``"max_conf"`` (default) keeps
          ``anomaly_score`` = max instance confidence; ``"top_frac_mean"``
          computes it as the mean of the top ``top_frac`` fraction of the
          pasted score map (integer-floor top-k — e.g. exactly 400 px of a
          987x405 map at 0.001), directly comparable to dense segmentation
          models scored the same way.
        """
        variant_key = str(variant).lower()
        if variant_key not in _SEG_VARIANT_CLASS_NAMES:
            raise ValueError(
                f"RFDETRSegmenter: variant must be one of "
                f"{sorted(_SEG_VARIANT_CLASS_NAMES)}, got {variant!r}."
            )
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"RFDETRSegmenter: threshold must be within [0, 1], got {threshold}.")
        if resolution is not None:
            resolution = int(resolution)
            if resolution <= 0:
                raise ValueError(
                    f"RFDETRSegmenter: resolution must be a positive int, got {resolution}."
                )
        tiling = str(tiling).lower()
        if tiling not in ("tiled", "whole"):
            raise ValueError(f"RFDETRSegmenter: tiling must be 'tiled' or 'whole', got {tiling!r}.")
        tile_rows = int(tile_rows)
        if tile_rows <= 0:
            raise ValueError(f"RFDETRSegmenter: tile_rows must be a positive int, got {tile_rows}.")
        row_starts = tuple(int(r) for r in row_starts)
        nms_iou = float(nms_iou)
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError(f"RFDETRSegmenter: nms_iou must be within [0, 1], got {nms_iou}.")
        jpeg_roundtrip = bool(jpeg_roundtrip)
        jpeg_quality = int(jpeg_quality)
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"RFDETRSegmenter: jpeg_quality must be within [1, 100], got {jpeg_quality}."
            )
        if class_filter is not None:
            class_filter = int(class_filter)
        score_reduction = str(score_reduction).lower()
        if score_reduction not in ("max_conf", "top_frac_mean"):
            raise ValueError(
                f"RFDETRSegmenter: score_reduction must be 'max_conf' or "
                f"'top_frac_mean', got {score_reduction!r}."
            )
        top_frac = float(top_frac)
        if not 0.0 < top_frac <= 1.0:
            raise ValueError(f"RFDETRSegmenter: top_frac must be within (0, 1], got {top_frac}.")
        checkpoint_loader = str(checkpoint_loader).lower()
        if checkpoint_loader not in ("constructor", "from_checkpoint"):
            raise ValueError(
                f"RFDETRSegmenter: checkpoint_loader must be 'constructor' or "
                f"'from_checkpoint', got {checkpoint_loader!r}."
            )

        self.checkpoint_path = checkpoint_path
        self.variant = variant_key
        self.threshold = threshold
        self.resolution = resolution
        self.tiling = tiling
        self.tile_rows = tile_rows
        self.row_starts = row_starts
        self.nms_iou = nms_iou
        self.jpeg_roundtrip = jpeg_roundtrip
        self.jpeg_quality = jpeg_quality
        self.class_filter = class_filter
        self.score_reduction = score_reduction
        self.top_frac = top_frac
        self.checkpoint_loader = checkpoint_loader

        name, execution_stages = Node.consume_base_kwargs(kwargs)
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            threshold=self.threshold,
            resolution=self.resolution,
            tiling=self.tiling,
            tile_rows=self.tile_rows,
            row_starts=self.row_starts,
            nms_iou=self.nms_iou,
            jpeg_roundtrip=self.jpeg_roundtrip,
            jpeg_quality=self.jpeg_quality,
            class_filter=self.class_filter,
            score_reduction=self.score_reduction,
            top_frac=self.top_frac,
            checkpoint_loader=self.checkpoint_loader,
            **kwargs,
        )

        # Lazily constructed on the first forward; rfdetr manages its own
        # device, so the model must NOT become a registered submodule.
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self) -> Any:
        try:
            import rfdetr
        except ImportError as exc:
            raise ImportError(
                "RFDETRSegmenter requires the 'rfdetr' package at inference time. "
                "Install the Apache-2.0 tier with: pip install 'rfdetr>=1.8,<2'"
            ) from exc

        class_name = _SEG_VARIANT_CLASS_NAMES[self.variant]
        if self.checkpoint_path is not None and self.checkpoint_loader == "from_checkpoint":
            loader = getattr(rfdetr, "RFDETR", None)
            if loader is None or not hasattr(loader, "from_checkpoint"):
                raise RuntimeError(
                    "RFDETRSegmenter: checkpoint_loader='from_checkpoint' requires "
                    "rfdetr>=1.8 (rfdetr.RFDETR.from_checkpoint not found)."
                )
            loader_kwargs: dict[str, Any] = {}
            if self.resolution is not None:
                loader_kwargs["resolution"] = int(self.resolution)
            model = loader.from_checkpoint(str(self.checkpoint_path), **loader_kwargs)
            loaded = type(model).__name__
            if loaded != class_name:
                raise RuntimeError(
                    f"RFDETRSegmenter: checkpoint resolved to {loaded}, but "
                    f"variant={self.variant!r} expects {class_name}. Set variant to "
                    f"match the checkpoint or use checkpoint_loader='constructor'."
                )
            return model

        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            raise RuntimeError(
                f"RFDETRSegmenter: the installed 'rfdetr' package does not export "
                f"{class_name}; rfdetr>=1.8 is required."
            )
        model_kwargs: dict[str, Any] = {}
        if self.checkpoint_path is not None:
            model_kwargs["pretrain_weights"] = str(self.checkpoint_path)
        if self.resolution is not None:
            model_kwargs["resolution"] = int(self.resolution)
        return model_cls(**model_kwargs)

    def _predict_frame(self, model: Any, frame: Any) -> list[tuple]:
        """One uint8 HWC frame -> list of (x1, y1, x2, y2, conf, class_id, mask|None)."""
        result = model.predict(frame, threshold=self.threshold)
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        if result is None:
            return []
        xyxy = getattr(result, "xyxy", None)
        if xyxy is None:
            return []
        confidence = getattr(result, "confidence", None)
        class_id = getattr(result, "class_id", None)
        inst_masks = getattr(result, "mask", None)  # supervision: [N, H, W] bool

        rows: list[tuple] = []
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[j])
            conf = float(confidence[j]) if confidence is not None else 0.0
            cid = int(class_id[j]) if class_id is not None else -1
            m = inst_masks[j] if inst_masks is not None else None
            rows.append((x1, y1, x2, y2, conf, cid, m))
        return rows

    def forward(
        self,
        rgb_image: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Segment instances per frame; rasterize mask x confidence into a score map."""
        if rgb_image.dim() != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"RFDETRSegmenter expects rgb_image of shape [B, H, W, 3], "
                f"got {tuple(rgb_image.shape)}."
            )
        model = self._ensure_model()
        batch, height, width = rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]
        device = rgb_image.device
        frames = to_uint8_frames(rgb_image)

        scores = torch.zeros((batch, height, width, 1), dtype=torch.float32)
        anomaly_score = torch.zeros((batch,), dtype=torch.float32)
        detections: list[list[dict[str, Any]]] = []

        for idx in range(batch):
            frame = frames[idx]
            rows: list[tuple] = []
            if self.tiling == "whole":
                model_input = (
                    _jpeg_roundtrip(frame, self.jpeg_quality) if self.jpeg_roundtrip else frame
                )
                for x1, y1, x2, y2, conf, cid, m in self._predict_frame(model, model_input):
                    if self.class_filter is not None and cid != self.class_filter:
                        continue
                    rows.append((x1, y1, x2, y2, conf, cid))
                    self._paste(scores[idx, :, :, 0], m, conf, 0, x1, y1, x2, y2)
            else:
                for r0 in self.row_starts:
                    r1 = min(r0 + self.tile_rows, height)
                    if r1 <= r0:
                        continue
                    tile = frame[r0:r1, :, :]
                    if self.jpeg_roundtrip:
                        tile = _jpeg_roundtrip(tile, self.jpeg_quality)
                    for x1, y1, x2, y2, conf, cid, m in self._predict_frame(model, tile):
                        if self.class_filter is not None and cid != self.class_filter:
                            continue
                        rows.append((x1, y1 + r0, x2, y2 + r0, conf, cid))
                        self._paste(scores[idx, :, :, 0], m, conf, r0, x1, y1, x2, y2)
                rows = nms_rows(rows, self.nms_iou)

            items = [
                {"xyxy": [r[0], r[1], r[2], r[3]], "confidence": r[4], "class_id": r[5]}
                for r in rows
            ]
            detections.append(items)
            if self.score_reduction == "top_frac_mean":
                anomaly_score[idx] = top_frac_mean(scores[idx, :, :, 0], self.top_frac)
            else:
                anomaly_score[idx] = max((r[4] for r in rows), default=0.0)

        return {
            "scores": scores.to(device),
            "detections": detections,
            "anomaly_score": anomaly_score.to(device),
        }

    @staticmethod
    def _paste(canvas: Tensor, inst_mask, conf: float, r0: int, x1, y1, x2, y2) -> None:
        """Max-paste one instance (mask if present, else its box) into the canvas."""
        h, w = canvas.shape
        if inst_mask is not None:
            m = torch.as_tensor(inst_mask, dtype=torch.bool)
            rows = min(m.shape[0], h - r0)
            if rows <= 0:
                return
            region = canvas[r0 : r0 + rows, : m.shape[1]]
            mm = m[:rows, : region.shape[1]]
            region[mm] = torch.maximum(region[mm], torch.tensor(conf))
            return
        col0 = min(max(int(math.floor(x1)), 0), w)
        col1 = min(max(int(math.ceil(x2)), 0), w)
        row0 = min(max(int(math.floor(y1)) + r0, 0), h)
        row1 = min(max(int(math.ceil(y2)) + r0, 0), h)
        if col1 > col0 and row1 > row0:
            canvas[row0:row1, col0:col1].clamp_(min=conf)
