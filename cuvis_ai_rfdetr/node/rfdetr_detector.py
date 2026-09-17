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

from cuvis_ai_rfdetr._compat import base_kwargs
from cuvis_ai_rfdetr.functional import (
    jpeg_roundtrip as _jpeg_roundtrip,
)
from cuvis_ai_rfdetr.functional import (
    top_frac_mean,
)

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
            description="Image-level score [B]: max box confidence (default), "
            "or the mean of the top-fraction pixels of the rasterized map "
            "when score_reduction='top_frac_mean' (0.0 when empty).",
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
        """Configure the detector without loading any weights yet.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to fine-tuned RF-DETR weights (e.g. the
            ``checkpoint_best_total.pth`` written by RF-DETR's own trainer).
            ``None`` loads the official COCO-pretrained weights for the
            selected variant.
        variant : str
            RF-DETR size variant: ``"nano"``, ``"small"``, ``"medium"``, or
            ``"large"`` (the Apache-2.0 tier).
        threshold : float
            Confidence threshold in ``[0, 1]`` passed to ``model.predict``.
        resolution : int, optional
            Optional input resolution forwarded to the RF-DETR constructor.
            ``None`` keeps the variant's default (RF-DETR enforces its own
            divisibility constraints). RF-DETR accepts any resolution divisible
            by the patch size (14) and interpolates positional encodings, so a
            higher value (e.g. 640 / 728) can be tried for small-object recall.
        checkpoint_loader : str
            Where the model *configuration* comes from when loading a
            fine-tuned checkpoint — the same checkpoint can yield materially
            different scores between the two loaders. ``"constructor"``
            (default): build the configured variant at ``resolution`` (or the
            class default) and load the checkpoint's weights into it.
            ``"from_checkpoint"``: delegate to
            ``rfdetr.RFDETR.from_checkpoint`` — class and configuration come
            from the checkpoint, falling back to **class defaults for fields
            the checkpoint does not carry, including resolution** (fine-tuned
            checkpoints do not necessarily record their training resolution).
            Use this to reproduce harnesses that load checkpoints the same
            way; ``resolution`` (if set) is forwarded as an explicit override,
            and ``variant`` must match the class resolved from the checkpoint.
        tiling : str
            ``"tiled"`` (default) splits each frame into overlapping full-width
            row-strips (``tile_rows`` high) at ``row_starts``, runs the model
            per tile, offsets boxes back to frame coordinates, and merges them
            with NMS at ``nms_iou`` — reproducing the tiled evaluation protocol
            for tall/narrow lane crops. ``"whole"`` runs the model once on the
            full frame (relying on RF-DETR's internal resize).
        tile_rows : int
            Row height of each tile in ``"tiled"`` mode (default 405).
        row_starts : tuple[int, ...]
            Top-row offsets of the tiles in ``"tiled"`` mode (default
            ``(0, 291, 582)`` = three overlapping 405-row tiles over a 987-row
            lane crop, matching the training/eval protocol).
        nms_iou : float
            IoU threshold for merging boxes across tiles (default 0.5).
        jpeg_roundtrip : bool
            Route each model input (every tile in ``"tiled"`` mode, the frame
            in ``"whole"`` mode) through an in-memory JPEG encode/decode at
            ``jpeg_quality`` before prediction. Lossy on purpose: evaluation
            harnesses that persist model inputs as ``.jpg`` files make the
            compression part of the score definition, and this reproduces
            their scores exactly.
        jpeg_quality : int
            JPEG quality for ``jpeg_roundtrip`` (default 95, Pillow defaults
            otherwise — pin Pillow for byte-exact reproduction).
        class_filter : int, optional
            Keep only detections with this ``class_id`` (dropped before NMS,
            rasterization, and scoring), matching harnesses that score a
            single foreground class.
        score_reduction : str
            ``"max_conf"`` (default): ``anomaly_score`` = max box confidence.
            ``"top_frac_mean"``: mean of the top ``top_frac`` fraction of the
            rasterized score map (integer-floor top-k), comparable to dense
            models scored the same way.
        top_frac : float
            Fraction for ``"top_frac_mean"`` (default 0.001).
        """
        variant_key = str(variant).lower()
        if variant_key not in _VARIANT_CLASS_NAMES:
            raise ValueError(
                f"RFDETRDetector: variant must be one of "
                f"{sorted(_VARIANT_CLASS_NAMES)} (Apache-2.0 tier), got {variant!r}."
            )
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"RFDETRDetector: threshold must be within [0, 1], got {threshold}.")
        if resolution is not None:
            resolution = int(resolution)
            if resolution <= 0:
                raise ValueError(
                    f"RFDETRDetector: resolution must be a positive int, got {resolution}."
                )
        tiling = str(tiling).lower()
        if tiling not in ("tiled", "whole"):
            raise ValueError(f"RFDETRDetector: tiling must be 'tiled' or 'whole', got {tiling!r}.")
        tile_rows = int(tile_rows)
        if tile_rows <= 0:
            raise ValueError(f"RFDETRDetector: tile_rows must be a positive int, got {tile_rows}.")
        row_starts = tuple(int(r) for r in row_starts)
        nms_iou = float(nms_iou)
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError(f"RFDETRDetector: nms_iou must be within [0, 1], got {nms_iou}.")
        jpeg_roundtrip = bool(jpeg_roundtrip)
        jpeg_quality = int(jpeg_quality)
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"RFDETRDetector: jpeg_quality must be within [1, 100], got {jpeg_quality}."
            )
        if class_filter is not None:
            class_filter = int(class_filter)
        score_reduction = str(score_reduction).lower()
        if score_reduction not in ("max_conf", "top_frac_mean"):
            raise ValueError(
                f"RFDETRDetector: score_reduction must be 'max_conf' or "
                f"'top_frac_mean', got {score_reduction!r}."
            )
        top_frac = float(top_frac)
        if not 0.0 < top_frac <= 1.0:
            raise ValueError(f"RFDETRDetector: top_frac must be within (0, 1], got {top_frac}.")
        checkpoint_loader = str(checkpoint_loader).lower()
        if checkpoint_loader not in ("constructor", "from_checkpoint"):
            raise ValueError(
                f"RFDETRDetector: checkpoint_loader must be 'constructor' or "
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

        super().__init__(
            **base_kwargs(kwargs),
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
        if self.checkpoint_path is not None and self.checkpoint_loader == "from_checkpoint":
            loader = getattr(rfdetr, "RFDETR", None)
            if loader is None or not hasattr(loader, "from_checkpoint"):
                raise RuntimeError(
                    "RFDETRDetector: checkpoint_loader='from_checkpoint' requires "
                    "rfdetr>=1.8 (rfdetr.RFDETR.from_checkpoint not found)."
                )
            loader_kwargs: dict[str, Any] = {}
            if self.resolution is not None:
                loader_kwargs["resolution"] = int(self.resolution)
            model = loader.from_checkpoint(str(self.checkpoint_path), **loader_kwargs)
            loaded = type(model).__name__
            if loaded != class_name:
                raise RuntimeError(
                    f"RFDETRDetector: checkpoint resolved to {loaded}, but "
                    f"variant={self.variant!r} expects {class_name}. Set variant to "
                    f"match the checkpoint or use checkpoint_loader='constructor'."
                )
            return model

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

    @staticmethod
    def _box_iou(a: tuple, b: tuple) -> float:
        """IoU of two ``[x1, y1, x2, y2]`` boxes."""
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def _nms(self, rows: list) -> list:
        """Greedy NMS by descending confidence at ``self.nms_iou`` (matches the eval)."""
        keep: list = []
        for b in sorted(rows, key=lambda t: -t[4]):
            if all(self._box_iou(b[:4], k[:4]) < self.nms_iou for k in keep):
                keep.append(b)
        return keep

    def _boxes_for_frame(self, model: Any, frame: Any, height: int) -> list:
        """Boxes for one HWC uint8 frame: whole-frame, or tiled + NMS-merged.

        ``"tiled"`` reproduces the training/eval protocol: full-width row-strips
        of ``tile_rows`` at each ``row_starts`` offset, boxes shifted back into
        frame coordinates by the tile's top row, then merged with NMS. The
        ``class_filter`` (if set) drops foreign-class detections before NMS;
        ``jpeg_roundtrip`` compresses each model input first.
        """
        if self.tiling == "whole":
            model_input = (
                _jpeg_roundtrip(frame, self.jpeg_quality) if self.jpeg_roundtrip else frame
            )
            rows = self._predict_frame(model, model_input)
            if self.class_filter is not None:
                rows = [r for r in rows if r[5] == self.class_filter]
            return rows
        rows = []
        for r0 in self.row_starts:
            r1 = min(r0 + self.tile_rows, height)
            if r1 <= r0:
                continue
            tile = frame[r0:r1, :, :]
            if self.jpeg_roundtrip:
                tile = _jpeg_roundtrip(tile, self.jpeg_quality)
            for x1, y1, x2, y2, conf, cid in self._predict_frame(model, tile):
                if self.class_filter is not None and cid != self.class_filter:
                    continue
                rows.append((x1, y1 + r0, x2, y2 + r0, conf, cid))
        return self._nms(rows)

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
            for x1, y1, x2, y2, conf, cid in self._boxes_for_frame(model, frames[idx], height):
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
            if self.score_reduction == "top_frac_mean":
                anomaly_score[idx] = top_frac_mean(scores[idx, :, :, 0], self.top_frac)
            else:
                anomaly_score[idx] = best_confidence

        return {
            "scores": scores.to(device),
            "detections": detections,
            "anomaly_score": anomaly_score.to(device),
        }
