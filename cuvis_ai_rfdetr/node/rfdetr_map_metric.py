"""Validation COCO mAP of an RF-DETR training run, computed as RF-DETR's trainer does."""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor, nn
from torchmetrics import Metric as TorchMetric

from cuvis_ai_rfdetr._compat import base_kwargs
from cuvis_ai_rfdetr.node.rfdetr_trainable import _DET_CLASS_NAMES, _SEG_CLASS_NAMES

_METRIC_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

#: Logged name -> (weights stream, torchmetrics key with a bbox+segm metric, key with bbox only).
_KEYS: dict[str, tuple[str, str, str]] = {
    "mAP_50_95": ("regular", "bbox_map", "map"),
    "mAP_50": ("regular", "bbox_map_50", "map_50"),
    "segm_mAP_50_95": ("regular", "segm_map", ""),
    "segm_mAP_50": ("regular", "segm_map_50", ""),
    "ema_mAP_50_95": ("ema", "bbox_map", "map"),
    "ema_mAP_50": ("ema", "bbox_map_50", "map_50"),
    "ema_segm_mAP_50_95": ("ema", "segm_map", ""),
    "ema_segm_mAP_50": ("ema", "segm_map_50", ""),
}
_BEST = "best_mAP_50_95"


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def convert_preds(results: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
    """RF-DETR's ``PostProcess`` results -> torchmetrics predictions (masks ``[K, H, W]``), CPU."""
    out = []
    for result in results:
        entry = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        masks = entry.get("masks")
        if masks is not None and masks.ndim == 4 and masks.shape[1] == 1:
            entry["masks"] = masks.squeeze(1)
        out.append(entry)
    return out


def convert_targets(
    targets: list[dict[str, Tensor]], preds: list[dict[str, Tensor]] | None = None
) -> list[dict[str, Tensor]]:
    """DETR targets -> torchmetrics ground truth, as RF-DETR's ``COCOEvalCallback`` converts them.

    Boxes: normalized cxcywh -> absolute xyxy at ``orig_size``. Masks: resized (nearest)
    to the matching prediction's mask grid when one is given, else to ``orig_size``. CPU.
    """
    out = []
    for index, target in enumerate(targets):
        height, width = (int(v) for v in target["orig_size"].tolist())
        boxes = target["boxes"].detach().cpu()
        scale = boxes.new_tensor([width, height, width, height])
        entry: dict[str, Tensor] = {
            "boxes": _cxcywh_to_xyxy(boxes) * scale,
            "labels": target["labels"].detach().cpu(),
        }
        if "masks" in target:
            masks = target["masks"].detach().cpu().bool()
            size = (height, width)
            if preds is not None and "masks" in preds[index]:
                size = (int(preds[index]["masks"].shape[-2]), int(preds[index]["masks"].shape[-1]))
            if tuple(masks.shape[-2:]) != size:
                masks = (
                    F.interpolate(masks.float().unsqueeze(1), size=size, mode="nearest")
                    .squeeze(1)
                    .bool()
                )
            entry["masks"] = masks
        if "iscrowd" in target:
            entry["iscrowd"] = target["iscrowd"].detach().cpu()
        out.append(entry)
    return out


class _MapStore:
    """Per-stream ``MeanAveragePrecision`` objects for one pass, computed once and cached."""

    def __init__(self, segmentation: bool, max_dets: int) -> None:
        self.segmentation = segmentation
        self.max_dets = max_dets
        self.metrics: dict[str, Any] = {}
        self._results: dict[str, dict[str, Tensor]] | None = None

    def _new_metric(self) -> Any:
        from torchmetrics.detection import MeanAveragePrecision

        return MeanAveragePrecision(
            iou_type=["bbox", "segm"] if self.segmentation else "bbox",
            class_metrics=False,
            max_detection_thresholds=[1, 10, self.max_dets],
            backend="faster_coco_eval",
            sync_on_compute=False,
        )

    def reset(self) -> None:
        self.metrics = {}
        self._results = None

    def update(self, stream: str, preds: list[dict], targets: list[dict]) -> None:
        if stream not in self.metrics:
            self.metrics[stream] = self._new_metric()
        self.metrics[stream].update(preds, targets)
        self._results = None

    def has(self, stream: str) -> bool:
        return stream in self.metrics

    def results(self) -> dict[str, dict[str, Tensor]]:
        if self._results is None:
            self._results = {stream: metric.compute() for stream, metric in self.metrics.items()}
        return self._results


class _MapView(TorchMetric):
    """One logged number of the shared store (the store holds the state, this the name)."""

    full_state_update = False
    higher_is_better = True

    def __init__(self, compute_fn: Any) -> None:
        super().__init__()
        self._compute_fn = compute_fn

    def update(self) -> None:  # the node feeds the store; this only counts batches
        return None

    def compute(self) -> Tensor:
        return self._compute_fn()


class RFDETRMapMetric(Node):
    """Validation COCO mAP for :class:`RFDETRTrainable`, as RF-DETR's own trainer computes it.

    Consumes the training node's ``outputs`` (and ``ema_outputs`` when it keeps an EMA)
    and ``targets``, turns the outputs into detections with RF-DETR's ``PostProcess``
    and pools ``torchmetrics.detection.MeanAveragePrecision`` over the whole val or test
    pass (bbox and, for segmentation models, segm; max detections ``1 / 10 / max_dets``;
    ``faster_coco_eval`` backend), the settings of RF-DETR's ``COCOEvalCallback``.

    Logged per pass: ``mAP_50_95``, ``mAP_50`` (bbox), ``segm_mAP_50_95``,
    ``segm_mAP_50`` and the same with an ``ema_`` prefix, plus ``best_mAP_50_95``: the
    higher of the regular and EMA value of RF-DETR's selection metric (segm mAP 50:95
    for segmentation models, bbox mAP 50:95 otherwise). Monitoring ``best_mAP_50_95``
    (mode ``max``) in the checkpoint callback picks the epoch RF-DETR's best-of-both
    checkpoint (``checkpoint_best_total``) would pick. The buffer ``selected_stream``
    (0 regular, 1 EMA) records which weights reached that value, so it is saved with
    the best checkpoint and tells a loader which weights to use.

    Configure ``variant`` / ``segmentation`` / ``resolution`` / ``class_names`` like the
    training node. Category METRIC, runs at val and test only.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DETECTION, NodeTag.SEGMENTATION})
    EXECUTION_STAGES = frozenset(_METRIC_STAGES)

    INPUT_SPECS = {
        "outputs": PortSpec(
            dtype=dict, shape=(), description="Raw RF-DETR outputs of the live weights."
        ),
        "targets": PortSpec(
            dtype=list, shape=(), description="DETR targets with orig_size per image."
        ),
        "ema_outputs": PortSpec(
            dtype=dict,
            shape=(),
            optional=True,
            description="Raw RF-DETR outputs of the EMA weights.",
        ),
        "context": PortSpec(dtype=Context, shape=(), optional=True),
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            description="Empty per batch; the mAP values are pooled over the pass.",
        ),
    }
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset({*_KEYS, _BEST})

    def __init__(
        self,
        dataset_dir: str = ".",
        variant: str = "medium",
        segmentation: bool = False,
        resolution: int | None = None,
        class_names: list[str] | None = None,
        max_dets: int = 500,
        **kwargs: Any,
    ) -> None:
        variant_key = str(variant).lower()
        table = _SEG_CLASS_NAMES if segmentation else _DET_CLASS_NAMES
        if variant_key not in table:
            raise ValueError(
                f"RFDETRMapMetric: variant must be one of {sorted(table)} for "
                f"segmentation={bool(segmentation)}, got {variant!r}."
            )
        if int(max_dets) < 1:
            raise ValueError(f"RFDETRMapMetric: max_dets must be >= 1, got {max_dets}.")
        if class_names is not None:
            class_names = [str(name) for name in class_names]
        self.dataset_dir = str(dataset_dir)
        self.variant = variant_key
        self.segmentation = bool(segmentation)
        self.resolution = int(resolution) if resolution is not None else None
        self.class_names = class_names
        self.max_dets = int(max_dets)

        super().__init__(
            **base_kwargs(kwargs, execution_stages=_METRIC_STAGES),
            dataset_dir=self.dataset_dir,
            variant=self.variant,
            segmentation=self.segmentation,
            resolution=self.resolution,
            class_names=self.class_names,
            max_dets=self.max_dets,
            **kwargs,
        )

        try:
            import rfdetr
            from rfdetr.models.lwdetr import build_criterion_from_config
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "RFDETRMapMetric requires the 'rfdetr' training stack at construction "
                "time: pip install 'rfdetr[train]>=1.8,<2'"
            ) from exc

        wrapper_kwargs: dict[str, Any] = {}
        if self.resolution is not None:
            wrapper_kwargs["resolution"] = self.resolution
        if self.class_names is not None:
            wrapper_kwargs["num_classes"] = len(self.class_names)
        wrapper = getattr(rfdetr, table[variant_key])(**wrapper_kwargs)
        train_config = wrapper.get_train_config(dataset_dir=self.dataset_dir, epochs=1)
        _criterion, self.postprocess = build_criterion_from_config(
            wrapper.model_config, train_config
        )

        self.register_buffer("selected_stream", torch.zeros((), dtype=torch.long))
        self._store = _MapStore(self.segmentation, self.max_dets)
        self.views = nn.ModuleDict(
            {name: _MapView(partial(self._value, name)) for name in self.POOLED_METRIC_NAMES}
        )
        self._last_key: Any = None

    def forward(
        self,
        outputs: dict,
        targets: list,
        ema_outputs: dict | None = None,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Add one batch to the pooled mAP of the current val / test pass."""
        key = (getattr(context, "stage", None), getattr(context, "epoch", None))
        if key != self._last_key or getattr(context, "batch_idx", None) == 0:
            self._store.reset()
            self._last_key = key
        if not targets:
            return {"metrics": []}
        device = outputs["pred_logits"].device
        orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        with torch.no_grad():
            for stream, raw in (("regular", outputs), ("ema", ema_outputs)):
                if raw is None:
                    continue
                preds = convert_preds(self.postprocess(raw, orig_sizes))
                gts = convert_targets(targets, preds if self.segmentation else None)
                self._store.update(stream, preds, gts)
        for view in self.views.values():
            view.update()
        return {"metrics": []}

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """The live metric objects of the streams seen in the current pass."""
        names = [name for name, (stream, *_k) in _KEYS.items() if self._store.has(stream)]
        if not self.segmentation:
            names = [name for name in names if "segm" not in name]
        if names:
            names.append(_BEST)
        return {name: self.views[name] for name in names}

    def _value(self, name: str) -> Tensor:
        results = self._store.results()
        if name == _BEST:
            key = "segm_map" if self.segmentation else "map"
            values = {stream: float(res[key]) for stream, res in results.items()}
            best = (
                "ema"
                if values.get("ema", float("-inf")) > values.get("regular", float("-inf"))
                else "regular"
            )
            self.selected_stream.fill_(1 if best == "ema" else 0)
            return torch.tensor(values[best], dtype=torch.float32)
        stream, both_key, bbox_key = _KEYS[name]
        res = results.get(stream)
        if res is None:
            return torch.tensor(float("nan"))
        return torch.as_tensor(
            res[both_key if self.segmentation else bbox_key], dtype=torch.float32
        )
