"""RF-DETR criterion as a cuvis-ai loss node (Hungarian-matched DETR losses)."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_rfdetr.node.rfdetr_trainable import _DET_CLASS_NAMES, _SEG_CLASS_NAMES

_LOSS_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}


class RFDETRCriterionLoss(Node):
    """Wraps RF-DETR's ``SetCriterion`` (Hungarian matcher + weighted DETR
    losses: cls / bbox / giou, plus mask CE / Dice for segmentation variants)
    as a cuvis-ai loss node.

    Consumes the ``outputs`` + ``targets`` ports of
    :class:`~cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable` — configure
    both nodes with the **same** ``variant`` / ``segmentation`` /
    ``dataset_dir`` so the criterion matches the model — and emits the scalar
    weighted-sum loss exactly as Roboflow's own trainer computes it
    (``sum(loss_dict[k] * weight_dict[k])``).

    Category LOSS, runs in train/val/test only (never inference), mirroring
    the sibling plugins' loss-node convention. The criterion has no trainable
    parameters. Constructing this node requires ``rfdetr[train]`` installed.
    """

    _category = NodeCategory.LOSS
    _tags = frozenset(
        {NodeTag.DETECTION, NodeTag.TRAINING, NodeTag.DIFFERENTIABLE, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "outputs": PortSpec(
            dtype=dict,
            shape=(),
            description="Raw RF-DETR outputs dict from RFDETRTrainable.",
        ),
        "targets": PortSpec(
            dtype=list,
            shape=(),
            description="DETR targets list from RFDETRTrainable.",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar weighted DETR loss."),
    }

    def __init__(
        self,
        dataset_dir: str,
        variant: str = "medium",
        segmentation: bool = False,
        resolution: int | None = None,
        **kwargs: Any,
    ) -> None:
        variant_key = str(variant).lower()
        table = _SEG_CLASS_NAMES if segmentation else _DET_CLASS_NAMES
        if variant_key not in table:
            raise ValueError(
                f"RFDETRCriterionLoss: variant must be one of {sorted(table)} for "
                f"segmentation={bool(segmentation)}, got {variant!r}."
            )
        self.dataset_dir = str(dataset_dir)
        self.variant = variant_key
        self.segmentation = bool(segmentation)
        self.resolution = int(resolution) if resolution is not None else None

        assert "execution_stages" not in kwargs, "loss nodes fix their own execution stages"
        name, _ = Node.consume_base_kwargs(kwargs)
        super().__init__(
            name=name,
            execution_stages=_LOSS_STAGES,
            dataset_dir=self.dataset_dir,
            variant=self.variant,
            segmentation=self.segmentation,
            resolution=self.resolution,
            **kwargs,
        )

        try:
            import rfdetr
            from rfdetr.models.lwdetr import build_criterion_from_config
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "RFDETRCriterionLoss requires the 'rfdetr' training stack at "
                "construction time: pip install 'rfdetr[train]>=1.8,<2'"
            ) from exc

        wrapper_cls = getattr(rfdetr, table[variant_key])
        wrapper_kwargs: dict[str, Any] = {}
        if self.resolution is not None:
            wrapper_kwargs["resolution"] = self.resolution
        wrapper = wrapper_cls(**wrapper_kwargs)
        model_config = wrapper.model_config
        train_config = wrapper.get_train_config(dataset_dir=self.dataset_dir, epochs=1)
        criterion, _postprocess = build_criterion_from_config(model_config, train_config)
        self.criterion = criterion  # nn.Module (no trainable params)

    def forward(self, outputs: dict, targets: list, **_: Any) -> dict[str, torch.Tensor]:
        """Hungarian-match and reduce to the Roboflow-weighted scalar loss."""
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        terms = [loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]
        if not terms:
            raise RuntimeError(
                "RFDETRCriterionLoss: no weighted loss terms produced — "
                "criterion/model variant mismatch?"
            )
        return {"loss": torch.stack([t.reshape(()) for t in terms]).sum()}
