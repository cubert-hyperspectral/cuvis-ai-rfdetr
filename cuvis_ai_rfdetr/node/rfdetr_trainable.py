"""Trainable RF-DETR node: train RF-DETR (detection or segmentation) inside a
cuvis-ai pipeline via ``GradientTrainer``, with weight transfer from Roboflow
``.pth`` checkpoints in both directions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr.functional import targets_from_mask

#: Detection tier (Apache-2.0 sizes only — XL/2XL detection is platform-licensed).
_DET_CLASS_NAMES: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
#: Segmentation tier (fully Apache-2.0).
_SEG_CLASS_NAMES: dict[str, str] = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}


class RFDETRTrainable(Node):
    """RF-DETR (LW-DETR) as a *trainable* cuvis-ai node.

    Unlike the inference-only ``RFDETRDetector`` / ``RFDETRSegmenter`` (which
    keep the Roboflow wrapper as an unregistered attribute), this node builds
    the underlying LW-DETR ``nn.Module`` **eagerly in the constructor and
    registers it as a submodule** — so its parameters are visible to
    ``GradientTrainer`` optimizers, serialize through ``state_dict`` /
    ``pipeline.save_to_file``, and reload with the pipeline.

    Weight transfer:

    - **In:** ``checkpoint_path`` is forwarded to RF-DETR's ``pretrain_weights``
      — the exact loading path Roboflow's own trainer uses, so ``.pth``
      checkpoints trained outside cuvis-ai drop straight in.
    - **Out:** the trained weights live in this node's ``state_dict`` and are
      saved/restored with the pipeline; they can also be loaded back into the
      inference nodes via their ``checkpoint_path`` after an export.

    Wire it to :class:`~cuvis_ai_rfdetr.node.rfdetr_loss.RFDETRCriterionLoss`
    (same ``variant`` / ``segmentation`` / ``dataset_dir``) for training. The
    forward emits RF-DETR's raw outputs dict plus the DETR targets built from
    ``targets_mask`` so the criterion sees exactly what Roboflow's trainer sees.

    Notes
    -----
    - Constructing this node **requires ``rfdetr`` installed** (unlike the lazy
      inference nodes) — parameters must exist before the trainer starts.
    - ``dataset_dir`` must point at a Roboflow-COCO export (its annotations
      size ``num_classes`` and the criterion config — the same mechanism
      RF-DETR's own ``get_train_config`` uses).
    - Input images are float BHWC in ``[0, 1]``; the node resizes to the model
      resolution and applies ImageNet mean/std internally (replicating the
      Roboflow train transform).
    - A custom ``resolution`` must satisfy the variant backbone's block-size
      divisibility (e.g. medium: multiple of 24 — the default 576 qualifies);
      rfdetr asserts this at the first forward otherwise.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.IMAGE,
            NodeTag.RGB,
            NodeTag.BBOX,
            NodeTag.DETECTION,
            NodeTag.SEGMENTATION,
            NodeTag.TRAINING,
            NodeTag.DIFFERENTIABLE,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="False-colour image [B, H, W, num_channels] float32 in [0, 1] "
            "(3 by default; set num_channels for >3-band composites).",
        ),
        "targets_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            optional=True,
            description="Integer class/instance mask [B, H, W], 0 = background. "
            "Connected components become DETR targets (single FO class). "
            "Required in TRAIN (denoising queries need targets).",
        ),
        "context": PortSpec(dtype=Context, shape=()),
    }

    OUTPUT_SPECS = {
        "outputs": PortSpec(
            dtype=dict,
            shape=(),
            description="Raw RF-DETR outputs dict (pred_logits, pred_boxes, "
            "aux_outputs, ... ; pred_masks for segmentation variants).",
        ),
        "targets": PortSpec(
            dtype=list,
            shape=(),
            description="DETR targets built from targets_mask (per-image dicts "
            "with normalized cxcywh boxes + labels [+ masks]); empty list when "
            "no mask was provided.",
        ),
    }

    def __init__(
        self,
        dataset_dir: str,
        checkpoint_path: str | None = None,
        variant: str = "medium",
        segmentation: bool = False,
        resolution: int | None = None,
        num_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        variant_key = str(variant).lower()
        table = _SEG_CLASS_NAMES if segmentation else _DET_CLASS_NAMES
        if variant_key not in table:
            raise ValueError(
                f"RFDETRTrainable: variant must be one of {sorted(table)} for "
                f"segmentation={bool(segmentation)}, got {variant!r}."
            )
        if resolution is not None:
            resolution = int(resolution)
            if resolution <= 0:
                raise ValueError(
                    f"RFDETRTrainable: resolution must be a positive int, got {resolution}."
                )
        num_channels = int(num_channels)
        if num_channels < 1:
            raise ValueError(f"RFDETRTrainable: num_channels must be >= 1, got {num_channels}.")

        self.dataset_dir = str(dataset_dir)
        self.checkpoint_path = checkpoint_path
        self.variant = variant_key
        self.segmentation = bool(segmentation)
        self.resolution = resolution
        self.num_channels = num_channels

        name, execution_stages = Node.consume_base_kwargs(kwargs)
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            dataset_dir=self.dataset_dir,
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            segmentation=self.segmentation,
            resolution=self.resolution,
            num_channels=self.num_channels,
            **kwargs,
        )

        try:
            import rfdetr
            from rfdetr.training.module_model import RFDETRModelModule
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "RFDETRTrainable requires the 'rfdetr' training stack at "
                "construction time: pip install 'rfdetr[train]>=1.8,<2'"
            ) from exc

        wrapper_cls = getattr(rfdetr, table[variant_key])
        wrapper_kwargs: dict[str, Any] = {}
        if self.checkpoint_path is not None:
            wrapper_kwargs["pretrain_weights"] = str(self.checkpoint_path)
        if self.resolution is not None:
            wrapper_kwargs["resolution"] = int(self.resolution)
        if self.num_channels != 3:
            wrapper_kwargs["num_channels"] = self.num_channels
        wrapper = wrapper_cls(**wrapper_kwargs)
        model_config = wrapper.model_config
        if int(getattr(model_config, "num_channels", 3)) != self.num_channels:
            raise RuntimeError(
                f"RFDETRTrainable: model built with num_channels="
                f"{getattr(model_config, 'num_channels', 3)} but {self.num_channels} requested."
            )
        train_config = wrapper.get_train_config(dataset_dir=self.dataset_dir, epochs=1)
        module = RFDETRModelModule(model_config, train_config)
        # Registered submodule: parameters/state_dict flow through the pipeline.
        self.model = module.model
        self._input_resolution = int(model_config.resolution)

        # rfdetr's num_channels config is NOT propagated to the DINOv2 patch-embed
        # (its conv + assert stay at 3). Inflate the pretrained RGB patch-embed
        # conv to num_channels — tile the 3 input-channel filters and rescale to
        # preserve activation magnitude — and fix the module's channel assert.
        if self.num_channels != 3:
            from torch import nn

            patched = []
            for _name, mod in self.model.named_modules():
                proj = getattr(mod, "projection", None)
                if (
                    _name.endswith("patch_embeddings")
                    and isinstance(proj, nn.Conv2d)
                    and proj.in_channels == 3
                    and getattr(mod, "num_channels", None) == 3
                ):
                    out_c, _, kh, kw = proj.weight.shape
                    new = nn.Conv2d(
                        self.num_channels,
                        out_c,
                        (kh, kw),
                        stride=proj.stride,
                        padding=proj.padding,
                        bias=proj.bias is not None,
                    ).to(proj.weight.device)
                    reps = (self.num_channels + 2) // 3
                    w = proj.weight.data.repeat(1, reps, 1, 1)[:, : self.num_channels]
                    new.weight.data.copy_((w * (3.0 / self.num_channels)).to(new.weight.dtype))
                    if proj.bias is not None:
                        new.bias.data.copy_(proj.bias.data)
                    mod.projection = new
                    mod.num_channels = self.num_channels
                    patched.append(_name)
            if not patched:
                raise RuntimeError(
                    "RFDETRTrainable: no 3-channel DINOv2 patch-embed found to inflate "
                    f"to num_channels={self.num_channels}."
                )
            self._inflated_patch_embed = patched
        # rfdetr cycles the 3 ImageNet stats to num_channels for non-RGB input.
        base_m = list(getattr(wrapper, "means", [0.485, 0.456, 0.406]))
        base_s = list(getattr(wrapper, "stds", [0.229, 0.224, 0.225]))
        means = [base_m[i % len(base_m)] for i in range(self.num_channels)]
        stds = [base_s[i % len(base_s)] for i in range(self.num_channels)]
        self.register_buffer("_means", torch.tensor(means).view(1, self.num_channels, 1, 1))
        self.register_buffer("_stds", torch.tensor(stds).view(1, self.num_channels, 1, 1))

    # -- freeze / unfreeze must reach the registered submodule -----------------
    def unfreeze(self) -> None:
        super().unfreeze()
        for p in self.model.parameters():
            p.requires_grad_(True)

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        super().freeze()

    def forward(
        self,
        rgb_image: Tensor,
        targets_mask: Tensor | None = None,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Run RF-DETR on one batch, returning raw outputs + DETR targets."""
        if rgb_image.dim() != 4 or rgb_image.shape[-1] != self.num_channels:
            raise ValueError(
                f"RFDETRTrainable expects rgb_image of shape [B, H, W, {self.num_channels}], "
                f"got {tuple(rgb_image.shape)}."
            )
        from rfdetr.util.misc import NestedTensor  # lazy: heavy package

        res = self._input_resolution
        x = rgb_image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        x = (x - self._means) / self._stds
        mask = torch.zeros(x.shape[0], res, res, dtype=torch.bool, device=x.device)
        samples = NestedTensor(x, mask)

        targets: list[dict[str, Tensor]] = []
        if targets_mask is not None:
            targets = targets_from_mask(targets_mask, with_masks=self.segmentation)
            targets = [{k: v.to(x.device) for k, v in t.items()} for t in targets]

        stage = context.stage if context is not None else None
        if (stage == ExecutionStage.TRAIN or self.training) and targets_mask is None:
            raise RuntimeError(
                "RFDETRTrainable: targets_mask is required in TRAIN "
                "(RF-DETR's denoising queries need ground-truth targets)."
            )

        outputs = self.model(samples, targets if targets else None)
        return {"outputs": outputs, "targets": targets}
