"""Trainable RF-DETR node: train RF-DETR (detection or segmentation) inside a
cuvis-ai pipeline via ``GradientTrainer``, with weight transfer from Roboflow
``.pth`` checkpoints in both directions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_rfdetr import optim as _optim
from cuvis_ai_rfdetr._compat import base_kwargs
from cuvis_ai_rfdetr.ema import ema_update_
from cuvis_ai_rfdetr.functional import (
    compute_multi_scale_scales,
    jpeg_roundtrip,
    targets_from_mask,
    tile_rows,
    to_uint8_frames,
)

# Training runs select the RF-DETR optimizer by name (``optimizer.name: rfdetr_adamw``);
# the node module is imported whenever a pipeline uses the node, which registers it.
_optim.register_optimizer()

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

    Training as RF-DETR's own trainer does (all off by default, existing runs unchanged):

    - ``class_names``: size the class head from a class list, as RF-DETR sizes it from
      the dataset's categories; ``target_class_id`` is the label every object gets in
      the single-class case (``1`` for a Roboflow-style ``["fo-background", "fo"]``).
    - ``jpeg_roundtrip`` / ``jpeg_quality``: encode and decode every image as JPEG
      before the model, for recipes whose original training images were JPEG files.
    - ``native_transforms``: run RF-DETR's own train / val transforms per image (from
      the model and train configs; ``aug_config`` picks the augmentation, e.g. the
      ``"rfdetr_1_8"`` preset) and pad the batch like RF-DETR's data loader. Replaces
      the fixed resize and the internal normalization.
    - ``eval_tile_rows`` / ``eval_row_starts``: at val/test, cut each frame into these
      row tiles before the model (training crops come from an upstream crop transform).
    - ``ema_decay`` / ``ema_tau``: keep RF-DETR's weight average (``ema_model``),
      updated after each optimizer step of the ``rfdetr_adamw`` optimizer and once per
      epoch; val/test also emit ``ema_outputs`` for :class:`RFDETRMapMetric`.
    - ``lr_encoder`` / ``lr_vit_layer_decay`` / ``lr_component_decay``: the per-layer
      learning rates ``rfdetr_adamw`` uses (defaults from RF-DETR's train config).
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
            "Connected components become DETR targets — one FO class by default, or "
            "per-class labels (id - 1) when multiclass_targets=True. "
            "Required in TRAIN (denoising queries need targets).",
        ),
        # The executor injects the Context as a call-site kwarg to every node
        # (pipeline.forward(..., context=context)), never through a port, so this
        # must be optional — otherwise graph pre-flight validation flags it as an
        # unsatisfied required input (it is in neither the batch nor a connection).
        "context": PortSpec(dtype=Context, shape=(), optional=True),
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
        "ema_outputs": PortSpec(
            dtype=dict,
            shape=(),
            optional=True,
            description="Raw outputs of the EMA weights at val/test (only with "
            "ema_decay set and after the first EMA update).",
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
        checkpoint_loader: str = "constructor",
        multiclass_targets: bool = False,
        class_names: list[str] | None = None,
        target_class_id: int = 0,
        jpeg_roundtrip: bool = False,
        jpeg_quality: int = 95,
        native_transforms: bool = False,
        aug_config: dict[str, Any] | str | None = None,
        eval_tile_rows: int | None = None,
        eval_row_starts: list[int] | None = None,
        ema_decay: float | None = None,
        ema_tau: float = 100.0,
        lr_encoder: float | None = None,
        lr_vit_layer_decay: float | None = None,
        lr_component_decay: float | None = None,
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
        if checkpoint_loader not in ("constructor", "from_checkpoint"):
            raise ValueError(
                "RFDETRTrainable: checkpoint_loader must be 'constructor' or "
                f"'from_checkpoint', got {checkpoint_loader!r}."
            )
        if checkpoint_loader == "from_checkpoint" and num_channels != 3:
            raise ValueError(
                "RFDETRTrainable: checkpoint_loader='from_checkpoint' supports only "
                "num_channels=3 (rfdetr's loader has no channel-inflation path); "
                "use checkpoint_loader='constructor' for >3-band composites."
            )

        if class_names is not None:
            class_names = [str(name) for name in class_names]
            if not class_names:
                raise ValueError("RFDETRTrainable: class_names must not be empty.")
        target_class_id = int(target_class_id)
        if target_class_id < 0 or (
            class_names is not None
            and not multiclass_targets
            and target_class_id >= len(class_names)
        ):
            raise ValueError(
                f"RFDETRTrainable: target_class_id={target_class_id} is not a class of "
                f"class_names={class_names}."
            )
        jpeg_quality = int(jpeg_quality)
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"RFDETRTrainable: jpeg_quality must be in 1..100, got {jpeg_quality}."
            )
        if (jpeg_roundtrip or native_transforms) and num_channels != 3:
            raise ValueError(
                "RFDETRTrainable: jpeg_roundtrip and native_transforms need 3-channel images."
            )
        if (eval_tile_rows is None) != (eval_row_starts is None):
            raise ValueError(
                "RFDETRTrainable: set both eval_tile_rows and eval_row_starts, or neither."
            )
        if eval_tile_rows is not None:
            eval_tile_rows = int(eval_tile_rows)
            eval_row_starts = [int(s) for s in eval_row_starts]
            if eval_tile_rows <= 0 or not eval_row_starts or min(eval_row_starts) < 0:
                raise ValueError(
                    "RFDETRTrainable: eval_tile_rows must be > 0 and eval_row_starts a "
                    "non-empty list of row offsets >= 0."
                )
        if ema_decay is not None:
            ema_decay = float(ema_decay)
            if not 0.0 < ema_decay <= 1.0:
                raise ValueError(f"RFDETRTrainable: ema_decay must be in (0, 1], got {ema_decay}.")
        ema_tau = float(ema_tau)
        if ema_tau < 0.0:
            raise ValueError(f"RFDETRTrainable: ema_tau must be >= 0, got {ema_tau}.")
        lr_knobs = {
            "lr_encoder": lr_encoder,
            "lr_vit_layer_decay": lr_vit_layer_decay,
            "lr_component_decay": lr_component_decay,
        }
        for knob, value in lr_knobs.items():
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"RFDETRTrainable: {knob} must be > 0, got {value}.")
        from cuvis_ai_rfdetr.native_data import resolve_aug_config

        resolve_aug_config(aug_config)  # validate the preset name / type early

        self.dataset_dir = str(dataset_dir)
        self.checkpoint_path = checkpoint_path
        self.variant = variant_key
        self.segmentation = bool(segmentation)
        self.resolution = resolution
        self.num_channels = num_channels
        self.checkpoint_loader = checkpoint_loader
        self.multiclass_targets = bool(multiclass_targets)
        self.class_names = class_names
        self.target_class_id = target_class_id
        self.jpeg_roundtrip = bool(jpeg_roundtrip)
        self.jpeg_quality = jpeg_quality
        self.native_transforms = bool(native_transforms)
        self.aug_config = aug_config
        self.eval_tile_rows = eval_tile_rows
        self.eval_row_starts = eval_row_starts
        self.ema_decay = ema_decay
        self.ema_tau = ema_tau
        self.lr_encoder = None if lr_encoder is None else float(lr_encoder)
        self.lr_vit_layer_decay = None if lr_vit_layer_decay is None else float(lr_vit_layer_decay)
        self.lr_component_decay = None if lr_component_decay is None else float(lr_component_decay)

        super().__init__(
            **base_kwargs(kwargs),
            dataset_dir=self.dataset_dir,
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            segmentation=self.segmentation,
            resolution=self.resolution,
            num_channels=self.num_channels,
            checkpoint_loader=self.checkpoint_loader,
            multiclass_targets=self.multiclass_targets,
            class_names=self.class_names,
            target_class_id=self.target_class_id,
            jpeg_roundtrip=self.jpeg_roundtrip,
            jpeg_quality=self.jpeg_quality,
            native_transforms=self.native_transforms,
            aug_config=self.aug_config,
            eval_tile_rows=self.eval_tile_rows,
            eval_row_starts=self.eval_row_starts,
            ema_decay=self.ema_decay,
            ema_tau=self.ema_tau,
            lr_encoder=self.lr_encoder,
            lr_vit_layer_decay=self.lr_vit_layer_decay,
            lr_component_decay=self.lr_component_decay,
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

        wrapper = self._build_wrapper(rfdetr, table)
        model_config = wrapper.model_config
        if int(getattr(model_config, "num_channels", 3)) != self.num_channels:
            raise RuntimeError(
                f"RFDETRTrainable: model built with num_channels="
                f"{getattr(model_config, 'num_channels', 3)} but {self.num_channels} requested."
            )
        train_config = wrapper.get_train_config(dataset_dir=self.dataset_dir, epochs=1)
        # Kept for get_param_groups: the native param-group builder reads a flat
        # args namespace mixing BOTH configs (lr knobs from the train config,
        # backbone fields like out_feature_indexes from the model config).
        self._model_config = model_config
        self._train_config = train_config
        module = RFDETRModelModule(model_config, train_config)
        # Registered submodule: parameters/state_dict flow through the pipeline.
        self.model = module.model
        self._input_resolution = int(model_config.resolution)
        # The model's spatial divisibility unit (windowed attention): inputs whose
        # side is a multiple of this are valid without resizing (multi-scale train).
        self._spatial_unit = int(getattr(model_config, "patch_size", 16)) * int(
            getattr(model_config, "num_windows", 4)
        )

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

        # RF-DETR's weight average: a frozen copy of the model, saved with the node.
        # ``_ema_count`` counts applied updates (0 = still the initial copy),
        # ``_ema_epoch`` the last training epoch seen (for the per-epoch update).
        self.ema_model = None
        if self.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
        self.register_buffer("_ema_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_ema_epoch", torch.full((), -1, dtype=torch.long))
        self._native_tf_cache: dict[str, Any] = {}
        _optim.track_trainable(self)

    # -- freeze / unfreeze must reach the registered submodule -----------------
    def unfreeze(self) -> None:
        super().unfreeze()
        for p in self.model.parameters():
            p.requires_grad_(True)
        if self.ema_model is not None:  # the average is never trained directly
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        super().freeze()
        if self.ema_model is not None:
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    # -- RF-DETR optimizer setup and weight average -----------------------------
    def optimizer_overrides(self) -> dict[str, float]:
        """The per-layer learning-rate knobs set on this node (others: RF-DETR defaults)."""
        knobs = {
            "lr_encoder": self.lr_encoder,
            "lr_vit_layer_decay": self.lr_vit_layer_decay,
            "lr_component_decay": self.lr_component_decay,
        }
        return {k: v for k, v in knobs.items() if v is not None}

    @torch.no_grad()
    def update_ema(self) -> None:
        """One RF-DETR EMA update of ``ema_model`` towards the live weights."""
        if self.ema_model is None:
            return
        ema_update_(self.ema_model, self.model, int(self._ema_count), self.ema_decay, self.ema_tau)
        self._ema_count += 1

    def _build_wrapper(self, rfdetr: Any, table: dict[str, str]) -> Any:
        """Build the RF-DETR wrapper via the configured checkpoint loader.

        ``"constructor"`` (default, unchanged behavior): the variant class with
        ``pretrain_weights=checkpoint_path``. Its generic weight loader falls
        back to a flat slice when a checkpoint lacks ``args.num_queries`` /
        ``args.group_detr`` — with ``group_detr > 1`` that can scramble the
        per-group query structure of a fine-tuned checkpoint.

        ``"from_checkpoint"``: delegate to ``rfdetr.RFDETR.from_checkpoint`` —
        the same loader-faithful path the inference nodes use. The wrapper class
        and configuration (including the query structure) come from the
        checkpoint itself, so fine-tuned weights load without the flat-slice
        fallback; ``variant`` / ``segmentation`` must match the checkpoint and
        are validated against the resolved class.
        """
        class_name = table[self.variant]
        if self.checkpoint_path is not None and self.checkpoint_loader == "from_checkpoint":
            loader = getattr(rfdetr, "RFDETR", None)
            if loader is None or not hasattr(loader, "from_checkpoint"):
                raise RuntimeError(
                    "RFDETRTrainable: checkpoint_loader='from_checkpoint' requires "
                    "rfdetr>=1.8 (rfdetr.RFDETR.from_checkpoint not found)."
                )
            loader_kwargs: dict[str, Any] = {}
            if self.resolution is not None:
                loader_kwargs["resolution"] = int(self.resolution)
            if self.class_names is not None:
                loader_kwargs["num_classes"] = len(self.class_names)
            wrapper = loader.from_checkpoint(str(self.checkpoint_path), **loader_kwargs)
            loaded = type(wrapper).__name__
            if loaded != class_name:
                raise RuntimeError(
                    f"RFDETRTrainable: checkpoint resolved to {loaded}, but "
                    f"variant={self.variant!r} (segmentation={self.segmentation}) expects "
                    f"{class_name}. Set variant/segmentation to match the checkpoint or "
                    "use checkpoint_loader='constructor'."
                )
            return wrapper

        wrapper_cls = getattr(rfdetr, class_name)
        wrapper_kwargs: dict[str, Any] = {}
        if self.checkpoint_path is not None:
            wrapper_kwargs["pretrain_weights"] = str(self.checkpoint_path)
        if self.resolution is not None:
            wrapper_kwargs["resolution"] = int(self.resolution)
        if self.num_channels != 3:
            wrapper_kwargs["num_channels"] = self.num_channels
        if self.class_names is not None:
            # As RF-DETR's trainer aligns num_classes to the dataset's categories: the
            # pretrained head is loaded, then cut to num_classes + 1 outputs.
            wrapper_kwargs["num_classes"] = len(self.class_names)
        return wrapper_cls(**wrapper_kwargs)

    def get_param_groups(
        self,
        lr: float | None = None,
        lr_encoder: float | None = None,
        lr_vit_layer_decay: float | None = None,
        lr_component_decay: float | None = None,
    ) -> list[dict[str, Any]]:
        """Native LW-DETR optimizer param groups for this node's model.

        Delegates to rfdetr's own ``get_param_dict`` — the exact grouping the
        Roboflow trainer uses: encoder params at ``lr_encoder`` with per-block
        ViT layer decay (``lr_vit_layer_decay``), decoder params at
        ``lr * lr_component_decay``, everything else at ``lr``. Feed the result
        to a torch optimizer (param-group dicts) so a cuvis-ai training run
        reproduces the native loop's learning-rate structure.

        Overrides default to the wrapper's train config; pass explicit values
        to match a specific run (e.g. a fine-tuning recipe of ``lr=1e-4,
        lr_encoder=1.5e-4, lr_vit_layer_decay=0.8, lr_component_decay=0.7``).

        Like the native trainer's ``args``, the namespace handed to
        ``get_param_dict`` is a flat merge of the **model** config (backbone
        fields such as ``out_feature_indexes``) and the **train** config (lr
        knobs), with explicit overrides on top.
        """
        from types import SimpleNamespace

        try:  # lazy: importing rfdetr.training pulls the full train stack
            from rfdetr.training.param_groups import get_param_dict
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "get_param_groups requires the rfdetr train stack: "
                "pip install 'rfdetr[train]>=1.8,<2'"
            ) from exc

        overrides = {
            "lr": lr,
            "lr_encoder": lr_encoder,
            "lr_vit_layer_decay": lr_vit_layer_decay,
            "lr_component_decay": lr_component_decay,
        }
        overrides = {k: float(v) for k, v in overrides.items() if v is not None}

        def as_dict(cfg: Any) -> dict[str, Any]:
            return dict(cfg.model_dump()) if hasattr(cfg, "model_dump") else dict(vars(cfg))

        merged = {**as_dict(self._model_config), **as_dict(self._train_config), **overrides}
        return get_param_dict(SimpleNamespace(**merged), self.model)

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
        stage = context.stage if context is not None else None
        in_train = stage == ExecutionStage.TRAIN or (stage is None and self.training)
        if in_train and targets_mask is None:
            raise RuntimeError(
                "RFDETRTrainable: targets_mask is required in TRAIN "
                "(RF-DETR's denoising queries need ground-truth targets)."
            )
        if in_train:
            self._epoch_end_ema_update(context)
        elif self.eval_tile_rows is not None:
            rgb_image = tile_rows(rgb_image, self.eval_tile_rows, self.eval_row_starts)
            if targets_mask is not None:
                targets_mask = tile_rows(targets_mask, self.eval_tile_rows, self.eval_row_starts)

        if self.native_transforms:
            samples, targets = self._native_samples(rgb_image, targets_mask, in_train)
        else:
            samples, targets = self._resized_samples(rgb_image, targets_mask, stage)

        outputs = self.model(samples, targets if targets else None)
        result: dict[str, Any] = {"outputs": outputs, "targets": targets}
        if not in_train and self.ema_model is not None and int(self._ema_count) > 0:
            was_training = self.ema_model.training
            self.ema_model.eval()  # as RF-DETR evaluates its averaged model
            try:
                with torch.no_grad():
                    result["ema_outputs"] = self.ema_model(samples)
            finally:
                self.ema_model.train(was_training)
        return result

    def _epoch_end_ema_update(self, context: Context | None) -> None:
        """RF-DETR's extra EMA update at each epoch end, applied at the next epoch's first step.

        RF-DETR averages once more in ``on_train_epoch_end`` (after that epoch's
        validation). The live weights do not change between that point and the next
        epoch's first optimizer step, so the update is the same when applied here.
        """
        if self.ema_model is None or context is None:
            return
        epoch = getattr(context, "epoch", None)
        if epoch is None:
            return
        epoch = int(epoch)
        last = int(self._ema_epoch)
        if 0 <= last < epoch and int(self._ema_count) > 0:
            self.update_ema()
        if epoch != last:
            self._ema_epoch.fill_(epoch)

    def _jpeg_frames(self, rgb_image: Tensor) -> np.ndarray:
        """``[B, H, W, 3]`` -> uint8 frames, JPEG encoded and decoded when switched on."""
        frames = to_uint8_frames(rgb_image)
        if self.jpeg_roundtrip:
            frames = np.stack([jpeg_roundtrip(frame, self.jpeg_quality) for frame in frames])
        return frames

    def _target_kwargs(self) -> dict[str, Any]:
        return {
            "with_masks": self.segmentation,
            "multiclass": self.multiclass_targets,
            "single_label": self.target_class_id,
        }

    def _native_samples(
        self, rgb_image: Tensor, targets_mask: Tensor | None, in_train: bool
    ) -> tuple[Any, list[dict[str, Tensor]]]:
        """RF-DETR's own transforms per image, batch padded like RF-DETR's data loader."""
        from cuvis_ai_rfdetr.native_data import build_transforms, native_batch, targets_to

        image_set = "train" if in_train else "val"
        transform = self._native_tf_cache.get(image_set)
        if transform is None:
            transform = build_transforms(
                image_set, self._model_config, self._train_config, self.aug_config
            )
            self._native_tf_cache[image_set] = transform
        masks = None
        if targets_mask is not None:
            masks = targets_mask.detach().to("cpu").numpy()
        samples, targets = native_batch(
            self._jpeg_frames(rgb_image),
            masks,
            transform,
            block_size=self._spatial_unit,
            target_kwargs=self._target_kwargs(),
        )
        device = rgb_image.device
        samples = samples.to(device)
        if targets_mask is None:
            targets = []
        return samples, targets_to(targets, device)

    def _resized_samples(
        self, rgb_image: Tensor, targets_mask: Tensor | None, stage: ExecutionStage | None
    ) -> tuple[Any, list[dict[str, Tensor]]]:
        """The node's own input path: fixed resize (or multi-scale pass-through) + ImageNet stats."""
        # Lazy import (heavy package). rfdetr.util was removed in RF-DETR 1.9.0.
        from rfdetr.utilities.tensors import NestedTensor

        height, width = int(rgb_image.shape[1]), int(rgb_image.shape[2])
        if self.jpeg_roundtrip:
            frames = torch.from_numpy(self._jpeg_frames(rgb_image)).to(rgb_image.device)
            rgb_image = frames.to(torch.float32) / torch.tensor(255.0, device=rgb_image.device)
        x = rgb_image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = self._resize_for_stage(x, stage)
        x = (x - self._means) / self._stds
        side_h, side_w = int(x.shape[-2]), int(x.shape[-1])
        mask = torch.zeros(x.shape[0], side_h, side_w, dtype=torch.bool, device=x.device)
        samples = NestedTensor(x, mask)

        targets: list[dict[str, Tensor]] = []
        if targets_mask is not None:
            targets = targets_from_mask(
                targets_mask,
                with_masks=self.segmentation,
                multiclass=self.multiclass_targets,
                single_label=self.target_class_id,
            )
            size = torch.tensor([height, width], dtype=torch.int64)
            targets = [
                {**{k: v.to(x.device) for k, v in t.items()}, "orig_size": size.to(x.device)}
                for t in targets
            ]
        return samples, targets

    def _resize_for_stage(self, x: Tensor, stage: ExecutionStage | None) -> Tensor:
        """Resize BCHW input to the model resolution — except multi-scale train input.

        At TRAIN (stage or ``module.training``), an already-square input whose
        side is a multiple of the model's spatial unit
        (``patch_size * num_windows``) passes through unresized — this is what
        lets ``RandomMultiScaleResize`` feed true multi-scale sizes to the
        model, exactly like the native dataloader. Everything else — all
        val/test/inference input, and train input at arbitrary sizes — gets the
        fixed resize to the model resolution, unchanged behavior.
        """
        h, w = int(x.shape[-2]), int(x.shape[-1])
        in_train = stage == ExecutionStage.TRAIN or self.training
        if in_train and h == w and h % self._spatial_unit == 0:
            return x
        res = self._input_resolution
        return F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)

    def multi_scale_scales(self, expanded_scales: bool | None = None) -> list[int]:
        """The native multi-scale training sizes for THIS model's configuration.

        Reads ``resolution`` / ``patch_size`` / ``num_windows`` from the built
        model config (``expanded_scales`` defaults to the wrapper train
        config's setting) — hand the result to ``RandomMultiScaleResize`` so
        the transform draws from exactly the sizes the native dataloader would.
        """
        if expanded_scales is None:
            expanded_scales = bool(getattr(self._train_config, "expanded_scales", False))
        mc = self._model_config
        return compute_multi_scale_scales(
            int(mc.resolution),
            expanded_scales=bool(expanded_scales),
            patch_size=int(getattr(mc, "patch_size", 16)),
            num_windows=int(getattr(mc, "num_windows", 4)),
        )
