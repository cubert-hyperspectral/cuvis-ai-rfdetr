"""RF-DETR's own data pipeline, run inside :class:`RFDETRTrainable`.

RF-DETR's trainer reads each image and its annotations, builds a target in its
dataset format, runs its train or val transforms on the pair and pads the batch to the
backbone's block size. The functions here reproduce that per batch inside the training
node, calling RF-DETR's own transform builder, so a cuvis-ai training run feeds the
model exactly what RF-DETR's data loader would for the same images.

``rfdetr`` is imported lazily inside the functions; importing this module is cheap.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cuvis_ai_rfdetr.functional import native_targets_from_mask

#: Named augmentation configs for ``aug_config``. ``"rfdetr_1_8"`` is the default RF-DETR
#: used before 1.9 (``rfdetr.datasets.aug_configs.AUG_CONFIG``: horizontal flip), which runs
#: through Albumentations (cv2 bilinear resize, no antialias). RF-DETR 1.9 moved its default
#: to torchvision (bilinear + antialias); passing this preset restores the 1.8 behaviour and
#: needs Albumentations (the ``rfdetr[augment]`` extra).
AUG_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "rfdetr_1_8": {"HorizontalFlip": {"p": 0.5}},
}


def resolve_aug_config(aug_config: Any) -> dict[str, Any] | None:
    """``None`` (RF-DETR's installed default), a preset name, or an explicit config dict."""
    if aug_config is None:
        return None
    if isinstance(aug_config, str):
        key = aug_config.lower()
        if key not in AUG_PRESETS:
            raise ValueError(
                f"aug_config: unknown preset {aug_config!r}; use one of {sorted(AUG_PRESETS)}, "
                "a config dict, or None for RF-DETR's default."
            )
        return copy.deepcopy(AUG_PRESETS[key])
    if isinstance(aug_config, dict):
        return copy.deepcopy(aug_config)
    raise TypeError(
        f"aug_config must be None, a preset name or a dict, got {type(aug_config).__name__}."
    )


def build_transforms(image_set: str, model_config: Any, train_config: Any, aug_config: Any = None):
    """RF-DETR's own transform pipeline for ``"train"`` or ``"val"``.

    Mirrors the arguments RF-DETR's Roboflow-COCO dataset builder passes, read from the
    model and train configs: square resize divisible by 64 (or not), multi-scale and
    expanded scales, ``skip_random_resize = not do_random_resize_via_padding`` (so with the
    default configs training uses only the largest multi-scale size), patch size and window
    count, scale jitter, CPU augmentation. Only keyword arguments the installed RF-DETR
    accepts are passed, so 1.8 and 1.9 both work.
    """
    from rfdetr.datasets import coco as coco_datasets

    square = bool(getattr(train_config, "square_resize_div_64", False))
    builder = (
        coco_datasets.make_coco_transforms_square_div_64
        if square
        else coco_datasets.make_coco_transforms
    )
    wanted = {
        "resolution": int(model_config.resolution),
        "multi_scale": bool(getattr(train_config, "multi_scale", False)),
        "expanded_scales": bool(getattr(train_config, "expanded_scales", False)),
        "skip_random_resize": not bool(
            getattr(train_config, "do_random_resize_via_padding", False)
        ),
        "patch_size": int(getattr(model_config, "patch_size", 16)),
        "num_windows": int(getattr(model_config, "num_windows", 4)),
        "aug_config": resolve_aug_config(aug_config),
        "scale_jitter": bool(getattr(train_config, "scale_jitter", True)),
        "gpu_postprocess": False,
    }
    accepted = inspect.signature(builder).parameters
    return builder(image_set, **{k: v for k, v in wanted.items() if k in accepted})


def native_batch(
    frames_u8: np.ndarray,
    masks: np.ndarray | None,
    transform: Any,
    *,
    block_size: int | None,
    target_kwargs: dict[str, Any],
) -> tuple[Any, list[dict[str, Tensor]]]:
    """Run RF-DETR's transforms per image and pad the batch like RF-DETR's collate.

    ``frames_u8`` is ``[B, H, W, 3]`` uint8, ``masks`` ``[B, H, W]`` integer instance/class
    masks (``None`` at inference: empty targets). Returns RF-DETR's ``NestedTensor`` of
    normalized images and the transformed targets (normalized cxcywh boxes, resized masks,
    ``orig_size`` / ``size``), both on the CPU.
    """
    from PIL import Image
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    images: list[Tensor] = []
    targets: list[dict[str, Tensor]] = []
    for index, frame in enumerate(frames_u8):
        height, width = int(frame.shape[0]), int(frame.shape[1])
        mask = masks[index] if masks is not None else np.zeros((height, width), dtype=np.int32)
        target = native_targets_from_mask(mask, image_id=index, **target_kwargs)
        image, target = transform(Image.fromarray(frame), target)
        images.append(image)
        targets.append(target)
    if (
        block_size is not None
        and "block_size" in inspect.signature(nested_tensor_from_tensor_list).parameters
    ):
        samples = nested_tensor_from_tensor_list(images, block_size=int(block_size))
    else:
        samples = nested_tensor_from_tensor_list(images)
    return samples, targets


def targets_to(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    """Move every tensor of every target dict to ``device``."""
    return [
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()}
        for target in targets
    ]
