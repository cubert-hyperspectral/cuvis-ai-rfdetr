"""Shared functional helpers for the cuvis-ai-rfdetr nodes.

Pure functions used by the inference nodes (tile merging) and the trainable
node / criterion loss (DETR target construction). Kept import-light: ``rfdetr``
is never imported here; ``scipy`` is imported lazily inside
:func:`targets_from_mask`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def box_iou(a, b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes (plain floats, no torch)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms_rows(rows: list, iou_thr: float) -> list:
    """Greedy NMS by descending confidence over ``(x1, y1, x2, y2, conf, ...)`` rows."""
    keep: list = []
    for r in sorted(rows, key=lambda t: -t[4]):
        if all(box_iou(r[:4], k[:4]) < iou_thr for k in keep):
            keep.append(r)
    return keep


def to_uint8_frames(rgb_image: Tensor) -> np.ndarray:
    """``[B, H, W, 3]`` float input -> uint8 HWC numpy batch.

    Auto-detects the value range: inputs whose maximum is <= 1.5 are treated
    as 0-1 and scaled by 255; everything else is assumed to already be 0-255.
    """
    x = rgb_image.detach().to(device="cpu", dtype=torch.float32)
    if x.numel() > 0 and float(x.max()) <= 1.5:
        x = x * 255.0
    return x.clamp(0.0, 255.0).round().to(torch.uint8).contiguous().numpy()


def targets_from_mask(
    mask: Tensor,
    with_masks: bool = False,
    min_pixels: int = 1,
) -> list[dict[str, Tensor]]:
    """Build DETR-style targets from an integer instance/class mask.

    Parameters
    ----------
    mask : Tensor
        ``[B, H, W]`` integer mask, ``0`` = background. Connected components of
        ``mask > 0`` become one target instance each (single foreign-object
        class, label ``0``).
    with_masks : bool
        Also emit per-instance boolean ``masks`` ``[N, H, W]`` (required by the
        segmentation criterion's mask losses).
    min_pixels : int
        Components smaller than this are ignored.

    Returns
    -------
    list[dict[str, Tensor]]
        One dict per batch item: ``boxes`` ``[N, 4]`` normalized cxcywh
        (resize-invariant), ``labels`` ``[N]`` int64 zeros, and optionally
        ``masks`` ``[N, H, W]`` bool. Tensors live on ``mask``'s device.
    """
    from scipy import ndimage  # heavy import kept out of module import time

    device = mask.device
    m_np = mask.detach().to("cpu").numpy()
    if m_np.ndim != 3:
        raise ValueError(f"targets_from_mask expects [B, H, W], got {m_np.shape}.")
    _, height, width = m_np.shape
    out: list[dict[str, Tensor]] = []
    for b in range(m_np.shape[0]):
        lbl, n = ndimage.label(m_np[b] > 0)
        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        for i in range(1, n + 1):
            comp = lbl == i
            if int(comp.sum()) < min_pixels:
                continue
            ys, xs = np.nonzero(comp)
            x0, x1 = float(xs.min()), float(xs.max()) + 1.0
            y0, y1 = float(ys.min()), float(ys.max()) + 1.0
            boxes.append(
                [
                    (x0 + x1) / 2.0 / width,
                    (y0 + y1) / 2.0 / height,
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ]
            )
            if with_masks:
                masks.append(comp)
        target: dict[str, Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "labels": torch.zeros(len(boxes), dtype=torch.int64, device=device),
        }
        if with_masks:
            target["masks"] = (
                torch.as_tensor(np.stack(masks)).to(device=device, dtype=torch.bool)
                if masks
                else torch.zeros((0, height, width), dtype=torch.bool, device=device)
            )
        out.append(target)
    return out
