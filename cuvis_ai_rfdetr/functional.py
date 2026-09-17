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


def compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    """The native RF-DETR multi-scale training sizes for a given resolution.

    Faithful reimplementation of ``rfdetr.datasets.coco.compute_multi_scale_scales``
    (kept here so the transform module stays importable without the rfdetr train
    stack; equality against the native function is unit-tested). Sizes are
    multiples of ``patch_size * num_windows`` — the model's spatial divisibility
    unit — centred on ``resolution``: offsets ``[-3..4]`` (or ``[-5..5]`` with
    ``expanded_scales``) around ``resolution // (patch_size * num_windows)``,
    with a minimum of two units.
    """
    unit = patch_size * num_windows
    base = resolution // unit
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else list(range(-5, 6))
    proposed = [(base + off) * unit for off in offsets]
    return [scale for scale in proposed if scale >= unit * 2]


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


def jpeg_roundtrip(frame_u8: np.ndarray, quality: int = 95) -> np.ndarray:
    """Encode/decode one uint8 HWC RGB frame through an in-memory JPEG.

    Byte-equivalent to saving the frame as a ``.jpg`` with Pillow and reading
    it back (same encoder, same default 4:2:0 chroma subsampling), without
    touching the filesystem. Evaluation harnesses that persist model inputs as
    JPEG tiles make the (lossy) compression part of the score definition;
    routing inference through this function reproduces their scores exactly.

    Determinism note: the output depends on the Pillow version — pin Pillow
    when byte-exact reproduction across environments is required.
    """
    from io import BytesIO

    from PIL import Image

    if frame_u8.dtype != np.uint8 or frame_u8.ndim != 3 or frame_u8.shape[-1] != 3:
        raise ValueError(
            f"jpeg_roundtrip expects a uint8 HWC RGB frame, "
            f"got dtype={frame_u8.dtype}, shape={frame_u8.shape}."
        )
    buf = BytesIO()
    Image.fromarray(frame_u8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def top_frac_mean(score_map, top_frac: float = 0.001) -> float:
    """Image-level score = mean of the top ``top_frac`` fraction of pixels.

    Exact arithmetic of the tiled evaluation protocol: flatten, ascending
    sort, ``k = clamp(int((1 - top_frac) * N), 0, N - 1)``, mean of
    ``flat[k:]``. With the integer floor this selects e.g. exactly the top
    400 pixels of a 987x405 map at ``top_frac=0.001``. Robust to sparse maps:
    zeros participate, so a single confident blob scores far below its peak
    confidence unless it covers the whole top fraction.
    """
    a = (
        score_map.detach().to("cpu").numpy()
        if isinstance(score_map, Tensor)
        else np.asarray(score_map)
    )
    flat = np.sort(a.astype(np.float32, copy=False).ravel())
    if flat.size == 0:
        return 0.0
    k = min(max(int((1.0 - float(top_frac)) * flat.size), 0), flat.size - 1)
    return float(flat[k:].mean())


def resolve_band_indices(wavelengths: np.ndarray, bands_nm) -> list[int]:
    """Nearest-channel index for each requested wavelength (int64 distance).

    Mirrors the band-resolution rule of composite exporters:
    ``argmin(|wavelengths - nm|)`` per requested band, on integer-cast
    wavelengths.
    """
    wl = np.asarray(wavelengths).astype(np.int64).ravel()
    if wl.size == 0:
        raise ValueError("resolve_band_indices: empty wavelengths array.")
    return [int(np.argmin(np.abs(wl - float(nm)))) for nm in bands_nm]


def targets_from_mask(
    mask: Tensor,
    with_masks: bool = False,
    min_pixels: int = 1,
    multiclass: bool = False,
    label_offset: int = 1,
) -> list[dict[str, Tensor]]:
    """Build DETR-style targets from an integer instance/class mask.

    Parameters
    ----------
    mask : Tensor
        ``[B, H, W]`` integer mask, ``0`` = background, positive values are class ids.
    with_masks : bool
        Also emit per-instance boolean ``masks`` ``[N, H, W]`` (required by the
        segmentation criterion's mask losses).
    min_pixels : int
        Components smaller than this are ignored.
    multiclass : bool
        ``False`` (default): connected components of ``mask > 0`` become one target
        instance each, all with label ``0`` — the single-class (foreign-object) case.
        ``True``: for each distinct class value ``c > 0`` in the mask, connected
        components of ``mask == c`` become instances with label ``c - label_offset``.
    label_offset : int
        Subtracted from each class value to map mask class ids to 0-indexed model
        labels (default ``1``: COCO category ids 1/2/3 -> model classes 0/1/2, matching
        RF-DETR's roboflow loader). Only used when ``multiclass`` is True.

    Returns
    -------
    list[dict[str, Tensor]]
        One dict per batch item: ``boxes`` ``[N, 4]`` normalized cxcywh
        (resize-invariant), ``labels`` ``[N]`` int64, and optionally
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
        m = m_np[b]
        # (binary_selector, class_label) pairs whose connected components become instances
        if multiclass:
            groups = [(m == c, int(c) - label_offset) for c in np.unique(m) if c > 0]
        else:
            groups = [(m > 0, 0)]
        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        for selector, class_label in groups:
            lbl, n = ndimage.label(selector)
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
                labels.append(class_label)
                if with_masks:
                    masks.append(comp)
        target: dict[str, Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64, device=device),
        }
        if with_masks:
            target["masks"] = (
                torch.as_tensor(np.stack(masks)).to(device=device, dtype=torch.bool)
                if masks
                else torch.zeros((0, height, width), dtype=torch.bool, device=device)
            )
        out.append(target)
    return out
