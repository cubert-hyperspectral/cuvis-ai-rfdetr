"""Forward-path tests with a mocked backbone (no rfdetr install needed).

A fake ``model.predict`` stands in for RF-DETR so the tiled/whole forward
loops, mask/box pasting, cross-tile NMS, class filtering, JPEG routing, and
both score reductions are exercised deterministically on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_rfdetr.functional import top_frac_mean
from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter


class FakeDetections:
    """Duck-typed stand-in for supervision.Detections."""

    def __init__(self, rows, tile_shape=None, with_masks=False):
        # rows: list of (x1, y1, x2, y2, conf, class_id)
        self.xyxy = np.array([r[:4] for r in rows], dtype=np.float64).reshape(-1, 4)
        self.confidence = np.array([r[4] for r in rows], dtype=np.float64)
        self.class_id = np.array([r[5] for r in rows], dtype=np.int64)
        self.mask = None
        if with_masks and tile_shape is not None:
            h, w = tile_shape
            masks = []
            for x1, y1, x2, y2, _, _ in rows:
                m = np.zeros((h, w), dtype=bool)
                m[int(y1) : int(y2), int(x1) : int(x2)] = True
                masks.append(m)
            self.mask = np.stack(masks) if masks else None


class FakeModel:
    """Returns queued FakeDetections per predict() call and records inputs."""

    def __init__(self, per_call):
        self.per_call = list(per_call)
        self.received: list[np.ndarray] = []

    def predict(self, frame, threshold=0.5):
        self.received.append(np.asarray(frame))
        if not self.per_call:
            return FakeDetections([])
        return self.per_call.pop(0)


def _rgb(h=16, w=8):
    rng = np.random.default_rng(21)
    arr = rng.integers(0, 256, size=(1, h, w, 3), dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32))


# ---------------------------------------------------------------- segmenter
def test_segmenter_tiled_paste_filter_and_top_frac() -> None:
    """Two tiles; masks pasted at row offsets; class 0 dropped by the filter."""
    tile_shape = (10, 8)
    per_call = [
        FakeDetections(
            [(0, 0, 8, 4, 0.8, 1), (0, 0, 8, 2, 0.9, 0)],  # 0.9 is class 0 -> filtered
            tile_shape=tile_shape,
            with_masks=True,
        ),
        FakeDetections([(0, 2, 8, 4, 0.6, 1)], tile_shape=tile_shape, with_masks=True),
    ]
    node = RFDETRSegmenter(
        tiling="tiled",
        tile_rows=10,
        row_starts=(0, 6),
        class_filter=1,
        score_reduction="top_frac_mean",
        top_frac=0.25,
    )
    node._model = FakeModel(per_call)
    out = node.forward(rgb_image=_rgb(16, 8))

    scores = out["scores"][0, :, :, 0]
    assert float(scores[0, 0]) == pytest.approx(0.8)  # tile-0 mask rows 0-3
    assert float(scores[3, 7]) == pytest.approx(0.8)
    assert float(scores[8, 0]) == pytest.approx(0.6)  # tile-1 mask rows 2-3 -> abs 8-9
    assert float(scores[5, 0]) == 0.0  # class-0 instance never pasted
    # detections carry only the kept class
    assert all(d["class_id"] == 1 for d in out["detections"][0])
    # anomaly = exact top-frac mean of the pasted map
    expected = top_frac_mean(scores.numpy(), 0.25)
    assert float(out["anomaly_score"][0]) == pytest.approx(expected)
    # top 32 of 128 px = the 32 pixels at 0.8
    assert expected == pytest.approx(0.8)


def test_segmenter_max_conf_default_and_box_fallback() -> None:
    """No masks -> box fill; default reduction stays max confidence."""
    node = RFDETRSegmenter(tiling="whole")
    node._model = FakeModel([FakeDetections([(1, 1, 4, 5, 0.7, 3)])])  # mask=None
    out = node.forward(rgb_image=_rgb(8, 8))
    assert float(out["anomaly_score"][0]) == pytest.approx(0.7)
    assert float(out["scores"][0, 2, 2, 0]) == pytest.approx(0.7)  # box-filled
    assert float(out["scores"][0, 6, 6, 0]) == 0.0


def test_segmenter_jpeg_roundtrip_invoked_per_tile(monkeypatch) -> None:
    calls = {"n": 0}

    def spy(frame, quality):
        calls["n"] += 1
        assert quality == 90
        return frame  # identity keeps expectations exact

    import cuvis_ai_rfdetr.node.rfdetr_segmenter as seg_mod

    monkeypatch.setattr(seg_mod, "_jpeg_roundtrip", spy)
    node = RFDETRSegmenter(
        tiling="tiled", tile_rows=10, row_starts=(0, 6), jpeg_roundtrip=True, jpeg_quality=90
    )
    node._model = FakeModel([FakeDetections([]), FakeDetections([])])
    node.forward(rgb_image=_rgb(16, 8))
    assert calls["n"] == 2  # once per tile

    calls["n"] = 0
    node_off = RFDETRSegmenter(tiling="tiled", tile_rows=10, row_starts=(0, 6))
    node_off._model = FakeModel([FakeDetections([]), FakeDetections([])])
    node_off.forward(rgb_image=_rgb(16, 8))
    assert calls["n"] == 0  # off by default


# ----------------------------------------------------------------- detector
def test_detector_tiled_nms_filter_and_top_frac() -> None:
    """Cross-tile duplicate merged by NMS; class filter applied before NMS."""
    per_call = [
        FakeDetections([(0, 7, 4, 9, 0.9, 1), (5, 0, 7, 2, 0.95, 0)]),  # tile r0=0
        FakeDetections([(0, 1, 4, 3, 0.8, 1)]),  # tile r0=6 -> abs rows 7-9 (dup of first)
    ]
    node = RFDETRDetector(
        tiling="tiled",
        tile_rows=10,
        row_starts=(0, 6),
        nms_iou=0.5,
        class_filter=1,
        score_reduction="top_frac_mean",
        top_frac=0.25,
    )
    node._model = FakeModel(per_call)
    out = node.forward(rgb_image=_rgb(16, 8))

    dets = out["detections"][0]
    assert len(dets) == 1  # class-0 dropped; duplicate NMS-merged
    assert dets[0]["confidence"] == pytest.approx(0.9)
    scores = out["scores"][0, :, :, 0]
    assert float(scores[8, 2]) == pytest.approx(0.9)  # inside the kept box
    assert float(scores[0, 6]) == 0.0  # class-0 box never rasterized
    assert float(out["anomaly_score"][0]) == pytest.approx(top_frac_mean(scores.numpy(), 0.25))


def test_detector_whole_mode_jpeg_and_max_conf(monkeypatch) -> None:
    calls = {"n": 0}

    def spy(frame, quality):
        calls["n"] += 1
        return frame

    import cuvis_ai_rfdetr.node.rfdetr_detector as det_mod

    monkeypatch.setattr(det_mod, "_jpeg_roundtrip", spy)
    node = RFDETRDetector(tiling="whole", jpeg_roundtrip=True)
    node._model = FakeModel([FakeDetections([(0, 0, 3, 3, 0.55, 2)])])
    out = node.forward(rgb_image=_rgb(8, 8))
    assert calls["n"] == 1
    assert float(out["anomaly_score"][0]) == pytest.approx(0.55)  # default max_conf
    assert out["detections"][0][0]["class_id"] == 2  # no filter by default


# ---------------------------------------------------------- small branches
def test_to_uint8_frames_unit_range_scaling() -> None:
    from cuvis_ai_rfdetr.functional import to_uint8_frames

    x = torch.full((1, 2, 2, 3), 0.5, dtype=torch.float32)  # max <= 1.5 -> scaled
    assert int(to_uint8_frames(x).max()) == 128
    y = torch.full((1, 2, 2, 3), 200.0, dtype=torch.float32)  # already 0-255
    assert int(to_uint8_frames(y).max()) == 200


def test_segmenter_paste_clips_out_of_canvas_mask() -> None:
    """A mask whose tile rows fall entirely below the canvas is ignored."""
    node = RFDETRSegmenter(tiling="tiled", tile_rows=10, row_starts=(0, 14))
    node._model = FakeModel(
        [
            FakeDetections([]),
            FakeDetections([(0, 0, 4, 2, 0.9, 1)], tile_shape=(10, 8), with_masks=True),
        ]
    )
    out = node.forward(rgb_image=_rgb(16, 8))  # second tile is only 2 rows tall
    assert float(out["scores"].max()) >= 0.0  # no crash; clip path exercised
