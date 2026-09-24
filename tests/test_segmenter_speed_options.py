"""Speed options of RFDETRSegmenter with a mocked model (no rfdetr install needed).

``fast_paste`` must reproduce the per-instance paste bit for bit (overlaps, row offsets, clipping,
box fallback, class filter); ``gpu_input`` must hand rfdetr the uint8 frame's exact values as a
channel-first float tensor; ``precision`` / ``jit_trace`` must reach rfdetr's ``inference`` call.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_rfdetr.functional import max_paste_masks, to_uint8_frames, to_unit_frames
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

CUDA = torch.cuda.is_available()


class FakeDetections:
    """supervision.Detections stand-in carrying explicit masks (or none)."""

    def __init__(self, rows, masks=None):
        self.xyxy = np.array([r[:4] for r in rows], dtype=np.float64).reshape(-1, 4)
        self.confidence = np.array([r[4] for r in rows], dtype=np.float64)
        self.class_id = np.array([r[5] for r in rows], dtype=np.int64)
        self.mask = masks


class FakeModel:
    """Replays queued detections; records the inputs and kwargs of every predict() call."""

    def __init__(self, per_call, accepts_source_flag=False):
        self.per_call = list(per_call)
        self.received: list = []
        self.kwargs: list[dict] = []
        if accepts_source_flag:
            self.predict = self._predict_with_flag

    def predict(self, frame, threshold=0.5):
        return self._record(frame, {"threshold": threshold})

    def _predict_with_flag(self, frame, threshold=0.5, include_source_image=True):
        return self._record(
            frame, {"threshold": threshold, "include_source_image": include_source_image}
        )

    def _record(self, frame, kwargs):
        self.received.append(frame)
        self.kwargs.append(kwargs)
        return self.per_call.pop(0) if self.per_call else FakeDetections([])


def _random_detections(rng, n, shape, n_classes=2):
    """``n`` overlapping instances with random masks, boxes and confidences."""
    h, w = shape
    rows, masks = [], []
    for _ in range(n):
        x1, y1 = rng.integers(0, w - 1), rng.integers(0, h - 1)
        x2, y2 = rng.integers(x1 + 1, w + 1), rng.integers(y1 + 1, h + 1)
        rows.append((x1, y1, x2, y2, float(rng.uniform(0.05, 1.0)), int(rng.integers(n_classes))))
        masks.append(rng.random((h, w)) < 0.4)
    return FakeDetections(rows, np.stack(masks) if masks else None)


def _run(node, per_call, rgb):
    node._model = FakeModel(per_call)
    return node.forward(rgb_image=rgb)


def _rgb(h, w, seed=3):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((1, h, w, 3)).astype(np.float32))


# ------------------------------------------------------------------ fast paste
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_paste_matches_per_instance_paste_whole(seed) -> None:
    rng = np.random.default_rng(seed)
    per_call = [_random_detections(rng, 25, (12, 9))]  # > one paste chunk of 16
    rgb = _rgb(12, 9)
    fast = _run(RFDETRSegmenter(tiling="whole", class_filter=1), list(per_call), rgb)
    slow = _run(
        RFDETRSegmenter(tiling="whole", class_filter=1, fast_paste=False), list(per_call), rgb
    )
    assert torch.equal(fast["scores"], slow["scores"])
    assert fast["detections"] == slow["detections"]
    assert torch.equal(fast["anomaly_score"], slow["anomaly_score"])
    assert float(fast["scores"].max()) > 0.0  # something was pasted


def test_fast_paste_matches_per_instance_paste_tiled_with_clipping_and_boxes() -> None:
    rng = np.random.default_rng(7)
    tile = (10, 8)
    boxes_only = FakeDetections([(1, 1, 5, 4, 0.66, 0), (2, 0, 9, 3, 0.41, 1)])  # mask=None
    per_call = [
        _random_detections(rng, 6, tile),
        boxes_only,
        _random_detections(rng, 4, tile),  # tile at row 14 of 16 -> masks clipped to 2 rows
    ]
    kw = {
        "tiling": "tiled",
        "tile_rows": 10,
        "row_starts": (0, 5, 14),
        "score_reduction": "top_frac_mean",
    }
    rgb = _rgb(16, 8)
    fast = _run(RFDETRSegmenter(**kw), list(per_call), rgb)
    slow = _run(RFDETRSegmenter(fast_paste=False, **kw), list(per_call), rgb)
    assert torch.equal(fast["scores"], slow["scores"])
    assert fast["detections"] == slow["detections"]
    assert torch.equal(fast["anomaly_score"], slow["anomaly_score"])


def test_max_paste_masks_golden_and_negative_canvas() -> None:
    """Hand-built overlap: each pixel keeps the max confidence of the masks covering it."""
    canvas = torch.full((3, 4), -1.0)  # negative background: untouched pixels must stay -1
    m1 = np.zeros((2, 4), dtype=bool)
    m1[0, :2] = True
    m2 = np.zeros((2, 4), dtype=bool)
    m2[0, 1:3] = True
    max_paste_masks(canvas, [m1, m2], [0.3, 0.7], row_offset=1)
    expected = torch.full((3, 4), -1.0)
    expected[1, 0] = 0.3
    expected[1, 1:3] = 0.7
    assert torch.equal(canvas, expected)
    max_paste_masks(canvas, [], [], row_offset=0)  # no instances: no-op
    assert torch.equal(canvas, expected)
    max_paste_masks(canvas, [m1], [0.9], row_offset=3)  # entirely below the canvas
    assert torch.equal(canvas, expected)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fast_paste_builds_scores_on_the_input_device() -> None:
    rng = np.random.default_rng(11)
    per_call = [_random_detections(rng, 5, (12, 9))]
    rgb = _rgb(12, 9)
    out = _run(RFDETRSegmenter(tiling="whole"), list(per_call), rgb.cuda())
    ref = _run(RFDETRSegmenter(tiling="whole", fast_paste=False), list(per_call), rgb)
    assert out["scores"].device.type == "cuda"
    assert torch.equal(out["scores"].cpu(), ref["scores"])


# ------------------------------------------------------------------ gpu input
def test_to_unit_frames_equals_uint8_frames_over_255() -> None:
    levels = torch.arange(256, dtype=torch.float32).repeat(3).view(1, 16, 16, 3)
    for x in (levels, levels / 255.0, levels + 0.4, levels * 1.7 - 20.0):
        ref = torch.from_numpy(to_uint8_frames(x)).float() / torch.tensor(255.0)
        assert torch.equal(to_unit_frames(x), ref)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_to_unit_frames_is_bit_identical_on_cuda() -> None:
    """Every byte value divides the same on CUDA as on the host (tensor divisor)."""
    levels = torch.arange(256, dtype=torch.float32).repeat(3).view(1, 16, 16, 3)
    assert torch.equal(to_unit_frames(levels.cuda()).cpu(), to_unit_frames(levels))


def test_gpu_input_hands_rfdetr_the_uint8_values_channel_first() -> None:
    rgb = _rgb(12, 9)
    node = RFDETRSegmenter(tiling="tiled", tile_rows=8, row_starts=(0, 4), gpu_input=True)
    model = FakeModel([FakeDetections([]), FakeDetections([])], accepts_source_flag=True)
    node._model = model
    node.forward(rgb_image=rgb)
    u8 = to_uint8_frames(rgb)[0]
    for received, r0 in zip(model.received, (0, 4), strict=True):
        assert isinstance(received, torch.Tensor)
        assert received.shape == (3, 8, 9)  # CHW tile
        assert received.is_contiguous()
        back = (received * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        assert np.array_equal(back, u8[r0 : r0 + 8])
    assert all(kw["include_source_image"] is False for kw in model.kwargs)


def test_source_image_flag_only_passed_when_supported() -> None:
    node = RFDETRSegmenter(tiling="whole")
    legacy = FakeModel([FakeDetections([])])
    node._model = legacy
    node.forward(rgb_image=_rgb(8, 8))
    assert legacy.kwargs == [{"threshold": 0.5}]
    node._model = modern = FakeModel([FakeDetections([])], accepts_source_flag=True)
    node.forward(rgb_image=_rgb(8, 8))
    assert modern.kwargs == [{"threshold": 0.5, "include_source_image": False}]


# ------------------------------------------------------------ precision / jit
class FakeRFDETR:
    def __init__(self, with_inference=True):
        self.calls: list[dict] = []
        if with_inference:
            self.inference = self._record
        else:
            self.optimize_for_inference = self._record

    def _record(self, compile=True, batch_size=1, dtype=torch.float32):
        self.calls.append({"compile": compile, "batch_size": batch_size, "dtype": dtype})


@pytest.mark.parametrize(
    "precision, jit, expected",
    [
        ("fp16", False, {"compile": False, "batch_size": 1, "dtype": torch.float16}),
        ("bf16", True, {"compile": True, "batch_size": 1, "dtype": torch.bfloat16}),
        ("fp32", True, {"compile": True, "batch_size": 1, "dtype": torch.float32}),
    ],
)
def test_precision_and_jit_reach_rfdetr_inference(precision, jit, expected) -> None:
    node = RFDETRSegmenter(precision=precision, jit_trace=jit)
    model = FakeRFDETR()
    assert node._apply_inference_options(model) is model
    assert model.calls == [expected]
    legacy = FakeRFDETR(with_inference=False)  # releases before the inference() rename
    node._apply_inference_options(legacy)
    assert legacy.calls == [expected]


def test_default_leaves_the_network_untouched() -> None:
    model = FakeRFDETR()
    RFDETRSegmenter()._apply_inference_options(model)
    assert model.calls == []


def test_ensure_model_applies_the_options_once(monkeypatch) -> None:
    node = RFDETRSegmenter(precision="fp16", jit_trace=True)
    model = FakeRFDETR()
    monkeypatch.setattr(node, "_build_model", lambda: model)
    assert node._ensure_model() is model
    assert node._ensure_model() is model
    assert len(model.calls) == 1


def test_speed_hparams_defaults_round_trip_and_validation() -> None:
    node = RFDETRSegmenter()
    assert (node.fast_paste, node.precision, node.jit_trace, node.gpu_input) == (
        True,
        "fp32",
        False,
        False,
    )
    node = RFDETRSegmenter(fast_paste=False, precision="FP16", jit_trace=True, gpu_input=True)
    assert node.hparams["fast_paste"] is False
    assert node.hparams["precision"] == "fp16"
    assert node.hparams["jit_trace"] is True
    assert node.hparams["gpu_input"] is True
    with pytest.raises(ValueError, match="precision"):
        RFDETRSegmenter(precision="int8")
    with pytest.raises(ValueError, match="gpu_input"):
        RFDETRSegmenter(gpu_input=True, jpeg_roundtrip=True)
    with pytest.raises(RuntimeError, match="inference"):
        RFDETRSegmenter(precision="fp16")._apply_inference_options(object())
