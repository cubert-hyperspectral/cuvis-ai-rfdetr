"""TensorRT backend of RFDETRSegmenter and the engine helpers (mocked; no TensorRT or rfdetr model needed).

The TensorRT path must hand the engine exactly rfdetr's preprocessed frame (uint8 / 255, bilinear resize without
antialias, mean / std), hand rfdetr's post-processing the engine outputs under rfdetr's names, and feed the same rows
into the node's paste / merge as the PyTorch path, so ``scores`` / ``detections`` / ``anomaly_score`` are identical
for identical instances. The ``slow`` test builds and runs a real engine where TensorRT is installed.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as tvf

from cuvis_ai_rfdetr import trt_engine
from cuvis_ai_rfdetr.functional import to_uint8_frames
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

CUDA = torch.cuda.is_available()
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FakeDetections:
    """supervision.Detections stand-in (what rfdetr's predict returns)."""

    def __init__(self, rows, masks):
        self.xyxy = np.array([r[:4] for r in rows], dtype=np.float64).reshape(-1, 4)
        self.confidence = np.array([r[4] for r in rows], dtype=np.float64)
        self.class_id = np.array([r[5] for r in rows], dtype=np.int64)
        self.mask = masks


class FakeTorchModel:
    """rfdetr model for backend='torch': predict() replays queued detections."""

    def __init__(self, per_call):
        self.per_call = list(per_call)

    def predict(self, frame, threshold=0.5):
        return self.per_call.pop(0)


class FakePostprocess:
    """rfdetr's ``model.model.postprocess``: records its arguments, replays queued results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, predictions, target_sizes, score_threshold=None):
        self.calls.append((predictions, target_sizes, score_threshold))
        return [self.results.pop(0)]


class FakeTRTModel:
    """rfdetr model object as the TensorRT path uses it: resolution, device, postprocess, means / stds."""

    def __init__(self, results, resolution=8, device="cpu"):
        self.model = SimpleNamespace(
            resolution=resolution,
            device=torch.device(device),
            postprocess=FakePostprocess(results),
        )
        self.means, self.stds = MEANS, STDS


class FakeEngine:
    """TensorRTEngine stand-in: records the input batch, returns fixed output buffers."""

    def __init__(self, device="cpu", input_shape=(1, 3, 8, 8)):
        self.device = torch.device(device)
        self.input_shape = input_shape
        self.inputs: list[torch.Tensor] = []
        self.outputs = {
            "dets": torch.rand(1, 5, 4),
            "labels": torch.rand(1, 5, 3),
            "masks": torch.rand(1, 5, 4, 4),
        }

    def __call__(self, x):
        self.inputs.append(x.clone())
        return self.outputs


def _f32(x) -> float:
    return float(np.float32(x))


def _instances(rng, n, shape, threshold=0.5, below=1):
    """``n`` instances above ``threshold`` (+ ``below`` under it) as (predict detections, postprocess result).

    Values are float32-exact so the float32 TensorRT rows and the float64 predict rows compare equal.
    """
    h, w = shape
    rows, masks = [], []
    for k in range(n + below):
        x1, y1 = int(rng.integers(0, w - 1)), int(rng.integers(0, h - 1))
        x2, y2 = int(rng.integers(x1 + 1, w + 1)), int(rng.integers(y1 + 1, h + 1))
        conf = _f32(rng.uniform(threshold + 0.01, 1.0) if k < n else rng.uniform(0.0, threshold))
        rows.append((_f32(x1), _f32(y1), _f32(x2), _f32(y2), conf, int(rng.integers(2))))
        masks.append(rng.random((h, w)) < 0.4)
    masks = np.stack(masks)
    kept = [i for i, r in enumerate(rows) if r[4] > threshold]
    detections = FakeDetections([rows[i] for i in kept], masks[kept])
    result = {
        "scores": torch.tensor([r[4] for r in rows], dtype=torch.float32),
        "labels": torch.tensor([r[5] for r in rows], dtype=torch.int64),
        "boxes": torch.tensor([r[:4] for r in rows], dtype=torch.float32),
        "masks": torch.from_numpy(masks)[:, None],  # [N, 1, H, W] bool, like rfdetr
    }
    return detections, result


def _rgb(h, w, seed=3):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((1, h, w, 3)).astype(np.float32))


def _trt_node(results, engine=None, **kw):
    node = RFDETRSegmenter(backend="tensorrt", **kw)
    node._model = FakeTRTModel(results)
    node._engine = engine or FakeEngine()
    return node


# ------------------------------------------------------------------ forward
def test_trt_engine_input_is_rfdetrs_preprocessing() -> None:
    rgb = _rgb(12, 9)
    node = _trt_node([_instances(np.random.default_rng(0), 2, (12, 9))[1]], tiling="whole")
    node.forward(rgb_image=rgb)
    u8 = to_uint8_frames(rgb)[0]
    # rfdetr: to_tensor (uint8 / 255), bilinear resize without antialias, normalize.
    golden = tvf.normalize(
        tvf.resize(tvf.to_tensor(u8), [8, 8], antialias=False)[None], MEANS, STDS
    )
    (fed,) = node._engine.inputs
    assert fed.shape == (1, 3, 8, 8)
    assert torch.equal(fed, golden)
    predictions, target_sizes, threshold = node._model.model.postprocess.calls[0]
    engine_out = node._engine.outputs
    assert predictions["pred_boxes"] is engine_out["dets"]
    assert predictions["pred_logits"] is engine_out["labels"]
    assert predictions["pred_masks"] is engine_out["masks"]
    assert target_sizes.tolist() == [[12, 9]]
    assert threshold == node.threshold


def test_trt_float_and_uint8_frames_feed_the_same_input() -> None:
    rgb = _rgb(10, 7, seed=5)
    rng = np.random.default_rng(1)
    results = [_instances(rng, 1, (10, 7))[1] for _ in range(2)]
    on_host = _trt_node([results[0]], tiling="whole")
    on_device = _trt_node([results[1]], tiling="whole", gpu_input=True)
    on_host.forward(rgb_image=rgb)
    on_device.forward(rgb_image=rgb)
    assert torch.equal(on_host._engine.inputs[0], on_device._engine.inputs[0])


@pytest.mark.parametrize("tiling", ["whole", "tiled"])
@pytest.mark.parametrize("fast_paste", [True, False])
def test_trt_forward_equals_torch_forward_for_the_same_instances(tiling, fast_paste) -> None:
    rng = np.random.default_rng(7)
    kw = {"tiling": tiling, "tile_rows": 10, "row_starts": (0, 5, 14), "class_filter": 1}
    kw["fast_paste"] = fast_paste
    tiles = [(16, 8)] if tiling == "whole" else [(10, 8), (10, 8), (2, 8)]
    pairs = [_instances(rng, 6, shape) for shape in tiles]
    rgb = _rgb(16, 8)
    torch_node = RFDETRSegmenter(**kw)
    torch_node._model = FakeTorchModel([d for d, _ in pairs])
    ref = torch_node.forward(rgb_image=rgb)
    out = _trt_node([r for _, r in pairs], **kw).forward(rgb_image=rgb)
    assert torch.equal(out["scores"], ref["scores"])
    assert out["detections"] == ref["detections"]
    assert torch.equal(out["anomaly_score"], ref["anomaly_score"])
    assert float(ref["scores"].max()) > 0.0  # something was pasted


def test_trt_rows_drop_scores_at_or_below_the_threshold() -> None:
    result = {
        "scores": torch.tensor([0.9, 0.5, 0.2]),
        "labels": torch.tensor([0, 1, 0]),
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [0.0, 1.0, 1.0, 2.0]]),
        "masks": torch.ones(3, 1, 4, 4, dtype=torch.bool),
    }
    node = _trt_node([result], tiling="whole")
    rows = node._predict_frame(node._model, np.zeros((4, 4, 3), dtype=np.uint8))
    assert [r[:6] for r in rows] == [(0.0, 0.0, 2.0, 2.0, pytest.approx(0.9), 0)]
    assert rows[0][6].shape == (4, 4) and rows[0][6].dtype == torch.bool


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("fast_paste", [True, False])
def test_trt_forward_with_masks_on_cuda(fast_paste) -> None:
    rng = np.random.default_rng(9)
    detections, result = _instances(rng, 5, (12, 9))
    cuda_result = {k: v.cuda() for k, v in result.items()}
    rgb = _rgb(12, 9)
    ref_node = RFDETRSegmenter(tiling="whole")
    ref_node._model = FakeTorchModel([detections])
    ref = ref_node.forward(rgb_image=rgb)
    node = _trt_node([cuda_result], FakeEngine("cuda"), tiling="whole", fast_paste=fast_paste)
    out = node.forward(rgb_image=rgb.cuda())
    assert out["scores"].device.type == "cuda"
    assert torch.equal(out["scores"].cpu(), ref["scores"])
    assert out["detections"] == ref["detections"]


# ------------------------------------------------------------------ hparams / engine loading
def test_backend_hparams_defaults_round_trip_and_validation() -> None:
    node = RFDETRSegmenter()
    assert (node.backend, node.engine_dir) == ("torch", None)
    node = RFDETRSegmenter(backend="TensorRT", precision="fp16", engine_dir="/e")
    assert node.hparams["backend"] == "tensorrt"
    assert node.hparams["precision"] == "fp16"
    assert node.hparams["engine_dir"] == "/e"
    with pytest.raises(ValueError, match="backend"):
        RFDETRSegmenter(backend="onnx")
    with pytest.raises(ValueError, match="precision"):
        RFDETRSegmenter(backend="tensorrt", precision="bf16")
    with pytest.raises(ValueError, match="jit_trace"):
        RFDETRSegmenter(backend="tensorrt", jit_trace=True)


class _LoadedEngine:
    def __init__(self, path, device, input_shape=(1, 3, 8, 8)):
        self.path, self.device, self.input_shape = path, device, input_shape


def _patch_engine_lookup(monkeypatch, engine_cls=_LoadedEngine):
    monkeypatch.setattr(
        trt_engine,
        "engine_file_name",
        lambda precision, res, device=None: f"{precision}_r{res}.engine",
    )
    monkeypatch.setattr(trt_engine, "TensorRTEngine", engine_cls)


def test_ensure_model_loads_the_engine_once_and_skips_inference_options(monkeypatch, tmp_path):
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"weights")
    (tmp_path / "w.pth.trt").mkdir()
    (tmp_path / "w.pth.trt" / "fp16_r8.engine").write_bytes(b"engine")
    node = RFDETRSegmenter(checkpoint_path=str(checkpoint), backend="tensorrt", precision="fp16")
    model = FakeTRTModel([], device="cuda")
    builds = []
    monkeypatch.setattr(node, "_build_model", lambda: builds.append(1) or model)
    monkeypatch.setattr(node, "_apply_inference_options", lambda m: pytest.fail("torch-only"))
    _patch_engine_lookup(monkeypatch)
    assert node._ensure_model() is model
    assert node._ensure_model() is model
    assert builds == [1]
    assert node._engine.path == str(tmp_path / "w.pth.trt" / "fp16_r8.engine")


def test_missing_engine_names_the_build_command(monkeypatch, tmp_path):
    node = RFDETRSegmenter(
        checkpoint_path=str(tmp_path / "w.pth"),
        variant="large",
        backend="tensorrt",
        precision="fp32",
        engine_dir=str(tmp_path / "engines"),
    )
    monkeypatch.setattr(node, "_build_model", lambda: FakeTRTModel([], device="cuda"))
    _patch_engine_lookup(monkeypatch)
    with pytest.raises(FileNotFoundError) as err:
        node._ensure_model()
    msg = str(err.value)
    assert "python -m cuvis_ai_rfdetr.trt_engine build" in msg
    assert "--variant large --resolution 8 --precision fp32" in msg
    assert f"--engine-dir {tmp_path / 'engines'}" in msg
    assert node._model is None  # nothing half-initialised: the next forward retries


def test_engine_with_the_wrong_input_shape_is_refused(monkeypatch, tmp_path):
    (tmp_path / "fp32_r8.engine").write_bytes(b"engine")
    node = RFDETRSegmenter(backend="tensorrt", engine_dir=str(tmp_path))
    monkeypatch.setattr(node, "_build_model", lambda: FakeTRTModel([], device="cuda"))
    _patch_engine_lookup(monkeypatch, lambda p, d: _LoadedEngine(p, d, (1, 3, 16, 16)))
    with pytest.raises(RuntimeError, match="rebuild"):
        node._ensure_model()


def test_tensorrt_backend_needs_cuda_and_an_engine_location(monkeypatch):
    node = RFDETRSegmenter(backend="tensorrt", engine_dir="/nowhere")
    monkeypatch.setattr(node, "_build_model", lambda: FakeTRTModel([], device="cpu"))
    with pytest.raises(RuntimeError, match="CUDA"):
        node._ensure_model()
    with pytest.raises(ValueError, match="checkpoint_path or engine_dir"):
        trt_engine.default_engine_dir(None)
    assert trt_engine.default_engine_dir("/w/rgb.pth") == "/w/rgb.pth.trt"


# ------------------------------------------------------------------ trt_engine helpers
def _fake_tensorrt(monkeypatch, version="10.15.1.29", fp16_flag=True, parse_ok=True, blob=b"plan"):
    """Minimal ``tensorrt`` module: records builder flags; the parse / build outcome is configurable."""
    trt = types.ModuleType("tensorrt")
    trt.__version__ = version
    flags = ["TF32"] + (["FP16"] if fp16_flag else [])
    trt.BuilderFlag = SimpleNamespace(**{f: f for f in flags})
    trt.Logger = type("Logger", (), {"WARNING": 2, "__init__": lambda self, level: None})
    trt.set_flags = []

    class Config:
        def set_flag(self, flag):
            trt.set_flags.append(flag)

        def get_flag(self, flag):
            return flag == "TF32" or flag in trt.set_flags

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return "network"

        def create_builder_config(self):
            return Config()

        def build_serialized_network(self, network, config):
            return blob

    class Parser:
        num_errors = 1

        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            trt.parsed = path
            return parse_ok

        def get_error(self, i):
            return "bad node"

    trt.Builder, trt.OnnxParser = Builder, Parser
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "_LOGGER", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA Thor (x)")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    return trt


class FakeExportModel:
    """rfdetr model object for building: export() writes one ONNX file (optionally returns its path)."""

    def __init__(self, returns_path=True, n_files=1):
        self.model = SimpleNamespace(resolution=504)
        self.returns_path, self.n_files = returns_path, n_files

    def export(self, output_dir, verbose=True):
        paths = [f"{output_dir}/net{i}.onnx" for i in range(self.n_files)]
        for p in paths:
            open(p, "wb").close()
        return paths[0] if self.returns_path and paths else None


def test_gpu_tag_and_engine_file_name(monkeypatch) -> None:
    _fake_tensorrt(monkeypatch)
    assert trt_engine.gpu_tag() == "NVIDIA-Thor-x-sm110"
    assert (
        trt_engine.engine_file_name("fp16", 504)
        == "fp16_r504_NVIDIA-Thor-x-sm110_trt10.15.1.29.engine"
    )


def test_tensorrt_import_errors(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import tensorrt -> ImportError
    with pytest.raises(ImportError, match="tensorrt-cu12"):
        trt_engine._tensorrt()
    _fake_tensorrt(monkeypatch, version="8.6.1")
    with pytest.raises(ImportError, match=">= 10"):
        trt_engine._tensorrt()


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_build_engine_writes_the_engine_and_its_record(monkeypatch, tmp_path, precision) -> None:
    trt = _fake_tensorrt(monkeypatch)
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"weights")
    engine = tmp_path / "engines" / "e.engine"
    record = trt_engine.build_engine(
        FakeExportModel(), precision, str(engine), checkpoint_path=str(checkpoint), variant="large"
    )
    assert engine.read_bytes() == b"plan"
    assert trt.parsed.endswith("net0.onnx")
    assert trt.set_flags == (["FP16"] if precision == "fp16" else [])
    saved = json.loads((tmp_path / "engines" / "e.engine.json").read_text())
    assert saved == record
    assert (saved["precision"], saved["resolution"], saved["variant"]) == (precision, 504, "large")
    assert saved["checkpoint_md5"] == trt_engine.file_md5(str(checkpoint))
    assert saved["tf32"] is True and saved["tensorrt"] == "10.15.1.29"


def test_build_engine_failures(monkeypatch, tmp_path) -> None:
    _fake_tensorrt(monkeypatch, fp16_flag=False)
    with pytest.raises(RuntimeError, match="FP16 builder flag"):
        trt_engine.build_engine(FakeExportModel(), "fp16", str(tmp_path / "e"))
    with pytest.raises(ValueError, match="precision"):
        trt_engine.build_engine(FakeExportModel(), "int8", str(tmp_path / "e"))
    _fake_tensorrt(monkeypatch, parse_ok=False)
    with pytest.raises(RuntimeError, match="bad node"):
        trt_engine.build_engine(FakeExportModel(), "fp32", str(tmp_path / "e"))
    _fake_tensorrt(monkeypatch, blob=None)
    with pytest.raises(RuntimeError, match="build failed"):
        trt_engine.build_engine(FakeExportModel(), "fp32", str(tmp_path / "e"))


def test_export_onnx_finds_rfdetrs_file(tmp_path) -> None:
    assert trt_engine.export_onnx(FakeExportModel(), str(tmp_path)).endswith("net0.onnx")
    older = tmp_path / "older"
    older.mkdir()
    found = trt_engine.export_onnx(FakeExportModel(returns_path=False), str(older))
    assert found.endswith("net0.onnx")
    two = tmp_path / "two"
    two.mkdir()
    with pytest.raises(RuntimeError, match="expected one"):
        trt_engine.export_onnx(FakeExportModel(returns_path=False, n_files=2), str(two))


def test_engine_built_from_another_checkpoint_is_refused(tmp_path) -> None:
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"weights v2")
    engine = tmp_path / "e.engine"
    trt_engine.check_engine_matches(str(engine), str(checkpoint))  # no record: accepted
    (tmp_path / "e.engine.json").write_text(json.dumps({"checkpoint_md5": "0" * 32}))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        trt_engine.check_engine_matches(str(engine), str(checkpoint))
    md5 = trt_engine.file_md5(str(checkpoint))
    (tmp_path / "e.engine.json").write_text(json.dumps({"checkpoint_md5": md5}))
    trt_engine.check_engine_matches(str(engine), str(checkpoint))


PIPELINE = """
nodes:
- name: Selector
  class_name: cuvis_ai.node.channel_selector.FixedWavelengthSelector
  hparams: {}
- name: SegRGB
  class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
  hparams: {checkpoint_path: /w/rgb.pth, variant: large, resolution: 504, backend: tensorrt,
            precision: fp16}
- name: SegCIR
  class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
  hparams: {checkpoint_path: /w/cir.pth, variant: large, backend: torch}
"""


def test_build_pipeline_cli_builds_each_tensorrt_segmenter_once(monkeypatch, tmp_path) -> None:
    (tmp_path / "p.yaml").write_text(PIPELINE)
    specs = trt_engine.engines_for_pipeline(str(tmp_path / "p.yaml"))
    assert specs == [
        {
            "checkpoint": "/w/rgb.pth",
            "variant": "large",
            "resolution": 504,
            "checkpoint_loader": "constructor",
            "precision": "fp16",
            "engine_dir": None,
        }
    ]
    built = []
    monkeypatch.setattr(trt_engine, "_build_one", lambda *a, **k: built.append((a, k)))
    yaml = str(tmp_path / "p.yaml")
    assert trt_engine.main(["build-pipeline", yaml, yaml]) == 0
    assert built == [((), {**specs[0], "force": False})]
    built.clear()
    assert trt_engine.main(["build", "--checkpoint", "/w/rgb.pth", "--variant", "large"]) == 0
    assert [a[4] for a, _ in built] == ["fp32", "fp16"]


# ------------------------------------------------------------------ real TensorRT
@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("precision, atol", [("fp32", 1e-3), ("fp16", 2e-2)])
def test_real_tensorrt_build_and_run_a_small_network(tmp_path, precision, atol) -> None:
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            y = self.conv(x)
            return y.mean((2, 3)), y.amax(1), torch.sigmoid(y)

    torch.manual_seed(0)
    net = Net().eval()

    class Exportable:
        model = SimpleNamespace(resolution=16)

        def export(self, output_dir, verbose=True):
            path = f"{output_dir}/net.onnx"
            torch.onnx.export(
                net,
                (torch.rand(1, 3, 16, 16),),
                path,
                input_names=["input"],
                output_names=["dets", "labels", "masks"],
                opset_version=17,
                dynamo=False,
            )
            return path

    path = str(tmp_path / "net.engine")
    trt_engine.build_engine(Exportable(), precision, path)
    engine = trt_engine.TensorRTEngine(path, "cuda")
    assert tuple(engine.input_shape) == (1, 3, 16, 16)
    x = torch.rand(1, 3, 16, 16, device="cuda")
    out = engine(x)
    with torch.no_grad():
        ref = net.cuda()(x)
    for name, r in zip(("dets", "labels", "masks"), ref, strict=True):
        torch.testing.assert_close(out[name], r, atol=atol, rtol=0)
