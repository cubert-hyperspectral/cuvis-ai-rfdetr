"""TensorRT engine lookup of RFDETRSegmenter: behaviour that building engines at load must keep.

Characterization tests, written against 0.5.1 before the engine build moved into the loader:

- an engine that already exists loads as it is, without a build;
- an engine whose build record names another checkpoint is refused, and nothing is built in its place;
- the torch backend never touches the TensorRT helpers.

Mocked throughout (no TensorRT, no rfdetr model, no GPU needed).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from cuvis_ai_rfdetr import trt_engine
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter


def _fail(*args, **kwargs):
    pytest.fail("the TensorRT path must not be touched here")


def _model(device: str = "cuda") -> SimpleNamespace:
    """The parts of an rfdetr model object the engine lookup reads."""
    return SimpleNamespace(model=SimpleNamespace(resolution=8, device=torch.device(device)))


class _LoadedEngine:
    def __init__(self, path, device, **_):
        self.path, self.device, self.input_shape = path, device, (1, 3, 8, 8)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Short engine file names, no real engine loads or builds, no writes outside tmp_path."""
    monkeypatch.setattr(
        trt_engine,
        "engine_file_name",
        lambda precision, resolution, device=None: f"{precision}_r{resolution}.engine",
    )
    monkeypatch.setattr(trt_engine, "compile_network", _fail)
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))


@pytest.mark.parametrize("explicit_dir", [False, True], ids=["next_to_checkpoint", "engine_dir"])
def test_existing_engine_loads_without_a_build(monkeypatch, tmp_path, explicit_dir) -> None:
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"weights")
    folder = tmp_path / ("engines" if explicit_dir else "w.pth.trt")
    folder.mkdir()
    engine = folder / "fp16_r8.engine"
    engine.write_bytes(b"engine")
    record = {"checkpoint_md5": trt_engine.file_md5(str(checkpoint))}
    (folder / "fp16_r8.engine.json").write_text(json.dumps(record))
    node = RFDETRSegmenter(
        checkpoint_path=str(checkpoint),
        backend="tensorrt",
        precision="fp16",
        engine_dir=str(folder) if explicit_dir else None,
    )
    model = _model()
    monkeypatch.setattr(node, "_build_model", lambda: model)
    monkeypatch.setattr(trt_engine, "TensorRTEngine", _LoadedEngine)

    assert node._ensure_model() is model
    assert node._engine.path == str(engine)
    assert engine.read_bytes() == b"engine"  # left as it was


def test_engine_built_from_another_checkpoint_is_refused(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"weights v2")
    folder = tmp_path / "w.pth.trt"
    folder.mkdir()
    (folder / "fp32_r8.engine").write_bytes(b"engine")
    (folder / "fp32_r8.engine.json").write_text(json.dumps({"checkpoint_md5": "0" * 32}))
    node = RFDETRSegmenter(checkpoint_path=str(checkpoint), backend="tensorrt")
    monkeypatch.setattr(node, "_build_model", _model)
    monkeypatch.setattr(trt_engine, "TensorRTEngine", _fail)

    with pytest.raises(RuntimeError, match="other weights"):
        node._ensure_model()
    assert node._model is None and node._engine is None
    assert sorted(p.name for p in folder.iterdir()) == ["fp32_r8.engine", "fp32_r8.engine.json"]


def test_torch_backend_never_touches_tensorrt(monkeypatch) -> None:
    node = RFDETRSegmenter(checkpoint_path="w.pth", precision="fp16")
    model = object()
    applied = []
    monkeypatch.setattr(node, "_build_model", lambda: model)
    monkeypatch.setattr(node, "_apply_inference_options", lambda m: applied.append(m) or m)
    for name in (
        "engine_file_name",
        "default_engine_dir",
        "check_engine_matches",
        "TensorRTEngine",
        "_tensorrt",
    ):
        monkeypatch.setattr(trt_engine, name, _fail)

    assert node._ensure_model() is model
    assert applied == [model]
    assert node._engine is None
