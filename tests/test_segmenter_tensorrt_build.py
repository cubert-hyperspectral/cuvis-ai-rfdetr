"""RFDETRSegmenter builds a missing TensorRT engine when the pipeline loads (``prepare_inference``).

Mocked: no TensorRT, no rfdetr model and no GPU needed. ``compile_network`` is replaced by a fake that returns an
engine and its record, which cuvis-ai-core's ``build_engine_once`` writes, so the lookup, the folder choice, the lock
and the error paths run as in production. The fingerprint cache and the atomic writer are core's and tested there.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cuvis_ai_core.utils import trt_engine as core_trt
from loguru import logger

from cuvis_ai_rfdetr import trt_engine
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter


def _fail(*args, **kwargs):
    pytest.fail("must not be called here")


def _model(device: str = "cuda") -> SimpleNamespace:
    """The parts of an rfdetr model object the engine lookup and the build read."""
    return SimpleNamespace(model=SimpleNamespace(resolution=8, device=torch.device(device)))


class _LoadedEngine:
    def __init__(self, path, device, **_):
        self.path, self.device, self.input_shape = path, device, (1, 3, 8, 8)


class FakeBuilds:
    """``compile_network`` stand-in: counts builds and returns an engine plus its RF-DETR record."""

    def __init__(self, seconds: float = 0.0, error: Exception | None = None):
        self.paths: list[str] = []
        self.seconds, self.error = seconds, error
        self.lock = threading.Lock()

    def __call__(self, model, precision, *, checkpoint_path=None, variant=None):
        with self.lock:
            self.paths.append(precision)
        time.sleep(self.seconds)
        if self.error is not None:
            raise self.error
        record = {
            "precision": precision,
            "checkpoint": checkpoint_path,
            "checkpoint_md5": trt_engine.checkpoint_md5(checkpoint_path),
        }
        return b"plan", record


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A checkpoint, a model cache in tmp_path, short engine names, a fake engine loader and a log capture."""
    checkpoint = tmp_path / "weights" / "w.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights v1")
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    monkeypatch.setattr(
        trt_engine,
        "engine_file_name",
        lambda precision, resolution, device=None: f"{precision}_r{resolution}.engine",
    )
    monkeypatch.setattr(trt_engine, "TensorRTEngine", _LoadedEngine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    builds = FakeBuilds()
    monkeypatch.setattr(trt_engine, "compile_network", builds)
    # core's build adds the TensorRT, torch and GPU fields to the record
    monkeypatch.setattr(core_trt, "import_tensorrt", lambda *a: SimpleNamespace(__version__="10.x"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield SimpleNamespace(
        checkpoint=checkpoint, cache=tmp_path / "model_cache", builds=builds, messages=messages
    )
    logger.remove(sink)


def _node(env, monkeypatch, **kw):
    node = RFDETRSegmenter(
        checkpoint_path=str(env.checkpoint), backend="tensorrt", precision="fp16", **kw
    )
    models = []
    monkeypatch.setattr(node, "_build_model", lambda: models.append(1) or _model())
    node.models_built = models
    return node


def _files(folder) -> list[str]:
    """The files in ``folder`` apart from lock files (filelock removes them on Windows only)."""
    return sorted(n for n in os.listdir(folder) if not n.endswith(".lock"))


def _keyed(env, folder=None):
    base = folder or f"{env.checkpoint}.trt"
    return os.path.join(base, trt_engine.weights_fingerprint(str(env.checkpoint)))


# ------------------------------------------------------------------ lookup and layout
def test_engine_location_order(env) -> None:
    key = trt_engine.weights_fingerprint(str(env.checkpoint))
    base = Path(f"{env.checkpoint}.trt")
    root = env.cache / "trt_engines" / "rfdetr"
    loc = trt_engine.engine_location("e.engine", str(env.checkpoint))
    assert loc.candidates == (
        base / key / "e.engine",
        base / "e.engine",
        root / key / "e.engine",
        root / "e.engine",
    )
    assert loc.build_dirs == (base / key, root / key)
    explicit = trt_engine.engine_location("e.engine", str(env.checkpoint), "/engines")
    assert explicit.candidates[:2] == (
        Path("/engines") / key / "e.engine",
        Path("/engines") / "e.engine",
    )
    for checkpoint in (None, str(env.checkpoint) + ".missing"):
        flat = trt_engine.engine_location("e.engine", checkpoint, "/engines")
        assert flat == trt_engine.EngineLocation(
            (Path("/engines") / "e.engine", root / "e.engine"), ()
        )


def test_fingerprint_folder_wins_over_a_stale_engine_of_0_5_1(env, monkeypatch) -> None:
    """Retrained weights under the same name: the new build is found, the old flat engine is never used."""
    base = f"{env.checkpoint}.trt"
    os.makedirs(base)
    open(os.path.join(base, "fp16_r8.engine"), "wb").close()
    with open(os.path.join(base, "fp16_r8.engine.json"), "w") as f:
        json.dump({"checkpoint_md5": "0" * 32}, f)
    node = _node(env, monkeypatch)
    with pytest.raises(RuntimeError, match="other weights"):
        node.prepare_inference()  # refused, not silently rebuilt
    assert env.builds.paths == []
    # the fix the error names: a forced build, which goes to the fingerprint folder
    path = trt_engine.build_missing_engine(
        _model(), "fp16", "fp16_r8.engine", checkpoint_path=str(env.checkpoint), force=True
    )
    assert path == os.path.join(_keyed(env), "fp16_r8.engine")
    fresh = _node(env, monkeypatch)
    fresh.prepare_inference()
    assert fresh._engine.path == path


# ------------------------------------------------------------------ prepare_inference
def test_prepare_inference_builds_a_missing_engine_once(env, monkeypatch) -> None:
    node = _node(env, monkeypatch)
    node.prepare_inference()
    expected = os.path.join(_keyed(env), "fp16_r8.engine")
    assert env.builds.paths == ["fp16"]
    assert node._engine.path == expected
    with open(f"{expected}.json") as f:
        assert json.load(f)["checkpoint_md5"] == trt_engine.file_md5(str(env.checkpoint))
    assert _files(_keyed(env)) == ["fp16_r8.engine", "fp16_r8.engine.json"]
    assert any(m.startswith(f"Building TensorRT engine {expected}") for m in env.messages)
    assert any(m.startswith(f"Built TensorRT engine {expected}") for m in env.messages)

    node.prepare_inference()  # after a weights load: everything is current
    node._ensure_model()  # the first frame
    assert len(env.builds.paths) == 1 and node.models_built == [1]

    other = _node(env, monkeypatch)  # a second pipeline with the same checkpoint
    other.prepare_inference()
    assert len(env.builds.paths) == 1 and other._engine.path == expected


def test_prepare_inference_uses_the_explicit_engine_dir(env, monkeypatch, tmp_path) -> None:
    folder = tmp_path / "engines"
    node = _node(env, monkeypatch, engine_dir=str(folder))
    node.prepare_inference()
    assert node._engine.path == os.path.join(_keyed(env, str(folder)), "fp16_r8.engine")


def test_unwritable_checkpoint_folder_falls_back_to_the_model_cache(env, monkeypatch) -> None:
    real = core_trt._not_writable
    keyed = _keyed(env)
    monkeypatch.setattr(
        core_trt,
        "_not_writable",
        lambda folder: f"{folder}: read-only" if str(folder) == keyed else real(folder),
    )
    node = _node(env, monkeypatch)
    node.prepare_inference()
    cached = os.path.join(
        str(env.cache),
        "trt_engines",
        "rfdetr",
        trt_engine.weights_fingerprint(str(env.checkpoint)),
        "fp16_r8.engine",
    )
    assert env.builds.paths == ["fp16"] and node._engine.path == cached
    assert not os.path.exists(keyed)

    monkeypatch.setattr(core_trt, "_not_writable", real)  # the next load finds it without a build
    fresh = _node(env, monkeypatch)
    fresh.prepare_inference()
    assert fresh._engine.path == cached and len(env.builds.paths) == 1


def test_failed_build_raises_one_error_that_names_the_command(env, monkeypatch) -> None:
    env.builds.error = RuntimeError("TensorRT engine build failed (fp16).")
    node = _node(env, monkeypatch, variant="large")
    with pytest.raises(RuntimeError) as err:
        node.prepare_inference()
    msg = str(err.value)
    assert "could not build TensorRT engine" in msg and "fp16_r8.engine" in msg
    assert "RuntimeError: TensorRT engine build failed (fp16).)" in msg
    assert (
        f"Build it by hand with: python -m cuvis_ai_rfdetr.trt_engine build --checkpoint "
        f"{env.checkpoint} --variant large --resolution 8 --precision fp16"
    ) in msg
    assert isinstance(err.value.__cause__, RuntimeError)
    assert node._model is None and node._engine is None  # the next load or frame retries
    assert _files(_keyed(env)) == []  # no engine, no record, no temporary file


def test_no_writable_folder_names_the_command(env, monkeypatch) -> None:
    monkeypatch.setattr(core_trt, "_not_writable", lambda folder: f"{folder}: read-only")
    node = _node(env, monkeypatch)
    with pytest.raises(RuntimeError, match="no folder to build") as err:
        node.prepare_inference()
    assert "Build it by hand with: python -m cuvis_ai_rfdetr.trt_engine build" in str(err.value)
    assert env.builds.paths == []


def test_prepare_inference_without_cuda_prepares_nothing(env, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    node = RFDETRSegmenter(checkpoint_path=str(env.checkpoint), backend="tensorrt")
    monkeypatch.setattr(node, "_build_model", _fail)
    node.prepare_inference()
    assert node._model is None and env.builds.paths == []
    assert any("no CUDA device" in m for m in env.messages)


def test_prepare_inference_does_nothing_for_the_torch_backend(env, monkeypatch) -> None:
    node = RFDETRSegmenter(checkpoint_path=str(env.checkpoint), precision="fp16")
    monkeypatch.setattr(node, "_build_model", _fail)
    monkeypatch.setattr(trt_engine, "find_engine", _fail)
    node.prepare_inference()
    assert node._model is None and node._engine is None


def test_concurrent_loads_build_the_engine_once(env, monkeypatch) -> None:
    env.builds.seconds = 0.5
    paths: list[str] = []
    errors: list[BaseException] = []

    def load() -> None:
        try:
            paths.append(
                trt_engine.build_missing_engine(
                    _model(), "fp16", "fp16_r8.engine", checkpoint_path=str(env.checkpoint)
                )
            )
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert len(env.builds.paths) == 1
    assert paths == [os.path.join(_keyed(env), "fp16_r8.engine")] * 3


# ------------------------------------------------------------------ writing and the CLI
def test_cli_builds_into_the_fingerprint_folder_and_reports_existing_engines(
    env, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(RFDETRSegmenter, "_build_model", lambda self: _model())
    args = ["build", "--checkpoint", str(env.checkpoint), "--precision", "fp16"]
    expected = os.path.join(_keyed(env), "fp16_r8.engine")
    assert trt_engine.main(args) == 0
    assert env.builds.paths == ["fp16"]
    assert trt_engine.main(args) == 0  # exists: nothing built
    assert len(env.builds.paths) == 1
    assert trt_engine.main([*args, "--force"]) == 0
    assert env.builds.paths == ["fp16", "fp16"]
    out = capsys.readouterr().out
    assert f"built: {expected}" in out and f"exists: {expected}" in out


# ------------------------------------------------------------------ real TensorRT
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_engine_is_built_at_load_and_found_by_the_next_load(monkeypatch, tmp_path) -> None:
    """A real (tiny) network: prepare_inference builds the engine once, a second node loads it as it is."""
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
        model = SimpleNamespace(resolution=16, device=torch.device("cuda"))

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

    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"tiny weights")
    builds = []
    real_compile = trt_engine.compile_network
    monkeypatch.setattr(
        trt_engine, "compile_network", lambda *a, **k: builds.append(1) or real_compile(*a, **k)
    )

    def node():
        n = RFDETRSegmenter(
            checkpoint_path=str(checkpoint), backend="tensorrt", precision="fp32", resolution=16
        )
        monkeypatch.setattr(n, "_build_model", Exportable)
        return n

    first = node()
    first.prepare_inference()
    second = node()
    second.prepare_inference()
    assert len(builds) == 1
    built = first._engine.path
    assert second._engine.path == built
    assert os.path.dirname(built) == _keyed(SimpleNamespace(checkpoint=checkpoint))
    with open(f"{built}.json") as f:
        assert json.load(f)["checkpoint_md5"] == trt_engine.file_md5(str(checkpoint))
    x = torch.rand(1, 3, 16, 16, device="cuda")
    out = second._engine(x)
    with torch.no_grad():
        ref = net.cuda()(x)
    for name, r in zip(("dets", "labels", "masks"), ref, strict=True):
        torch.testing.assert_close(out[name], r, atol=1e-3, rtol=0)
