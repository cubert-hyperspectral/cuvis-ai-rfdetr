"""TensorRT engines for the RF-DETR segmentation network: build them per machine, run them on torch CUDA tensors.

The network is exported with rfdetr's own ONNX exporter (``model.export()``: static ``1 x 3 x resolution x
resolution`` input ``input``, outputs ``dets`` / ``labels`` / ``masks``) and compiled by the installed TensorRT into an
engine for THIS GPU and TensorRT version. Engines are not portable, so their file names carry the precision, the
resolution, the GPU and the TensorRT version, and one engine directory can hold the engines of several machines side by
side (``<checkpoint>.trt/fp16_r504_NVIDIA-Thor-sm110_trt10.15.1.29.engine``, ...). A JSON file next to each engine
records how it was built, including the checksum of the checkpoint it was built from.

Precision: ``fp32`` is TensorRT's default float build, which allows TF32 tensor-core math - like PyTorch after
``import rfdetr``, which sets ``torch.set_float32_matmul_precision("high")``. ``fp16`` sets TensorRT's FP16 builder
flag (mixed precision: TensorRT keeps a layer in fp32 where that is faster or needed); TensorRT 11 dropped that flag,
so fp16 engines need TensorRT 10.

Optional dependencies, not installed with the plugin: the TensorRT Python package matching torch's CUDA
(``tensorrt-cu12`` / ``tensorrt-cu13``, version 10) and, for building, ``onnx``. Build engines with::

    python -m cuvis_ai_rfdetr.trt_engine build --checkpoint W.pth --variant large --resolution 504 --precision fp16
    python -m cuvis_ai_rfdetr.trt_engine build-pipeline pipeline.yaml [more.yaml ...]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
from typing import Any

import torch
from torch import Tensor

PRECISIONS = ("fp32", "fp16")
_LOGGER: Any = None


def _tensorrt() -> Any:
    try:
        import tensorrt
    except ImportError as exc:
        raise ImportError(
            "RFDETRSegmenter backend='tensorrt' needs the TensorRT Python package (version 10) matching "
            "torch's CUDA: pip install tensorrt-cu12 (CUDA 12 torch) or tensorrt-cu13 (CUDA 13 torch); "
            "building engines also needs onnx."
        ) from exc
    if int(str(tensorrt.__version__).split(".")[0]) < 10:
        raise ImportError(f"TensorRT >= 10 is required, found {tensorrt.__version__}.")
    return tensorrt


def _logger(trt: Any) -> Any:
    """One TensorRT logger per process (TensorRT keeps the first one it sees and warns about others)."""
    global _LOGGER
    if _LOGGER is None or _LOGGER[0] is not trt:
        _LOGGER = (trt, trt.Logger(trt.Logger.WARNING))
    return _LOGGER[1]


def gpu_tag(device: torch.device | str | int | None = None) -> str:
    """``<device name>-sm<capability>``, filesystem-safe (e.g. ``NVIDIA-Thor-sm110``)."""
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')}-sm{major}{minor}"


def engine_file_name(
    precision: str, resolution: int, device: torch.device | str | int | None = None
) -> str:
    """This machine's engine file for ``precision`` / ``resolution``.

    ``<precision>_r<resolution>_<gpu tag>_trt<TensorRT version>.engine``.
    """
    return f"{precision}_r{int(resolution)}_{gpu_tag(device)}_trt{_tensorrt().__version__}.engine"


def default_engine_dir(checkpoint_path: str | None) -> str:
    """Engines live next to their checkpoint by default: ``<checkpoint>.trt/``."""
    if checkpoint_path is None:
        raise ValueError("RFDETRSegmenter backend='tensorrt' needs checkpoint_path or engine_dir.")
    return f"{checkpoint_path}.trt"


def file_md5(path: str, chunk: int = 1 << 22) -> str:
    """MD5 of a file (identity check of a local checkpoint, not a security measure)."""
    digest = hashlib.md5()  # noqa: S324
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_engine_matches(engine_path: str, checkpoint_path: str | None) -> None:
    """Refuse an engine whose build record says it was compiled from a different checkpoint."""
    record_path = f"{engine_path}.json"
    if checkpoint_path is None or not os.path.exists(record_path):
        return
    with open(record_path, encoding="utf-8") as f:
        built_from = json.load(f).get("checkpoint_md5")
    if built_from and built_from != file_md5(checkpoint_path):
        raise RuntimeError(
            f"TensorRT engine {engine_path} was built from a different checkpoint than "
            f"{checkpoint_path}; rebuild it (python -m cuvis_ai_rfdetr.trt_engine build ... --force)."
        )


def export_onnx(model: Any, output_dir: str) -> str:
    """Export the network of ``model`` (an rfdetr model object) to ONNX with rfdetr's own exporter."""
    returned = model.export(output_dir=output_dir, verbose=False)
    if returned is not None and str(returned).endswith(".onnx") and os.path.exists(returned):
        return str(returned)
    found = sorted(glob.glob(os.path.join(output_dir, "*.onnx")))
    if len(found) != 1:
        raise RuntimeError(
            f"rfdetr's export wrote {len(found)} ONNX files to {output_dir}, expected one."
        )
    return found[0]


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_engine(
    model: Any,
    precision: str,
    engine_path: str,
    *,
    checkpoint_path: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Build and save a TensorRT engine for the network of ``model``; returns (and saves) its build record."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}.")
    trt = _tensorrt()
    if precision == "fp16" and not hasattr(trt.BuilderFlag, "FP16"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} has no FP16 builder flag (dropped in TensorRT 11): build fp16 "
            "engines with TensorRT 10, or use precision 'fp32'."
        )
    t0 = time.perf_counter()
    logger = _logger(trt)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model, tmp)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the exported network: {errors}")
        config = builder.create_builder_config()
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        tf32 = config.get_flag(trt.BuilderFlag.TF32) if hasattr(trt.BuilderFlag, "TF32") else None
        blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT engine build failed ({precision}).")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(blob)
    have_checkpoint = checkpoint_path is not None and os.path.exists(checkpoint_path)
    record = {
        "engine": os.path.basename(engine_path),
        "precision": precision,
        "tf32": tf32,
        "resolution": int(model.model.resolution),
        "variant": variant,
        "checkpoint": checkpoint_path,
        "checkpoint_md5": file_md5(checkpoint_path) if have_checkpoint else None,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "rfdetr": _version("rfdetr"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_tag": gpu_tag(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(f"{engine_path}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return record


class TensorRTEngine:
    """Run a serialized TensorRT engine (static shapes, one input) on torch CUDA tensors.

    The engine runs on its own CUDA stream (TensorRT adds a host synchronisation to every call on the default
    stream), ordered after the work already queued on torch's current stream, and torch's current stream waits for
    it, so torch ops that follow see the results in order. The output tensors are reused buffers, overwritten by the
    next call - consume or copy them before calling again.
    """

    def __init__(self, path: str, device: torch.device | str = "cuda") -> None:
        trt = _tensorrt()
        self.path = path
        self.device = torch.device(device)
        dtypes = {
            getattr(trt, name): dtype
            for name, dtype in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
                ("int32", torch.int32),
                ("int64", torch.int64),
                ("int8", torch.int8),
                ("uint8", torch.uint8),
                ("bool", torch.bool),
            )
            if hasattr(trt, name)
        }
        self._runtime = trt.Runtime(_logger(trt))
        with open(path, "rb") as f, torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._stream = torch.cuda.Stream(self.device)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}: an engine only runs on the GPU and TensorRT version it "
                "was built with - rebuild it on this machine (python -m cuvis_ai_rfdetr.trt_engine build)."
            )
        self._context = self._engine.create_execution_context()
        inputs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        self.outputs: dict[str, Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = dtypes[self._engine.get_tensor_dtype(name)]
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = (dtype, shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.outputs[name] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
        if len(inputs) != 1:
            raise RuntimeError(f"{path}: expected one engine input, found {sorted(inputs)}.")
        self.input_name, (self.input_dtype, self.input_shape) = next(iter(inputs.items()))
        self._held: Tensor | None = None

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()
        self._context.set_tensor_address(self.input_name, x.data_ptr())
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)  # x written, previous outputs consumed
        with torch.cuda.device(self.device):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        current.wait_stream(self._stream)  # everything queued after this sees the outputs
        self._held = x  # the engine reads x asynchronously; keep it alive until the next call
        return self.outputs


def _build_one(
    checkpoint: str,
    variant: str,
    resolution: int | None,
    checkpoint_loader: str,
    precision: str,
    engine_dir: str | None,
    force: bool = False,
) -> str:
    from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

    model = RFDETRSegmenter(
        checkpoint_path=checkpoint,
        variant=variant,
        resolution=resolution,
        checkpoint_loader=checkpoint_loader,
    )._build_model()
    engine_dir = engine_dir or default_engine_dir(checkpoint)
    path = os.path.join(engine_dir, engine_file_name(precision, int(model.model.resolution)))
    if os.path.exists(path) and not force:
        print(f"exists: {path}", flush=True)
        return path
    record = build_engine(model, precision, path, checkpoint_path=checkpoint, variant=variant)
    print(f"built: {path} ({record['build_seconds']} s)", flush=True)
    return path


def engines_for_pipeline(pipeline_yaml: str) -> list[dict[str, Any]]:
    """Build specs of the ``backend: tensorrt`` RFDETRSegmenter nodes of a pipeline yaml."""
    import yaml

    with open(pipeline_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    specs = []
    for node in doc.get("nodes", []):
        hp = node.get("hparams") or {}
        if not str(node.get("class_name", "")).endswith("RFDETRSegmenter"):
            continue
        if hp.get("backend") != "tensorrt":
            continue
        specs.append(
            {
                "checkpoint": hp.get("checkpoint_path"),
                "variant": hp.get("variant", "medium"),
                "resolution": hp.get("resolution"),
                "checkpoint_loader": hp.get("checkpoint_loader", "constructor"),
                "precision": hp.get("precision", "fp32"),
                "engine_dir": hp.get("engine_dir"),
            }
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m cuvis_ai_rfdetr.trt_engine",
        description="Build TensorRT engines for RFDETRSegmenter(backend='tensorrt') on this machine.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("build", help="build the engine(s) of one checkpoint")
    one.add_argument("--checkpoint", required=True)
    one.add_argument("--variant", default="medium")
    one.add_argument("--resolution", type=int, default=None)
    one.add_argument("--checkpoint-loader", default="constructor")
    one.add_argument(
        "--precision",
        choices=PRECISIONS,
        action="append",
        help="repeatable; default: fp32 and fp16",
    )
    one.add_argument("--engine-dir", default=None, help="default: <checkpoint>.trt/")
    one.add_argument("--force", action="store_true", help="rebuild an existing engine")
    pipe = sub.add_parser(
        "build-pipeline",
        help="build the engines the backend: tensorrt segmenters of the given pipeline yamls need",
    )
    pipe.add_argument("pipelines", nargs="+")
    pipe.add_argument("--force", action="store_true", help="rebuild existing engines")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for precision in args.precision or list(PRECISIONS):
            _build_one(
                args.checkpoint,
                args.variant,
                args.resolution,
                args.checkpoint_loader,
                precision,
                args.engine_dir,
                args.force,
            )
        return 0
    seen = set()
    for pipeline_yaml in args.pipelines:
        for spec in engines_for_pipeline(pipeline_yaml):
            key = tuple(sorted(spec.items()))
            if key not in seen:
                seen.add(key)
                _build_one(**spec, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
