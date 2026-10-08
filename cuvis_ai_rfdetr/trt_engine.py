"""TensorRT engines for the RF-DETR segmentation network: build them per machine, run them on torch CUDA tensors.

The network is exported with rfdetr's own ONNX exporter (``model.export()``: static ``1 x 3 x resolution x
resolution`` input ``input``, outputs ``dets`` / ``labels`` / ``masks``) and compiled by the installed TensorRT into an
engine for THIS GPU and TensorRT version. Engines are not portable, so their file names carry the precision, the
resolution, the GPU and the TensorRT version, and one engine directory can hold the engines of several machines side by
side (``<checkpoint>.trt/fp16_r504_NVIDIA-Thor-sm110_trt10.15.1.29.engine``, ...). A JSON file next to each engine
records how it was built, including the checksum of the checkpoint it was built from.

Where engines live: next to their checkpoint by default, in ``<checkpoint>.trt/<fingerprint>/``, where the
fingerprint is the start of the checkpoint's MD5, so retrained weights saved under the same name get their own
engines. When that folder cannot be written: ``<model cache>/trt_engines/cuvis_ai_rfdetr/<fingerprint>/`` (the
cuvis-ai model cache, ``$CUVIS_MODEL_CACHE_DIR``). An explicit ``engine_dir`` takes the place of ``<checkpoint>.trt``.
Engines of 0.5.1 and earlier, directly in ``<checkpoint>.trt/`` (or ``engine_dir``), are still found.

A pipeline loader builds a missing engine on the machine itself (``RFDETRSegmenter.prepare_inference``, the
cuvis-ai-core 0.18.2 hook): once per checkpoint, precision and GPU, under a file lock per engine, so concurrent loads
build it once.

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
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from torch import Tensor

PRECISIONS = ("fp32", "fp16")
#: Hex digits of the checkpoint MD5 that name its engine folder.
FINGERPRINT_CHARS = 16
#: Seconds a load waits for another process that builds the same engine (a build takes minutes).
LOCK_TIMEOUT_S = 1800.0
_LOGGER: Any = None
#: Full MD5 per (absolute path, size, modification time): a pipeline that is prepared again does not
#: re-read its checkpoint.
_MD5_CACHE: dict[tuple[str, int, int], str] = {}


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


def checkpoint_md5(path: str) -> str:
    """:func:`file_md5` of a checkpoint, computed once per path, size and modification time."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = file_md5(path)
    return _MD5_CACHE[key]


def weights_fingerprint(checkpoint_path: str) -> str:
    """Name of a checkpoint's engine folder: the first :data:`FINGERPRINT_CHARS` hex digits of its MD5."""
    return checkpoint_md5(checkpoint_path)[:FINGERPRINT_CHARS]


def cache_engine_root() -> str:
    """Engine folders of checkpoints whose own folder is not writable.

    ``<model cache>/trt_engines/cuvis_ai_rfdetr``, in the cuvis-ai model cache: ``$CUVIS_MODEL_CACHE_DIR`` (which the
    CuvisNEXT runtime sets for its child environments), else ``~/.cuvis_runs/model_cache``.
    """
    from cuvis_ai_core.orchestrator.model_cache import model_cache_dir

    return os.path.join(str(model_cache_dir()), "trt_engines", "cuvis_ai_rfdetr")


def check_engine_matches(engine_path: str, checkpoint_path: str | None) -> None:
    """Refuse an engine whose build record says it was compiled from a different checkpoint."""
    record_path = f"{engine_path}.json"
    if checkpoint_path is None or not os.path.exists(record_path):
        return
    with open(record_path, encoding="utf-8") as f:
        built_from = json.load(f).get("checkpoint_md5")
    if built_from and built_from != checkpoint_md5(checkpoint_path):
        raise RuntimeError(
            f"TensorRT engine {engine_path} was built from a different checkpoint than "
            f"{checkpoint_path}; rebuild it (python -m cuvis_ai_rfdetr.trt_engine build ... --force) "
            "or delete it."
        )


@dataclass(frozen=True)
class EngineLocation:
    """Where one engine is looked up, and where a missing one is built.

    ``candidates``, in lookup order: the fingerprint folder, the 0.5.1 location directly in the engine folder, the
    fingerprint folder in the model cache. ``build_dirs``: the fingerprint folder, then the one in the model cache;
    a build goes to the first that is writable. Without a checkpoint there is nothing to fingerprint: only the engine
    folder itself is searched, and nothing is built; the same when the checkpoint file does not exist.
    """

    candidates: tuple[str, ...]
    build_dirs: tuple[str, ...]


def engine_location(
    file_name: str, checkpoint_path: str | None, engine_dir: str | None = None
) -> EngineLocation:
    """The :class:`EngineLocation` of the engine ``file_name`` for a checkpoint and an optional ``engine_dir``."""
    base = engine_dir or default_engine_dir(checkpoint_path)
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return EngineLocation(candidates=(os.path.join(base, file_name),), build_dirs=())
    key = weights_fingerprint(checkpoint_path)
    keyed = os.path.join(base, key)
    cached = os.path.join(cache_engine_root(), key)
    return EngineLocation(
        candidates=(
            os.path.join(keyed, file_name),
            os.path.join(base, file_name),
            os.path.join(cached, file_name),
        ),
        build_dirs=(keyed, cached),
    )


def find_engine(
    file_name: str, checkpoint_path: str | None, engine_dir: str | None = None
) -> str | None:
    """The first existing engine of :func:`engine_location`, checked against its checkpoint; ``None`` if none.

    An engine whose build record names another checkpoint is refused (:func:`check_engine_matches`), not skipped.
    """
    for path in engine_location(file_name, checkpoint_path, engine_dir).candidates:
        if os.path.exists(path):
            check_engine_matches(path, checkpoint_path)
            return path
    return None


def _not_writable(folder: str) -> str | None:
    """``None`` when ``folder`` exists (or can be created) and takes a new file, else why it does not."""
    try:
        os.makedirs(folder, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as exc:
        return f"{folder}: {exc}"
    return None


def build_missing_engine(
    model: Any,
    precision: str,
    file_name: str,
    *,
    checkpoint_path: str,
    engine_dir: str | None = None,
    variant: str | None = None,
    force: bool = False,
) -> str:
    """Build the engine ``file_name`` where :func:`find_engine` looks first, unless another process already has.

    The build runs in the first writable folder of :func:`engine_location`, under a file lock per engine
    (``<engine>.lock``): a second process that needs the same engine waits for the first and then loads its result.
    Start and end are logged. ``force`` rebuilds even when an engine exists. Returns the engine path.
    """
    from filelock import FileLock, Timeout

    location = engine_location(file_name, checkpoint_path, engine_dir)
    reasons = []
    for folder in location.build_dirs:
        reason = _not_writable(folder)
        if reason is not None:
            reasons.append(reason)
            continue
        path = os.path.join(folder, file_name)
        try:
            with FileLock(f"{path}.lock", timeout=LOCK_TIMEOUT_S):
                if not force:  # built by another process while this one waited for the lock
                    found = find_engine(file_name, checkpoint_path, engine_dir)
                    if found is not None:
                        return found
                logger.info(
                    f"Building TensorRT engine {path} from {checkpoint_path} ({precision}); this "
                    "takes a few minutes, once per checkpoint, precision and GPU."
                )
                record = build_engine(
                    model, precision, path, checkpoint_path=checkpoint_path, variant=variant
                )
                logger.info(f"Built TensorRT engine {path} in {record['build_seconds']} s.")
                return path
        except Timeout as exc:
            raise RuntimeError(
                f"waited {LOCK_TIMEOUT_S:.0f} s for another process that builds {path} "
                f"(lock file {path}.lock)"
            ) from exc
    raise PermissionError(
        "no writable folder for the TensorRT engine: " + "; ".join(reasons or ["no checkpoint"])
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
    have_checkpoint = checkpoint_path is not None and os.path.exists(checkpoint_path)
    record = {
        "engine": os.path.basename(engine_path),
        "precision": precision,
        "tf32": tf32,
        "resolution": int(model.model.resolution),
        "variant": variant,
        "checkpoint": checkpoint_path,
        "checkpoint_md5": checkpoint_md5(checkpoint_path) if have_checkpoint else None,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "rfdetr": _version("rfdetr"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_tag": gpu_tag(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    _write_engine_and_record(engine_path, blob, record)
    return record


def _write_engine_and_record(engine_path: str, blob: Any, record: dict[str, Any]) -> None:
    """Write both files under temporary names next to their target, then move the record and last the engine.

    A half-written engine is never visible under its name, and a visible engine always has its record.
    """
    folder = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.basename(engine_path)
    contents = ((".json", json.dumps(record, indent=1).encode("utf-8")), ("", blob))
    moves: list[tuple[str, str]] = []
    try:
        for suffix, data in contents:
            fd, tmp = tempfile.mkstemp(dir=folder, prefix=f"{stem}{suffix}.", suffix=".tmp")
            moves.append((tmp, f"{engine_path}{suffix}"))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        for tmp, final in moves:  # the record first, the engine last
            os.replace(tmp, final)
    finally:
        for tmp, _ in moves:
            if os.path.exists(tmp):
                os.remove(tmp)


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
    name = engine_file_name(precision, int(model.model.resolution))
    if not force:
        found = find_engine(name, checkpoint, engine_dir)
        if found is not None:
            print(f"exists: {found}", flush=True)
            return found
    path = build_missing_engine(
        model,
        precision,
        name,
        checkpoint_path=checkpoint,
        engine_dir=engine_dir,
        variant=variant,
        force=force,
    )
    print(f"built: {path}", flush=True)
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
    one.add_argument(
        "--engine-dir",
        default=None,
        help="default: <checkpoint>.trt/ (the engine goes into its <fingerprint>/ subfolder)",
    )
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
