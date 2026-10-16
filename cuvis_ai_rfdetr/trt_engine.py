"""TensorRT engines for the RF-DETR segmentation network: build them per machine, run them on torch CUDA tensors.

The network is exported with rfdetr's own ONNX exporter (``model.export()``: static ``1 x 3 x resolution x
resolution`` input ``input``, outputs ``dets`` / ``labels`` / ``masks``) and compiled by the installed TensorRT into an
engine for THIS GPU and TensorRT version. Where an engine lives, how it is built once per machine and how it runs
follow cuvis-ai-core's ``cuvis_ai_core.utils.trt_engine`` (core 0.18.5), the helper every TensorRT node shares; this
module keeps what is specific to RF-DETR: the ONNX export, the resolution in the file name, the checkpoint as the
weights an engine is keyed by, and the build command.

Engine names carry the precision, the resolution, the GPU and the TensorRT version, so one folder holds the engines of
several machines side by side (``fp16_r504_NVIDIA-Thor-sm110_trt10.15.1.29.engine``, ...). A JSON build record next to
each engine records how it was built, including the MD5 of the checkpoint; an engine whose record names another
checkpoint is refused.

Where engines live: next to their checkpoint by default, in ``<checkpoint>.trt/<fingerprint>/``, where the
fingerprint is the start of the checkpoint's MD5, so retrained weights saved under the same name get their own
engines. When that folder cannot be written: ``<model cache>/trt_engines/rfdetr/<fingerprint>/`` (the cuvis-ai model
cache, ``$CUVIS_MODEL_CACHE_DIR``, or ``$CUVIS_AI_TRT_ENGINE_DIR/rfdetr``). An explicit ``engine_dir`` takes the place
of ``<checkpoint>.trt``. Engines of 0.5.1 and earlier, directly in ``<checkpoint>.trt/`` (or ``engine_dir``), are
still found.

A pipeline loader builds a missing engine on the machine itself (``RFDETRSegmenter.prepare_inference``, the
cuvis-ai-core hook): once per checkpoint, precision and GPU, under a file lock per engine, so concurrent loads build
it once.

Precision: ``fp32`` is TensorRT's default float build, which allows TF32 tensor-core math - like PyTorch after
``import rfdetr``, which sets ``torch.set_float32_matmul_precision("high")``. ``fp16`` sets TensorRT's FP16 builder
flag (mixed precision: TensorRT keeps a layer in fp32 where that is faster or needed); TensorRT 11 dropped that flag,
so fp16 engines need TensorRT 10.

Optional dependencies, not installed with the plugin: the TensorRT Python package matching torch's CUDA
(``tensorrt-cu12`` / ``tensorrt-cu13``, version 10; the ``tensorrt`` extra) and, for building, ``onnx``. Build engines
with::

    python -m cuvis_ai_rfdetr.trt_engine build --checkpoint W.pth --variant large --resolution 504 --precision fp16
    python -m cuvis_ai_rfdetr.trt_engine build-pipeline pipeline.yaml [more.yaml ...]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib.metadata
import os
import tempfile
import time
from typing import Any

import torch
from cuvis_ai_core.utils import trt_engine as core
from cuvis_ai_core.utils.trt_engine import (
    FINGERPRINT_CHARS,
    EngineLocation,
    TensorRTEngine,
    file_md5,
    gpu_tag,
)
from cuvis_ai_core.utils.trt_engine import cached_file_md5 as checkpoint_md5
from cuvis_ai_core.utils.trt_engine import file_fingerprint as weights_fingerprint

__all__ = [
    "FINGERPRINT_CHARS",
    "NAMESPACE",
    "PRECISIONS",
    "EngineLocation",
    "TensorRTEngine",
    "build_command",
    "build_engine",
    "build_missing_engine",
    "check_engine_matches",
    "checkpoint_md5",
    "compile_network",
    "default_engine_dir",
    "engine_file_name",
    "engine_location",
    "engines_for_pipeline",
    "export_onnx",
    "file_md5",
    "find_engine",
    "gpu_tag",
    "main",
    "weights_fingerprint",
]

PRECISIONS = ("fp32", "fp16")
#: The plugin's folder under the engine root of cuvis-ai-core (``<model cache>/trt_engines/rfdetr``).
NAMESPACE = "rfdetr"


def _tensorrt() -> Any:
    """The ``tensorrt`` module (version 10), or an ImportError that says how to install it."""
    return core.import_tensorrt("RFDETRSegmenter backend='tensorrt'")


def engine_file_name(
    precision: str, resolution: int, device: torch.device | str | int | None = None
) -> str:
    """This machine's engine file for ``precision`` / ``resolution``.

    ``<precision>_r<resolution>_<gpu tag>_trt<TensorRT version>.engine``.
    """
    return core.engine_file_name(precision, f"r{int(resolution)}", device=device, trt=_tensorrt())


def default_engine_dir(checkpoint_path: str | None) -> str:
    """Engines live next to their checkpoint by default: ``<checkpoint>.trt/``."""
    if checkpoint_path is None:
        raise ValueError("RFDETRSegmenter backend='tensorrt' needs checkpoint_path or engine_dir.")
    return f"{checkpoint_path}.trt"


def _has_checkpoint(checkpoint_path: str | None) -> bool:
    return checkpoint_path is not None and os.path.isfile(checkpoint_path)


def _expected(checkpoint_path: str | None) -> dict[str, Any]:
    """What an engine's build record must say: the MD5 of the checkpoint, when there is one."""
    return (
        {"checkpoint_md5": checkpoint_md5(checkpoint_path)}
        if _has_checkpoint(checkpoint_path)
        else {}
    )


def check_engine_matches(engine_path: str, checkpoint_path: str | None) -> None:
    """Refuse an engine whose build record says it was compiled from a different checkpoint."""
    core.check_engine_record(engine_path, _expected(checkpoint_path))


def engine_location(
    file_name: str, checkpoint_path: str | None, engine_dir: str | None = None
) -> EngineLocation:
    """Where the engine ``file_name`` of a checkpoint is looked up and built (core's :class:`EngineLocation`).

    Lookup: ``<engine folder>/<fingerprint>/``, ``<engine folder>/`` (0.5.1), ``<engine root>/<fingerprint>/``,
    ``<engine root>/``; the engine folder is ``engine_dir`` or ``<checkpoint>.trt``. Without a checkpoint file there is
    nothing to fingerprint: the flat folders are searched and nothing is built.
    """
    fingerprint = weights_fingerprint(checkpoint_path) if _has_checkpoint(checkpoint_path) else None
    return core.engine_location(
        file_name,
        NAMESPACE,
        fingerprint,
        preferred_dir=engine_dir or default_engine_dir(checkpoint_path),
    )


def find_engine(
    file_name: str, checkpoint_path: str | None, engine_dir: str | None = None
) -> str | None:
    """The first existing engine of :func:`engine_location`, checked against its checkpoint; ``None`` if none.

    An engine whose build record names another checkpoint is refused (:func:`check_engine_matches`), not skipped.
    """
    found = core.find_engine(
        engine_location(file_name, checkpoint_path, engine_dir), _expected(checkpoint_path)
    )
    return None if found is None else str(found)


def build_command(
    checkpoint_path: str | None,
    variant: str | None,
    resolution: int,
    precision: str,
    *,
    checkpoint_loader: str = "constructor",
    engine_dir: str | None = None,
) -> str:
    """The command that builds one engine by hand."""
    command = (
        f"python -m cuvis_ai_rfdetr.trt_engine build --checkpoint {checkpoint_path} "
        f"--variant {variant} --resolution {int(resolution)} --precision {precision}"
    )
    if checkpoint_loader != "constructor":
        command += f" --checkpoint-loader {checkpoint_loader}"
    if engine_dir:
        command += f" --engine-dir {engine_dir}"
    return command


def compile_network(
    model: Any,
    precision: str,
    *,
    checkpoint_path: str | None = None,
    variant: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Export the network of ``model`` to ONNX and compile it; the serialized engine and the RF-DETR build record.

    The record holds what is specific to RF-DETR (precision, TF32, resolution, variant, checkpoint and its MD5,
    rfdetr version); :func:`cuvis_ai_core.utils.trt_engine.build_engine_once` adds the engine name, the TensorRT,
    torch and GPU fields and the build time.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}.")
    trt = _tensorrt()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model, tmp)
        blob, info = core.build_serialized_network(onnx_path, fp16=precision == "fp16", trt=trt)
    record = {
        "precision": precision,
        "tf32": info["tf32"],
        "resolution": int(model.model.resolution),
        "variant": variant,
        "checkpoint": checkpoint_path,
        "checkpoint_md5": checkpoint_md5(checkpoint_path)
        if _has_checkpoint(checkpoint_path)
        else None,
        "rfdetr": _version("rfdetr"),
    }
    return blob, record


def build_engine(
    model: Any,
    precision: str,
    engine_path: str,
    *,
    checkpoint_path: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Build and save a TensorRT engine for the network of ``model`` at ``engine_path``; returns its build record.

    No lookup and no lock: this writes exactly ``engine_path`` (with its record, as one unit). A pipeline load uses
    :func:`build_missing_engine` instead.
    """
    t0 = time.perf_counter()
    blob, plugin_record = compile_network(
        model, precision, checkpoint_path=checkpoint_path, variant=variant
    )
    trt = _tensorrt()
    record = {
        "engine": os.path.basename(engine_path),
        **plugin_record,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "gpu_tag": gpu_tag(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    core.write_engine_and_record(engine_path, blob, record)
    return record


def build_missing_engine(
    model: Any,
    precision: str,
    file_name: str,
    *,
    checkpoint_path: str,
    engine_dir: str | None = None,
    variant: str | None = None,
    force: bool = False,
    command: str | None = None,
) -> str:
    """Build the engine ``file_name`` where :func:`find_engine` looks first, unless another process already has.

    :func:`cuvis_ai_core.utils.trt_engine.build_engine_once`: the first writable folder of :func:`engine_location`,
    one file lock per engine, engine and record written as one unit, start and end logged, and one error that names
    ``command`` (default: :func:`build_command` for these arguments). ``force`` rebuilds an existing engine.
    """
    if command is None:
        command = build_command(
            checkpoint_path,
            variant,
            int(model.model.resolution),
            precision,
            engine_dir=engine_dir,
        )
    path = core.build_engine_once(
        engine_location(file_name, checkpoint_path, engine_dir),
        file_name,
        lambda: compile_network(model, precision, checkpoint_path=checkpoint_path, variant=variant),
        command=command,
        expected=_expected(checkpoint_path),
        force=force,
    )
    return str(path)


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
