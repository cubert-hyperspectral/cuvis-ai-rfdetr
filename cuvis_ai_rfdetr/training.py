"""Training-loop extras for RF-DETR nodes: the native EMA as a Lightning callback.

RF-DETR's own trainer keeps an exponential moving average of the model weights and
selects its best checkpoints from BOTH the regular and the EMA weights — the
published Roboflow checkpoints are typically the EMA ones. ``EmaCallback`` brings
that ingredient to cuvis-ai's ``GradientTrainer``, which already accepts explicit
Lightning callbacks::

    trainable = RFDETRTrainable(name="RFDETR", dataset_dir=..., segmentation=True)
    ema = EmaCallback(node_name="RFDETR", decay=0.993, tau=100)
    GradientTrainer(pipeline=pipeline, datamodule=dm, loss_nodes=[loss],
                    training_config=cfg, callbacks=[ema]).fit()
    ema.save("rfdetr_ema.pth")               # or pass save_path= to save on fit end

The EMA math is rfdetr's own ``ModelEma`` (decay warm-up ``decay * (1 -
exp(-updates / tau))``), applied to the ``RFDETRTrainable``'s registered LW-DETR
module — so a cuvis-ai run reproduces the native loop's averaging exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from cuvis_ai_core.training import GradientTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

__all__ = ["EmaCallback", "MapEvalCallback", "RFDETRGradientTrainer"]
# (MapEvalCallback is defined after EmaCallback, which it discovers among the
# trainer callbacks to evaluate the EMA weight stream.)


def _find_node(pipeline: Any, node_name: str):
    """Return the pipeline node called ``node_name`` or raise a clear error."""
    for node in pipeline.nodes:
        if getattr(node, "name", None) == node_name:
            return node
    raise RuntimeError(f"no node named {node_name!r} in the pipeline.")


class RFDETRGradientTrainer(GradientTrainer):
    """``GradientTrainer`` whose optimizer uses the native RF-DETR param groups.

    ``configure_optimizers`` asks the named ``RFDETRTrainable`` for its native
    param groups (:meth:`RFDETRTrainable.get_param_groups` — encoder at
    ``lr_encoder`` with ViT layer decay, decoder at ``lr * lr_component_decay``,
    rest at the base lr) and feeds them to the standard optimizer/scheduler
    registry, so everything else (optimizer type, weight decay, scheduler,
    callbacks) behaves exactly like the base trainer. Any *other* unfrozen
    pipeline parameters not covered by the node's groups are appended as a
    final base-lr group, preserving the base trainer's "optimize everything
    trainable" contract.

    The base learning rate is taken from ``training_config.optimizer.lr``; the
    three structural knobs are constructor arguments (defaults = the wrapper's
    train config; the BonBack champion trained with ``lr_encoder=1.5e-4,
    lr_vit_layer_decay=0.8, lr_component_decay=0.7``).

    This lives in the plugin (not cuvis-ai-core) by design for now — candidate
    for later migration into ``GradientTrainer`` as an optional node
    param-group protocol.
    """

    def __init__(
        self,
        *args: Any,
        node_name: str = "RFDETR",
        lr_encoder: float | None = None,
        lr_vit_layer_decay: float | None = None,
        lr_component_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.node_name = str(node_name)
        self._group_overrides = {
            "lr_encoder": lr_encoder,
            "lr_vit_layer_decay": lr_vit_layer_decay,
            "lr_component_decay": lr_component_decay,
        }

    def configure_optimizers(self):
        from cuvis_ai_core.training.optimizer_registry import (
            create_optimizer,
            create_scheduler,
            wrap_scheduler_for_lightning,
        )

        node = _find_node(self.pipeline, self.node_name)
        if not hasattr(node, "get_param_groups"):
            raise RuntimeError(
                f"RFDETRGradientTrainer: node {self.node_name!r} has no "
                "get_param_groups — expected an RFDETRTrainable."
            )
        groups = node.get_param_groups(lr=self.optimizer_config.lr, **self._group_overrides)
        # Preserve the base contract: every trainable pipeline parameter is
        # optimized. Anything outside the node's groups joins at the base lr.
        grouped = {
            id(p)
            for g in groups
            for p in (g["params"] if isinstance(g["params"], (list, tuple)) else [g["params"]])
        }
        extras = [p for p in self.pipeline.parameters() if p.requires_grad and id(p) not in grouped]
        if extras:
            groups = [*groups, {"params": extras}]

        optimizer = create_optimizer(self.optimizer_config, groups)
        scheduler = create_scheduler(
            self.scheduler_config, optimizer, self.training_config.max_epochs
        )
        if scheduler is None:
            return optimizer
        monitor = self.scheduler_config.monitor if self.scheduler_config else None
        return {
            "optimizer": optimizer,
            "lr_scheduler": wrap_scheduler_for_lightning(scheduler, monitor),
        }


class EmaCallback(pl.Callback):
    """Exponential moving average of an ``RFDETRTrainable``'s weights.

    Parameters
    ----------
    node_name : str
        Name of the ``RFDETRTrainable`` node inside the trained pipeline.
    decay, tau : float
        Native EMA schedule: effective decay is ``decay * (1 - exp(-updates/tau))``
        (``tau=0`` means a constant ``decay``). Defaults match RF-DETR's trainer
        defaults for fine-tuning (decay ``0.993``, tau ``100``).
    update_interval : int
        Update the average every N optimizer batches (native ``ema_update_interval``).
    save_path : str or None
        When set, the EMA weights are written there at ``fit`` end (same payload
        as :meth:`save`).
    ema_cls : type or None
        EMA implementation (``ModelEma``-compatible: ``module``/``updates``
        attributes, ``update(model)``). Defaults to rfdetr's native ``ModelEma``
        (lazy import). Injection point for tests and alternative schedules.
    """

    def __init__(
        self,
        node_name: str = "RFDETR",
        decay: float = 0.993,
        tau: float = 100.0,
        update_interval: int = 1,
        save_path: str | None = None,
        ema_cls: type | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError(f"EmaCallback: decay must be in (0, 1], got {decay!r}")
        if float(tau) < 0.0:
            raise ValueError(f"EmaCallback: tau must be >= 0, got {tau!r}")
        if int(update_interval) < 1:
            raise ValueError(f"EmaCallback: update_interval must be >= 1, got {update_interval!r}")
        self.node_name = str(node_name)
        self.decay = float(decay)
        self.tau = float(tau)
        self.update_interval = int(update_interval)
        self.save_path = str(save_path) if save_path is not None else None
        self.ema_cls = ema_cls
        self.ema = None  # EMA instance, built on fit start
        self._node = None
        self._batches_seen = 0
        self._pending_state: dict[str, Any] | None = None  # restored ckpt state

    # ------------------------------------------------------------- resolution
    def _resolve_node(self, pl_module: pl.LightningModule):
        pipeline = getattr(pl_module, "pipeline", None)
        if pipeline is None:
            raise RuntimeError(
                "EmaCallback expects the LightningModule to expose `.pipeline` "
                "(cuvis-ai GradientTrainer does)."
            )
        try:
            node = _find_node(pipeline, self.node_name)
        except RuntimeError as exc:
            raise RuntimeError(f"EmaCallback: {exc}") from None
        if not hasattr(node, "model"):
            raise RuntimeError(
                f"EmaCallback: node {self.node_name!r} has no `.model` "
                "submodule — expected an RFDETRTrainable."
            )
        return node

    # ------------------------------------------------------------- lifecycle
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ema_cls = self.ema_cls
        if ema_cls is None:
            try:  # lazy: importing rfdetr.training pulls the full train stack
                from rfdetr.training.model_ema import ModelEma as ema_cls
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "EmaCallback requires the rfdetr train stack: "
                    "pip install 'rfdetr[train]>=1.8,<2'"
                ) from exc

        self._node = self._resolve_node(pl_module)
        self.ema = ema_cls(self._node.model, decay=self.decay, tau=self.tau)
        if self._pending_state is not None:  # resume: restore averaged weights
            self.ema.module.load_state_dict(self._pending_state["ema_state"])
            self.ema.updates = int(self._pending_state["updates"])
            self._batches_seen = int(self._pending_state["batches_seen"])
            self._pending_state = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batches_seen += 1
        if self._batches_seen % self.update_interval == 0:
            self.ema.update(self._node.model)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.save_path is not None:
            self.save(self.save_path)

    # ------------------------------------------------------------- state / IO
    @property
    def ema_module(self) -> torch.nn.Module:
        """The averaged LW-DETR module (eval mode); None-safe only after fit start."""
        if self.ema is None:
            raise RuntimeError("EmaCallback: EMA not initialized yet (fit not started).")
        return self.ema.module

    def state_dict(self) -> dict[str, Any]:
        if self.ema is None:
            return self._pending_state or {}
        return {
            "ema_state": self.ema.module.state_dict(),
            "updates": int(self.ema.updates),
            "batches_seen": int(self._batches_seen),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Lightning restores callback state before `on_fit_start`; the EMA module
        # does not exist yet, so stash and apply once it is built.
        if state_dict:
            self._pending_state = state_dict

    def save(self, path: str) -> None:
        """Write the EMA weights (``{"model": state_dict, "updates": int}``)."""
        payload = {"model": self.ema_module.state_dict(), "updates": int(self.ema.updates)}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)


def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """Normalized cxcywh (DETR targets) -> absolute xyxy at a square size."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * float(size)


class MapEvalCallback(pl.Callback):
    """Native RF-DETR best-checkpoint selection for cuvis-ai training runs.

    Each validation epoch, evaluates COCO mAP over the val dataloader for the
    **regular** weights and — when an :class:`EmaCallback` is present — for the
    **EMA** weights (swapped in for the pass, restored after), tracks the best
    value of each stream, and writes the native-style checkpoints:
    ``checkpoint_best_regular.pth`` / ``checkpoint_best_ema.pth`` /
    ``checkpoint_best_total.pth`` (best across both streams — the artifact the
    native loop selects its shipped weights from). A callback rather than a
    metric node deliberately: it must evaluate two weight sets per epoch, which
    a streaming metric node cannot.

    mAP is ``map@[.5:.95]`` from torchmetrics' ``MeanAveragePrecision`` with the
    ``faster_coco_eval`` backend (the evaluator family the native loop uses).
    Predictions come from rfdetr's own ``PostProcess`` (built from the node's
    stored model/train configs unless one is injected); ground truth from the
    node's DETR ``targets``. Both are scaled to the model input resolution, so
    COCO area buckets stay consistent across checkpoints.

    Parameters
    ----------
    node_name : str
        The ``RFDETRTrainable`` inside the trained pipeline.
    output_dir : str
        Where the three checkpoints are written.
    iou_type : str
        ``"bbox"`` (default) or ``"segm"`` (needs masks in the postprocess
        output).
    postprocess : callable or None
        ``(outputs, target_sizes) -> list[{boxes, scores, labels}]``. Default:
        rfdetr's ``PostProcess`` built at fit start (needs the train stack).
    map_backend : str
        torchmetrics backend; default ``faster_coco_eval``.
    """

    def __init__(
        self,
        node_name: str = "RFDETR",
        output_dir: str = ".",
        iou_type: str = "bbox",
        postprocess: Callable | None = None,
        map_backend: str = "faster_coco_eval",
    ) -> None:
        super().__init__()
        self.node_name = str(node_name)
        self.output_dir = Path(output_dir)
        self.iou_type = str(iou_type)
        self.postprocess = postprocess
        self.map_backend = str(map_backend)
        self._node = None
        self._ema: EmaCallback | None = None
        self.best: dict[str, dict[str, Any]] = {}  # stream -> {"map", "epoch", ...}
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------- lifecycle
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._node = _find_node(pl_module.pipeline, self.node_name)
        if self.postprocess is None:
            try:  # lazy: the train stack
                from rfdetr.models.lwdetr import build_criterion_from_config
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "MapEvalCallback needs rfdetr's PostProcess (pip install "
                    "'rfdetr[train]>=1.8,<2') or an injected `postprocess`."
                ) from exc
            _, self.postprocess = build_criterion_from_config(
                self._node._model_config, self._node._train_config
            )
        self._ema = next((cb for cb in trainer.callbacks if isinstance(cb, EmaCallback)), None)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch)

        self._record("regular", self._evaluate(trainer, pl_module), epoch)

        if self._ema is not None and self._ema.ema is not None:
            model = self._node.model
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(self._ema.ema_module.state_dict())
            try:
                map_ema = self._evaluate(trainer, pl_module)
            finally:
                model.load_state_dict(backup)
            self._record("ema", map_ema, epoch)

    # ------------------------------------------------------------- evaluation
    def _evaluate(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> float:
        from torchmetrics.detection import MeanAveragePrecision

        metric = MeanAveragePrecision(iou_type=self.iou_type, backend=self.map_backend)
        loader = trainer.datamodule.val_dataloader()
        pipeline = pl_module.pipeline
        device = next(self._node.model.parameters()).device
        res = int(getattr(self._node, "_input_resolution", 0)) or 1
        was_training = self._node.model.training
        self._node.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = {
                        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                    }
                    out = pipeline.forward(batch=batch, context=Context(stage=ExecutionStage.VAL))
                    raw = out[(self.node_name, "outputs")]
                    targets = out[(self.node_name, "targets")]
                    sizes = torch.tensor([[res, res]] * len(targets), device=device)
                    results = self.postprocess(raw, sizes)
                    preds, gts = [], []
                    for result, target in zip(results, targets, strict=True):
                        preds.append(
                            {
                                "boxes": result["boxes"].detach().cpu(),
                                "scores": result["scores"].detach().cpu(),
                                "labels": result["labels"].detach().cpu(),
                            }
                        )
                        gts.append(
                            {
                                "boxes": _cxcywh_norm_to_xyxy_abs(target["boxes"], res).cpu(),
                                "labels": target["labels"].detach().cpu(),
                            }
                        )
                    metric.update(preds, gts)
        finally:
            self._node.model.train(was_training)
        return float(metric.compute()["map"])

    # ------------------------------------------------------------- bookkeeping
    def _record(self, stream: str, value: float, epoch: int) -> None:
        """Strict-max per stream + across streams (native BestMetricHolder rule)."""
        self.history.append({"stream": stream, "map": value, "epoch": epoch})
        payload = None
        best_stream = self.best.get(stream)
        if best_stream is None or value > best_stream["map"]:
            self.best[stream] = {"map": value, "epoch": epoch}
            payload = self._payload(stream, value, epoch)
            torch.save(payload, self.output_dir / f"checkpoint_best_{stream}.pth")
        best_total = self.best.get("total")
        if best_total is None or value > best_total["map"]:
            self.best["total"] = {"map": value, "epoch": epoch, "stream": stream}
            payload = payload if payload is not None else self._payload(stream, value, epoch)
            torch.save(payload, self.output_dir / "checkpoint_best_total.pth")

    def _payload(self, stream: str, value: float, epoch: int) -> dict[str, Any]:
        model = self._node.model if stream == "regular" else self._ema.ema_module
        return {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "map": float(value),
            "epoch": int(epoch),
            "stream": stream,
        }
