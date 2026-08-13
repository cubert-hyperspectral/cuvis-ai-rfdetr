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

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from cuvis_ai_core.training import GradientTrainer

__all__ = ["EmaCallback", "RFDETRGradientTrainer"]


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
