"""``rfdetr_adamw``: RF-DETR's optimizer setup for cuvis-ai training runs.

RF-DETR's trainer gives its weights per-layer learning rates (encoder at
``lr_encoder`` with ViT layer decay, decoder at ``lr * lr_component_decay``) and keeps
an EMA of the weights. cuvis-ai-core (0.19, 0.20) builds one optimizer over all
trainable pipeline parameters at a single learning rate and has no hook for a node to
contribute learning-rate groups or callbacks. Until it has one, this module registers an
extra optimizer name in core's optimizer registry, so a trainrun can select it with

    optimizer:
      name: rfdetr_adamw
      lr: 0.0001
      weight_decay: 0.0001

``RFDETRAdamW`` is ``torch.optim.AdamW`` over RF-DETR's native parameter groups (taken
from every :class:`~cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable` whose weights
it receives) and, after every optimizer step, applies the node's EMA update when the node
has EMA enabled. Parameters of other nodes join at the base learning rate.

The registration relies on ``cuvis_ai_core.training.optimizer_registry.SUPPORTED_OPTIMIZERS``
being a plain dict. It is meant to be removed once core accepts learning-rate groups and
EMA from the training config.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

#: Name of the optimizer in core's registry.
OPTIMIZER_NAME = "rfdetr_adamw"

_LIVE_TRAINABLES: weakref.WeakSet = weakref.WeakSet()


def track_trainable(node: Any) -> None:
    """Remember a constructed training node so ``RFDETRAdamW`` can find its parameters."""
    _LIVE_TRAINABLES.add(node)


class RFDETRAdamW(torch.optim.AdamW):
    """AdamW over RF-DETR's native parameter groups, with the nodes' EMA update after each step."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        **kwargs: Any,
    ) -> None:
        params = list(params)
        if params and isinstance(params[0], dict):  # already grouped by the caller
            super().__init__(params, lr=lr, weight_decay=weight_decay, betas=betas, **kwargs)
            self._ema_nodes: list[weakref.ref] = []
            return
        offered = {id(p) for p in params}
        groups: list[dict[str, Any]] = []
        claimed: set[int] = set()
        ema_nodes: list[weakref.ref] = []
        for node in list(_LIVE_TRAINABLES):
            node_params = [p for p in node.model.parameters() if p.requires_grad]
            if not node_params or any(id(p) not in offered for p in node_params):
                continue  # this node's weights are not being trained by this optimizer
            for group in node.get_param_groups(lr=lr, **node.optimizer_overrides()):
                members = group["params"]
                members = list(members) if isinstance(members, (list, tuple)) else [members]
                members = [p for p in members if id(p) in offered and id(p) not in claimed]
                if not members:
                    continue
                claimed.update(id(p) for p in members)
                groups.append({**group, "params": members})
            if getattr(node, "ema_model", None) is not None:
                ema_nodes.append(weakref.ref(node))
        rest = [p for p in params if id(p) not in claimed]
        if rest:
            groups.append({"params": rest})
        super().__init__(groups, lr=lr, weight_decay=weight_decay, betas=betas, **kwargs)
        self._ema_nodes = ema_nodes

    def step(self, closure: Any = None) -> Any:
        loss = super().step(closure)
        for ref in self._ema_nodes:
            node = ref()
            if node is not None:
                node.update_ema()
        return loss


def register_optimizer() -> bool:
    """Add ``rfdetr_adamw`` to core's optimizer registry; ``False`` when core is absent."""
    try:
        from cuvis_ai_core.training import optimizer_registry
    except ImportError:  # pragma: no cover - core is a hard dependency of the plugin
        return False
    registry = getattr(optimizer_registry, "SUPPORTED_OPTIMIZERS", None)
    if not isinstance(registry, dict):
        return False
    registry.setdefault(
        OPTIMIZER_NAME,
        {
            "class": RFDETRAdamW,
            "description": "AdamW over RF-DETR's native learning-rate groups, with the "
            "training node's EMA update after each step (cuvis-ai-rfdetr)",
            "params": {
                "lr": {"required": True, "type": "float"},
                "weight_decay": {"required": False, "default": 0.0, "type": "float"},
            },
        },
    )
    return True
