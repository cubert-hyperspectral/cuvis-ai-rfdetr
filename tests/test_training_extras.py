"""Tests for EmaCallback and RFDETRTrainable.get_param_groups (no model build).

The EMA tests drive the callback through fake trainer/pl_module objects with a tiny
linear module, asserting the averaged weights against rfdetr's own schedule
(decay * (1 - exp(-updates / tau))). The param-group test bypasses the heavy
constructor via ``__new__`` and monkeypatches rfdetr's ``get_param_dict`` to verify
the config-override plumbing.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from cuvis_ai_rfdetr.training import EmaCallback

try:  # rfdetr.training pulls the full train stack (faster_coco_eval, ...)
    import rfdetr.training.model_ema  # noqa: F401

    TRAIN_STACK = True
except ImportError:  # pragma: no cover
    TRAIN_STACK = False

needs_train_stack = pytest.mark.skipif(
    not TRAIN_STACK, reason="needs the rfdetr train stack (pip install 'rfdetr[train]')"
)


class _Node(torch.nn.Module):
    """Minimal stand-in for RFDETRTrainable: a named node with a .model."""

    def __init__(self, name: str = "RFDETR") -> None:
        super().__init__()
        self.name = name
        self.model = torch.nn.Linear(2, 2, bias=False)


def _fake_pl(node: _Node):
    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace()
    return trainer, pl_module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decay": 0.0},
        {"decay": 1.5},
        {"tau": -1.0},
        {"update_interval": 0},
    ],
)
def test_invalid_args_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EmaCallback(**kwargs)


def test_resolves_node_by_name_and_requires_model() -> None:
    cb = EmaCallback(node_name="missing")
    _, pl_module = _fake_pl(_Node("RFDETR"))
    with pytest.raises(RuntimeError, match="no node named"):
        cb._resolve_node(pl_module)

    bare = SimpleNamespace(name="RFDETR")  # no .model
    pl_module = SimpleNamespace(pipeline=SimpleNamespace(nodes=[bare]))
    with pytest.raises(RuntimeError, match="no `.model`"):
        EmaCallback()._resolve_node(pl_module)


@needs_train_stack
def test_ema_matches_native_schedule() -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(decay=0.9, tau=2.0)
    cb.on_fit_start(trainer, pl_module)

    # EMA starts as a copy of the model
    assert torch.equal(cb.ema_module.weight, node.model.weight)

    with torch.no_grad():
        node.model.weight += 1.0
    expected = cb.ema_module.weight.clone()
    for updates in (1, 2, 3):
        d = 0.9 * (1 - math.exp(-updates / 2.0))  # rfdetr: decay*(1-exp(-updates/tau))
        expected = d * expected + (1 - d) * node.model.weight
        cb.on_train_batch_end(trainer, pl_module, None, None, updates - 1)
    assert torch.allclose(cb.ema_module.weight, expected, atol=1e-7)


@needs_train_stack
def test_update_interval_gates_updates() -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(update_interval=3, decay=0.5, tau=0.0)
    cb.on_fit_start(trainer, pl_module)
    with torch.no_grad():
        node.model.weight += 1.0
    before = cb.ema_module.weight.clone()
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    cb.on_train_batch_end(trainer, pl_module, None, None, 1)
    assert torch.equal(cb.ema_module.weight, before)  # gated: 2 < interval
    cb.on_train_batch_end(trainer, pl_module, None, None, 2)
    assert not torch.equal(cb.ema_module.weight, before)  # third batch updates


@needs_train_stack
def test_state_roundtrip_restores_average(tmp_path) -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(decay=0.9, tau=0.0, save_path=str(tmp_path / "ema.pth"))
    cb.on_fit_start(trainer, pl_module)
    with torch.no_grad():
        node.model.weight += 1.0
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    state = cb.state_dict()

    # fresh callback + fresh (different) model: resume must restore the average
    node2 = _Node()
    trainer2, pl_module2 = _fake_pl(node2)
    cb2 = EmaCallback(decay=0.9, tau=0.0)
    cb2.load_state_dict(state)
    cb2.on_fit_start(trainer2, pl_module2)
    assert torch.allclose(cb2.ema_module.weight, cb.ema_module.weight)
    assert cb2.ema.updates == cb.ema.updates
    assert cb2._batches_seen == cb._batches_seen

    # save() writes the documented payload
    cb.on_fit_end(trainer, pl_module)
    payload = torch.load(tmp_path / "ema.pth", weights_only=True)
    assert set(payload) == {"model", "updates"}


@needs_train_stack
def test_get_param_groups_overrides_config(monkeypatch) -> None:
    from cuvis_ai_rfdetr.node import rfdetr_trainable as mod

    node = mod.RFDETRTrainable.__new__(mod.RFDETRTrainable)  # bypass heavy __init__
    train_cfg = SimpleNamespace(
        lr=1e-4, lr_encoder=1.5e-4, lr_vit_layer_decay=1.0, lr_component_decay=1.0
    )
    model_cfg = SimpleNamespace(out_feature_indexes=[2, 5, 8, 11], resolution=432)
    node_model = torch.nn.Linear(2, 2)
    # __init__ was bypassed, so set attrs without nn.Module.__setattr__ machinery
    object.__setattr__(node, "_train_config", train_cfg)
    object.__setattr__(node, "_model_config", model_cfg)
    object.__setattr__(node, "model", node_model)

    captured = {}

    def fake_get_param_dict(cfg, model):
        captured["cfg"] = cfg
        captured["model"] = model
        return [{"params": list(model.parameters()), "lr": cfg.lr}]

    import rfdetr.training.param_groups as pg

    monkeypatch.setattr(pg, "get_param_dict", fake_get_param_dict)

    groups = mod.RFDETRTrainable.get_param_groups(
        node, lr=2e-4, lr_vit_layer_decay=0.8, lr_component_decay=0.7
    )
    assert captured["model"] is node_model
    assert captured["cfg"].lr == 2e-4
    assert captured["cfg"].lr_encoder == 1.5e-4  # not overridden -> config value
    assert captured["cfg"].lr_vit_layer_decay == 0.8
    assert captured["cfg"].lr_component_decay == 0.7
    # model-config fields are merged into the flat args (native-loop shape)
    assert captured["cfg"].out_feature_indexes == [2, 5, 8, 11]
    # the original stored configs are untouched (merge builds a new namespace)
    assert node._train_config.lr == 1e-4
    assert node._model_config.resolution == 432
    assert groups[0]["lr"] == 2e-4
