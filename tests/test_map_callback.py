"""Tests for MapEvalCallback: selection policy, EMA weight-swap, and mAP evaluation.

All CI-runnable: the postprocess is injected (no rfdetr stack), the EMA uses the
ModelEma-compatible stub, and mAP comes from torchmetrics (a lightning dependency).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cuvis_ai_rfdetr.training import EmaCallback, MapEvalCallback, _cxcywh_norm_to_xyxy_abs
from tests.test_training_extras import _Node, _StubEma


def _rig(tmp_path, node=None):
    """A wired (callback, trainer, pl_module, node) rig with a live stub EMA."""
    node = node or _Node()
    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    ema = EmaCallback(node_name=node.name, decay=0.5, tau=0.0, ema_cls=_StubEma)
    trainer = SimpleNamespace(
        callbacks=[ema],
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: []),
    )
    ema.on_fit_start(trainer, pl_module)
    cb = MapEvalCallback(
        node_name=node.name, output_dir=str(tmp_path), postprocess=lambda raw, sizes: raw
    )
    cb.on_fit_start(trainer, pl_module)
    return cb, trainer, pl_module, node


def test_best_checkpoint_policy(tmp_path, monkeypatch) -> None:
    cb, trainer, pl_module, _ = _rig(tmp_path)
    script = iter([0.3, 0.2, 0.25, 0.4, 0.35, 0.1])  # (regular, ema) per epoch
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: next(script))

    for epoch in range(3):
        trainer.current_epoch = epoch
        cb.on_validation_epoch_end(trainer, pl_module)

    assert cb.best["regular"] == {"map": 0.35, "epoch": 2}
    assert cb.best["ema"] == {"map": 0.4, "epoch": 1}
    assert cb.best["total"] == {"map": 0.4, "epoch": 1, "stream": "ema"}

    regular = torch.load(tmp_path / "checkpoint_best_regular.pth", weights_only=True)
    ema = torch.load(tmp_path / "checkpoint_best_ema.pth", weights_only=True)
    total = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=True)
    assert (regular["map"], regular["epoch"], regular["stream"]) == (0.35, 2, "regular")
    assert (ema["map"], ema["epoch"], ema["stream"]) == (0.4, 1, "ema")
    # best_total is the 0.4 EMA epoch — NOT overwritten by the later 0.35 regular
    assert (total["map"], total["epoch"], total["stream"]) == (0.4, 1, "ema")
    assert set(total["model"]) == set(regular["model"])


def test_sanity_checking_is_skipped(tmp_path, monkeypatch) -> None:
    cb, trainer, pl_module, _ = _rig(tmp_path)
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 1.0)
    trainer.sanity_checking = True
    cb.on_validation_epoch_end(trainer, pl_module)
    assert cb.history == [] and cb.best == {}


def test_ema_weights_swapped_in_and_restored(tmp_path) -> None:
    cb, trainer, pl_module, node = _rig(tmp_path)
    with torch.no_grad():
        node.model.weight += 1.0  # model has drifted; EMA still holds the init copy
    regular_w = node.model.weight.detach().clone()
    ema_w = cb._ema.ema_module.weight.detach().clone()
    assert not torch.equal(regular_w, ema_w)

    seen: list[torch.Tensor] = []

    def spy_evaluate(*_a, **_k) -> float:
        seen.append(node.model.weight.detach().clone())
        return 0.5

    cb._evaluate = spy_evaluate
    cb.on_validation_epoch_end(trainer, pl_module)
    assert len(seen) == 2
    assert torch.equal(seen[0], regular_w)  # first pass: regular weights
    assert torch.equal(seen[1], ema_w)  # second pass: EMA weights swapped in
    assert torch.equal(node.model.weight, regular_w)  # restored afterwards


def test_evaluate_perfect_predictions_score_map_one(tmp_path) -> None:
    node = _Node()
    object.__setattr__(node, "_input_resolution", 100)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.3, 0.1, 0.1]]),
            "labels": torch.tensor([0, 0]),
        }
    ]

    def perfect_postprocess(raw, sizes):
        assert raw == "RAW_OUTPUTS"
        assert sizes.tolist() == [[100, 100]]
        return [
            {
                "boxes": _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": targets[0]["labels"].clone(),
            }
        ]

    pipeline = SimpleNamespace(
        nodes=[node],
        forward=lambda batch, context: {
            (node.name, "outputs"): "RAW_OUTPUTS",
            (node.name, "targets"): targets,
        },
    )
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: [{"data": torch.zeros(1)}]),
    )
    cb = MapEvalCallback(
        node_name=node.name, output_dir=str(tmp_path), postprocess=perfect_postprocess
    )
    cb.on_fit_start(trainer, pl_module)
    assert cb._evaluate(trainer, pl_module) == pytest.approx(1.0)


def test_evaluate_wrong_boxes_score_below_one(tmp_path) -> None:
    node = _Node()
    object.__setattr__(node, "_input_resolution", 100)
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])}]

    def offset_postprocess(raw, sizes):
        boxes = _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100) + 15.0  # shifted
        return [{"boxes": boxes, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]

    pipeline = SimpleNamespace(
        nodes=[node],
        forward=lambda batch, context: {
            (node.name, "outputs"): None,
            (node.name, "targets"): targets,
        },
    )
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: [{"data": torch.zeros(1)}]),
    )
    cb = MapEvalCallback(
        node_name=node.name, output_dir=str(tmp_path), postprocess=offset_postprocess
    )
    cb.on_fit_start(trainer, pl_module)
    assert cb._evaluate(trainer, pl_module) < 1.0


def test_cxcywh_conversion() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    out = _cxcywh_norm_to_xyxy_abs(boxes, 100)
    assert torch.allclose(out, torch.tensor([[40.0, 30.0, 60.0, 70.0]]))
    assert _cxcywh_norm_to_xyxy_abs(torch.zeros(0, 4), 100).shape == (0, 4)
