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


try:  # the mAP backend MapEvalCallback defaults to is an optional extra
    import faster_coco_eval  # noqa: F401

    HAS_MAP_BACKEND = True
except ImportError:  # pragma: no cover
    HAS_MAP_BACKEND = False


def _eval_rig(tmp_path, targets, postprocess):
    node = _Node()
    object.__setattr__(node, "_input_resolution", 100)
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
    cb = MapEvalCallback(node_name=node.name, output_dir=str(tmp_path), postprocess=postprocess)
    cb.on_fit_start(trainer, pl_module)
    return cb, trainer, pl_module


def test_evaluate_plumbing_with_mocked_metric(tmp_path, monkeypatch) -> None:
    # Cover _evaluate everywhere (no MAP backend needed): assert it feeds the
    # postprocessed preds and the cxcywh->xyxy-converted GT to the metric, one
    # update per val batch, and returns metric.compute()["map"].
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.3, 0.1, 0.1]]),
            "labels": torch.tensor([0, 0]),
        },
    ]

    def postprocess(raw, sizes):
        assert raw == "RAW_OUTPUTS" and sizes.tolist() == [[100, 100]]
        return [
            {
                "boxes": _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": targets[0]["labels"].clone(),
            }
        ]

    captured = {"preds": [], "gts": [], "updates": 0}

    class FakeMAP:
        def __init__(self, iou_type, backend):
            captured["iou_type"], captured["backend"] = iou_type, backend

        def update(self, preds, gts):
            captured["preds"] += preds
            captured["gts"] += gts
            captured["updates"] += 1

        def compute(self):
            return {"map": torch.tensor(0.4242)}

    import torchmetrics.detection

    monkeypatch.setattr(torchmetrics.detection, "MeanAveragePrecision", FakeMAP)

    cb, trainer, pl_module = _eval_rig(tmp_path, targets, postprocess)
    result = cb._evaluate(trainer, pl_module)
    assert result == pytest.approx(0.4242)  # returns the metric's map
    assert captured["updates"] == 1  # one val batch
    assert captured["backend"] == "faster_coco_eval"  # node's default backend forwarded
    # GT boxes were converted cxcywh(norm) -> xyxy(abs @ res 100)
    assert torch.allclose(
        captured["gts"][0]["boxes"], _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100)
    )
    assert torch.equal(captured["preds"][0]["scores"], torch.tensor([0.9, 0.8]))


@pytest.mark.skipif(
    not HAS_MAP_BACKEND, reason="needs a torchmetrics MAP backend (faster-coco-eval)"
)
def test_evaluate_real_backend_perfect_and_wrong(tmp_path) -> None:
    # With the real backend present: perfect boxes -> mAP 1.0; shifted -> < 1.0.
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])}]
    cb, tr, plm = _eval_rig(
        tmp_path,
        targets,
        lambda raw, s: [
            {
                "boxes": _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ],
    )
    assert cb._evaluate(tr, plm) == pytest.approx(1.0)

    cb2, tr2, plm2 = _eval_rig(
        tmp_path,
        targets,
        lambda raw, s: [
            {
                "boxes": _cxcywh_norm_to_xyxy_abs(targets[0]["boxes"], 100) + 15.0,
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ],
    )
    assert cb2._evaluate(tr2, plm2) < 1.0


def test_cxcywh_conversion() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    out = _cxcywh_norm_to_xyxy_abs(boxes, 100)
    assert torch.allclose(out, torch.tensor([[40.0, 30.0, 60.0, 70.0]]))
    assert _cxcywh_norm_to_xyxy_abs(torch.zeros(0, 4), 100).shape == (0, 4)


def test_payload_is_self_describing_when_node_has_configs(tmp_path, monkeypatch) -> None:
    # With model/train configs + variant on the node, checkpoints must carry
    # model_name + a native-style flat args namespace (num_queries, group_detr,
    # resolution, ...) so rfdetr's loaders can rebuild the architecture instead
    # of falling back to a flat weight slice.
    node = _Node()
    object.__setattr__(
        node,
        "_model_config",
        SimpleNamespace(num_queries=200, group_detr=13, resolution=624, patch_size=12),
    )
    object.__setattr__(node, "_train_config", SimpleNamespace(expanded_scales=True, lr=1e-4))
    object.__setattr__(node, "variant", "medium")
    object.__setattr__(node, "segmentation", True)

    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],
        sanity_checking=False,
        current_epoch=3,
        datamodule=SimpleNamespace(val_dataloader=lambda: []),
    )
    cb = MapEvalCallback(node_name=node.name, output_dir=str(tmp_path), postprocess=lambda r, s: r)
    cb.on_fit_start(trainer, pl_module)
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 0.7)
    cb.on_validation_epoch_end(trainer, pl_module)

    # args is a pickled namespace -> weights_only=False, exactly like native ckpts
    payload = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=False)
    assert payload["model_name"] == "RFDETRSegMedium"
    assert payload["args"].num_queries == 200
    assert payload["args"].group_detr == 13
    assert payload["args"].resolution == 624
    assert payload["args"].expanded_scales is True  # train config merged in
    assert payload["stream"] == "regular" and payload["epoch"] == 3


# ------------------------------------------------------- MapEval edge cases
def test_without_ema_callback_only_regular_stream(tmp_path, monkeypatch) -> None:
    node = _Node()
    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],  # no EmaCallback attached
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: []),
    )
    cb = MapEvalCallback(node_name=node.name, output_dir=str(tmp_path), postprocess=lambda r, s: r)
    cb.on_fit_start(trainer, pl_module)
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 0.6)
    cb.on_validation_epoch_end(trainer, pl_module)

    assert [h["stream"] for h in cb.history] == ["regular"]  # no ema evaluation
    assert cb.best["total"]["stream"] == "regular"
    assert (tmp_path / "checkpoint_best_regular.pth").exists()
    assert not (tmp_path / "checkpoint_best_ema.pth").exists()


def test_equal_map_keeps_first_epoch(tmp_path, monkeypatch) -> None:
    # strict-max: a later EQUAL value must not overwrite the earlier checkpoint
    cb, trainer, pl_module, _ = _rig(tmp_path)
    script = iter([0.5, 0.1, 0.5, 0.1])  # regular repeats the same best at epoch 1
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: next(script))
    for epoch in range(2):
        trainer.current_epoch = epoch
        cb.on_validation_epoch_end(trainer, pl_module)
    assert cb.best["regular"] == {"map": 0.5, "epoch": 0}  # first epoch retained
    assert cb.best["total"]["epoch"] == 0


def test_evaluate_counts_one_update_per_val_batch(tmp_path, monkeypatch) -> None:
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    node = _Node()
    object.__setattr__(node, "_input_resolution", 50)
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
        datamodule=SimpleNamespace(
            val_dataloader=lambda: [
                {"d": torch.zeros(1)},
                {"d": torch.zeros(1)},
                {"d": torch.zeros(1)},
            ]
        ),
    )

    calls = {"updates": 0}

    class FakeMAP:
        def __init__(self, iou_type, backend): ...

        def update(self, preds, gts):
            calls["updates"] += 1

        def compute(self):
            return {"map": torch.tensor(0.0)}

    import torchmetrics.detection

    monkeypatch.setattr(torchmetrics.detection, "MeanAveragePrecision", FakeMAP)
    cb = MapEvalCallback(
        node_name=node.name,
        output_dir=str(tmp_path),
        postprocess=lambda r, s: [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            }
        ],
    )
    cb.on_fit_start(trainer, pl_module)
    cb._evaluate(trainer, pl_module)
    assert calls["updates"] == 3  # one metric update per val batch


def test_bare_node_payload_stays_weights_only_loadable(tmp_path, monkeypatch) -> None:
    # nodes without configs (back-compat) -> no args/model_name, still weights_only=True
    cb, trainer, pl_module, _ = _rig(tmp_path)  # _Node has no configs/variant
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 0.5)
    cb.on_validation_epoch_end(trainer, pl_module)
    payload = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=True)
    assert "args" not in payload and "model_name" not in payload


def test_detection_variant_model_name(tmp_path, monkeypatch) -> None:
    node = _Node()
    object.__setattr__(node, "_model_config", SimpleNamespace(num_queries=300))
    object.__setattr__(node, "_train_config", SimpleNamespace(lr=1e-4))
    object.__setattr__(node, "variant", "large")
    object.__setattr__(node, "segmentation", False)  # detection tier
    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: []),
    )
    cb = MapEvalCallback(node_name=node.name, output_dir=str(tmp_path), postprocess=lambda r, s: r)
    cb.on_fit_start(trainer, pl_module)
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 0.3)
    cb.on_validation_epoch_end(trainer, pl_module)
    payload = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=False)
    assert payload["model_name"] == "RFDETRLarge"


def test_unknown_variant_omits_model_name(tmp_path, monkeypatch) -> None:
    node = _Node()
    object.__setattr__(node, "_model_config", SimpleNamespace(num_queries=300))
    object.__setattr__(node, "_train_config", SimpleNamespace(lr=1e-4))
    object.__setattr__(node, "variant", "gigantic")  # not in any table
    object.__setattr__(node, "segmentation", True)
    pipeline = SimpleNamespace(nodes=[node])
    pl_module = SimpleNamespace(pipeline=pipeline)
    trainer = SimpleNamespace(
        callbacks=[],
        sanity_checking=False,
        current_epoch=0,
        datamodule=SimpleNamespace(val_dataloader=lambda: []),
    )
    cb = MapEvalCallback(node_name=node.name, output_dir=str(tmp_path), postprocess=lambda r, s: r)
    cb.on_fit_start(trainer, pl_module)
    monkeypatch.setattr(cb, "_evaluate", lambda *a, **k: 0.3)
    cb.on_validation_epoch_end(trainer, pl_module)
    payload = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=False)
    assert "model_name" not in payload
    assert payload["args"].num_queries == 300  # args still attached
