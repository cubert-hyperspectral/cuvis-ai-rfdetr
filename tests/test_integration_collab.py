"""Cross-component integration tests: the training extras working in collaboration.

The centerpiece runs a REAL Lightning fit on CPU through a real CuvisPipeline with
the full callback stack at once — RFDETRGradientTrainer (native param groups),
EmaCallback (stub EMA), and MapEvalCallback (mocked metric, injected postprocess,
self-describing checkpoints) — asserting the pieces cooperate exactly as they do in
the production training script. No rfdetr model build, no GPU, no MAP backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.test_training_extras import _StubEma

try:
    import cuvis_ai_augment  # noqa: F401

    AUGMENT = True
except ImportError:  # pragma: no cover
    AUGMENT = False

needs_augment = pytest.mark.skipif(not AUGMENT, reason="needs cuvis-ai-augment")

FIXED_TARGET = {
    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    "labels": torch.tensor([0]),
}


def _build_tiny_pipeline():
    """A real CuvisPipeline with an RFDETRTrainable-shaped tiny node + a loss node."""
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_schemas.pipeline import PortSpec

    class TinyTrainable(Node):
        INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
        OUTPUT_SPECS = {
            "pred": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "outputs": PortSpec(dtype=dict, shape=()),
            "targets": PortSpec(dtype=list, shape=()),
        }

        def __init__(self, **kwargs):
            name, stages = Node.consume_base_kwargs(kwargs)
            super().__init__(name=name, execution_stages=stages, **kwargs)
            self.model = torch.nn.Linear(4, 2)
            # RFDETRTrainable-shaped attributes the callbacks rely on:
            self._input_resolution = 100
            self._spatial_unit = 24
            self._model_config = SimpleNamespace(num_queries=200, group_detr=13, resolution=624)
            self._train_config = SimpleNamespace(expanded_scales=True, lr=1e-4)
            self.variant = "medium"
            self.segmentation = True

        def unfreeze(self):
            super().unfreeze()
            for p in self.model.parameters():
                p.requires_grad_(True)

        def get_param_groups(self, lr, **_):
            return [{"params": [self.model.weight], "lr": lr * 0.5}]

        def forward(self, data, context=None, **_):
            pred = self.model(data)
            return {
                "pred": pred,
                "outputs": {"raw": pred.detach()},
                "targets": [FIXED_TARGET] * data.shape[0],
            }

    class TinyLoss(Node):
        INPUT_SPECS = {
            "pred": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "target": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }
        OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

        def __init__(self, **kwargs):
            name, stages = Node.consume_base_kwargs(kwargs)
            super().__init__(name=name, execution_stages=stages, **kwargs)

        def forward(self, pred, target, context=None, **_):
            return {"loss": torch.nn.functional.mse_loss(pred, target)}

    trainable, loss = TinyTrainable(name="RFDETR"), TinyLoss(name="loss")
    pipe = CuvisPipeline("collab_integration")
    pipe.connect((trainable.outputs.pred, loss.inputs.pred))
    pipe.unfreeze_nodes_by_name(["RFDETR"])
    return pipe, trainable, loss


def _tiny_datamodule():
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    class TinyData(pl.LightningDataModule):
        def _loader(self):
            ds = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
            return DataLoader(
                ds,
                batch_size=4,
                collate_fn=lambda items: {
                    "data": torch.stack([i[0] for i in items]),
                    "target": torch.stack([i[1] for i in items]),
                },
            )

        train_dataloader = _loader
        val_dataloader = _loader

    return TinyData()


def test_full_fit_with_all_three_callbacks(tmp_path, monkeypatch) -> None:
    """EmaCallback + MapEvalCallback + RFDETRGradientTrainer in one real 2-epoch fit."""
    from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig

    from cuvis_ai_rfdetr.training import EmaCallback, MapEvalCallback, RFDETRGradientTrainer

    # scripted metric values: (regular, ema) x 2 epochs -> best reg .4@e1, ema .5@e0
    script = [0.3, 0.5, 0.4, 0.2]

    class FakeMAP:
        def __init__(self, iou_type, backend): ...

        def update(self, preds, gts):
            # collaboration contract: preds come from the injected postprocess,
            # GT boxes from the node's targets, converted to abs xyxy @ res 100
            assert preds[0]["scores"].numel() == 1
            assert gts[0]["boxes"].shape == (1, 4)
            assert float(gts[0]["boxes"][0, 2]) == pytest.approx(60.0)  # (0.5+0.1)*100

        def compute(self):
            return {"map": torch.tensor(script.pop(0))}

    import torchmetrics.detection

    monkeypatch.setattr(torchmetrics.detection, "MeanAveragePrecision", FakeMAP)

    pipe, trainable, loss = _build_tiny_pipeline()
    initial_weight = trainable.model.weight.detach().clone()

    ema_cb = EmaCallback(
        node_name="RFDETR",
        decay=0.9,
        tau=0.0,
        ema_cls=_StubEma,
        save_path=str(tmp_path / "ema_last.pth"),
    )
    map_cb = MapEvalCallback(
        node_name="RFDETR",
        output_dir=str(tmp_path / "best"),
        postprocess=lambda raw, sizes: [
            {
                "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in range(len(sizes))
        ],
    )
    trainer = RFDETRGradientTrainer(
        pipeline=pipe,
        datamodule=_tiny_datamodule(),
        loss_nodes=[loss],
        metric_nodes=[],
        training_config=TrainingConfig(
            seed=7,
            optimizer=OptimizerConfig(name="adamw", lr=1e-2),
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
        ),
        callbacks=[ema_cb, map_cb],
        node_name="RFDETR",
    )

    # the grouped optimizer: half-lr weight group + leftover (bias) at base lr
    optimizer = trainer.configure_optimizers()
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 5e-3

    trainer.fit()

    # training moved the weights; EMA tracked and saved
    assert not torch.equal(trainable.model.weight.detach(), initial_weight)
    ema_payload = torch.load(tmp_path / "ema_last.pth", weights_only=True)
    assert ema_payload["updates"] > 1

    # selection: scripted values consumed in order regular->ema per epoch
    assert script == []  # exactly 4 evaluations (2 epochs x 2 streams), sanity skipped
    assert map_cb.best["regular"]["map"] == pytest.approx(0.4)  # float32 round-trip
    assert map_cb.best["regular"]["epoch"] == 1
    assert map_cb.best["ema"]["map"] == pytest.approx(0.5)
    assert map_cb.best["ema"]["epoch"] == 0
    assert map_cb.best["total"]["stream"] == "ema"
    assert map_cb.best["total"]["epoch"] == 0
    assert [h["stream"] for h in map_cb.history] == ["regular", "ema", "regular", "ema"]

    # the three native-style checkpoints exist and are self-describing
    for stream in ("regular", "ema", "total"):
        payload = torch.load(
            tmp_path / "best" / f"checkpoint_best_{stream}.pth", weights_only=False
        )
        assert payload["model_name"] == "RFDETRSegMedium"
        assert payload["args"].num_queries == 200 and payload["args"].group_detr == 13
        assert all(v.device.type == "cpu" for v in payload["model"].values())
    total = torch.load(tmp_path / "best" / "checkpoint_best_total.pth", weights_only=False)
    assert total["stream"] == "ema" and total["map"] == pytest.approx(0.5)


@needs_augment
def test_compose_applies_multiscale_at_train_and_passes_val() -> None:
    """RandomMultiScaleResize inside AugmentationCompose: TRAIN applies, VAL identity."""
    from cuvis_ai_augment.node.compose import AugmentationCompose
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    compose = AugmentationCompose(
        name="Aug",
        transforms=[{"type": "RandomMultiScaleResize", "scales": [48, 72]}],
        extra_transform_modules=["cuvis_ai_rfdetr.transforms"],
        seed=0,
    )
    cube = torch.rand(2, 30, 30, 3)
    mask = torch.randint(0, 2, (2, 30, 30), dtype=torch.int32)

    out = compose.forward(cube=cube, mask=mask, context=Context(stage=ExecutionStage.TRAIN))
    assert out["cube"].shape[1] in (48, 72)
    assert out["mask"].shape[1:] == out["cube"].shape[1:3]

    val = compose.forward(cube=cube, mask=mask, context=Context(stage=ExecutionStage.VAL))
    assert torch.equal(val["cube"], cube)  # stage-gated identity outside TRAIN
    assert torch.equal(val["mask"], mask)


def test_targets_from_mask_invariant_under_multiscale_resize() -> None:
    """The normalized DETR boxes survive the transform's nearest-neighbour mask resize."""
    import torch.nn.functional as F

    from cuvis_ai_rfdetr.functional import targets_from_mask

    mask = torch.zeros(1, 100, 100, dtype=torch.int32)
    mask[0, 20:40, 30:60] = 1  # one rectangle
    base = targets_from_mask(mask)[0]["boxes"]

    resized = (
        F.interpolate(mask.unsqueeze(1).float(), size=(200, 200), mode="nearest")
        .squeeze(1)
        .to(torch.int32)
    )
    scaled = targets_from_mask(resized)[0]["boxes"]

    assert base.shape == scaled.shape == (1, 4)
    # normalized cxcywh is resize-invariant up to nearest-neighbour quantization
    assert torch.allclose(base, scaled, atol=0.02)
