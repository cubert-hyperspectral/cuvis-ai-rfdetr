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


class _StubEma(torch.nn.Module):
    """ModelEma-compatible stub (constant decay) so callback logic tests run in CI
    without the rfdetr train stack; the native decay schedule is asserted separately
    under the train-stack marker."""

    def __init__(self, model, decay=0.999, tau=0.0, device=None) -> None:
        super().__init__()
        from copy import deepcopy

        self.module = deepcopy(model)
        self.module.eval()
        self.decay, self.tau, self.updates = float(decay), float(tau), 1

    def update(self, model) -> None:
        with torch.no_grad():
            for e, m in zip(
                self.module.state_dict().values(), model.state_dict().values(), strict=True
            ):
                e.copy_(self.decay * e + (1.0 - self.decay) * m)
        self.updates += 1


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


def test_update_interval_gates_updates() -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(update_interval=3, decay=0.5, tau=0.0, ema_cls=_StubEma)
    cb.on_fit_start(trainer, pl_module)
    with torch.no_grad():
        node.model.weight += 1.0
    before = cb.ema_module.weight.clone()
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    cb.on_train_batch_end(trainer, pl_module, None, None, 1)
    assert torch.equal(cb.ema_module.weight, before)  # gated: 2 < interval
    cb.on_train_batch_end(trainer, pl_module, None, None, 2)
    assert not torch.equal(cb.ema_module.weight, before)  # third batch updates


def test_state_roundtrip_restores_average(tmp_path) -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(decay=0.9, tau=0.0, save_path=str(tmp_path / "ema.pth"), ema_cls=_StubEma)
    cb.on_fit_start(trainer, pl_module)
    with torch.no_grad():
        node.model.weight += 1.0
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    state = cb.state_dict()

    # fresh callback + fresh (different) model: resume must restore the average
    node2 = _Node()
    trainer2, pl_module2 = _fake_pl(node2)
    cb2 = EmaCallback(decay=0.9, tau=0.0, ema_cls=_StubEma)
    cb2.load_state_dict(state)
    cb2.on_fit_start(trainer2, pl_module2)
    assert torch.allclose(cb2.ema_module.weight, cb.ema_module.weight)
    assert cb2.ema.updates == cb.ema.updates
    assert cb2._batches_seen == cb._batches_seen

    # save() writes the documented payload
    cb.on_fit_end(trainer, pl_module)
    payload = torch.load(tmp_path / "ema.pth", weights_only=True)
    assert set(payload) == {"model", "updates"}


def test_rfdetr_gradient_trainer_builds_grouped_optimizer() -> None:
    from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig

    from cuvis_ai_rfdetr.training import RFDETRGradientTrainer

    p_grouped = torch.nn.Parameter(torch.zeros(2))
    p_extra = torch.nn.Parameter(torch.zeros(3))
    p_frozen = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    class _TrainableStub:
        name = "RFDETR"

        @staticmethod
        def get_param_groups(lr, lr_encoder=None, lr_vit_layer_decay=None, lr_component_decay=None):
            assert lr == 1e-4  # base lr comes from the optimizer config
            assert lr_encoder == 1.5e-4  # constructor knob forwarded
            return [{"params": [p_grouped], "lr": 5e-5}]

    pipeline = SimpleNamespace(
        nodes=[_TrainableStub()],
        parameters=lambda: iter([p_grouped, p_extra, p_frozen]),
    )

    trainer = RFDETRGradientTrainer.__new__(RFDETRGradientTrainer)  # skip Lightning init
    cfg = TrainingConfig(optimizer=OptimizerConfig(name="adamw", lr=1e-4))
    for attr, value in {
        "pipeline": pipeline,
        "optimizer_config": cfg.optimizer,
        "scheduler_config": None,
        "training_config": cfg,
        "node_name": "RFDETR",
        "_group_overrides": {
            "lr_encoder": 1.5e-4,
            "lr_vit_layer_decay": None,
            "lr_component_decay": None,
        },
    }.items():
        object.__setattr__(trainer, attr, value)

    optimizer = trainer.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[0]["params"] == [p_grouped]
    assert optimizer.param_groups[1]["params"] == [p_extra]  # trainable leftover, base lr
    assert optimizer.param_groups[1]["lr"] == 1e-4
    # the frozen parameter is optimized by neither group
    assert all(p_frozen is not q for g in optimizer.param_groups for q in g["params"])


def test_rfdetr_gradient_trainer_requires_param_group_node() -> None:
    from cuvis_ai_rfdetr.training import RFDETRGradientTrainer

    trainer = RFDETRGradientTrainer.__new__(RFDETRGradientTrainer)
    object.__setattr__(trainer, "pipeline", SimpleNamespace(nodes=[SimpleNamespace(name="RFDETR")]))
    object.__setattr__(trainer, "node_name", "RFDETR")
    with pytest.raises(RuntimeError, match="get_param_groups"):
        trainer.configure_optimizers()


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


# ------------------------------------------------------------ CPU integration
def test_integration_fit_with_ema_and_param_groups(tmp_path) -> None:
    """End-to-end on CPU: a real CuvisPipeline through RFDETRGradientTrainer with
    EmaCallback — param groups reach the optimizer, training moves the weights,
    the EMA tracks them and is written on fit end. No rfdetr stack needed."""
    import pytorch_lightning as pl
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig
    from cuvis_ai_schemas.pipeline import PortSpec
    from torch.utils.data import DataLoader, TensorDataset

    from cuvis_ai_rfdetr.training import RFDETRGradientTrainer

    class TinyTrainable(Node):
        INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
        OUTPUT_SPECS = {"pred": PortSpec(dtype=torch.float32, shape=(-1, -1))}

        def __init__(self, **kwargs):
            name, stages = Node.consume_base_kwargs(kwargs)
            super().__init__(name=name, execution_stages=stages, **kwargs)
            self.model = torch.nn.Linear(4, 2)

        def unfreeze(self):
            super().unfreeze()
            for p in self.model.parameters():
                p.requires_grad_(True)

        def get_param_groups(self, lr, **_):
            # weight in a half-lr group; bias left for the trainer's leftover path
            return [{"params": [self.model.weight], "lr": lr * 0.5}]

        def forward(self, data, context=None, **_):
            return {"pred": self.model(data)}

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

    class TinyData(pl.LightningDataModule):
        def _loader(self):
            ds = TensorDataset(torch.randn(16, 4), torch.randn(16, 2))
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

    trainable, loss = TinyTrainable(name="tiny"), TinyLoss(name="loss")
    pipe = CuvisPipeline("tiny_integration")
    pipe.connect((trainable.outputs.pred, loss.inputs.pred))
    pipe.unfreeze_nodes_by_name(["tiny"])

    initial = trainable.model.weight.detach().clone()
    ema_path = tmp_path / "tiny_ema.pth"
    trainer = RFDETRGradientTrainer(
        pipeline=pipe,
        datamodule=TinyData(),
        loss_nodes=[loss],
        metric_nodes=[],
        training_config=TrainingConfig(
            seed=7,
            optimizer=OptimizerConfig(name="adamw", lr=1e-2),
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
        ),
        callbacks=[
            EmaCallback(
                node_name="tiny", decay=0.9, tau=0.0, save_path=str(ema_path), ema_cls=_StubEma
            )
        ],
        node_name="tiny",
    )

    optimizer = trainer.configure_optimizers()
    assert len(optimizer.param_groups) == 2  # half-lr weight group + leftover bias
    assert optimizer.param_groups[0]["lr"] == 5e-3

    trainer.fit()

    final = trainable.model.weight.detach()
    assert not torch.equal(final, initial), "training did not move the weights"
    assert ema_path.exists()
    payload = torch.load(ema_path, weights_only=True)
    ema_weight = payload["model"]["weight"]
    assert not torch.equal(ema_weight, final), "EMA identical to final weights"
    assert payload["updates"] > 1


# ------------------------------------------------- checkpoint_loader (fidelity)
def test_trainable_checkpoint_loader_validation() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    with pytest.raises(ValueError, match="checkpoint_loader must be"):
        RFDETRTrainable(dataset_dir="x", checkpoint_loader="bogus")
    with pytest.raises(ValueError, match="num_channels=3"):
        RFDETRTrainable(
            dataset_dir="x",
            checkpoint_path="ck.pth",
            checkpoint_loader="from_checkpoint",
            num_channels=6,
        )


@needs_train_stack
def test_trainable_from_checkpoint_forwards_kwargs_and_validates_class(monkeypatch) -> None:
    import rfdetr

    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    captured: dict = {}

    class _WrongClass:  # resolved class name will not match the requested variant
        pass

    def fake_from_checkpoint(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _WrongClass()

    monkeypatch.setattr(
        rfdetr.RFDETR,
        "from_checkpoint",
        classmethod(lambda cls, path, **kw: fake_from_checkpoint(path, **kw)),
    )

    with pytest.raises(RuntimeError, match="checkpoint resolved to _WrongClass"):
        RFDETRTrainable(
            dataset_dir="x",
            checkpoint_path="champion.pth",
            variant="medium",
            segmentation=True,
            resolution=624,
            checkpoint_loader="from_checkpoint",
        )
    assert captured["path"] == "champion.pth"
    assert captured["kwargs"] == {"resolution": 624}  # loader kwargs forwarded


# ------------------------------------------------------- EmaCallback edge cases
def test_ema_module_property_raises_before_fit() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = EmaCallback().ema_module


def test_empty_load_state_dict_is_noop_and_fresh_fit_works() -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(ema_cls=_StubEma)
    cb.load_state_dict({})  # e.g. restored from a checkpoint saved before any fit
    cb.on_fit_start(trainer, pl_module)
    assert torch.equal(cb.ema_module.weight, node.model.weight)  # fresh copy, no restore


def test_state_dict_before_fit_passes_pending_through() -> None:
    cb = EmaCallback(ema_cls=_StubEma)
    assert cb.state_dict() == {}  # nothing yet
    pending = {"ema_state": {}, "updates": 7, "batches_seen": 3}
    cb.load_state_dict(pending)
    assert cb.state_dict() == pending  # survives a save-before-fit round trip


def test_interval_larger_than_run_saves_initial_copy(tmp_path) -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(update_interval=100, ema_cls=_StubEma, save_path=str(tmp_path / "e.pth"))
    cb.on_fit_start(trainer, pl_module)
    initial = cb.ema_module.weight.detach().clone()
    with torch.no_grad():
        node.model.weight += 1.0
    for i in range(5):  # fewer batches than the interval -> EMA never updates
        cb.on_train_batch_end(trainer, pl_module, None, None, i)
    cb.on_fit_end(trainer, pl_module)
    payload = torch.load(tmp_path / "e.pth", weights_only=True)
    assert torch.equal(payload["model"]["weight"], initial)
    assert payload["updates"] == 1  # stub starts at 1, no update happened


def test_ema_save_writes_cpu_tensors(tmp_path) -> None:
    node = _Node()
    trainer, pl_module = _fake_pl(node)
    cb = EmaCallback(ema_cls=_StubEma, save_path=str(tmp_path / "e.pth"))
    cb.on_fit_start(trainer, pl_module)
    cb.on_fit_end(trainer, pl_module)
    payload = torch.load(tmp_path / "e.pth", weights_only=True)
    assert all(v.device.type == "cpu" for v in payload["model"].values())


# --------------------------------------------------- get_param_groups edge cases
@needs_train_stack
def test_get_param_groups_no_overrides_uses_config_values(monkeypatch) -> None:
    from cuvis_ai_rfdetr.node import rfdetr_trainable as mod

    node = mod.RFDETRTrainable.__new__(mod.RFDETRTrainable)
    object.__setattr__(node, "_train_config", SimpleNamespace(lr=3e-4, lr_encoder=9e-4))
    object.__setattr__(node, "_model_config", SimpleNamespace(out_feature_indexes=[1]))
    object.__setattr__(node, "model", torch.nn.Linear(2, 2))

    captured = {}
    import rfdetr.training.param_groups as pg

    monkeypatch.setattr(pg, "get_param_dict", lambda cfg, m: captured.update(cfg=cfg) or [])
    mod.RFDETRTrainable.get_param_groups(node, lr=None)
    assert captured["cfg"].lr == 3e-4  # config value passes through untouched
    assert captured["cfg"].lr_encoder == 9e-4


@needs_train_stack
def test_get_param_groups_train_config_wins_key_collisions(monkeypatch) -> None:
    from cuvis_ai_rfdetr.node import rfdetr_trainable as mod

    node = mod.RFDETRTrainable.__new__(mod.RFDETRTrainable)
    # both configs define `resolution`: the train config must win the flat merge
    object.__setattr__(node, "_model_config", SimpleNamespace(resolution=432, patch_size=12))
    object.__setattr__(node, "_train_config", SimpleNamespace(resolution=624, lr=1e-4))
    object.__setattr__(node, "model", torch.nn.Linear(2, 2))

    captured = {}
    import rfdetr.training.param_groups as pg

    monkeypatch.setattr(pg, "get_param_dict", lambda cfg, m: captured.update(cfg=cfg) or [])
    mod.RFDETRTrainable.get_param_groups(node)
    assert captured["cfg"].resolution == 624  # train config overrode model config
    assert captured["cfg"].patch_size == 12  # model-only field survives


@needs_train_stack
def test_get_param_groups_model_dump_branch(monkeypatch) -> None:
    from cuvis_ai_rfdetr.node import rfdetr_trainable as mod

    class PydanticLike:
        def __init__(self, **kw):
            self._kw = kw

        def model_dump(self):
            return dict(self._kw)

    node = mod.RFDETRTrainable.__new__(mod.RFDETRTrainable)
    object.__setattr__(node, "_model_config", PydanticLike(num_queries=200))
    object.__setattr__(node, "_train_config", PydanticLike(lr=1e-4))
    object.__setattr__(node, "model", torch.nn.Linear(2, 2))

    captured = {}
    import rfdetr.training.param_groups as pg

    monkeypatch.setattr(pg, "get_param_dict", lambda cfg, m: captured.update(cfg=cfg) or [])
    mod.RFDETRTrainable.get_param_groups(node)
    assert captured["cfg"].num_queries == 200 and captured["cfg"].lr == 1e-4


# ------------------------------------------- RFDETRGradientTrainer edge cases
def _bare_grouped_trainer(pipeline, scheduler_config=None):
    from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig

    from cuvis_ai_rfdetr.training import RFDETRGradientTrainer

    trainer = RFDETRGradientTrainer.__new__(RFDETRGradientTrainer)
    cfg = TrainingConfig(optimizer=OptimizerConfig(name="adamw", lr=1e-4))
    for attr, value in {
        "pipeline": pipeline,
        "optimizer_config": cfg.optimizer,
        "scheduler_config": scheduler_config,
        "training_config": cfg,
        "node_name": "RFDETR",
        "_group_overrides": {
            "lr_encoder": None,
            "lr_vit_layer_decay": None,
            "lr_component_decay": None,
        },
    }.items():
        object.__setattr__(trainer, attr, value)
    return trainer


def test_trainer_no_leftovers_single_group() -> None:
    p = torch.nn.Parameter(torch.zeros(2))

    class _Stub:
        name = "RFDETR"

        @staticmethod
        def get_param_groups(lr, **_):
            return [{"params": [p], "lr": lr}]

    pipeline = SimpleNamespace(nodes=[_Stub()], parameters=lambda: iter([p]))
    optimizer = _bare_grouped_trainer(pipeline).configure_optimizers()
    assert len(optimizer.param_groups) == 1  # everything grouped -> no leftover group


def test_trainer_handles_scalar_params_entry() -> None:
    # torch accepts {"params": <single tensor>}; the leftover-dedup must too
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(3))

    class _Stub:
        name = "RFDETR"

        @staticmethod
        def get_param_groups(lr, **_):
            return [{"params": p1, "lr": lr * 0.5}]  # scalar, not a list

    pipeline = SimpleNamespace(nodes=[_Stub()], parameters=lambda: iter([p1, p2]))
    optimizer = _bare_grouped_trainer(pipeline).configure_optimizers()
    assert len(optimizer.param_groups) == 2
    leftover = optimizer.param_groups[1]["params"]
    assert leftover == [p2]  # p1 correctly recognized as already grouped


def test_trainer_scheduler_path_returns_lightning_dict() -> None:
    from cuvis_ai_schemas.training.scheduler import SchedulerConfig

    p = torch.nn.Parameter(torch.zeros(2))

    class _Stub:
        name = "RFDETR"

        @staticmethod
        def get_param_groups(lr, **_):
            return [{"params": [p], "lr": lr}]

    pipeline = SimpleNamespace(nodes=[_Stub()], parameters=lambda: iter([p]))
    trainer = _bare_grouped_trainer(pipeline, SchedulerConfig(name="step", step_size=100))
    out = trainer.configure_optimizers()
    assert isinstance(out, dict) and "optimizer" in out and "lr_scheduler" in out
