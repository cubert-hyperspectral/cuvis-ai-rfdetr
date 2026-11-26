"""Tests for training RF-DETR in a pipeline the way RF-DETR's own trainer does.

Covers the RF-DETR-format targets and row tiles, the transform-builder arguments, the
weight-average update against ``torch.optim.swa_utils.AveragedModel`` with RF-DETR's
average function, the ``rfdetr_adamw`` optimizer (groups, EMA hook, core registry), the
new training-node arguments, the mAP metric's conversions and best-of-both selection,
the prediction node's ``"pipeline"`` loader (with a stand-in ``rfdetr`` module), and the
EMA callback's per-step timing. Tests that need RF-DETR's train stack are skipped
without it; the end-to-end CPU fit is marked slow.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from cuvis_ai_rfdetr import optim
from cuvis_ai_rfdetr.ema import ema_update_
from cuvis_ai_rfdetr.functional import native_targets_from_mask, targets_from_mask, tile_rows
from cuvis_ai_rfdetr.native_data import AUG_PRESETS, resolve_aug_config

try:  # RF-DETR's training stack (faster_coco_eval, ...)
    import rfdetr.training.model_ema  # noqa: F401

    TRAIN_STACK = True
except ImportError:  # pragma: no cover
    TRAIN_STACK = False

needs_train_stack = pytest.mark.skipif(
    not TRAIN_STACK, reason="needs the rfdetr train stack (pip install 'rfdetr[train]')"
)


# ----------------------------------------------------------------- targets and tiles
def _mask(height: int = 20, width: int = 30) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.int32)
    mask[2:5, 3:9] = 1  # 3 x 6 object
    mask[10:12, 20:21] = 1  # 2 x 1 object
    return mask


def test_targets_from_mask_single_label() -> None:
    targets = targets_from_mask(torch.from_numpy(_mask())[None], single_label=1)
    assert targets[0]["labels"].tolist() == [1, 1]


def test_native_targets_from_mask_format() -> None:
    target = native_targets_from_mask(_mask(), image_id=7, with_masks=True, single_label=1)
    assert set(target) == {
        "boxes",
        "labels",
        "image_id",
        "area",
        "iscrowd",
        "orig_size",
        "size",
        "masks",
    }
    # absolute xyxy, exclusive right / bottom edge (COCO x + w)
    assert target["boxes"].tolist() == [[3.0, 2.0, 9.0, 5.0], [20.0, 10.0, 21.0, 12.0]]
    assert target["labels"].tolist() == [1, 1]
    assert target["area"].tolist() == [18.0, 2.0]
    assert target["iscrowd"].tolist() == [0, 0]
    assert target["image_id"].tolist() == [7]
    assert target["orig_size"].tolist() == [20, 30] == target["size"].tolist()
    assert target["masks"].shape == (2, 20, 30) and target["masks"].dtype == torch.bool
    assert int(target["masks"][0].sum()) == 18


def test_native_targets_from_empty_mask() -> None:
    target = native_targets_from_mask(np.zeros((8, 8), dtype=np.int32), with_masks=True)
    assert target["boxes"].shape == (0, 4)
    assert target["masks"].shape == (0, 8, 8)


def test_tile_rows_order_and_content() -> None:
    frames = torch.arange(2 * 987 * 4, dtype=torch.float32).reshape(2, 987, 4, 1)
    tiles = tile_rows(frames, 405, [0, 291, 582])
    assert tiles.shape == (6, 405, 4, 1)
    assert torch.equal(tiles[0], frames[0, 0:405])
    assert torch.equal(tiles[2], frames[0, 582:987])
    assert torch.equal(tiles[3], frames[1, 0:405])  # frame-major order


def test_tile_rows_rejects_tiles_outside_the_frame() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        tile_rows(torch.zeros(1, 900, 4), 405, [0, 582])


# ----------------------------------------------------------------- augmentation config
def test_resolve_aug_config() -> None:
    assert resolve_aug_config(None) is None
    preset = resolve_aug_config("rfdetr_1_8")
    assert preset == {"HorizontalFlip": {"p": 0.5}}
    preset["HorizontalFlip"]["p"] = 1.0  # a copy: the preset itself is unchanged
    assert AUG_PRESETS["rfdetr_1_8"]["HorizontalFlip"]["p"] == 0.5
    assert resolve_aug_config({"VerticalFlip": {"p": 0.5}}) == {"VerticalFlip": {"p": 0.5}}
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_aug_config("nope")
    with pytest.raises(TypeError):
        resolve_aug_config(3)


@needs_train_stack
def test_build_transforms_passes_rfdetr_dataset_arguments(monkeypatch) -> None:
    from rfdetr.datasets import coco

    from cuvis_ai_rfdetr.native_data import build_transforms

    seen: dict = {}

    def fake_builder(
        image_set,
        resolution,
        multi_scale=False,
        expanded_scales=False,
        skip_random_resize=False,
        patch_size=16,
        num_windows=4,
        aug_config=None,
    ):
        seen.update(
            image_set=image_set,
            resolution=resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=skip_random_resize,
            patch_size=patch_size,
            num_windows=num_windows,
            aug_config=aug_config,
        )
        return "transform"

    monkeypatch.setattr(coco, "make_coco_transforms_square_div_64", fake_builder)
    model_config = SimpleNamespace(resolution=624, patch_size=12, num_windows=2)
    train_config = SimpleNamespace(
        square_resize_div_64=True,
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=False,
        scale_jitter=True,
    )
    assert build_transforms("train", model_config, train_config, "rfdetr_1_8") == "transform"
    # only accepted keyword arguments are passed (the fake takes no scale_jitter)
    assert seen == {
        "image_set": "train",
        "resolution": 624,
        "multi_scale": True,
        "expanded_scales": True,
        "skip_random_resize": True,  # not do_random_resize_via_padding
        "patch_size": 12,
        "num_windows": 2,
        "aug_config": {"HorizontalFlip": {"p": 0.5}},
    }


# ----------------------------------------------------------------- weight average
def _rfdetr_avg_fn(decay: float, tau: float):
    def avg_fn(averaged, live, num_averaged):
        updates = int(num_averaged) + 1
        d = decay * (1 - math.exp(-updates / tau)) if tau > 0 else decay
        return averaged * d + live * (1.0 - d)

    return avg_fn


def test_ema_update_matches_averaged_model() -> None:
    from torch.optim.swa_utils import AveragedModel

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    reference = AveragedModel(model, use_buffers=True, avg_fn=_rfdetr_avg_fn(0.993, 100.0))
    ours = __import__("copy").deepcopy(model)
    for step in range(6):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
            model[1].running_mean.add_(1.0)
        reference.update_parameters(model)
        ema_update_(ours, model, step, 0.993, 100.0)
        for ref, mine in zip(
            reference.module.state_dict().values(), ours.state_dict().values(), strict=True
        ):
            assert torch.allclose(ref.float(), mine.float(), atol=1e-6)


def test_ema_update_first_call_copies_and_constant_decay() -> None:
    avg, live = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    ema_update_(avg, live, 0, 0.5, 0.0)
    assert torch.equal(avg.weight, live.weight)
    before = avg.weight.detach().clone()
    with torch.no_grad():
        live.weight.add_(2.0)
    ema_update_(avg, live, 1, 0.5, 0.0)  # tau 0: constant decay
    assert torch.allclose(avg.weight, 0.5 * before + 0.5 * live.weight)


# ----------------------------------------------------------------- rfdetr_adamw
class _FakeTrainable:
    def __init__(self, ema: bool = True) -> None:
        self.model = torch.nn.Linear(4, 2)
        self.ema_model = torch.nn.Linear(4, 2) if ema else None
        self.ema_updates = 0

    def get_param_groups(self, lr, **overrides):
        self.overrides = overrides
        return [{"params": [self.model.weight], "lr": lr * 0.5}]

    def optimizer_overrides(self):
        return {"lr_encoder": 1.5e-4}

    def update_ema(self):
        self.ema_updates += 1


def test_rfdetr_adamw_builds_native_groups_and_updates_ema() -> None:
    node = _FakeTrainable()
    optim.track_trainable(node)
    other = torch.nn.Parameter(torch.zeros(3))
    params = [*node.model.parameters(), other]
    optimizer = optim.RFDETRAdamW(params, lr=1e-4, weight_decay=1e-4)
    assert [len(g["params"]) for g in optimizer.param_groups] == [1, 2]  # weight; bias + other
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[1]["lr"] == 1e-4
    assert node.overrides == {"lr_encoder": 1.5e-4}
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.step()
    assert node.ema_updates == 2


def test_rfdetr_adamw_ignores_nodes_it_does_not_train() -> None:
    node = _FakeTrainable()
    optim.track_trainable(node)
    other = torch.nn.Parameter(torch.zeros(3))
    optimizer = optim.RFDETRAdamW([other], lr=1e-3)
    assert len(optimizer.param_groups) == 1
    other.grad = torch.ones_like(other)
    optimizer.step()
    assert node.ema_updates == 0


def test_rfdetr_adamw_is_registered_in_core() -> None:
    from cuvis_ai_core.training.config import OptimizerConfig
    from cuvis_ai_core.training.optimizer_registry import SUPPORTED_OPTIMIZERS, create_optimizer

    import cuvis_ai_rfdetr.node.rfdetr_trainable  # noqa: F401  (registers on import)

    assert optim.OPTIMIZER_NAME in SUPPORTED_OPTIMIZERS
    param = torch.nn.Parameter(torch.zeros(2))
    created = create_optimizer(
        OptimizerConfig(name="rfdetr_adamw", lr=1e-4, weight_decay=1e-4), [param]
    )
    assert isinstance(created, optim.RFDETRAdamW)
    assert created.param_groups[0]["weight_decay"] == 1e-4


# ----------------------------------------------------------------- training node arguments
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"class_names": []}, "class_names"),
        ({"class_names": ["fo-background", "fo"], "target_class_id": 2}, "target_class_id"),
        ({"jpeg_quality": 0}, "jpeg_quality"),
        ({"native_transforms": True, "num_channels": 6}, "3-channel"),
        ({"eval_tile_rows": 405}, "eval_row_starts"),
        ({"eval_tile_rows": 0, "eval_row_starts": [0]}, "eval_tile_rows"),
        ({"ema_decay": 0.0}, "ema_decay"),
        ({"ema_tau": -1.0}, "ema_tau"),
        ({"lr_encoder": 0.0}, "lr_encoder"),
        ({"aug_config": "nope"}, "unknown preset"),
    ],
)
def test_trainable_rejects_bad_native_arguments(kwargs: dict, match: str) -> None:
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    with pytest.raises(ValueError, match=match):
        RFDETRTrainable(dataset_dir=".", segmentation=True, **kwargs)


# ----------------------------------------------------------------- mAP metric
def test_convert_targets_scales_boxes_and_resizes_masks() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_map_metric import convert_preds, convert_targets

    masks = torch.zeros(1, 40, 20, dtype=torch.bool)
    masks[0, :20] = True
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.25, 1.0, 0.5]]),  # cxcywh, normalized
            "labels": torch.tensor([1]),
            "masks": masks,
            "iscrowd": torch.tensor([0]),
            "orig_size": torch.tensor([40, 20]),
        }
    ]
    preds = convert_preds(
        [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros(0, 1, 10, 5, dtype=torch.bool),
            }
        ]
    )
    assert preds[0]["masks"].shape == (0, 10, 5)  # [K, 1, H, W] -> [K, H, W]
    gts = convert_targets(targets, preds)
    assert gts[0]["boxes"].tolist() == [[0.0, 0.0, 20.0, 20.0]]  # absolute xyxy at orig_size
    assert gts[0]["masks"].shape == (1, 10, 5)  # the prediction's mask grid
    assert gts[0]["iscrowd"].tolist() == [0]
    assert convert_targets(targets)[0]["masks"].shape == (1, 40, 20)  # orig_size grid


def _bare_metric(segmentation: bool = True):
    from cuvis_ai_rfdetr.node.rfdetr_map_metric import RFDETRMapMetric, _MapStore

    node = RFDETRMapMetric.__new__(RFDETRMapMetric)
    torch.nn.Module.__init__(node)
    node.segmentation = segmentation
    node.register_buffer("selected_stream", torch.zeros((), dtype=torch.long))
    node._store = _MapStore(segmentation, 500)
    return node


def test_metric_best_of_both_sets_selected_stream() -> None:
    node = _bare_metric()
    node._store.metrics = {"regular": object(), "ema": object()}
    node._store._results = {
        "regular": {"segm_map": torch.tensor(0.40), "bbox_map": torch.tensor(0.9)},
        "ema": {"segm_map": torch.tensor(0.55), "bbox_map": torch.tensor(0.5)},
    }
    assert float(node._value("best_mAP_50_95")) == pytest.approx(0.55)
    assert int(node.selected_stream) == 1
    assert float(node._value("segm_mAP_50_95")) == pytest.approx(0.40)
    assert float(node._value("ema_mAP_50_95")) == pytest.approx(0.5)  # bbox key in list mode

    node._store._results["ema"]["segm_map"] = torch.tensor(0.40)  # tie keeps the live weights
    node._value("best_mAP_50_95")
    assert int(node.selected_stream) == 0


def test_metric_detection_uses_bbox_keys_and_hides_segm_names() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_map_metric import _MapView

    node = _bare_metric(segmentation=False)
    node.views = torch.nn.ModuleDict(
        {name: _MapView(lambda: torch.tensor(0.0)) for name in node.POOLED_METRIC_NAMES}
    )
    node._store.metrics = {"regular": object()}
    node._store._results = {"regular": {"map": torch.tensor(0.3), "map_50": torch.tensor(0.6)}}
    assert set(node.pooled_metrics()) == {"mAP_50_95", "mAP_50", "best_mAP_50_95"}
    assert float(node._value("mAP_50")) == pytest.approx(0.6)
    assert float(node._value("best_mAP_50_95")) == pytest.approx(0.3)


# ----------------------------------------------------------------- pipeline loader
def _write_run(tmp_path, stream_choice: int | None = 1, segmentation: bool = True):
    live, ema = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    config = {
        "nodes": [
            {
                "name": "RFDETR",
                "class_name": "cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable",
                "hparams": {
                    "variant": "medium",
                    "segmentation": segmentation,
                    "resolution": 624,
                    "num_channels": 3,
                    "class_names": ["fo-background", "fo"],
                },
            }
        ]
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    state = {f"model.{k}": v for k, v in live.state_dict().items()}
    state.update({f"ema_model.{k}": v for k, v in ema.state_dict().items()})
    states = {"RFDETR": state}
    if stream_choice is not None:
        states["MapMetric"] = {"selected_stream": torch.tensor(stream_choice)}
    torch.save({"state_dict": states}, tmp_path / "run.pt")
    return live, ema


class _FakeWrapper:
    created: list = []

    def __init__(self, **kwargs) -> None:
        _FakeWrapper.created.append(kwargs)
        self.model = SimpleNamespace(model=torch.nn.Linear(3, 2))


def _segmenter(tmp_path, **kwargs):
    from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

    return RFDETRSegmenter(
        checkpoint_path=str(tmp_path / "run.pt"), checkpoint_loader="pipeline", **kwargs
    )


@pytest.mark.parametrize(("choice", "expected"), [(1, "ema"), (0, "regular"), (None, "regular")])
def test_pipeline_loader_takes_settings_and_selected_weights(tmp_path, choice, expected) -> None:
    live, ema = _write_run(tmp_path, stream_choice=choice)
    _FakeWrapper.created.clear()
    seg = _segmenter(tmp_path)
    model = seg._build_from_pipeline(
        SimpleNamespace(RFDETRSegMedium=_FakeWrapper), "RFDETRSegMedium"
    )
    assert _FakeWrapper.created == [{"pretrain_weights": None, "resolution": 624, "num_classes": 2}]
    source = ema if expected == "ema" else live
    assert seg.loaded_stream == expected
    assert torch.equal(model.model.model.weight, source.weight)


def test_pipeline_loader_explicit_stream_and_resolution_override(tmp_path) -> None:
    live, _ = _write_run(tmp_path, stream_choice=1)
    _FakeWrapper.created.clear()
    seg = _segmenter(tmp_path, checkpoint_stream="regular", resolution=432)
    model = seg._build_from_pipeline(
        SimpleNamespace(RFDETRSegMedium=_FakeWrapper), "RFDETRSegMedium"
    )
    assert _FakeWrapper.created[0]["resolution"] == 432
    assert torch.equal(model.model.model.weight, live.weight)


def test_pipeline_loader_errors(tmp_path) -> None:
    from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

    with pytest.raises(ValueError, match="needs checkpoint_path"):
        RFDETRSegmenter(checkpoint_loader="pipeline")
    with pytest.raises(ValueError, match="checkpoint_stream"):
        RFDETRSegmenter(
            checkpoint_path="x.pt", checkpoint_loader="pipeline", checkpoint_stream="best"
        )
    fake = SimpleNamespace(RFDETRSegMedium=_FakeWrapper)
    with pytest.raises(FileNotFoundError, match="pipeline yaml"):
        _segmenter(tmp_path)._build_from_pipeline(fake, "RFDETRSegMedium")
    _write_run(tmp_path, segmentation=False)
    with pytest.raises(ValueError, match="detection"):
        _segmenter(tmp_path)._build_from_pipeline(fake, "RFDETRSegMedium")
    _write_run(tmp_path)
    with pytest.raises(ValueError, match="no node named"):
        _segmenter(tmp_path, checkpoint_node="Other")._build_from_pipeline(fake, "RFDETRSegMedium")
    with pytest.raises(ValueError, match="variant"):
        _segmenter(tmp_path, variant="large")._build_from_pipeline(fake, "RFDETRSegLarge")


# ----------------------------------------------------------------- EMA callback timing
def test_ema_callback_updates_once_per_optimizer_step_and_at_epoch_end() -> None:
    from cuvis_ai_rfdetr.training import EmaCallback

    class _Counting:
        def __init__(self, model, decay=0.9, tau=0.0) -> None:
            self.module = model
            self.updates = 0

        def update(self, model) -> None:
            self.updates += 1

    node = SimpleNamespace(name="RFDETR", model=torch.nn.Linear(2, 2))
    pl_module = SimpleNamespace(pipeline=SimpleNamespace(nodes=[node]))
    trainer = SimpleNamespace(global_step=0)
    cb = EmaCallback(ema_cls=_Counting)
    cb.on_fit_start(trainer, pl_module)
    for step in (0, 1, 1, 1, 1, 2):  # accumulation: global_step moves every few batches
        trainer.global_step = step
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    assert cb.ema.updates == 2  # one per optimizer step, not one per batch
    cb.on_train_epoch_end(trainer, pl_module)
    assert cb.ema.updates == 3  # plus RF-DETR's epoch-end update


def test_map_metric_runs_at_val_and_test_only() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    from cuvis_ai_rfdetr.node.rfdetr_map_metric import RFDETRMapMetric

    assert RFDETRMapMetric.EXECUTION_STAGES == {ExecutionStage.VAL, ExecutionStage.TEST}
    assert RFDETRMapMetric.INPUT_SPECS["ema_outputs"].optional
