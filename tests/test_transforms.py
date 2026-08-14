"""Tests for RandomMultiScaleResize, the scale math, and the stage-aware resize.

The scale formula is unit-tested for exact equality against rfdetr's native
``compute_multi_scale_scales`` (where the train stack is installed). Transform
tests need cuvis-ai-augment (the registry/base) and skip where it is absent —
same convention as the train-stack skips.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cuvis_ai_rfdetr.functional import compute_multi_scale_scales

try:
    import cuvis_ai_augment  # noqa: F401

    AUGMENT = True
except ImportError:  # pragma: no cover
    AUGMENT = False

try:
    from rfdetr.datasets.coco import compute_multi_scale_scales as native_scales

    TRAIN_STACK = True
except ImportError:  # pragma: no cover
    TRAIN_STACK = False

needs_augment = pytest.mark.skipif(not AUGMENT, reason="needs cuvis-ai-augment")
needs_train_stack = pytest.mark.skipif(not TRAIN_STACK, reason="needs rfdetr train stack")


# --------------------------------------------------------------- scale formula
def test_scales_champion_config() -> None:
    # SegMedium@624: patch 12 x windows 2 -> unit 24; expanded -> 11 sizes incl. 624
    scales = compute_multi_scale_scales(624, expanded_scales=True, patch_size=12, num_windows=2)
    assert scales == [504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744]
    assert all(s % 24 == 0 for s in scales)


def test_scales_min_size_filter() -> None:
    # tiny resolution: offsets below two units are filtered out
    scales = compute_multi_scale_scales(128, expanded_scales=True, patch_size=16, num_windows=4)
    assert scales and all(s >= 128 for s in scales)


@needs_train_stack
@pytest.mark.parametrize(
    ("resolution", "expanded", "patch", "windows"),
    [
        (624, True, 12, 2),
        (624, False, 12, 2),
        (432, True, 12, 2),
        (560, True, 16, 4),
        (128, True, 16, 4),
    ],
)
def test_scales_match_native(resolution: int, expanded: bool, patch: int, windows: int) -> None:
    ours = compute_multi_scale_scales(resolution, expanded, patch, windows)
    theirs = native_scales(resolution, expanded, patch, windows)
    assert ours == theirs


# --------------------------------------------------------------- the transform
@needs_augment
def test_registered_and_buildable() -> None:
    from cuvis_ai_augment.transforms.base import build_transform

    import cuvis_ai_rfdetr.transforms  # noqa: F401 — registers on import

    t = build_transform({"type": "RandomMultiScaleResize", "scales": [96, 120]})
    assert t.scales == [96, 120]


@needs_augment
def test_resizes_cube_and_mask_to_one_scale() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[96, 120, 144])
    cube = torch.rand(2, 64, 80, 5)
    mask = torch.randint(0, 3, (2, 64, 80), dtype=torch.int32)
    rng = torch.Generator().manual_seed(7)
    out_cube, out_mask = t(cube, mask, rng)
    size = out_cube.shape[1]
    assert size in (96, 120, 144)
    assert out_cube.shape == (2, size, size, 5)
    assert out_mask.shape == (2, size, size)
    assert out_mask.dtype == torch.int32
    # nearest-neighbour: no new label values invented
    assert set(out_mask.unique().tolist()) <= set(mask.unique().tolist())


@needs_augment
def test_deterministic_under_seed() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[96, 120, 144, 168])
    cube = torch.rand(1, 50, 50, 3)
    a, _ = t(cube, None, torch.Generator().manual_seed(123))
    b, _ = t(cube, None, torch.Generator().manual_seed(123))
    assert a.shape == b.shape
    assert torch.equal(a, b)


@needs_augment
def test_prob_zero_is_passthrough() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[96], prob=0.0)
    cube = torch.rand(1, 33, 44, 2)
    out, _ = t(cube, None, torch.Generator().manual_seed(0))
    assert torch.equal(out, cube)


@needs_augment
def test_scales_computed_from_resolution() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(resolution=624, expanded_scales=True, patch_size=12, num_windows=2)
    assert 624 in t.scales and len(t.scales) == 11
    with pytest.raises(ValueError, match="scales=.*or resolution"):
        RandomMultiScaleResize()


# ------------------------------------------------- trainable stage-aware resize
def _bare_trainable():
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    node = RFDETRTrainable.__new__(RFDETRTrainable)
    for attr, value in {
        "_input_resolution": 432,
        "_spatial_unit": 24,
        "training": False,
        "_model_config": SimpleNamespace(resolution=624, patch_size=12, num_windows=2),
        "_train_config": SimpleNamespace(expanded_scales=True),
    }.items():
        object.__setattr__(node, attr, value)
    return node


def test_resize_for_stage_multi_scale_train_passthrough() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    x = torch.rand(1, 3, 480, 480)  # 480 % 24 == 0
    assert node._resize_for_stage(x, ExecutionStage.TRAIN) is x


def test_resize_for_stage_train_arbitrary_size_resizes() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    x = torch.rand(1, 3, 405, 405)  # 405 % 24 != 0 -> fixed resize (P6 trainrun path)
    out = node._resize_for_stage(x, ExecutionStage.TRAIN)
    assert out.shape[-2:] == (432, 432)


def test_resize_for_stage_inference_always_fixed() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    x = torch.rand(1, 3, 480, 480)  # divisible, but NOT train -> certified fixed resize
    out = node._resize_for_stage(x, ExecutionStage.INFERENCE)
    assert out.shape[-2:] == (432, 432)


def test_resize_for_stage_module_training_flag() -> None:
    node = _bare_trainable()
    object.__setattr__(node, "training", True)
    x = torch.rand(1, 3, 456, 456)  # 456 % 24 == 0
    assert node._resize_for_stage(x, None) is x


def test_multi_scale_scales_reads_model_config() -> None:
    node = _bare_trainable()
    scales = node.multi_scale_scales()
    assert scales == compute_multi_scale_scales(624, True, 12, 2)
    assert node.multi_scale_scales(expanded_scales=False) == compute_multi_scale_scales(
        624, False, 12, 2
    )


# ----------------------------------------------------- scale-formula edge cases
def test_scales_floor_behavior_and_ordering() -> None:
    # resolution floors to the unit: 625 and 624 give identical sets
    assert compute_multi_scale_scales(625, True, 12, 2) == compute_multi_scale_scales(
        624, True, 12, 2
    )
    scales = compute_multi_scale_scales(624, True, 12, 2)
    assert scales == sorted(set(scales))  # strictly increasing, no duplicates


def test_scales_non_expanded_is_subset_of_expanded() -> None:
    exp = set(compute_multi_scale_scales(624, True, 12, 2))
    base = set(compute_multi_scale_scales(624, False, 12, 2))
    assert base <= exp


def test_scales_resolution_below_unit_still_nonempty() -> None:
    # base floor is 0; only offsets producing >= 2 units survive
    scales = compute_multi_scale_scales(20, True, 12, 2)  # unit 24, res < unit
    assert scales and min(scales) >= 48 and all(s % 24 == 0 for s in scales)


# ----------------------------------------------------- transform edge cases
@needs_augment
def test_mask_none_returns_none() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[96])
    out_cube, out_mask = t(torch.rand(1, 40, 40, 3), None, torch.Generator().manual_seed(0))
    assert out_mask is None
    assert out_cube.shape == (1, 96, 96, 3)


@needs_augment
def test_bool_mask_supported_and_binary_preserved() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[64])
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask[0, 8:16, 8:16] = True
    _, out_mask = t(torch.rand(1, 32, 32, 2), mask, torch.Generator().manual_seed(0))
    assert out_mask.dtype == torch.bool
    assert out_mask.any() and not out_mask.all()  # region survives nearest resize


@needs_augment
def test_upscale_and_downscale_both_work() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    up = RandomMultiScaleResize(scales=[128])(  # 50 -> 128 (upscale)
        torch.rand(1, 50, 50, 3), None, torch.Generator().manual_seed(0)
    )[0]
    down = RandomMultiScaleResize(scales=[24])(  # 50 -> 24 (downscale)
        torch.rand(1, 50, 50, 3), None, torch.Generator().manual_seed(0)
    )[0]
    assert up.shape[1:3] == (128, 128) and down.shape[1:3] == (24, 24)


@needs_augment
def test_wavelengths_argument_is_ignored() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[48])
    out, _ = t(torch.rand(1, 30, 30, 4), None, torch.Generator().manual_seed(0), [500.0] * 4)
    assert out.shape == (1, 48, 48, 4)


@needs_augment
def test_mask_shape_mismatch_raises() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[48])
    with pytest.raises(ValueError, match="spatial shapes must match"):
        t(
            torch.rand(1, 30, 30, 2),
            torch.zeros(1, 30, 29, dtype=torch.int32),
            torch.Generator().manual_seed(0),
        )


@needs_augment
def test_single_scale_always_that_scale() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    t = RandomMultiScaleResize(scales=[72])
    for seed in (0, 1, 2):
        out, _ = t(torch.rand(1, 30, 30, 1), None, torch.Generator().manual_seed(seed))
        assert out.shape[1:3] == (72, 72)


@needs_augment
def test_invalid_scale_values_raise() -> None:
    from cuvis_ai_rfdetr.transforms import RandomMultiScaleResize

    with pytest.raises(ValueError, match="invalid scales"):
        RandomMultiScaleResize(scales=[])
    with pytest.raises(ValueError, match="invalid scales"):
        RandomMultiScaleResize(scales=[96, 0])


# --------------------------------------------- stage-aware resize edge cases
def test_resize_for_stage_val_and_test_always_fixed() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    x = torch.rand(1, 3, 456, 456)  # divisible by 24
    for stage in (ExecutionStage.VAL, ExecutionStage.TEST):
        assert node._resize_for_stage(x, stage).shape[-2:] == (432, 432)


def test_resize_for_stage_train_nonsquare_divisible_resizes() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    x = torch.rand(1, 3, 480, 504)  # both divisible by 24 but NOT square
    assert node._resize_for_stage(x, ExecutionStage.TRAIN).shape[-2:] == (432, 432)


def test_every_native_scale_passes_through_at_train() -> None:
    # contract glue: every size the transform can emit is passthrough-compatible
    from cuvis_ai_schemas.enums import ExecutionStage

    node = _bare_trainable()
    for s in node.multi_scale_scales():
        x = torch.rand(1, 3, s, s)
        assert node._resize_for_stage(x, ExecutionStage.TRAIN) is x, f"scale {s} resized"
