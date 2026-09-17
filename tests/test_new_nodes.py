"""Contract tests for the segmentation / trainable / loss nodes and functional helpers.

Everything here runs WITHOUT the rfdetr package installed (module import is
lazy for inference nodes; trainable/loss validate hparams before importing
rfdetr). GPU / rfdetr-dependent behavior is covered by the slow smoke tests.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from cuvis_ai_rfdetr.functional import box_iou, nms_rows, targets_from_mask
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter

RFDETR_INSTALLED = importlib.util.find_spec("rfdetr") is not None


# ---------------------------------------------------------------- functional
def test_targets_from_mask_boxes_normalized() -> None:
    mask = torch.zeros(2, 100, 50, dtype=torch.int32)
    mask[0, 10:20, 5:15] = 1  # one 10x10 component
    mask[1, 0:4, 0:4] = 1
    mask[1, 90:100, 40:50] = 2  # two components in image 1
    targets = targets_from_mask(mask)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (2, 4)
    cx, cy, w, h = targets[0]["boxes"][0].tolist()
    assert abs(cx - 10.0 / 50.0) < 1e-6  # (5+15)/2/50
    assert abs(cy - 15.0 / 100.0) < 1e-6
    assert abs(w - 10.0 / 50.0) < 1e-6
    assert abs(h - 10.0 / 100.0) < 1e-6
    assert targets[0]["labels"].dtype == torch.int64
    assert (targets[0]["labels"] == 0).all()
    assert "masks" not in targets[0]


def test_targets_from_mask_with_masks_and_empty() -> None:
    mask = torch.zeros(1, 32, 32, dtype=torch.int32)
    targets = targets_from_mask(mask, with_masks=True)
    assert targets[0]["boxes"].shape == (0, 4)
    assert targets[0]["masks"].shape == (0, 32, 32)
    mask[0, 4:8, 4:8] = 7
    targets = targets_from_mask(mask, with_masks=True)
    assert targets[0]["masks"].shape == (1, 32, 32)
    assert targets[0]["masks"].sum() == 16


def test_nms_rows_merges_overlaps() -> None:
    rows = [(0, 0, 10, 10, 0.9, 0), (1, 1, 11, 11, 0.8, 0), (50, 50, 60, 60, 0.7, 0)]
    kept = nms_rows(rows, 0.5)
    assert len(kept) == 2
    assert kept[0][4] == 0.9
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


# ---------------------------------------------------------------- segmenter
def test_segmenter_constructs_without_rfdetr() -> None:
    node = RFDETRSegmenter()
    assert node.variant == "medium"
    assert node.tiling == "tiled"
    assert node._model is None


def test_segmenter_hparams_round_trip() -> None:
    node = RFDETRSegmenter(
        checkpoint_path="w.pth", variant="xlarge", threshold=0.25, tiling="whole"
    )
    assert node.hparams["variant"] == "xlarge"
    assert node.hparams["checkpoint_path"] == "w.pth"
    assert node.hparams["tiling"] == "whole"


def test_segmenter_parity_hparams_defaults_and_round_trip() -> None:
    node = RFDETRSegmenter()
    assert node.jpeg_roundtrip is False
    assert node.jpeg_quality == 95
    assert node.class_filter is None
    assert node.score_reduction == "max_conf"
    assert node.top_frac == 0.001

    node = RFDETRSegmenter(
        jpeg_roundtrip=True,
        jpeg_quality=90,
        class_filter=1,
        score_reduction="top_frac_mean",
        top_frac=0.002,
    )
    assert node.hparams["jpeg_roundtrip"] is True
    assert node.hparams["jpeg_quality"] == 90
    assert node.hparams["class_filter"] == 1
    assert node.hparams["score_reduction"] == "top_frac_mean"
    assert node.hparams["top_frac"] == 0.002


def test_segmenter_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="variant"):
        RFDETRSegmenter(variant="mega")
    with pytest.raises(ValueError, match="threshold"):
        RFDETRSegmenter(threshold=2.0)
    with pytest.raises(ValueError, match="tiling"):
        RFDETRSegmenter(tiling="mosaic")
    with pytest.raises(ValueError, match="jpeg_quality"):
        RFDETRSegmenter(jpeg_quality=0)
    with pytest.raises(ValueError, match="score_reduction"):
        RFDETRSegmenter(score_reduction="median")
    with pytest.raises(ValueError, match="top_frac"):
        RFDETRSegmenter(top_frac=0.0)


def test_checkpoint_loader_validation_and_round_trip() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector

    assert RFDETRSegmenter().checkpoint_loader == "constructor"  # back-compat default
    node = RFDETRSegmenter(checkpoint_loader="from_checkpoint")
    assert node.hparams["checkpoint_loader"] == "from_checkpoint"
    with pytest.raises(ValueError, match="checkpoint_loader"):
        RFDETRSegmenter(checkpoint_loader="magic")
    with pytest.raises(ValueError, match="checkpoint_loader"):
        RFDETRDetector(checkpoint_loader="auto")


@pytest.mark.skipif(not RFDETR_INSTALLED, reason="needs rfdetr to monkeypatch its loader")
def test_from_checkpoint_variant_mismatch_raises(monkeypatch) -> None:
    """A checkpoint resolving to a different class than `variant` must fail loudly."""
    import rfdetr

    class RFDETRSegLarge:  # noqa: N801 - mimics the resolved class name
        pass

    monkeypatch.setattr(
        rfdetr.RFDETR, "from_checkpoint", classmethod(lambda cls, p, **kw: RFDETRSegLarge())
    )
    node = RFDETRSegmenter(
        checkpoint_path="w.pth", variant="medium", checkpoint_loader="from_checkpoint"
    )
    with pytest.raises(RuntimeError, match="resolved to RFDETRSegLarge"):
        node._build_model()


@pytest.mark.skipif(not RFDETR_INSTALLED, reason="needs rfdetr to monkeypatch its loader")
def test_from_checkpoint_forwards_resolution_override(monkeypatch) -> None:
    import rfdetr

    seen: dict = {}

    class RFDETRSegMedium:  # noqa: N801
        pass

    def fake(cls, path, **kw):
        seen.update(kw, path=path)
        return RFDETRSegMedium()

    monkeypatch.setattr(rfdetr.RFDETR, "from_checkpoint", classmethod(fake))
    node = RFDETRSegmenter(
        checkpoint_path="w.pth", checkpoint_loader="from_checkpoint", resolution=624
    )
    assert type(node._build_model()).__name__ == "RFDETRSegMedium"
    assert seen == {"path": "w.pth", "resolution": 624}
    # without resolution set, nothing is forwarded (checkpoint/class defaults rule)
    seen.clear()
    RFDETRSegmenter(checkpoint_path="w.pth", checkpoint_loader="from_checkpoint")._build_model()
    assert seen == {"path": "w.pth"}


def test_detector_parity_hparams_and_validation() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector

    node = RFDETRDetector(jpeg_roundtrip=True, class_filter=1, score_reduction="top_frac_mean")
    assert node.hparams["jpeg_roundtrip"] is True
    assert node.hparams["class_filter"] == 1
    assert node.hparams["score_reduction"] == "top_frac_mean"
    assert node.hparams["top_frac"] == 0.001
    with pytest.raises(ValueError, match="jpeg_quality"):
        RFDETRDetector(jpeg_quality=101)
    with pytest.raises(ValueError, match="score_reduction"):
        RFDETRDetector(score_reduction="p99")
    with pytest.raises(ValueError, match="top_frac"):
        RFDETRDetector(top_frac=1.5)


def test_segmenter_specs() -> None:
    assert set(RFDETRSegmenter.OUTPUT_SPECS) == {"scores", "detections", "anomaly_score"}
    assert RFDETRSegmenter.OUTPUT_SPECS["scores"].shape == (-1, -1, -1, 1)


# ------------------------------------------------------- trainable + loss
def test_trainable_and_loss_import_without_rfdetr() -> None:
    """Class import must never require rfdetr (manifest registration)."""
    from cuvis_ai_rfdetr.node.rfdetr_loss import RFDETRCriterionLoss
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    assert set(RFDETRTrainable.OUTPUT_SPECS) == {"outputs", "targets"}
    assert set(RFDETRCriterionLoss.OUTPUT_SPECS) == {"loss"}
    assert RFDETRTrainable.INPUT_SPECS["targets_mask"].optional


def test_trainable_validates_variant_before_rfdetr_import() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    with pytest.raises(ValueError, match="variant"):
        RFDETRTrainable(dataset_dir="/tmp/ds", variant="xlarge", segmentation=False)


@pytest.mark.skipif(RFDETR_INSTALLED, reason="rfdetr installed; error path unreachable")
def test_trainable_raises_clear_error_without_rfdetr() -> None:
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    with pytest.raises(ImportError, match="rfdetr"):
        RFDETRTrainable(dataset_dir="/tmp/ds")


def test_manifest_lists_all_nodes() -> None:
    import pathlib

    import yaml

    manifest = yaml.safe_load((pathlib.Path(__file__).parent.parent / "plugins.yaml").read_text())
    names = {c["class_name"].rsplit(".", 1)[1] for c in manifest["capabilities"]}
    assert names == {
        "RFDETRDetector",
        "RFDETRSegmenter",
        "RFDETRTrainable",
        "RFDETRCriterionLoss",
        "PercentileComposite",
        "ScalarMinMaxBandSlice",
        "ScoreIntersection",
        "ScoreFusion",
        "FixedPCAProjection",
        "CarlSegmenter",
    }
