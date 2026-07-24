"""Manifest and node-contract tests.

Runnable with only ``cuvis-ai-core``, ``cuvis-ai-schemas`` and ``torch``
installed: the node imports ``rfdetr`` lazily inside ``forward``, so nothing
here requires the RF-DETR package. The missing-package error test runs only
when ``rfdetr`` is actually absent from the environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector

RFDETR_INSTALLED = importlib.util.find_spec("rfdetr") is not None


def test_node_constructs_without_rfdetr() -> None:
    """Constructing with default hparams must not import (or need) rfdetr."""
    node = RFDETRDetector()
    assert node.checkpoint_path is None
    assert node.variant == "medium"
    assert node.threshold == 0.5
    assert node.resolution is None
    assert node._model is None  # lazy: nothing constructed yet


def test_input_specs() -> None:
    """The node consumes a float32 [B, H, W, 3] RGB / false-color image."""
    spec = RFDETRDetector.INPUT_SPECS["rgb_image"]
    assert set(RFDETRDetector.INPUT_SPECS) == {"rgb_image"}
    assert spec.dtype is torch.float32
    assert spec.shape == (-1, -1, -1, 3)


def test_output_specs() -> None:
    """scores [B,H,W,1], detections (list), anomaly_score [B]."""
    specs = RFDETRDetector.OUTPUT_SPECS
    assert set(specs) == {"scores", "detections", "anomaly_score"}
    assert specs["scores"].dtype is torch.float32
    assert specs["scores"].shape == (-1, -1, -1, 1)
    assert specs["detections"].dtype is list
    assert specs["detections"].shape == ()
    assert specs["anomaly_score"].dtype is torch.float32
    assert specs["anomaly_score"].shape == (-1,)


def test_hparams_round_trip() -> None:
    """All hparams are captured for pipeline serialization."""
    node = RFDETRDetector(
        checkpoint_path="weights/checkpoint_best_total.pth",
        variant="small",
        threshold=0.25,
        resolution=512,
    )
    assert node.hparams["checkpoint_path"] == "weights/checkpoint_best_total.pth"
    assert node.hparams["variant"] == "small"
    assert node.hparams["threshold"] == 0.25
    assert node.hparams["resolution"] == 512


def test_invalid_variant_rejected() -> None:
    """Only the Apache-2.0 tier (nano/small/medium/large) is accepted."""
    with pytest.raises(ValueError, match="variant"):
        RFDETRDetector(variant="2xl")


def test_invalid_threshold_rejected() -> None:
    """Threshold must stay within [0, 1]."""
    with pytest.raises(ValueError, match="threshold"):
        RFDETRDetector(threshold=1.5)


def test_bad_input_shape_rejected_before_model_init() -> None:
    """Shape validation fires before any rfdetr import is attempted."""
    node = RFDETRDetector()
    with pytest.raises(ValueError, match=r"\[B, H, W, 3\]"):
        node.forward(rgb_image=torch.zeros(1, 8, 8, 4))


@pytest.mark.skipif(
    RFDETR_INSTALLED,
    reason="rfdetr is installed; the missing-package error path is unreachable",
)
def test_forward_without_rfdetr_raises_clear_error() -> None:
    """forward must fail with an ImportError that names the missing package."""
    node = RFDETRDetector()
    with pytest.raises(ImportError, match="rfdetr"):
        node.forward(rgb_image=torch.zeros(1, 8, 8, 3))


@pytest.mark.integration
def test_manifest_loads_and_registers_node() -> None:
    """NodeRegistry can load the bare manifest and resolve the node class."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    manifest = Path(__file__).resolve().parents[1] / "plugins.yaml"
    registry = NodeRegistry()
    registry.register_plugin(str(manifest))

    detector = registry.get("RFDETRDetector")
    assert detector.__name__ == "RFDETRDetector"
