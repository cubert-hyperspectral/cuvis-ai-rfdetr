"""ScoreFusion: each mode computes the right elementwise combination; port contract; hparam validation."""

import pytest
import torch

from cuvis_ai_rfdetr.node.score_fusion import ScoreFusion


def _maps():
    a = torch.tensor([0.2, 0.8, 0.9, 0.4]).reshape(1, 2, 2, 1)
    b = torch.tensor([0.6, 0.4, 0.1, 0.5]).reshape(1, 2, 2, 1)
    return a, b


def test_modes_match_reference():
    a, b = _maps()
    assert torch.allclose(ScoreFusion(mode="min")(a=a, b=b)["scores"], torch.minimum(a, b))
    assert torch.allclose(ScoreFusion(mode="max")(a=a, b=b)["scores"], torch.maximum(a, b))
    assert torch.allclose(ScoreFusion(mode="mean")(a=a, b=b)["scores"], 0.5 * (a + b))
    assert torch.allclose(
        ScoreFusion(mode="wmean", weight=0.6)(a=a, b=b)["scores"], 0.6 * a + 0.4 * b
    )
    assert torch.allclose(
        ScoreFusion(mode="gmean")(a=a, b=b)["scores"], torch.sqrt(a * b), atol=1e-6
    )


def test_port_contract():
    a, b = _maps()
    out = ScoreFusion(mode="gmean")(a=a, b=b)["scores"]
    assert out.shape == (1, 2, 2, 1) and out.dtype == torch.float32


def test_hparams_round_trip_and_validation():
    node = ScoreFusion(mode="mean", weight=0.7)
    assert node.hparams["mode"] == "mean" and node.hparams["weight"] == 0.7
    with pytest.raises(ValueError):
        ScoreFusion(mode="median")
    with pytest.raises(ValueError):
        ScoreFusion(mode="wmean", weight=1.5)
