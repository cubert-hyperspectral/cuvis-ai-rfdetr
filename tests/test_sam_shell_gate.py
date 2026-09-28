"""SamShellGate: raw-cosine spectral angle to a reference gates the score map; scale-invariance; port contract."""

import pytest
import torch

from cuvis_ai_rfdetr.node.sam_shell_gate import SamShellGate

pytestmark = pytest.mark.unit


def _cube_scores():
    # ref = [1,0,0]; 4 pixels: [1,0,0] (0deg), [2,0,0] (0deg, scaled), [0,1,0] (90deg), [1,1,0] (45deg)
    cube = torch.tensor([[[[1.0, 0, 0], [2.0, 0, 0]], [[0, 1.0, 0], [1.0, 1.0, 0]]]])  # [1,2,2,3]
    scores = torch.tensor([[[[0.9], [0.8]], [[0.7], [0.6]]]])  # [1,2,2,1]
    return cube, scores


def test_gate_matches_reference():
    cube, scores = _cube_scores()
    # T=45: keep 0deg,0deg,45deg ; drop 90deg
    out45 = SamShellGate(reference=[1.0, 0.0, 0.0], threshold_deg=45.0)(cube=cube, scores=scores)[
        "scores"
    ]
    assert torch.allclose(out45, torch.tensor([[[[0.9], [0.8]], [[0.0], [0.6]]]]), atol=1e-5)
    # T=30: keep only the two 0deg pixels ; drop 90deg and 45deg
    out30 = SamShellGate(reference=[1.0, 0.0, 0.0], threshold_deg=30.0)(cube=cube, scores=scores)[
        "scores"
    ]
    assert torch.allclose(out30, torch.tensor([[[[0.9], [0.8]], [[0.0], [0.0]]]]), atol=1e-5)


def test_scale_invariance():
    # a pixel and its scaled copy get the same gate decision (cosine ignores magnitude)
    cube, scores = _cube_scores()
    out = SamShellGate(reference=[1.0, 0.0, 0.0], threshold_deg=10.0)(cube=cube, scores=scores)[
        "scores"
    ]
    assert out[0, 0, 0, 0] == pytest.approx(0.9)  # [1,0,0]
    assert out[0, 0, 1, 0] == pytest.approx(0.8)  # [2,0,0] scaled -> same 0deg -> kept


def test_port_contract():
    cube, scores = _cube_scores()
    out = SamShellGate(reference=[1.0, 0.0, 0.0], threshold_deg=45.0)(cube=cube, scores=scores)[
        "scores"
    ]
    assert out.shape == (1, 2, 2, 1)
    assert out.dtype == torch.float32


def test_reference_validation():
    with pytest.raises(ValueError):
        SamShellGate(reference=[1.0], threshold_deg=11.0)
