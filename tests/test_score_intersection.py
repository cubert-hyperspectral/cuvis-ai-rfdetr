"""ScoreIntersection: golden reference (min), AND semantics under a threshold, port contract."""

import torch

from cuvis_ai_rfdetr.node.score_intersection import ScoreIntersection


def test_golden_reference_is_elementwise_min():
    g = torch.Generator().manual_seed(0)
    a = torch.rand((2, 4, 5, 1), generator=g)
    b = torch.rand((2, 4, 5, 1), generator=g)
    out = ScoreIntersection().forward(a=a, b=b)["scores"]
    assert torch.allclose(out, torch.minimum(a, b), atol=1e-7)


def test_threshold_equals_logical_and():
    a = torch.tensor([[[[0.9], [0.2], [0.7], [0.4]]]])
    b = torch.tensor([[[[0.8], [0.9], [0.3], [0.6]]]])
    out = ScoreIntersection().forward(a=a, b=b)["scores"] >= 0.5
    assert torch.equal(out, (a >= 0.5) & (b >= 0.5))


def test_port_contract_and_no_inplace():
    a = torch.rand((1, 3, 3, 1))
    b = torch.rand((1, 3, 3, 1))
    a0, b0 = a.clone(), b.clone()
    out = ScoreIntersection().forward(a=a, b=b)["scores"]
    assert out.shape == (1, 3, 3, 1) and out.dtype == torch.float32
    assert torch.equal(a, a0) and torch.equal(b, b0)
