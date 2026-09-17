"""base_kwargs: one construction path for cuvis-ai-core < 0.15 and >= 0.15."""

import pytest

import cuvis_ai_rfdetr._compat as compat
from cuvis_ai_rfdetr._compat import base_kwargs
from cuvis_ai_rfdetr.node.percentile_composite import PercentileComposite
from cuvis_ai_rfdetr.node.rfdetr_loss import _LOSS_STAGES, RFDETRCriterionLoss


def test_installed_core_pops_name_and_leaves_hparams():
    kw = {"name": "seg", "other": 1}
    out = base_kwargs(kw)
    assert out["name"] == "seg"
    assert kw == {"other": 1}
    assert ("execution_stages" in out) == compat.CORE_HAS_INSTANCE_STAGES


def test_fixed_stages_forwarded_on_old_core():
    if not compat.CORE_HAS_INSTANCE_STAGES:
        pytest.skip("stages are class-level on this core")
    out = base_kwargs({}, execution_stages=_LOSS_STAGES)
    assert out["execution_stages"] == _LOSS_STAGES


def test_new_core_branch(monkeypatch):
    monkeypatch.setattr(compat, "CORE_HAS_INSTANCE_STAGES", False)
    assert base_kwargs({"name": "n", "execution_stages": None}) == {"name": "n"}
    # fixed stages live on the class on new cores — nothing is forwarded
    assert base_kwargs({"name": "n"}, execution_stages=_LOSS_STAGES) == {"name": "n"}
    with pytest.raises(TypeError):
        base_kwargs({"execution_stages": ["inference"]})
    # nodes construct through the new-core branch without passing execution_stages
    node = PercentileComposite(name="pc")
    assert isinstance(node, PercentileComposite)


def test_loss_declares_class_level_stages():
    assert RFDETRCriterionLoss.EXECUTION_STAGES == _LOSS_STAGES
