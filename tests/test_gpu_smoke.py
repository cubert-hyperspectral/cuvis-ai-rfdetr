"""GPU smoke tests (slow): require CUDA + rfdetr[train]; excluded from CI (-m "not slow").

Prove the trainable path end to end on synthetic data: forward through
``RFDETRTrainable``, Hungarian loss via ``RFDETRCriterionLoss``, gradients
reach the model, one optimizer step reduces the loss, and the node state_dict
round-trips (the weight-transfer contract).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

RFDETR = importlib.util.find_spec("rfdetr") is not None
CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.slow

DATASET_DIR = "/mnt/data/dev/rfdetr_data/rgb"  # any Roboflow-COCO export


def _have_dataset() -> bool:
    import os

    return os.path.isdir(DATASET_DIR)


@pytest.mark.skipif(not (RFDETR and CUDA and _have_dataset()), reason="needs cuda+rfdetr+dataset")
def test_trainable_loss_backward_and_state_dict_round_trip() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    from cuvis_ai_rfdetr.node.rfdetr_loss import RFDETRCriterionLoss
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    node = RFDETRTrainable(dataset_dir=DATASET_DIR, variant="medium").to("cuda")
    loss_node = RFDETRCriterionLoss(dataset_dir=DATASET_DIR, variant="medium").to("cuda")
    node.unfreeze()
    node.train()
    loss_node.train()

    torch.manual_seed(0)
    rgb = torch.rand(2, 128, 96, 3, device="cuda")
    mask = torch.zeros(2, 128, 96, dtype=torch.int32, device="cuda")
    mask[0, 30:50, 20:40] = 1
    mask[1, 60:80, 50:70] = 1
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0)

    opt = torch.optim.AdamW([p for p in node.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for _ in range(4):
        out = node(rgb_image=rgb, targets_mask=mask, context=ctx)
        loss = loss_node(outputs=out["outputs"], targets=out["targets"])["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    grads = sum(1 for p in node.model.parameters() if p.grad is not None)
    assert grads > 0, "no gradients reached the RF-DETR model"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    # weight-transfer contract: state_dict round-trips into a fresh node
    sd = node.state_dict()
    node2 = RFDETRTrainable(dataset_dir=DATASET_DIR, variant="medium")
    node2.load_state_dict(sd)
    p1 = next(iter(node.model.parameters())).detach().cpu()
    p2 = next(iter(node2.model.parameters())).detach().cpu()
    assert torch.equal(p1, p2)


@pytest.mark.skipif(not (RFDETR and CUDA and _have_dataset()), reason="needs cuda+rfdetr+dataset")
def test_trainable_segmentation_variant_loss_runs() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    from cuvis_ai_rfdetr.node.rfdetr_loss import RFDETRCriterionLoss
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    node = RFDETRTrainable(dataset_dir=DATASET_DIR, variant="medium", segmentation=True).to("cuda")
    loss_node = RFDETRCriterionLoss(
        dataset_dir=DATASET_DIR, variant="medium", segmentation=True
    ).to("cuda")
    node.train()
    loss_node.train()
    rgb = torch.rand(1, 128, 96, 3, device="cuda")
    mask = torch.zeros(1, 128, 96, dtype=torch.int32, device="cuda")
    mask[0, 40:70, 30:60] = 1
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0)
    out = node(rgb_image=rgb, targets_mask=mask, context=ctx)
    assert "pred_masks" in out["outputs"] or any("mask" in k for k in out["outputs"])
    loss = loss_node(outputs=out["outputs"], targets=out["targets"])["loss"]
    loss.backward()
    assert torch.isfinite(loss)
