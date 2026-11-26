"""End-to-end CPU fit with RF-DETR's native recipe through core's GradientTrainer (slow).

Builds RFDETRTrainable (class list head, JPEG, RF-DETR's own transforms, fixed-row val
tiles, EMA) + RFDETRCriterionLoss + RFDETRMapMetric, trains two short epochs with the
``rfdetr_adamw`` optimizer and gradient accumulation, checkpoints on
``MapMetric/best_mAP_50_95``, saves the pipeline and loads it back with
``RFDETRSegmenter(checkpoint_loader="pipeline")``. Needs ``rfdetr[train]`` and the
RF-DETR segmentation weights (downloaded on first use); about 2 minutes on a CPU.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow

CLASSES = ["fo-background", "fo"]


def _sample(height: int, width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(height, width, 3, generator=generator) * 0.3 + 0.3
    mask = torch.zeros(height, width, dtype=torch.int32)
    y = int(torch.randint(20, height - 80, (1,), generator=generator))
    x = int(torch.randint(20, width - 80, (1,), generator=generator))
    image[y : y + 50, x : x + 50] = torch.tensor([0.9, 0.1, 0.1])
    mask[y : y + 50, x : x + 50] = 1
    return image, mask


def test_native_recipe_fit_save_and_pipeline_load(tmp_path) -> None:
    pytest.importorskip("rfdetr.training.model_ema")
    import pytorch_lightning as pl
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import GradientTrainer
    from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig
    from cuvis_ai_schemas.training.callbacks import CallbacksConfig, ModelCheckpointConfig
    from torch.utils.data import DataLoader

    from cuvis_ai_rfdetr.node.rfdetr_loss import RFDETRCriterionLoss
    from cuvis_ai_rfdetr.node.rfdetr_map_metric import RFDETRMapMetric
    from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter
    from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

    class Data(pl.LightningDataModule):
        @staticmethod
        def _loader(height, width, count, base):
            items = [_sample(height, width, base + i) for i in range(count)]
            return DataLoader(
                items,
                batch_size=2,
                collate_fn=lambda batch: {
                    "rgb_image": torch.stack([b[0] for b in batch]),
                    "targets_mask": torch.stack([b[1] for b in batch]),
                },
            )

        def train_dataloader(self):
            return self._loader(405, 405, 4, 0)

        def val_dataloader(self):
            return self._loader(987, 405, 2, 100)

    common = {
        "dataset_dir": ".",
        "variant": "medium",
        "segmentation": True,
        "resolution": 624,
        "class_names": CLASSES,
    }
    trainable = RFDETRTrainable(
        name="RFDETR",
        target_class_id=1,
        jpeg_roundtrip=True,
        native_transforms=True,
        eval_tile_rows=405,
        eval_row_starts=[0, 291, 582],
        ema_decay=0.993,
        ema_tau=100,
        **common,
    )
    loss = RFDETRCriterionLoss(name="CriterionLoss", **common)
    metric = RFDETRMapMetric(name="MapMetric", **common)
    pipeline = CuvisPipeline("native_recipe_fit")
    pipeline.connect(
        (trainable.outputs.outputs, loss.inputs.outputs),
        (trainable.outputs.targets, loss.inputs.targets),
        (trainable.outputs.outputs, metric.inputs.outputs),
        (trainable.outputs.targets, metric.inputs.targets),
        (trainable.outputs.ema_outputs, metric.inputs.ema_outputs),
    )
    pipeline.unfreeze_nodes_by_name(["RFDETR"])
    assert trainable.model.class_embed.out_features == len(CLASSES) + 1

    config = TrainingConfig(
        seed=0,
        optimizer=OptimizerConfig(name="rfdetr_adamw", lr=1e-4, weight_decay=1e-4),
        max_epochs=2,
        accelerator="cpu",
        accumulate_grad_batches=2,
        gradient_clip_val=0.1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
        callbacks=CallbacksConfig(
            checkpoint=ModelCheckpointConfig(
                dirpath=str(tmp_path / "checkpoints"),
                monitor="MapMetric/best_mAP_50_95",
                mode="max",
                save_top_k=1,
            )
        ),
    )
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=Data(),
        loss_nodes=[loss],
        metric_nodes=[metric],
        training_config=config,
    )
    trainer.fit()

    # 2 epochs x 2 batches, accumulation 2: one optimizer step per epoch (2 updates)
    # plus RF-DETR's epoch-end update of epoch 0 (applied at the start of epoch 1).
    assert int(trainable._ema_count) == 3
    logged = trainer.trainer.callback_metrics
    assert {"MapMetric/segm_mAP_50_95", "MapMetric/ema_segm_mAP_50_95"} <= set(logged)
    best = float(logged["MapMetric/best_mAP_50_95"])
    assert best == max(
        float(logged["MapMetric/segm_mAP_50_95"]), float(logged["MapMetric/ema_segm_mAP_50_95"])
    )

    pipeline.save_to_file(str(tmp_path / "run.yaml"))
    segmenter = RFDETRSegmenter(
        checkpoint_path=str(tmp_path / "run.pt"),
        checkpoint_loader="pipeline",
        class_filter=1,
        threshold=0.02,
    )
    model = segmenter._ensure_model()
    expected = trainable.ema_model if segmenter.loaded_stream == "ema" else trainable.model
    loaded = model.model.model.state_dict()
    assert all(torch.equal(loaded[k].cpu(), v.cpu()) for k, v in expected.state_dict().items())
    out = segmenter.forward(rgb_image=_sample(987, 405, 100)[0].unsqueeze(0))
    assert out["scores"].shape == (1, 987, 405, 1)
