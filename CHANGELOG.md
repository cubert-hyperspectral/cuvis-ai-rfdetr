# Changelog

## [Unreleased]

### Added
- Added `RFDETRSegmenter`: RF-DETR-Seg instance segmentation inference node (all sizes, Apache-2.0 tier) emitting a per-pixel mask-score map, detections, and image-level score.
- Added `RFDETRTrainable`: trainable RF-DETR node (detection + segmentation variants) with the LW-DETR module registered as a submodule — parameters visible to `GradientTrainer`, weights round-trip through pipeline save/load, Roboflow `.pth` checkpoints load via `checkpoint_path`.
- Added `RFDETRCriterionLoss`: RF-DETR `SetCriterion` (Hungarian matcher + weighted cls/bbox/giou and mask losses) as a train/val/test loss node.
- Added `cuvis_ai_rfdetr.functional`: shared tile-merge helpers and `targets_from_mask` (connected components → normalized-cxcywh DETR targets, optional per-instance masks).
- Added tiled inference to `RFDETRDetector` (`tiling="tiled"` default: full-width row strips + NMS merge, reproducing the tiled evaluation protocol; `"whole"` kept as option) plus `resolution` passthrough.
- Added a `[train]` extra installing `rfdetr[train]` for the trainable / loss nodes.

### Changed
- Changed `plugins.yaml` to register the four nodes via full module paths.
- Changed dependencies: added `scipy>=1.10` (connected-component target building).

## 0.1.0 - 2026-07-26

### Added
- Initial release: `RFDETRDetector` inference node, plugin manifest, contract tests.
