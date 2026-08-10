# Changelog

## [Unreleased]

### Added
- Added `PercentileComposite`: three-band false-color composite node with per-band full-frame percentile stretch and `*255 + 0.5` quantization — arithmetic-identical to JPEG tile exporters, so downstream uint8 conversion recovers the exact exporter bytes.
- Added parity hyperparameters to `RFDETRDetector` and `RFDETRSegmenter` for byte-faithful reproduction of file-based evaluation harnesses: `jpeg_roundtrip`/`jpeg_quality` (in-memory JPEG encode/decode of each model input — the compression is part of such harnesses' score definition), `class_filter` (single-foreground-class scoring, applied before NMS/paste), and `score_reduction="top_frac_mean"` + `top_frac` (image score = mean of the top pixel fraction of the score map, integer-floor top-k; default remains `"max_conf"`).
- Added `cuvis_ai_rfdetr.functional.jpeg_roundtrip`, `top_frac_mean`, and `resolve_band_indices` pure helpers (+ parity unit tests).

- Added `RFDETRSegmenter`: RF-DETR-Seg instance segmentation inference node (all sizes, Apache-2.0 tier) emitting a per-pixel mask-score map, detections, and image-level score.
- Added `RFDETRTrainable`: trainable RF-DETR node (detection + segmentation variants) with the LW-DETR module registered as a submodule — parameters visible to `GradientTrainer`, weights round-trip through pipeline save/load, Roboflow `.pth` checkpoints load via `checkpoint_path`.
- Added `RFDETRCriterionLoss`: RF-DETR `SetCriterion` (Hungarian matcher + weighted cls/bbox/giou and mask losses) as a train/val/test loss node.
- Added `cuvis_ai_rfdetr.functional`: shared tile-merge helpers and `targets_from_mask` (connected components → normalized-cxcywh DETR targets, optional per-instance masks).
- Added tiled inference to `RFDETRDetector` (`tiling="tiled"` default: full-width row strips + NMS merge, reproducing the tiled evaluation protocol; `"whole"` kept as option) plus `resolution` passthrough.
- Added a `[train]` extra installing `rfdetr[train]` for the trainable / loss nodes.

### Changed
- Changed `plugins.yaml` to register the five nodes via full module paths.
- Changed dependencies: added `scipy>=1.10` (connected-component target building) and `pillow` (JPEG round-trip helper).
- Documented that `resolution` must be set to a fine-tuned checkpoint's training resolution (the constructor does not read it from the file).

## 0.1.0 - 2026-07-26

### Added
- Initial release: `RFDETRDetector` inference node, plugin manifest, contract tests.
