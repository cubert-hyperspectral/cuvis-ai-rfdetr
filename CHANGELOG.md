# Changelog

## [Unreleased]

### Added
- `EmaCallback` (`cuvis_ai_rfdetr.training`): the native RF-DETR weight EMA as a Lightning
  callback for cuvis-ai's `GradientTrainer` (which accepts explicit callbacks). Wraps rfdetr's
  own `ModelEma` (decay warm-up `decay*(1-exp(-updates/tau))`), targets the `RFDETRTrainable`'s
  registered LW-DETR module by node name, updates every `update_interval` train batches,
  persists through Lightning checkpoint state (resume-safe), and can write the averaged
  weights on fit end (`save_path`) or on demand (`save()`).
- `RFDETRGradientTrainer` (`cuvis_ai_rfdetr.training`): a `GradientTrainer` whose
  `configure_optimizers` feeds the named `RFDETRTrainable`'s native param groups to the standard
  optimizer/scheduler registry (base lr from the optimizer config; `lr_encoder` /
  `lr_vit_layer_decay` / `lr_component_decay` as constructor knobs). Other unfrozen pipeline
  parameters join as a final base-lr group, preserving the base trainer's optimize-everything
  contract. Kept in this plugin by design (no cuvis-ai-core change); candidate for later
  migration into `GradientTrainer` as an optional node param-group protocol.
- `RFDETRTrainable.get_param_groups(...)`: the native LW-DETR optimizer param groups
  (encoder at `lr_encoder` with per-block ViT layer decay, decoder at
  `lr * lr_component_decay`, rest at `lr`) built by rfdetr's own `get_param_dict` against the
  node's model. Like the native trainer's `args`, the namespace handed to `get_param_dict` is
  a flat merge of the model config (e.g. `out_feature_indexes`) and the train config, with
  explicit overrides on top. Feed the returned param-group dicts to a torch optimizer to
  reproduce the native loop's learning-rate structure.

## [0.3.0] - 2026-08-10

### Added
- Added `checkpoint_loader` hyperparameter to `RFDETRDetector` and `RFDETRSegmenter` (`"constructor"` default / `"from_checkpoint"`). `"from_checkpoint"` delegates to `rfdetr.RFDETR.from_checkpoint`, taking the model class and configuration from the checkpoint itself and falling back to **class defaults for fields the checkpoint does not carry — including resolution**. This reproduces harnesses that load checkpoints the same way; the two loaders can yield materially different scores from the same checkpoint (verified: a fine-tuned checkpoint without a recorded resolution evaluates at the class default under `from_checkpoint`, not at its training resolution). `variant` is validated against the resolved class; an explicit `resolution` is forwarded as an override.

### Changed
- Corrected the `resolution` guidance: the previous note ("set resolution to the checkpoint's training resolution") reproduces the *constructor* path only. To reproduce `from_checkpoint`-based harnesses, use `checkpoint_loader="from_checkpoint"` and leave `resolution` unset.

## [0.2.0] - 2026-08-10

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
