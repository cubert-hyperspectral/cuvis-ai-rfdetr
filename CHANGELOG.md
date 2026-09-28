# Changelog

## [Unreleased]

### Added
- `tensorrt` extra for the TensorRT backend: `tensorrt-cu13==10.15.1.29` on Linux aarch64 (Jetson Thor-class,
  CUDA 13 torch), `tensorrt-cu12==10.15.1.29` on Windows and Linux x86_64 (CUDA 12 torch), plus `onnx` for
  building engines. Pinned to TensorRT 10 (TensorRT 11 dropped the FP16 builder flag); engines are tied to the
  TensorRT version. cuvis.next does not install extras of node plugins yet (cuvis-ai-core#89, #19).

## [0.5.0] - 2026-09-28

### Added
- `RFDETRSegmenter` TensorRT backend: `backend="tensorrt"` runs a TensorRT engine compiled from the network (via
  rfdetr's own ONNX export) in place of the PyTorch network, with rfdetr's pre- and post-processing unchanged
  around it. `precision` selects the engine: `fp32` is TensorRT's default build with TF32 allowed, `fp16` uses the
  FP16 builder flag (TensorRT 10). `engine_dir` defaults to `<checkpoint>.trt/` and holds per-machine engines
  named by precision, resolution, GPU and TensorRT version, each with a JSON build record; the checkpoint MD5 is
  checked at load. New module `cuvis_ai_rfdetr.trt_engine` builds engines and runs them on torch's CUDA stream,
  with a CLI: `python -m cuvis_ai_rfdetr.trt_engine build | build-pipeline`. TensorRT and onnx are optional,
  CUDA-specific installs (`tensorrt-cu12` / `tensorrt-cu13`), not plugin dependencies. Tests in
  `tests/test_segmenter_tensorrt.py` are mocked; one `slow` test builds and runs a real engine.
- `RFDETRSegmenter` speed hparams: `fast_paste` (default True) combines all instance masks of a prediction
  with one masked max on the input's device instead of one boolean-indexed CPU write per instance (bit-identical;
  `False` restores the per-instance paste); opt-in `gpu_input` hands rfdetr the frame as a float tensor on the
  input's device, quantized like the uint8 frame and bit-identical to the NumPy path on rfdetr 1.10 (not with
  `jpeg_roundtrip`); opt-in `precision` (`fp32` | `fp16` | `bf16`) and `jit_trace` run rfdetr's
  `model.inference(dtype=..., compile=...)` once when the model is built (`optimize_for_inference` on older
  releases). New helpers `functional.to_unit_frames` and `functional.max_paste_masks`; tests in
  `tests/test_segmenter_speed_options.py`.
- Fair-robust training transforms in `cuvis_ai_rfdetr.transforms` (registered for `AugmentationCompose` via
  `extra_transform_modules`, like `RandomMultiScaleResize`): `RandomZoom` (per-sample zoom out/in at a fixed
  output size — shrink onto a median-filled canvas or crop-and-enlarge, mask resampled nearest-neighbour — for
  camera-height changes), `RandomShading` (smooth spatial darkening field applied to all bands; unlike a global
  gain it survives per-frame min-max normalisation — uneven or missing light), `RandomGammaContrast` (per-sample
  gamma + contrast around the channel mean, floored at 0) and `RandomGaussianBlur` (per-sample separable Gaussian
  blur with reflect padding — defocus). All draw from the compose's shared generator, apply per sample with
  `prob`, keep shapes/dtypes and leave the mask aligned; tests in `tests/test_fair_transforms.py`.
  Planned move to `cuvis-ai-augment` (#13).
- `SamShellGate` node — gates a score map by the full-spectrum spectral angle to a fixed reference spectrum
  (raw cosine, so illumination-scale invariant): pixels whose angle exceeds `threshold_deg` are zeroed, so
  look-alike objects with a different spectrum drop out of the mask while the real ones stay; tests in
  `tests/test_sam_shell_gate.py`. Planned move to `cuvis-ai` as a generic `SpectralAngleGate` (#14).
- `RFDETRTrainable` gains `multiclass_targets` (default False): `targets_from_mask` builds per-class DETR targets
  (label = mask class id - 1) instead of collapsing all foreground to one class — needed to train multi-class (e.g.
  shell/fo/fake) models in-pipeline.
- `FixedPCAProjection` node — cuvis-ai's `TrainablePCA` with a frozen, file-loaded projection (mean/components/
  1-99% range from an `.npz` hparam) + fixed unit scaling and [0,1] clamp, so a downstream model always sees the exact
  projection it was trained on (the stateless `PCA` node refits per frame - unusable in front of a trained model).
  Reproduces the exporter's full input recipe: an `input_global_minmax` step (on by default) min-maxes each cube
  globally to [0,1] before projecting, so the fixed projection is invariant to the caller's absolute reflectance scale
  (cuvis.next's `CU3SDataNode` delivers raw-scale reflectance, not [0,1], which otherwise collapses the projection —
  everything clamps to 1 and the mask goes empty). The min-max is global (one min/max over all H*W*C), matching the
  export; per-channel scaling (`MinMaxNormalizer`) does not reproduce it. Exposes only the `projected` output port (the
  parent's `components`/`explained_variance_ratio` ports are dropped — non-image ports break cuvis.next's per-output
  display/mask handling). Planned move to `cuvis-ai` (#15).
- `ScoreFusion` node — combine two score maps by `min`/`max`/`mean`/`gmean`/`wmean` (generalizes
  `ScoreIntersection`) for two-segmenter ensembles; the soft rules keep recall that the hard AND (`min`) loses.
  Planned move to `cuvis-ai` (#14).
- `ScalarMinMaxBandSlice` (`cuvis_ai_rfdetr.node.scalar_minmax_bandslice`): cube -> 3-band false-colour
  composite using ONE scalar min-max over the whole cube (all bands together, so relative band
  intensities are preserved) followed by nearest-band selection (`argmin |wavelength - nm|`), output
  float32 in [0, 1]. This is the input recipe some 3-band RF-DETR checkpoints were trained on;
  `PercentileComposite` (per-band percentile stretch) and cuvis-ai's `FixedWavelengthSelector`
  (per-band per-frame / running normalisation) do not reproduce it. Verified byte-exact against the
  training exporter. Planned move to cuvis-ai's channel selectors (#15).
- `ScoreIntersection` (`cuvis_ai_rfdetr.node.score_intersection`): elementwise minimum of two
  `[B, H, W, 1]` score maps — the soft AND of two segmenters (thresholding the output at t is exactly
  "both >= t"). Enables two-model ensembles (e.g. RGB ∩ 870-nm RF-DETR) inside one pipeline; torch-native,
  stateless. Same as `ScoreFusion(mode="min")`; retirement tracked in #14.
- `CarlSegmenter` (`cuvis_ai_rfdetr.node.carl_segmenter`): the CARL (IMSY-DKFZ) 61-band ViT-Adapter/UperNet
  segmenter as a node — per-cube min-max + z-score + resize preprocessing (the training recipe), lazy import of the
  CARL repo from `carl_repo`, softmax score map of one class + argmax labels; preprocessing runs on the model
  device and the forward uses bf16 autocast by default (`precision` hparam: bf16 | fp16 | fp32) — 1.6x faster
  than fp32 with identical masks; `band_step` (every k-th band + its wavelengths; k=2 is ~1.45x faster at 0.999
  agreement) and opt-in `compile` + `compile_cache_dir` (torch.compile/inductor, ~1.6x, bit-identical; needs
  triton; warm cache restarts in ~30 s). Planned move to a `cuvis-ai-carl` plugin (#16).

### Changed
- `RFDETRSegmenter` passes `include_source_image=False` to `predict` when the installed rfdetr accepts it (the
  node never reads the attached source image; saves one frame copy per call, predictions unchanged).
- README documents every node (input builders, score fusion / gating, `CarlSegmenter`, the training
  transforms) and the planned moves of the general-purpose nodes to other ecosystem repos (#13-#17).
- `ScalarMinMaxBandSlice` routes its base kwargs through `base_kwargs` like the other nodes (a yaml
  `execution_stages: null` no longer reaches `Node` on cuvis-ai-core >= 0.15) and imports the palette
  enums unconditionally; node docstrings describe the nodes generically. No behaviour change.

### Fixed
- Compatible with cuvis-ai-core >= 0.15, where execution stages became class-level and
  `Node.consume_base_kwargs` / the `execution_stages=` constructor kwarg were removed: the nodes
  now forward base kwargs through `cuvis_ai_rfdetr._compat.base_kwargs`, which keeps the
  pre-0.15 behaviour on older cores (per-instance stage override from the yaml) and passes only
  `name` on newer ones. `RFDETRCriterionLoss` declares its `{train, val, test}` stages as
  `EXECUTION_STAGES` for the new mechanism (the constructor still fixes them on old cores). The shim goes
  once the core floor is >= 0.15 (#17).

## [0.4.0] - 2026-08-14

### Added
- `RFDETRTrainable` gained `checkpoint_loader` (`"constructor"` default | `"from_checkpoint"`),
  mirroring the inference nodes: `"from_checkpoint"` delegates to `rfdetr.RFDETR.from_checkpoint`,
  so the wrapper class and configuration (including the query structure) come from the checkpoint
  itself, with the resolved class validated against `variant`/`segmentation` and `resolution`
  forwarded. Verified tensor-exact against a fine-tuned segmentation checkpoint; note the generic
  constructor path also loads default-config checkpoints exactly — the loader-faithful path
  matters for checkpoints trained at non-default model configs. Incompatible with
  `num_channels != 3` (no channel-inflation in rfdetr's loader; validated).
- `RandomMultiScaleResize` (`cuvis_ai_rfdetr.transforms`): the native RF-DETR multi-scale
  training resize as a cuvis-ai-augment transform, contributed through augment's
  `extra_transform_modules` mechanism (no augment change; cuvis-ai-augment is deliberately not a
  pip dependency — the module import raises a clear hint when it is absent). Draws one square
  size per batch from the native scale set; cube bilinear, mask nearest. Scale math lives in
  `functional.compute_multi_scale_scales` (faithful reimplementation, unit-tested for equality
  against rfdetr's own function) and `RFDETRTrainable.multi_scale_scales()` returns the set for
  the node's actual model config (e.g. SegMedium@624: patch 12 × windows 2 → unit 24).
- `RFDETRTrainable` stage-aware input resize: at TRAIN, an already-square input whose side is a
  multiple of the model's spatial unit (`patch_size * num_windows`) passes through unresized —
  making upstream multi-scale augmentation real instead of being nullified by the fixed resize.
  Val/test/inference input and arbitrary-size train input keep the fixed model-resolution
  resize (unchanged behavior).
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
