# cuvis-ai-rfdetr

[RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's real-time detection
transformer — wrapped as a [cuvis.ai](https://github.com/cubert-hyperspectral)
plugin for object / foreign-object detection and instance segmentation on RGB
and false-color renderings of hyperspectral data.

Nodes:

- inference wrappers `RFDETRDetector`, `RFDETRSegmenter`;
- an in-graph training pair `RFDETRTrainable` + `RFDETRCriterionLoss`;
- false-colour input builders `PercentileComposite`, `ScalarMinMaxBandSlice`,
  `FixedPCAProjection`;
- score-map ensembles and gating `ScoreFusion`, `ScoreIntersection`,
  `SamShellGate`;
- `CarlSegmenter` (CARL 61-band hyperspectral segmenter, optional).

Plus training transforms for `cuvis-ai-augment`'s `AugmentationCompose`
(`cuvis_ai_rfdetr.transforms`). Several of these are general-purpose and are
planned to move to other ecosystem repos — see [Planned moves](#planned-moves).

## Licensing scope

This plugin wraps the **Apache-2.0 tier** of RF-DETR. For detection that is
`nano | small | medium | large`; the whole **segmentation** tier
(`nano … 2xlarge`) ships under Apache-2.0 and is fully exposed. The RF-DETR
detection **XL / 2XL** checkpoints are distributed under the non-open Roboflow
Platform Model License and are **not** included, wrapped, or downloaded by
this package. See `NOTICE` for attribution.

## Install

```bash
pip install "cuvis-ai-rfdetr @ git+https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git@v0.4.0"
```

Dependencies pull in the RF-DETR **core (inference) tier** only. The `rfdetr`
import happens lazily on the first `forward`, so pipelines can be built,
validated, and visualized without it. Add the `[train]` extra for the
in-graph training nodes (`rfdetr[train]`).

## Inference nodes: `RFDETRDetector` / `RFDETRSegmenter`

| Port | Direction | Type / shape | Description |
| --- | --- | --- | --- |
| `rgb_image` | input | `float32 [B, H, W, 3]` | RGB / false-color frame; values 0–1 or 0–255 (auto-detected) |
| `scores` | output | `float32 [B, H, W, 1]` | Detector: box regions filled with `max(existing, confidence)`. Segmenter: per-pixel `max(instance_mask × confidence)` |
| `detections` | output | `list` | Per-image list of `{"xyxy": [x1, y1, x2, y2], "confidence": float, "class_id": int}` |
| `anomaly_score` | output | `float32 [B]` | Image-level score (see `score_reduction`) |

Shared hyperparameters:

- `checkpoint_path` (`str | None`) — fine-tuned weights, passed to the RF-DETR
  model's `pretrain_weights`; `None` loads the official COCO-pretrained
  weights for the variant.
- `variant` (`str`, default `"medium"`) — detector: `nano | small | medium |
  large`; segmenter additionally `xlarge | 2xlarge`.
- `threshold` (`float`, default `0.5`) — confidence threshold for `predict`.
  Note: when downstream scoring uses the dense map (`top_frac_mean`), this
  threshold is part of the score definition — keep it at the evaluation
  protocol's value.
- `resolution` (`int | None`) — input resolution forwarded to the RF-DETR
  constructor; `None` keeps the class default.
- `checkpoint_loader` (`"constructor"` default / `"from_checkpoint"`) — where
  the model *configuration* comes from when loading a fine-tuned checkpoint.
  `"from_checkpoint"` delegates to `rfdetr.RFDETR.from_checkpoint`: class and
  configuration come from the checkpoint, **falling back to class defaults for
  fields it does not carry — including resolution** (fine-tuned checkpoints do
  not necessarily record their training resolution). The two loaders can yield
  materially different scores from the same checkpoint — match the loader your
  reference results were produced with.
- `tiling` (`"tiled"` default / `"whole"`), `tile_rows` (405), `row_starts`
  (`(0, 291, 582)`), `nms_iou` (0.5) — `"tiled"` splits each frame into
  full-width row strips, runs the model per strip, offsets results back, and
  NMS-merges boxes (the segmenter merges strip mask-maps by pixelwise max),
  reproducing tiled evaluation protocols for tall lane crops.

Parity hyperparameters (byte-faithful reproduction of evaluation harnesses
built around per-tile JPEG files and top-fraction image scores):

- `jpeg_roundtrip` (`bool`, default `False`) + `jpeg_quality` (default 95) —
  route every model input (each tile in `"tiled"` mode, the frame in
  `"whole"` mode) through an in-memory PIL JPEG encode/decode before
  prediction. Deliberately lossy: harnesses that persist tiles as `.jpg`
  make the compression part of the score definition; this reproduces their
  scores exactly (byte-identical to a disk save/reopen; pin your Pillow
  version for cross-environment reproducibility).
- `class_filter` (`int | None`) — keep only this `class_id`, dropped before
  NMS/paste/scoring (single-foreground-class protocols).
- `score_reduction` (`"max_conf"` default / `"top_frac_mean"`) + `top_frac`
  (default 0.001) — `"top_frac_mean"` scores an image as the mean of the top
  `top_frac` fraction of the score map with the integer-floor top-k rule
  (e.g. exactly 400 px of a 987×405 map), directly comparable to dense
  segmentation models scored the same way.

Segmenter speed hyperparameters (`RFDETRSegmenter` only). `fast_paste` is on
by default because it changes no number; the other three are opt-in:

| Hparam | Default | What it does | Output vs default |
| --- | --- | --- | --- |
| `fast_paste` | `True` | Combines all instance masks of a prediction with one masked max on the input image's device, instead of one boolean-indexed write per instance into a CPU canvas. `False` restores the per-instance CPU paste. | bit-identical |
| `gpu_input` | `False` | Hands each frame to rfdetr as a `[0, 1]` float tensor on the input's device, quantized exactly like the uint8 frame, instead of copying it to a CPU uint8 NumPy array that rfdetr then copies back. Not combinable with `jpeg_roundtrip`. | bit-identical (rfdetr 1.10, CUDA) |
| `precision` | `"fp32"` | `"fp16"` / `"bf16"`: rfdetr's `model.inference(dtype=...)` exports and casts the network once, when the model is built. Needs CUDA. | rounding: pixels at a downstream threshold can flip |
| `jit_trace` | `False` | `torch.jit.trace` of the exported network (rfdetr's `model.inference(compile=True)`) for its fixed `1 × 3 × resolution × resolution` input; removes the Python overhead of the forward pass. Traces on the first frame (a few seconds). | as `precision` |

The node never reads the source image rfdetr can attach to its predictions,
so it passes `include_source_image=False` whenever the installed rfdetr
accepts it. This saves one frame copy per call and leaves the predictions
unchanged.

The wrapped model manages its own device placement and is intentionally not a
registered submodule: `node.to(...)` does not move it.

## Input builder: `PercentileComposite`

`cube [B, H, W, C] float32` + `wavelengths [B, C] int32` → `rgb_image
[B, H, W, 3]` float32 (integer-valued 0–255).

Selects the channels nearest to `bands_nm` (nearest integer wavelength,
ties break low), stretches each band independently between its
`p_low`/`p_high` percentiles **over the full frame**, clips, and quantizes
with `*255 + 0.5`. This is arithmetic-identical to JPEG tile exporters, so a
downstream uint8 conversion (the detector/segmenter input path) recovers the
exact bytes such an exporter would have written. Crop before this node, tile
after it (the nodes' `tiling` does that).

- `bands_nm` (default `(650.0, 550.0, 450.0)`), `p_low` (1.0), `p_high` (99.0)

## In-graph training: `RFDETRTrainable` + `RFDETRCriterionLoss`

`RFDETRTrainable` registers the LW-DETR module as a real submodule — its
parameters are visible to `GradientTrainer`, weights round-trip through
pipeline save/load, and Roboflow `.pth` checkpoints load via
`checkpoint_path`. `RFDETRCriterionLoss` wraps RF-DETR's `SetCriterion`
(Hungarian matcher + cls/bbox/giou and mask losses) as a loss node;
`functional.targets_from_mask` builds DETR targets from integer masks.
Requires the `[train]` extra.

For runs that must match RF-DETR's own trainer exactly (EMA, LR schedule,
multi-scale augmentation), fine-tuning with the upstream trainer and loading
the checkpoint into the inference nodes remains the reference path.

`RFDETRTrainable(multiclass_targets=True)` turns each mask class id `c > 0`
into DETR label `c - 1` (e.g. COCO categories 1/2/3 → model classes 0/1/2);
the default keeps the single-class behaviour (all foreground = label 0).

## More input builders: `ScalarMinMaxBandSlice`, `FixedPCAProjection`

- `ScalarMinMaxBandSlice(bands_nm=(640, 550, 470))` — `cube` + `wavelengths` →
  `rgb_image [B, H, W, 3]` in [0, 1]: ONE min / max over the whole cube (band
  ratios preserved), then the three bands nearest `bands_nm`. Use it when a
  checkpoint was trained on that recipe; it is not the same image as a
  per-channel stretch (`FixedWavelengthSelector(norm_mode="per_frame")`).
- `FixedPCAProjection(projection_path=...)` — cuvis-ai's `TrainablePCA` with a
  frozen projection loaded from an `.npz` (`mean` [C], `comps` [K, C], `lo` /
  `hi` [K]); never refits at inference. `input_global_minmax` (default on)
  min-maxes each cube globally first, so raw-scale reflectance (as cuvis.next
  delivers it) and [0, 1] cubes give the same projection. Needs `cuvis-ai`
  installed (parent class).

## Ensembles and gating: `ScoreFusion`, `ScoreIntersection`, `SamShellGate`

All take and return `scores [B, H, W, 1]` float32; stateless and torch-native.

- `ScoreFusion(mode, weight)` — fuse two score maps (`a`, `b`): `min` (AND),
  `max` (OR), `mean`, `gmean` (geometric), `wmean` (`weight`·a +
  (1 − `weight`)·b). For ensembles feed un-thresholded maps (segmenter
  `threshold` ≈ 0.05) and threshold the fused map once downstream.
- `ScoreIntersection` — elementwise minimum; the same as
  `ScoreFusion(mode="min")`, kept for existing pipelines.
- `SamShellGate(reference, threshold_deg)` — zero the scores where the
  raw-cosine spectral angle of the pixel's full spectrum (`cube`) to a fixed
  reference spectrum exceeds `threshold_deg`; drops look-alike objects with a
  different spectrum. Illumination-scale invariant.

## Optional: `CarlSegmenter`

CARL (IMSY-DKFZ) ViT-Adapter / UperNet semantic segmentation on the full
cube (`cube` + `wavelengths`) → `scores` (probability of `score_class`) +
`labels` (argmax map). CARL itself is imported lazily from `carl_repo` on the
first forward, so pipelines build without it. Speed knobs: `precision`
(`bf16` default), `band_step` (every k-th band), `compile` (+
`compile_cache_dir` for a warm inductor cache).

## Training transforms for `AugmentationCompose`

`cuvis_ai_rfdetr.transforms` registers extra transforms with
`cuvis-ai-augment`'s registry; list the module in the compose node's
`extra_transform_modules` and use the names like built-in transforms. All
apply per sample with probability `prob`, draw from the compose's shared
generator (seeded runs are reproducible) and keep the mask aligned.

| name | what it simulates | key hparams |
| --- | --- | --- |
| `RandomMultiScaleResize` | RF-DETR's native multi-scale training sizes | `scales` or `resolution` / `patch_size` / `num_windows` |
| `RandomZoom` | camera height change (zoom out onto a median canvas / crop and enlarge) | `scale_range` (0.5, 2.0) |
| `RandomShading` | uneven or missing light (smooth darkening field on all bands) | `strength_range`, `grid` |
| `RandomGammaContrast` | tone-curve changes | `gamma_range`, `contrast_range` |
| `RandomGaussianBlur` | defocus | `sigma_range` |

`cuvis-ai-augment` is not a pip dependency of this plugin (it is released by
git tag); importing `cuvis_ai_rfdetr.transforms` without it raises a clear
error.

## Plugin manifest

The repository root ships a local-path manifest (`plugins.yaml`) exposing all
nodes. Released consumers should pin the git source instead (the nodes after
`PercentileComposite` ship from the first release after v0.4.0; until it is
tagged, use the local-path manifest):

```yaml
name: rfdetr
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
tag: "v0.5.0"
package_name: cuvis-ai-rfdetr
capabilities:
  - class_name: cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector
  - class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
  - class_name: cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable
  - class_name: cuvis_ai_rfdetr.node.rfdetr_loss.RFDETRCriterionLoss
  - class_name: cuvis_ai_rfdetr.node.percentile_composite.PercentileComposite
  - class_name: cuvis_ai_rfdetr.node.scalar_minmax_bandslice.ScalarMinMaxBandSlice
  - class_name: cuvis_ai_rfdetr.node.fixed_pca_projection.FixedPCAProjection
  - class_name: cuvis_ai_rfdetr.node.score_fusion.ScoreFusion
  - class_name: cuvis_ai_rfdetr.node.score_intersection.ScoreIntersection
  - class_name: cuvis_ai_rfdetr.node.sam_shell_gate.SamShellGate
  - class_name: cuvis_ai_rfdetr.node.carl_segmenter.CarlSegmenter
```

`package_name` lets cuvis.next's per-pipeline child env install the plugin.

## Example: tiled false-color pipeline (nodes only)

```yaml
nodes:
  composite:
    class_name: cuvis_ai_rfdetr.node.percentile_composite.PercentileComposite
    hparams: { bands_nm: [650.0, 550.0, 450.0] }
  segmenter:
    class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
    hparams:
      checkpoint_path: runs/ft/checkpoint_best_total.pth
      variant: medium
      checkpoint_loader: from_checkpoint   # reproduce a from_checkpoint-based harness
      threshold: 0.02
      tiling: tiled
      jpeg_roundtrip: true     # reproduce a JPEG-tile evaluation harness
      class_filter: 1
      score_reduction: top_frac_mean
connections:
  - { source: composite.outputs.rgb_image, target: segmenter.inputs.rgb_image }
```

## Planned moves

Several nodes here are general-purpose and only live in this repo so the
walnut deploy pipelines could ship together. Tracked moves:

| what | destination | issue |
| --- | --- | --- |
| `RandomZoom`, `RandomShading`, `RandomGammaContrast`, `RandomGaussianBlur` | `cuvis-ai-augment` | #13 |
| `ScoreFusion`, `SamShellGate` (→ generic `SpectralAngleGate`); retire `ScoreIntersection` | `cuvis-ai` builtins | #14 |
| `ScalarMinMaxBandSlice`, `FixedPCAProjection` (optional: `PercentileComposite`) | `cuvis-ai` builtins | #15 |
| `CarlSegmenter` | new `cuvis-ai-carl` plugin | #16 |
| `_compat.base_kwargs` (core < 0.15 shim) | delete once the core floor is ≥ 0.15 | #17 |

Moves happen in the destination first; pipelines and consumer manifests switch
over; the code is removed here last.

## Tests

```bash
uv run --no-sources --locked --extra dev pytest tests/ -m "not slow" -q
uv run --no-sources --locked --extra dev ruff format --check cuvis_ai_rfdetr tests
uv run --no-sources --locked --extra dev ruff check cuvis_ai_rfdetr tests
```

The suite runs with only `cuvis-ai-core`, `cuvis-ai-schemas`, and `torch`
installed (no `rfdetr` required); forward paths are covered with a mocked
backbone. GPU / rfdetr-dependent smokes are marked `slow`.

## License

Apache-2.0 — see `LICENSE` and `NOTICE`.
