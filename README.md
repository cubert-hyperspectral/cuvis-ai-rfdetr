# cuvis-ai-rfdetr

[RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's real-time detection
transformer — wrapped as a [cuvis.ai](https://github.com/cubert-hyperspectral)
plugin for object / foreign-object detection and instance segmentation on RGB
and false-color renderings of hyperspectral data.

Five nodes: two inference wrappers (`RFDETRDetector`, `RFDETRSegmenter`), a
false-color input builder (`PercentileComposite`), and an in-graph training
pair (`RFDETRTrainable`, `RFDETRCriterionLoss`).

## Licensing scope

This plugin wraps the **Apache-2.0 tier** of RF-DETR. For detection that is
`nano | small | medium | large`; the whole **segmentation** tier
(`nano … 2xlarge`) ships under Apache-2.0 and is fully exposed. The RF-DETR
detection **XL / 2XL** checkpoints are distributed under the non-open Roboflow
Platform Model License and are **not** included, wrapped, or downloaded by
this package. See `NOTICE` for attribution.

## Install

```bash
pip install "cuvis-ai-rfdetr @ git+https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git@v0.2.0"
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
  constructor. **When loading a fine-tuned checkpoint, set this to the
  checkpoint's training resolution** — the constructor does not read it from
  the file, and a mismatch silently changes every score.
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

## Plugin manifest

The repository root ships a local-path manifest (`plugins.yaml`) exposing all
five nodes. Released consumers should pin the git source instead:

```yaml
name: rfdetr
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
tag: "v0.2.0"
capabilities:
  - class_name: cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector
  - class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
  - class_name: cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable
  - class_name: cuvis_ai_rfdetr.node.rfdetr_loss.RFDETRCriterionLoss
  - class_name: cuvis_ai_rfdetr.node.percentile_composite.PercentileComposite
```

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
      resolution: 624          # = the checkpoint's training resolution
      threshold: 0.02
      tiling: tiled
      jpeg_roundtrip: true     # reproduce a JPEG-tile evaluation harness
      class_filter: 1
      score_reduction: top_frac_mean
connections:
  - { source: composite.outputs.rgb_image, target: segmenter.inputs.rgb_image }
```

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
