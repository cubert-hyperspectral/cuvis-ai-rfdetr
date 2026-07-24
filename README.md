# cuvis-ai-rfdetr

[RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's real-time detection
transformer — wrapped as a [cuvis.ai](https://github.com/cubert-hyperspectral)
plugin node for object / foreign-object detection on RGB and false-color
renderings of hyperspectral data.

## Licensing scope

This plugin wraps the **Apache-2.0 tier** of RF-DETR only: the `nano`, `small`,
`medium`, and `large` variants (code and checkpoints under Apache-2.0, matching
this repository's license). The RF-DETR **XL / 2XL** checkpoints are distributed
under the non-open Roboflow Platform Model License and are **not** included,
wrapped, or downloaded by this package. See `NOTICE` for attribution.

## Install

```bash
pip install "cuvis-ai-rfdetr @ git+https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
```

Dependencies pull in the RF-DETR **core (inference) tier** only — deliberately
not `rfdetr[train]`. The `rfdetr` import happens lazily on the first `forward`,
so pipelines can be built and validated without it.

## Node: `RFDETRDetector`

| Port | Direction | Type / shape | Description |
| --- | --- | --- | --- |
| `rgb_image` | input | `float32 [B, H, W, 3]` | RGB / false-color frame; values 0–1 or 0–255 (auto-detected) |
| `scores` | output | `float32 [B, H, W, 1]` | Rasterized detection map: box regions filled with `max(existing, confidence)` |
| `detections` | output | `list` | Per-image list of `{"xyxy": [x1, y1, x2, y2], "confidence": float, "class_id": int}` |
| `anomaly_score` | output | `float32 [B]` | Max box confidence per image (0.0 when no detections) |

Hyperparameters:

- `checkpoint_path` (`str | None`, default `None`) — fine-tuned weights, passed
  to the RF-DETR model's `pretrain_weights`; `None` loads the official
  COCO-pretrained weights for the variant.
- `variant` (`str`, default `"medium"`) — `nano | small | medium | large`.
- `threshold` (`float`, default `0.5`) — confidence threshold for `predict`.
- `resolution` (`int | None`, default `None`) — optional input resolution
  forwarded to the RF-DETR constructor (RF-DETR enforces its own divisibility
  rules); `None` keeps the variant default.

The wrapped model manages its own device placement and is intentionally not a
registered submodule: `node.to(...)` does not move it.

## Plugin manifest

The repository root ships a local-path manifest (`plugins.yaml`):

```yaml
name: rfdetr
path: "."
capabilities:
  - class_name: cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector
```

Released consumers should pin the git source instead:

```yaml
name: rfdetr
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
tag: "v0.1.0"
capabilities:
  - class_name: cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector
```

## Fine-tuning (outside the graph)

Training is not part of the pipeline graph. Fine-tune with RF-DETR's own
trainer, then point the node at the resulting checkpoint:

```bash
pip install "rfdetr[train]"   # in a separate training environment
```

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()
model.train(dataset_dir="/path/to/coco-format-dataset", epochs=50, output_dir="runs/ft")
```

```yaml
# pipeline node config
class_name: cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector
hparams:
  checkpoint_path: runs/ft/checkpoint_best_total.pth
  variant: medium
  threshold: 0.5
```

## Tests

```bash
pytest tests/ -q
```

The suite runs with only `cuvis-ai-core`, `cuvis-ai-schemas`, and `torch`
installed (no `rfdetr` required).

## License

Apache-2.0 — see `LICENSE` and `NOTICE`.
