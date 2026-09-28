"""CarlSegmenter: lazy construction (no CARL import) and the preprocessing golden reference."""

import torch

from cuvis_ai_rfdetr.node.carl_segmenter import CarlSegmenter


def test_construction_is_lazy_and_records_hparams():
    node = CarlSegmenter(
        checkpoint_path="c.ckpt",
        config_path="c.yaml",
        carl_repo="/nowhere",
        image_size=64,
        score_class=1,
    )
    assert node._model is None
    assert node.hparams["image_size"] == 64
    assert node.hparams["score_class"] == 1


def test_preprocess_matches_training_recipe():
    g = torch.Generator().manual_seed(0)
    cube = torch.rand((2, 10, 12, 61), generator=g) * 3000.0 + 100.0
    out = CarlSegmenter.preprocess(cube, 16)
    assert out.shape == (2, 61, 16, 16)
    x = cube.clone()
    for b in range(2):  # per-cube scalar min-max
        x[b] = (x[b] - x[b].min()) / (x[b].max() - x[b].min())
    x = x.permute(0, 3, 1, 2)
    for b in range(2):  # per-cube z-score
        x[b] = (x[b] - x[b].mean()) / (x[b].std() + 1e-6)
    ref = torch.nn.functional.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    assert torch.allclose(out, ref, atol=1e-5)


def test_band_step_and_compile_hparams_round_trip():
    node = CarlSegmenter(
        checkpoint_path="c.ckpt",
        config_path="c.yaml",
        carl_repo="/nowhere",
        band_step=2,
        compile=True,
        compile_cache_dir="/cache",
    )
    assert node.hparams["band_step"] == 2 and node.hparams["compile"] is True
    assert node.band_step == 2 and node.compile is True
    assert node.hparams["compile_cache_dir"] == "/cache"
