"""FixedPCAProjection: golden reference vs the exporter math (projection + fixed scaling + clamp); hparams; ports.
Needs cuvis_ai (TrainablePCA parent) — skipped in envs without it (the plugin venv lacks cuvis-ai's cv2 chain)."""

import numpy as np
import pytest
import torch

pytest.importorskip("cuvis_ai.node.dimensionality_reduction")
from cuvis_ai_rfdetr.node.fixed_pca_projection import FixedPCAProjection  # noqa: E402


def _proj(tmp_path):
    rng = np.random.default_rng(0)
    C, K = 7, 3
    mean = rng.normal(size=C).astype(np.float32)
    comps = np.linalg.qr(rng.normal(size=(C, C)))[0][:, :K].T.astype(np.float32)
    lo = np.array([-1.0, -2.0, -0.5], np.float32)
    hi = np.array([2.0, 1.5, 0.7], np.float32)
    p = tmp_path / "proj.npz"
    np.savez(
        p, mean=mean, comps=comps, lo=lo, hi=hi, explained=np.array([0.7, 0.2, 0.1], np.float32)
    )
    return str(p), mean, comps, lo, hi


def test_matches_exporter_math_with_scaling_and_clamp(tmp_path):
    path, mean, comps, lo, hi = _proj(tmp_path)
    # input_global_minmax=False isolates the pure projection math (the input min-max is tested separately below).
    node = FixedPCAProjection(projection_path=path, input_global_minmax=False)
    rng = np.random.default_rng(1)
    cube = (rng.normal(size=(2, 4, 5, 7)) * 3).astype(np.float32)
    out = node(data=torch.from_numpy(cube))["projected"].numpy()
    ref = (cube.reshape(-1, 7) - mean) @ comps.T
    ref = np.clip((ref - lo) / (hi - lo), 0, 1).reshape(2, 4, 5, 3)
    assert out.shape == (2, 4, 5, 3) and out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_raw_projection_when_scaling_disabled(tmp_path):
    path, mean, comps, _, _ = _proj(tmp_path)
    node = FixedPCAProjection(
        projection_path=path, scale_to_unit=False, clamp01=False, input_global_minmax=False
    )
    rng = np.random.default_rng(2)
    cube = (rng.normal(size=(1, 3, 3, 7)) * 2).astype(np.float32)
    out = node(data=torch.from_numpy(cube))["projected"].numpy()
    ref = ((cube.reshape(-1, 7) - mean) @ comps.T).reshape(1, 3, 3, 3)
    assert np.allclose(out, ref, atol=1e-5)


def test_input_global_minmax_is_scale_invariant_and_idempotent(tmp_path):
    # The exporter global-min-maxed each cube to [0, 1] before projecting, so the fixed projection must be
    # invariant to the caller's absolute scale — cuvis.next's CU3SDataNode delivers raw-scale reflectance,
    # not [0, 1]. With input_global_minmax on (default), a cube and any positive rescaling of it must project
    # identically, and an already-[0, 1] cube must be unchanged by the min-max.
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path)  # default input_global_minmax=True
    rng = np.random.default_rng(3)
    base = rng.random(size=(1, 4, 4, 7)).astype(np.float32)  # in [0, 1)
    out_unit = node(data=torch.from_numpy(base))["projected"].numpy()
    out_scaled = node(data=torch.from_numpy(base * 10000.0 + 5.0))["projected"].numpy()
    assert np.allclose(out_unit, out_scaled, atol=1e-5)  # scale/offset invariant
    # idempotent on an already-[0, 1] cube (min 0, max 1): min-max is identity there
    b01 = base.copy()
    b01.flat[0], b01.flat[1] = 0.0, 1.0
    gm = FixedPCAProjection._global_minmax(torch.from_numpy(b01)).numpy()
    assert np.allclose(gm, b01, atol=1e-6)


def test_hparams_round_trip_and_initialized(tmp_path):
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path, scale_to_unit=True, clamp01=True)
    assert node.hparams["projection_path"] == path
    assert node.hparams["scale_to_unit"] is True and node.hparams["clamp01"] is True
    assert node.hparams["input_global_minmax"] is True
    assert node._statistically_initialized is True
