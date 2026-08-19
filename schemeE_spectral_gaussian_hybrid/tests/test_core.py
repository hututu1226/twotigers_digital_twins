from __future__ import annotations

import numpy as np
import torch
import unittest
from pathlib import Path

from scheme_e.angle_delay import ChannelShape
from scheme_e.gp import (
    SharedMultiOutputGP,
    convex_cosine_weights,
    convex_grid_weights,
    convex_mse_weights,
    ensemble_log_power_predictions,
)
from scheme_e.projection import alternating_spectral_projection
from scheme_e.reference import build_reference_candidates
from scheme_e.rf_geometry import build_rf_gaussians, extract_geometry_features, feature_names
from scheme_e.spectral_targets import channel_spectral_targets, decode_pas_log, decode_pdp_log
from scheme_e.splits import spatial_block_folds
from scheme_e.hybrid_training import _build_model


def _shape() -> ChannelShape:
    return ChannelShape(m=8, m_h=2, m_v=2, m_p=2, n=2, s=8)


def test_spectral_targets_and_projection_are_finite() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(7)
    real = torch.randn(3, *shape.raw_shape, generator=generator)
    imag = torch.randn(3, *shape.raw_shape, generator=generator)
    channel = torch.complex(real, imag)
    target = channel_spectral_targets(channel, shape, proxy_count=2)
    proxy, mean_pas = decode_pas_log(target["pas_log"], shape, proxy_count=2)
    pdp = decode_pdp_log(target["pdp_log"], shape)
    assert target["pas_log"].shape == (3, 12)
    assert target["pdp_log"].shape == (3, 16)
    assert torch.allclose(proxy.sum((2, 3)), torch.ones(3, 2), atol=1e-5)
    assert torch.allclose(mean_pas.sum((1, 2)), torch.ones(3), atol=1e-5)
    assert torch.allclose(pdp.sum(2), torch.ones(3, 2), atol=1e-5)
    projected = alternating_spectral_projection(
        channel, target["pas_log"], target["pdp_log"], target["ue_log_energy"],
        shape, iterations=2, proxy_count=2,
    )
    assert projected.shape == channel.shape
    assert torch.isfinite(projected).all()


def test_rf_gaussians_produce_exactly_71_features() -> None:
    vertices = np.asarray(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [0, 0, 3], [4, 0, 3]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1]], dtype=np.int64)
    field = build_rf_gaussians(vertices, faces, target_count=4)
    features = extract_geometry_features(
        np.asarray([[2, 2, 1], [3, 1, 1]], dtype=np.float32),
        np.asarray([0, 1]),
        np.asarray([[-5, 0, 5], [8, 0, 5]], dtype=np.float32),
        field,
        corridor_samples=8,
        fresnel_radius_meters=1.0,
    )
    assert len(feature_names()) == 71
    assert features.shape == (2, 71)
    assert np.isfinite(features).all()


def test_gp_and_convex_ensemble() -> None:
    x = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    features = np.stack([x[:, 0], x[:, 0] ** 2], axis=1)
    targets = np.stack([np.sin(x[:, 0]), np.cos(x[:, 0])], axis=1)
    model = SharedMultiOutputGP("rq10", noise=1e-3).fit(x, features, targets)
    prediction, uncertainty = model.predict(x[:2], features[:2])
    assert prediction.shape == (2, 2)
    assert uncertainty.shape == (2,)
    assert np.isfinite(prediction).all()
    weights = convex_grid_weights(np.asarray([[1, 2, 3], [2, 1, 3]], dtype=np.float32), 0.5)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)

    scale = 1000.0
    target_power = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    model_power = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    target_log = np.log1p(scale * target_power)
    prediction_log = np.log1p(scale * model_power)
    cosine_weights = convex_cosine_weights(
        prediction_log, target_log, scale=scale, step=0.05
    )
    blended = ensemble_log_power_predictions(
        prediction_log, cosine_weights, scale=scale
    )
    assert cosine_weights[0] > 0.0 and cosine_weights[1] > 0.0
    assert np.allclose(
        np.expm1(blended) / scale,
        target_power * cosine_weights[:2].sum() / 2.0,
        atol=1e-5,
    )
    mse_weights = convex_mse_weights(
        np.asarray([[[-1.0]], [[1.0]], [[3.0]]], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        step=0.05,
    )
    self_consistent = float(
        np.sum(mse_weights * np.asarray([-1.0, 1.0, 3.0], dtype=np.float32))
    )
    assert abs(self_consistent) < 1e-5


def test_reference_candidates_exclude_self() -> None:
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    cells = np.zeros(3, dtype=np.int64)
    global_indices = np.arange(3)
    candidates, _ = build_reference_candidates(
        positions, cells, positions, cells, np.zeros(3, dtype=bool), top_k=1,
        target_global_indices=global_indices, observed_global_indices=global_indices,
    )
    assert np.all(candidates[:, 0] != global_indices)


def test_spatial_folds_are_nonempty_and_cell_balanced() -> None:
    x = np.arange(32, dtype=np.float32)
    positions = np.stack([x, x % 4, np.zeros_like(x)], axis=1)
    cells = np.repeat([0, 1], 16)
    folds = spatial_block_folds(positions, cells, fold_count=4, tile_meters=2, seed=5)
    assert set(folds.tolist()) == {0, 1, 2, 3}
    for fold in range(4):
        assert set(cells[folds == fold].tolist()) == {0, 1}


class SchemeECoreTests(unittest.TestCase):
    def test_spectral_targets(self) -> None:
        test_spectral_targets_and_projection_are_finite()

    def test_rf_features(self) -> None:
        test_rf_gaussians_produce_exactly_71_features()

    def test_gp(self) -> None:
        test_gp_and_convex_ensemble()

    def test_references(self) -> None:
        test_reference_candidates_exclude_self()

    def test_spatial_folds(self) -> None:
        test_spatial_folds_are_nonempty_and_cell_balanced()

    def test_final_projection_override_is_declared(self) -> None:
        import inspect

        signature = inspect.signature(_build_model)
        self.assertIn("section_override", signature.parameters)

    def test_formal_pipeline_preprocesses_before_architecture_inspection(self) -> None:
        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts" / "run_all_5090.sh").read_text(encoding="utf-8")
        self.assertLess(
            script.index("python scripts/preprocess.py"),
            script.index("python scripts/inspect_architecture.py"),
        )
