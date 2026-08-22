from __future__ import annotations

import numpy as np
import torch
import unittest
from pathlib import Path

from scheme_e.angle_delay import ChannelShape
from scheme_e.autoencoder import FactorizedResidualAutoencoder
from scheme_e.carrier_transport import (
    TRANSPORT_CONTEXT_DIM,
    build_transport_seed,
    select_transport_candidates,
)
from scheme_e.gp import (
    SharedMultiOutputGP,
    convex_cosine_weights,
    convex_grid_weights,
    convex_mse_weights,
    ensemble_log_power_predictions,
)
from scheme_e.power_safety import (
    apply_outage_policy,
    apply_power_calibration,
    clip_power_priors,
    compute_power_bounds,
    fit_power_calibration,
)
from scheme_e.projection import alternating_spectral_projection, relaxed_output_projection
from scheme_e.reference import build_reference_candidates
from scheme_e.reference_context import (
    REFERENCE_CONTEXT_DIM,
    build_reference_context,
    select_reference_candidates,
)
from scheme_e.rf_geometry import build_rf_gaussians, extract_geometry_features, feature_names
from scheme_e.spectral_targets import channel_spectral_targets, decode_pas_log, decode_pdp_log
from scheme_e.splits import spatial_block_folds
from scheme_e.hybrid_training import _build_model, _validation_mask
from scheme_e.hybrid_model import SpectralGaussianHybrid, StructuredSpectralFieldEncoder
from scheme_e.local_spectral import local_expert_settings, local_spectral_prediction


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


def test_relaxed_output_projection_preserves_requested_power() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(13)
    channel = torch.complex(
        torch.randn(2, *shape.raw_shape, generator=generator),
        torch.randn(2, *shape.raw_shape, generator=generator),
    )
    targets = channel_spectral_targets(channel, shape, proxy_count=2)
    requested_power = targets["log_power"] + torch.tensor([0.2, -0.1])
    projected = relaxed_output_projection(
        channel,
        targets["pas_log"],
        targets["pdp_log"],
        targets["ue_log_energy"],
        requested_power,
        shape,
        iterations=1,
        proxy_count=2,
        strength=torch.tensor([0.0, 0.75]),
    )
    measured = torch.log10(projected.abs().square().mean(dim=(1, 2, 3)))
    assert torch.allclose(measured, requested_power, atol=1e-5)
    expected_first = channel[0] * torch.pow(
        10.0, 0.5 * (requested_power[0] - targets["log_power"][0])
    )
    assert torch.allclose(projected[0], expected_first, atol=1e-5)


def test_structured_spectral_field_preserves_absolute_grid_axes() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(19)
    channel = torch.complex(
        torch.randn(2, *shape.raw_shape, generator=generator),
        torch.randn(2, *shape.raw_shape, generator=generator),
    )
    targets = channel_spectral_targets(channel, shape, proxy_count=2)
    encoder = StructuredSpectralFieldEncoder(
        shape,
        proxy_count=2,
        spectrum_channels=4,
        detail_channels=6,
        spectrum_size=(1, 1, 2),
        detail_size=(2, 2, 4),
    )
    spectrum, detail = encoder(targets["pas_log"], targets["pdp_log"])
    assert spectrum.shape == (2, 4, 1, 1, 2)
    assert detail.shape == (2, 6, 2, 2, 4)
    assert torch.isfinite(spectrum).all()
    assert torch.isfinite(detail).all()


def test_v4_structured_hybrid_forward_is_finite() -> None:
    shape = ChannelShape(m=16, m_h=4, m_v=4, m_p=1, n=1, s=16)
    autoencoder = FactorizedResidualAutoencoder(
        shape,
        spectrum_stem_channels=8,
        phase_stem_channels=8,
        spectrum_latent_channels=8,
        phase_latent_channels=8,
        residual_blocks=1,
        detail_hidden_channels=16,
        spectrum_decoder_channels=16,
        detail_decoder_channels=16,
    )
    model = SpectralGaussianHybrid(
        autoencoder,
        shape,
        proxy_count=2,
        geometry_dim=71,
        condition_width=16,
        spectrum_blocks=1,
        detail_blocks=1,
        projection_iterations=1,
        preserve_spectral_positions=True,
        structured_spectral_field=True,
    )
    generator = torch.Generator().manual_seed(23)
    channel = torch.complex(
        torch.randn(2, *shape.raw_shape, generator=generator),
        torch.randn(2, *shape.raw_shape, generator=generator),
    )
    targets = channel_spectral_targets(channel, shape, proxy_count=2)
    result = model(
        channel,
        targets["pas_log"],
        targets["pdp_log"],
        targets["ue_log_energy"],
        targets["log_power"],
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2, 71),
    )
    assert result["channel"].shape == channel.shape
    assert result["spectrum_field"] is not None
    assert result["detail_field"] is not None
    assert torch.isfinite(result["channel"]).all()


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


def test_local_spectral_expert_blends_physical_power() -> None:
    positions = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    queries = np.asarray([[1, 0, 0]], dtype=np.float32)
    scale_pas = 1000.0
    scale_pdp = 1000.0
    pas_power = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pdp_power = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    pas, pdp, ue, power, uncertainty = local_spectral_prediction(
        positions,
        queries,
        np.log1p(scale_pas * pas_power),
        np.log1p(scale_pdp * pdp_power),
        np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
        np.asarray([-4.0, -2.0], dtype=np.float32),
        neighbors=2,
        distance_power=1.0,
    )
    np.testing.assert_allclose(np.expm1(pas) / scale_pas, [[0.5, 0.5]], atol=1e-6)
    np.testing.assert_allclose(np.expm1(pdp) / scale_pdp, [[1.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(ue, [[2.0, 4.0]], atol=1e-6)
    np.testing.assert_allclose(power, [-3.0], atol=1e-6)
    assert 0.0 < float(uncertainty[0]) < 1.0
    assert local_expert_settings(
        {"local_spectral_experts": [{"name": "idw", "neighbors": 2}]}
    ) == [("idw", 2, 1.0)]


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


def test_v2_power_safety_is_per_cell_and_bounded() -> None:
    power = np.asarray([-4.0, -3.0, -2.0, -8.0, -7.0, -6.0], dtype=np.float32)
    outage = np.zeros(6, dtype=bool)
    cells = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bounds = compute_power_bounds(power, outage, cells, np.arange(6), 0.0, 1.0)
    clipped, ue = clip_power_priors(
        np.asarray([4.0, -20.0], dtype=np.float32),
        np.asarray([[4.0, 4.0], [-20.0, -20.0]], dtype=np.float32),
        np.asarray([0, 1]),
        bounds,
    )
    np.testing.assert_allclose(clipped, np.asarray([-2.0, -8.0]))
    assert np.isfinite(ue).all()
    channel = torch.ones(2, 2, 1, 2, dtype=torch.complex64)
    attenuated = apply_outage_policy(
        channel,
        torch.tensor([0.5, 0.99]),
        torch.tensor([0.9, 0.9]),
        torch.tensor([2.0, 2.0]),
    )
    assert torch.allclose(attenuated[0].abs(), torch.full_like(attenuated[0].abs(), 0.5))
    assert torch.count_nonzero(attenuated[1]) == 0


def test_v2_power_calibration_is_cell_specific_and_shifts_ue() -> None:
    prediction = np.asarray([-4.0, -3.0, -2.0, -8.0, -7.0, -6.0], dtype=np.float32)
    target = np.asarray([-5.0, -4.0, -3.0, -6.0, -5.0, -4.0], dtype=np.float32)
    cells = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    parameters = fit_power_calibration(
        prediction, target, cells, np.arange(len(cells)), slope_bounds=(0.5, 1.5)
    )
    calibrated, ue = apply_power_calibration(
        prediction,
        np.column_stack([prediction, prediction]),
        cells,
        parameters,
    )
    np.testing.assert_allclose(calibrated, target, atol=1e-5)
    np.testing.assert_allclose(ue[:, 0], target, atol=1e-5)


def test_v2_reference_context_and_spectral_selection() -> None:
    count = 2
    geometry = np.zeros((count, 71), dtype=np.float32)
    pas = np.log1p(1000.0 * np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    pdp = pas.copy()
    context = build_reference_context(
        np.asarray([[1, 0, 0], [0, 2, 0]], dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
        geometry,
        geometry,
        pas,
        pdp,
        pas,
        pdp,
        np.asarray([-3, -4], dtype=np.float32),
        np.asarray([-3, -4], dtype=np.float32),
        np.zeros(count, dtype=np.float32),
    )
    assert context.shape == (count, REFERENCE_CONTEXT_DIM)
    selected = select_reference_candidates(
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([[1.0, 1.1]], dtype=np.float32),
        geometry[:1],
        geometry,
        pas[1:2],
        pdp[1:2],
        pas,
        pdp,
        {
            "name": "spectral",
            "top_k": 2,
            "distance_weight": 0.1,
            "pas_weight": 2.0,
            "pdp_weight": 2.0,
            "geometry_weight": 0.0,
        },
    )
    np.testing.assert_array_equal(selected, np.asarray([1]))


def test_v3_transport_seed_and_context_are_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    real = torch.randn(2, 3, 8, 2, 8, generator=generator)
    imag = torch.randn(2, 3, 8, 2, 8, generator=generator)
    references = torch.complex(real, imag)
    seed, context = build_transport_seed(
        references,
        torch.tensor([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]),
        torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ),
        torch.tensor([0, 1]),
        torch.tensor([[2.0, 1.0, 0.5], [3.0, 2.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([-1.0, -2.0]),
        torch.tensor([0.8, 0.4]),
    )
    assert seed.shape == references[:, 0].shape
    assert context.shape == (2, TRANSPORT_CONTEXT_DIM)
    assert torch.isfinite(seed).all()
    assert torch.isfinite(context).all()


def test_v3_transport_selection_respects_guard_distance() -> None:
    candidates = np.asarray([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int64)
    distances = np.asarray([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=np.float32)
    selected, selected_distances = select_transport_candidates(
        candidates, distances, 2, np.asarray([3.0, 5.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(selected, np.asarray([[5, 6], [10, 11]]))
    assert np.all(selected_distances >= np.asarray([[3.0], [5.0]]))


class SchemeECoreTests(unittest.TestCase):
    def test_spectral_targets(self) -> None:
        test_spectral_targets_and_projection_are_finite()

    def test_rf_features(self) -> None:
        test_rf_gaussians_produce_exactly_71_features()

    def test_relaxed_output_projection(self) -> None:
        test_relaxed_output_projection_preserves_requested_power()

    def test_structured_spectral_field(self) -> None:
        test_structured_spectral_field_preserves_absolute_grid_axes()

    def test_v4_structured_hybrid_forward(self) -> None:
        test_v4_structured_hybrid_forward_is_finite()

    def test_gp(self) -> None:
        test_gp_and_convex_ensemble()

    def test_local_spectral_expert(self) -> None:
        test_local_spectral_expert_blends_physical_power()

    def test_references(self) -> None:
        test_reference_candidates_exclude_self()

    def test_spatial_folds(self) -> None:
        test_spatial_folds_are_nonempty_and_cell_balanced()

    def test_v2_power_safety(self) -> None:
        test_v2_power_safety_is_per_cell_and_bounded()

    def test_v2_power_calibration(self) -> None:
        test_v2_power_calibration_is_cell_specific_and_shifts_ue()

    def test_v2_reference_context(self) -> None:
        test_v2_reference_context_and_spectral_selection()

    def test_v3_transport_seed(self) -> None:
        test_v3_transport_seed_and_context_are_finite()

    def test_v3_transport_selection(self) -> None:
        test_v3_transport_selection_respects_guard_distance()

    def test_final_projection_override_is_declared(self) -> None:
        import inspect

        signature = inspect.signature(_build_model)
        self.assertIn("section_override", signature.parameters)

    def test_final_training_accepts_null_validation_fold(self) -> None:
        metadata = {
            "validation_masks": np.asarray([[True, False, True]], dtype=bool),
        }
        np.testing.assert_array_equal(
            _validation_mask(metadata, 3, validation_fold=None, final=True),
            np.zeros(3, dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "cannot be null"):
            _validation_mask(metadata, 3, validation_fold=None, final=False)

    def test_formal_pipeline_preprocesses_before_architecture_inspection(self) -> None:
        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts" / "run_all_5090.sh").read_text(encoding="utf-8")
        self.assertLess(
            script.index("python scripts/preprocess.py"),
            script.index("python scripts/inspect_architecture.py"),
        )

    def test_v2_pipeline_is_continuous_and_preserves_v1_output(self) -> None:
        import json

        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts" / "run_v2_5090.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            script.index("build_strict_fold_prior.py"),
            script.index("prepare_v2_attempts.py"),
        )
        self.assertLess(
            script.index("select_v2_attempt.py"),
            script.index("prepare_v2_final_config.py"),
        )
        self.assertLess(script.index("scripts/infer.py"), script.index("package_v2_results.sh"))
        config = json.loads((project / "configs" / "v2_5090.json").read_text())
        self.assertEqual(
            config["inference"]["output_path"],
            "outputs/v2/Round2_Test_Channel.npy",
        )

    def test_v3_pipeline_is_continuous_and_uses_independent_output(self) -> None:
        import json

        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts" / "run_v3_5090.sh").read_text(
            encoding="utf-8"
        )
        ordered = [
            "prepare_v3_config.py",
            "build_strict_fold_prior.py",
            "prepare_v3_attempts.py",
            "select_v3_attempt.py",
            "prepare_v3_final_config.py",
            "scripts/infer.py",
            "package_v3_results.sh",
        ]
        offsets = [script.index(value) for value in ordered]
        self.assertEqual(offsets, sorted(offsets))
        config = json.loads((project / "configs" / "v3_5090.json").read_text())
        self.assertTrue(config["hybrid"]["transport_seed"]["enabled"])
        self.assertEqual(config["hybrid"]["transport_seed"]["count"], 8)
        self.assertEqual(
            config["inference"]["output_path"],
            "outputs/v3/Round2_Test_Channel.npy",
        )
