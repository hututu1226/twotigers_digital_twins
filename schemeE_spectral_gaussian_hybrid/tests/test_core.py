from __future__ import annotations

import numpy as np
import torch
import unittest
from pathlib import Path

from scheme_e.angle_delay import ChannelShape
from scheme_e.adaptive_experiment import adaptive_hybrid_config
from scheme_e.autoencoder import FactorizedResidualAutoencoder
from scheme_e.carrier_transport import (
    CarrierFit,
    TRANSPORT_CONTEXT_DIM,
    build_transport_seed,
    quality_gated_carrier_fit,
    select_transport_candidates,
)
from scheme_e.complex_residual import (
    angle_delay_log_power,
    angle_delay_to_complex,
    complex_to_angle_delay,
    decode_low_rank_coefficients,
    project_low_rank_coefficients,
    reconstruct_low_rank_residual,
    replace_angle_delay_log_power,
    split_complex_correction,
)
from scheme_e.gp import (
    SharedMultiOutputGP,
    convex_cosine_weights,
    convex_grid_weights,
    convex_mse_weights,
    ensemble_log_power_predictions,
)
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    outage_threshold_oracle,
    sample_metric_batch,
    scale_oracle_predictions,
    target_informed_expert_oracle,
)
from scheme_e.metrics import ChannelMetricAccumulator
from scheme_e.magnitude_refiner import (
    FullResolutionMagnitudeRefiner,
    energy_weighted_log_power_loss,
    magnitude_marginal_cosine_loss,
    normalize_log_power_grid,
)
from scheme_e.local_magnitude import (
    estimate_magnitude_profile_shifts,
    same_cell_neighbors,
    transfer_log_power_residual,
)
from scheme_e.local_set_magnitude import QueryConditionedLocalSetMagnitudeRefiner
from scheme_e.marginal_projection import (
    alternating_marginal_projection,
    decode_pas_marginals,
)
from scheme_e.power_safety import (
    apply_outage_policy,
    apply_power_calibration,
    clip_power_priors,
    compute_power_bounds,
    fit_power_calibration,
)
from scheme_e.projection import (
    _pas_projection,
    alternating_spectral_projection,
    relaxed_output_projection,
)
from scheme_e.reference import build_reference_candidates
from scheme_e.reference_context import (
    REFERENCE_CONTEXT_DIM,
    build_reference_context,
    select_reference_candidates,
)
from scheme_e.rf_geometry import build_rf_gaussians, extract_geometry_features, feature_names
from scheme_e.spectral_targets import channel_spectral_targets, decode_pas_log, decode_pdp_log
from scheme_e.splits import spatial_block_folds
from scheme_e.hybrid_training import (
    _build_model,
    _output_projection_seed,
    _validation_mask,
)
from scheme_e.hybrid_model import SpectralGaussianHybrid, StructuredSpectralFieldEncoder
from scheme_e.local_spectral import local_expert_settings, local_spectral_prediction
from scheme_e.neural_spectral_refiner import SpectralNeighborRefiner
from scheme_e.residual_set_model import (
    ResidualCoefficientSetEncoder,
    spectrum_summary_features,
)


def _shape() -> ChannelShape:
    return ChannelShape(m=8, m_h=2, m_v=2, m_p=2, n=2, s=8)


def test_carrier_quality_gate_keeps_reliable_fit_and_replaces_weak_fit() -> None:
    fit = CarrierFit(
        wave_numbers=np.asarray([-140.4, -146.1], dtype=np.float64),
        qualities=np.asarray([0.78, 0.12], dtype=np.float64),
        pair_counts=np.asarray([1024, 1024], dtype=np.int64),
    )
    gated = quality_gated_carrier_fit(fit, -140.33, minimum_quality=0.5)
    np.testing.assert_allclose(gated.wave_numbers, [-140.4, -140.33])
    np.testing.assert_array_equal(gated.qualities, fit.qualities)
    np.testing.assert_array_equal(gated.pair_counts, fit.pair_counts)
    assert gated.wave_numbers is not fit.wave_numbers


def test_round1_marginal_projection_is_finite_and_power_aligned() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(2141)
    target = torch.complex(
        torch.randn(2, *shape.raw_shape, generator=generator),
        torch.randn(2, *shape.raw_shape, generator=generator),
    )
    seed = torch.complex(
        torch.randn(2, *shape.raw_shape, generator=generator),
        torch.randn(2, *shape.raw_shape, generator=generator),
    )
    targets = channel_spectral_targets(target, shape, proxy_count=2)
    horizontal, vertical = decode_pas_marginals(
        targets["pas_log"], shape, proxy_count=2
    )
    assert torch.allclose(horizontal.sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(vertical.sum(1), torch.ones(2), atol=1e-6)
    projected = alternating_marginal_projection(
        seed,
        targets["pas_log"],
        targets["pdp_log"],
        targets["ue_log_energy"],
        shape,
        iterations=2,
        proxy_count=2,
    )
    assert projected.shape == seed.shape
    assert torch.isfinite(projected).all()
    projected_energy = projected.abs().square().mean(dim=(1, 3))
    target_energy = torch.pow(10.0, targets["ue_log_energy"])
    assert torch.allclose(projected_energy, target_energy, rtol=1e-4, atol=1e-5)


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


def test_diagnostic_metrics_match_streaming_evaluator() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(31)
    target = torch.complex(
        torch.randn(5, *shape.raw_shape, generator=generator),
        torch.randn(5, *shape.raw_shape, generator=generator),
    )
    target[4] = 0.0
    prediction = target + 0.2 * torch.complex(
        torch.randn(5, *shape.raw_shape, generator=generator),
        torch.randn(5, *shape.raw_shape, generator=generator),
    )
    outage = torch.tensor([False, False, False, False, True])
    batches = [
        sample_metric_batch(prediction[:2], target[:2], shape, outage[:2]),
        sample_metric_batch(prediction[2:], target[2:], shape, outage[2:]),
    ]
    diagnostic = aggregate_sample_metrics(concatenate_metric_batches(batches))
    evaluator = ChannelMetricAccumulator(shape)
    evaluator.update(prediction[:2], target[:2], outage[:2])
    evaluator.update(prediction[2:], target[2:], outage[2:])
    streamed = evaluator.compute()
    for name in ("pas", "pdp", "nmse", "score"):
        assert abs(float(diagnostic[name]) - float(streamed[name])) < 1e-6


def test_scale_oracles_do_not_increase_nmse() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(37)
    target = torch.complex(
        torch.randn(3, *shape.raw_shape, generator=generator),
        torch.randn(3, *shape.raw_shape, generator=generator),
    )
    prediction = target * torch.tensor([0.25, 2.0, 1.0j])[:, None, None, None]
    baseline = sample_metric_batch(prediction, target, shape)
    for value in scale_oracle_predictions(prediction, target).values():
        oracle = sample_metric_batch(value, target, shape)
        assert np.all(oracle.sample_nmse <= baseline.sample_nmse + 1e-8)


def test_target_informed_oracle_selects_complementary_experts() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(41)
    target = torch.complex(
        torch.randn(4, *shape.raw_shape, generator=generator),
        torch.randn(4, *shape.raw_shape, generator=generator),
    )
    first = target.clone()
    second = target.clone()
    first[2:] = 0.0
    second[:2] = 0.0
    experts = {
        "first": sample_metric_batch(first, target, shape).as_dict(),
        "second": sample_metric_batch(second, target, shape).as_dict(),
    }
    oracle = target_informed_expert_oracle(experts)
    assert float(oracle["metrics"]["score"]) > 0.99999
    assert oracle["selection_counts"] == {"first": 2, "second": 2}


def test_outage_oracle_recovers_false_negative_energy() -> None:
    shape = ChannelShape(m=2, m_h=1, m_v=1, m_p=2, n=1, s=3)
    target = torch.ones(3, 2, 1, 3, dtype=torch.complex64)
    target[0] = 0.0
    prediction = target.clone()
    prediction[0] = 2.0
    batch = sample_metric_batch(
        prediction, target, shape, torch.tensor([True, False, False])
    ).as_dict()
    result = outage_threshold_oracle(
        batch,
        np.asarray([0.9, 0.1, 0.1]),
        np.asarray([True, False, False]),
        np.asarray([0, 0, 1]),
        threshold_steps=10,
    )
    assert abs(float(result["perfect_label_metrics"]["nmse"])) < 1e-8
    assert abs(float(result["perfect_label_metrics"]["score"]) - 1.0) < 1e-6
    assert abs(float(result["best_global_hard_metrics"]["score"]) - 1.0) < 1e-6


def test_residual_set_encoder_preserves_full_seed_grid() -> None:
    model = ResidualCoefficientSetEncoder(
        spectrum_shape=(8, 2, 2, 4),
        query_dim=11,
        neighbor_dim=17,
        coefficient_dim=6,
        width=32,
        dropout=0.0,
    )
    output = model(
        torch.randn(3, 8, 2, 2, 4),
        torch.randn(3, 11),
        torch.randn(3, 5, 17),
        torch.rand(3, 5, 1),
    )
    assert output["coefficients"].shape == (3, 6)
    assert output["attention"].shape == (3, 5)
    assert torch.allclose(output["attention"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.all(output["effective_neighbors"] >= 1.0)


def test_spectrum_summary_features_are_finite_and_channel_preserving() -> None:
    latent = np.asarray(
        [
            [[[-1.0, 2.0]], [[3.0, -4.0]]],
            [[[5.0, 7.0]], [[-2.0, -6.0]]],
        ],
        dtype=np.float32,
    )
    features = spectrum_summary_features(latent, channels=2)
    assert features.shape == (2, 6)
    np.testing.assert_allclose(features[:, 4:], [[2.0, 4.0], [7.0, 6.0]])
    assert np.isfinite(features).all()


def test_complex_residual_modes_preserve_requested_component() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(53)
    base = torch.randn(2, *shape.ad_shape, generator=generator)
    corrected = torch.randn(2, *shape.ad_shape, generator=generator)
    base_complex = angle_delay_to_complex(base, shape)
    corrected_complex = angle_delay_to_complex(corrected, shape)
    assert torch.allclose(
        complex_to_angle_delay(base_complex, shape), base, atol=1e-6
    )
    variants = split_complex_correction(base, corrected, shape)
    magnitude = angle_delay_to_complex(variants["magnitude"], shape)
    phase = angle_delay_to_complex(variants["phase"], shape)
    assert torch.allclose(magnitude.abs(), corrected_complex.abs(), atol=1e-5)
    assert torch.allclose(phase.abs(), base_complex.abs(), atol=1e-5)


def test_low_rank_residual_reconstruction_uses_only_selected_rank() -> None:
    residual = torch.tensor([[3.0, 4.0, 9.0], [5.0, 8.0, 7.0]])
    mean = torch.tensor([1.0, 2.0, 7.0])
    components = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rank0 = reconstruct_low_rank_residual(residual, mean, components, rank=0)
    rank1 = reconstruct_low_rank_residual(residual, mean, components, rank=1)
    rank2 = reconstruct_low_rank_residual(residual, mean, components, rank=2)
    coefficients = project_low_rank_coefficients(residual, mean, components, rank=2)
    decoded = decode_low_rank_coefficients(coefficients, mean, components)
    assert torch.allclose(rank0, mean.expand_as(residual))
    assert torch.allclose(rank1, torch.tensor([[3.0, 2.0, 7.0], [5.0, 2.0, 7.0]]))
    assert torch.allclose(rank2, torch.tensor([[3.0, 4.0, 7.0], [5.0, 8.0, 7.0]]))
    assert torch.allclose(decoded, rank2)


def test_log_power_replacement_preserves_base_phase() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(59)
    base = torch.randn(2, *shape.ad_shape, generator=generator)
    base_complex = angle_delay_to_complex(base, shape)
    target_log_power = angle_delay_log_power(base * 1.7, shape, scale=4.0)
    replaced = replace_angle_delay_log_power(
        base, target_log_power, shape, scale=4.0
    )
    replaced_complex = angle_delay_to_complex(replaced, shape)
    np.testing.assert_allclose(
        replaced_complex.abs().numpy(),
        (base_complex * 1.7).abs().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    phase_alignment = (
        replaced_complex * base_complex.conj()
    ).imag.abs().max()
    assert float(phase_alignment) < 1e-5


def test_full_resolution_magnitude_refiner_starts_from_identity() -> None:
    generator = torch.Generator().manual_seed(61)
    base = torch.rand(2, 4, 3, 2, 9, generator=generator) * 3.0
    geometry = torch.randn(2, 7, generator=generator)
    model = FullResolutionMagnitudeRefiner(
        input_channels=4,
        geometry_dim=7,
        cell_count=2,
        width=8,
        blocks=2,
        dropout=0.0,
    )
    output = model(base, geometry, torch.tensor([0, 1]))
    expected = base
    normalized = normalize_log_power_grid(base, scale=4.0)
    normalized_power = torch.expm1(normalized) / 4.0
    assert output["log_power"].shape == base.shape
    assert torch.allclose(output["correction"], torch.zeros_like(base))
    assert torch.allclose(output["log_power"], expected, atol=1e-6)
    assert torch.allclose(
        normalized_power.mean(dim=(1, 2, 3, 4)), torch.ones(2), atol=1e-5
    )
    loss = energy_weighted_log_power_loss(
        output["log_power"], expected * 1.05, scale=4.0
    )
    marginal_loss = magnitude_marginal_cosine_loss(
        output["log_power"], expected * 1.05, scale=4.0, frequency_groups=2
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(marginal_loss)
    loss.backward()
    assert model.output.weight.grad is not None


def test_local_set_magnitude_refiner_is_identity_and_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(67)
    base = torch.rand(2, 4, 3, 2, 9, generator=generator) * 3.0
    residual = torch.randn(2, 3, 4, 3, 2, 9, generator=generator)
    base_delta = torch.randn(2, 3, 4, 3, 2, 9, generator=generator)
    relative = torch.randn(2, 3, 6, generator=generator)
    geometry = torch.randn(2, 7, generator=generator)
    cells = torch.tensor([0, 1])
    model = QueryConditionedLocalSetMagnitudeRefiner(
        input_channels=4,
        geometry_dim=7,
        relative_dim=6,
        cell_count=2,
        width=8,
        blocks=2,
        dropout=0.0,
    )
    first = model(base, residual, base_delta, relative, geometry, cells)
    permutation = torch.tensor([2, 0, 1])
    second = model(
        base,
        residual[:, permutation],
        base_delta[:, permutation],
        relative[:, permutation],
        geometry,
        cells,
    )
    inverse = torch.argsort(permutation)
    assert first["log_power"].shape == base.shape
    assert first["attention"].shape == (2, 3, 3, 2, 9)
    assert torch.allclose(first["correction"], torch.zeros_like(base))
    assert torch.allclose(first["log_power"], base, atol=1e-6)
    assert torch.allclose(first["attention"].sum(dim=1), torch.ones(2, 3, 2, 9))
    assert torch.allclose(
        first["attention"], second["attention"][:, inverse], atol=1e-6
    )
    assert torch.allclose(first["log_power"], second["log_power"], atol=1e-6)
    loss = energy_weighted_log_power_loss(
        first["log_power"], base + 0.2, scale=4.0
    )
    loss.backward()
    assert model.output.weight.grad is not None
    assert model.transfer_scale.grad is not None


def test_local_magnitude_transfer_uses_same_cell_residuals() -> None:
    positions = np.asarray([[0, 0], [1, 0], [10, 0], [11, 0]], dtype=np.float32)
    cells = np.asarray([0, 0, 1, 1], dtype=np.int64)
    neighbors, distances = same_cell_neighbors(
        positions,
        cells,
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        count=1,
    )
    np.testing.assert_array_equal(neighbors[:, 0], np.asarray([0, 2]))
    np.testing.assert_allclose(distances[:, 0], 1.0)
    query = np.ones((2, 1, 1), dtype=np.float32)
    neighbor_base = np.zeros((2, 1, 1, 1), dtype=np.float32)
    neighbor_target = np.asarray([[[[2.0]]], [[[4.0]]]], dtype=np.float32)
    transferred = transfer_log_power_residual(
        query,
        neighbor_base,
        neighbor_target,
        distances,
        count=1,
        strength=0.5,
    )
    np.testing.assert_allclose(transferred[:, 0, 0], np.asarray([2.0, 3.0]))


def test_magnitude_profile_alignment_recovers_known_roll() -> None:
    source_power = np.zeros((1, 1, 4, 6, 16), dtype=np.float32)
    source_power[0, 0, 1, 2, 4] = 3.0
    source_power[0, 0, 2, 4, 7] = 1.0
    query_power = np.roll(source_power, (1, -2, 3), axis=(2, 3, 4))
    source_log = np.log1p(4.0 * source_power)[:, None]
    query_log = np.log1p(4.0 * query_power)
    shifts = estimate_magnitude_profile_shifts(
        query_log,
        source_log,
        maximum_vertical_shift=2,
        maximum_horizontal_shift=3,
        maximum_delay_shift=5,
    )
    np.testing.assert_array_equal(shifts[0, 0], np.asarray([1, -2, 3]))


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


def test_pas_projection_uses_mean_pas_target() -> None:
    shape = _shape()
    generator = torch.Generator().manual_seed(17)
    source = torch.complex(
        torch.randn(3, *shape.raw_shape, generator=generator),
        torch.randn(3, *shape.raw_shape, generator=generator),
    )
    target_channel = torch.complex(
        torch.randn(3, *shape.raw_shape, generator=generator),
        torch.randn(3, *shape.raw_shape, generator=generator),
    )
    target = channel_spectral_targets(target_channel, shape, proxy_count=2)
    target_proxy, target_mean = decode_pas_log(
        target["pas_log"], shape, proxy_count=2
    )
    source_mean = decode_pas_log(
        channel_spectral_targets(source, shape, proxy_count=2)["pas_log"],
        shape,
        proxy_count=2,
    )[1]
    projected = _pas_projection(
        source,
        target_proxy,
        target_mean,
        shape,
        proxy_count=2,
        minimum_scale=0.01,
        maximum_scale=100.0,
    )
    projected_mean = decode_pas_log(
        channel_spectral_targets(projected, shape, proxy_count=2)["pas_log"],
        shape,
        proxy_count=2,
    )[1]

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.flatten(1)
        right = right.flatten(1)
        return (left * right).sum(1) / (left.norm(dim=1) * right.norm(dim=1))

    assert torch.all(cosine(projected_mean, target_mean) > cosine(source_mean, target_mean))


def test_output_projection_can_start_from_reference_or_transport() -> None:
    model = torch.full((1, 2, 1, 2), 1.0 + 0.0j)
    reference = torch.full_like(model, 2.0 + 0.0j)
    transport = torch.full_like(model, 3.0 + 0.0j)
    assert _output_projection_seed("model", model, reference, transport) is model
    assert _output_projection_seed("reference", model, reference, transport) is reference
    assert _output_projection_seed("transport", model, reference, transport) is transport
    with np.testing.assert_raises_regex(ValueError, "requires a transport seed"):
        _output_projection_seed("transport", model, reference, None)


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


def test_v7_neural_teacher_preserves_full_latent_width() -> None:
    model = SpectralNeighborRefiner(
        latent_dim=12,
        pas_dim=8,
        query_feature_dim=5,
        neighbor_feature_dim=7,
        width=32,
        layers=2,
        heads=4,
        dropout=0.0,
    )
    output = model(
        torch.randn(3, 12),
        torch.randn(3, 5),
        torch.randn(3, 4, 12),
        torch.randn(3, 4, 7),
    )
    assert output["latent"].shape == (3, 12)
    assert output["residual"].shape == (3, 12)
    assert torch.isfinite(output["latent"]).all()


class SchemeECoreTests(unittest.TestCase):
    def test_carrier_quality_gate(self) -> None:
        test_carrier_quality_gate_keeps_reliable_fit_and_replaces_weak_fit()

    def test_round1_marginal_projection(self) -> None:
        test_round1_marginal_projection_is_finite_and_power_aligned()

    def test_diagnostic_metric_bridge(self) -> None:
        test_diagnostic_metrics_match_streaming_evaluator()

    def test_diagnostic_scale_oracles(self) -> None:
        test_scale_oracles_do_not_increase_nmse()

    def test_diagnostic_expert_oracle(self) -> None:
        test_target_informed_oracle_selects_complementary_experts()

    def test_diagnostic_outage_oracle(self) -> None:
        test_outage_oracle_recovers_false_negative_energy()

    def test_residual_set_encoder(self) -> None:
        test_residual_set_encoder_preserves_full_seed_grid()

    def test_spectrum_summary_features(self) -> None:
        test_spectrum_summary_features_are_finite_and_channel_preserving()

    def test_complex_residual_modes(self) -> None:
        test_complex_residual_modes_preserve_requested_component()

    def test_low_rank_residual_reconstruction(self) -> None:
        test_low_rank_residual_reconstruction_uses_only_selected_rank()

    def test_log_power_replacement(self) -> None:
        test_log_power_replacement_preserves_base_phase()

    def test_full_resolution_magnitude_refiner(self) -> None:
        test_full_resolution_magnitude_refiner_starts_from_identity()

    def test_local_set_magnitude_refiner(self) -> None:
        test_local_set_magnitude_refiner_is_identity_and_permutation_invariant()

    def test_local_magnitude_transfer(self) -> None:
        test_local_magnitude_transfer_uses_same_cell_residuals()

    def test_magnitude_profile_alignment(self) -> None:
        test_magnitude_profile_alignment_recovers_known_roll()

    def test_adaptive_hybrid_config_is_scoped(self) -> None:
        base = {
            "spectral_teacher": {"oof_output_path": "base.npz"},
            "hybrid": {
                "output_dir": "base",
                "learning_rate": 2e-4,
                "structured_spectral_field": True,
            },
        }
        prepared = adaptive_hybrid_config(
            base,
            adaptive_prior="adaptive.npz",
            initial_checkpoint="best.pt",
            output_dir="new",
        )
        self.assertEqual(
            prepared["spectral_teacher"]["oof_output_path"], "adaptive.npz"
        )
        self.assertEqual(prepared["hybrid"]["initial_checkpoint"], "best.pt")
        self.assertEqual(prepared["hybrid"]["output_dir"], "new")
        self.assertTrue(prepared["hybrid"]["structured_spectral_field"])
        self.assertEqual(base["spectral_teacher"]["oof_output_path"], "base.npz")

    def test_spectral_targets(self) -> None:
        test_spectral_targets_and_projection_are_finite()

    def test_rf_features(self) -> None:
        test_rf_gaussians_produce_exactly_71_features()

    def test_relaxed_output_projection(self) -> None:
        test_relaxed_output_projection_preserves_requested_power()

    def test_pas_projection_mean_target(self) -> None:
        test_pas_projection_uses_mean_pas_target()

    def test_output_projection_seed(self) -> None:
        test_output_projection_can_start_from_reference_or_transport()

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

    def test_v7_neural_teacher(self) -> None:
        test_v7_neural_teacher_preserves_full_latent_width()

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
