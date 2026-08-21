from __future__ import annotations

import unittest

import numpy as np
import torch

from scheme_g.angle_delay import (
    ChannelShape,
    angle_delay_to_channel,
    channel_to_angle_delay,
)
from scheme_g.autoencoder import (
    FactorizedResidualAutoencoder,
    MetricHighFidelityAutoencoder,
    StructuredAngleDelayAutoencoder,
)
from scheme_g.autoencoder_training import autoencoder_training_stage
from scheme_g.context_model import (
    CellTokenPool,
    FullResolutionContextField,
    StructuredSpectralPriorEncoder,
)
from scheme_g.context_data import ContextRepository, compact_spectral_prior
from scheme_g.context_diagnostics import hybrid_predictions, nearest_observed_indices
from scheme_g.data import balanced_limit
from scheme_g.losses import (
    complex_coherence_loss,
    energy_weighted_complex_direction_loss,
    metric_aligned_channel_losses,
)
from scheme_g.metrics import (
    cosine_accuracy,
    nmse,
    official_score,
    pas_accuracy,
    pdp_accuracy,
)
from scheme_g.spatial_grid import GridSpec, test_like_validation_masks
from scheme_g.projection import soft_shape_projection


class AngleDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = ChannelShape(m=64, m_h=8, m_v=4, m_p=2, n=1, s=16)

    def test_transform_round_trip(self) -> None:
        torch.manual_seed(1)
        channel = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        restored = angle_delay_to_channel(
            channel_to_angle_delay(channel, self.shape), self.shape
        )
        self.assertTrue(torch.allclose(channel, restored, atol=2e-5, rtol=2e-5))

    def test_metrics_are_exact_for_identical_channels(self) -> None:
        torch.manual_seed(2)
        channel = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        self.assertAlmostEqual(
            float(pas_accuracy(channel, channel, self.shape)), 1.0, places=5
        )
        self.assertAlmostEqual(float(pdp_accuracy(channel, channel)), 1.0, places=5)
        self.assertAlmostEqual(float(nmse(channel, channel)), 0.0, places=7)

    def test_structured_autoencoder_shapes_and_gradient(self) -> None:
        model = StructuredAngleDelayAutoencoder(
            self.shape,
            spectrum_stem_channels=2,
            phase_stem_channels=2,
            spectrum_latent_channels=2,
            phase_latent_channels=1,
        )
        value = torch.randn(2, *self.shape.ad_shape)
        output, spectrum, phase = model(value)
        spectrum_only = model.decode(spectrum, None)
        self.assertEqual(output.shape, value.shape)
        self.assertEqual(spectrum_only.shape, value.shape)
        self.assertEqual(spectrum.shape, (2, model.spectrum_latent_dim))
        self.assertEqual(phase.shape, (2, model.phase_latent_dim))
        output.square().mean().backward()
        self.assertIsNotNone(next(model.parameters()).grad)

    def test_high_fidelity_autoencoder_shapes_and_gradient(self) -> None:
        model = MetricHighFidelityAutoencoder(
            self.shape,
            spectrum_stem_channels=4,
            phase_stem_channels=2,
            spectrum_latent_channels=4,
            phase_latent_channels=2,
            residual_blocks=1,
        )
        value = torch.randn(1, *self.shape.ad_shape)
        output, spectrum, detail = model(value)
        spectrum_only = model.decode(spectrum, None)
        self.assertEqual(output.shape, value.shape)
        self.assertEqual(spectrum_only.shape, value.shape)
        self.assertEqual(spectrum.shape, (1, model.spectrum_latent_dim))
        self.assertEqual(detail.shape, (1, model.phase_latent_dim))
        self.assertEqual(model.spectrum_shape.tensor_shape, (4, 1, 2, 1))
        self.assertEqual(model.phase_shape.tensor_shape, (2, 2, 4, 2))
        (output.square().mean() + spectrum_only.square().mean()).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_factorized_residual_autoencoder_keeps_detail_effective(self) -> None:
        torch.manual_seed(5)
        model = FactorizedResidualAutoencoder(
            self.shape,
            spectrum_stem_channels=4,
            phase_stem_channels=4,
            spectrum_latent_channels=4,
            phase_latent_channels=2,
            residual_blocks=1,
            detail_hidden_channels=4,
            spectrum_decoder_channels=4,
            detail_decoder_channels=4,
        )
        value = torch.randn(1, *self.shape.ad_shape)
        output, spectrum, detail = model(value)
        coarse = model.decode(spectrum, None)
        weak_detail = model.decode(spectrum, detail, detail_scale=0.1)
        zero_latent = torch.zeros(1, *model.phase_shape.tensor_shape)
        zero_decoded = model.decoder.detail_decoder(zero_latent)

        self.assertEqual(output.shape, value.shape)
        self.assertEqual(model.spectrum_shape.tensor_shape, (4, 1, 2, 1))
        self.assertEqual(model.phase_shape.tensor_shape, (2, 2, 4, 2))
        self.assertEqual(float(zero_decoded.detach().abs().max()), 0.0)
        self.assertGreater(float((output - coarse).detach().square().mean()), 1e-3)
        self.assertGreater(float((output - weak_detail).detach().square().mean()), 1e-3)

        loss = (output - value).square().mean()
        loss.backward()
        detail_gradients = [
            parameter.grad
            for parameter in model.decoder.detail_decoder.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(detail_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in detail_gradients)
        )
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in detail_gradients), 0.0
        )

        model.set_trainable_stage("coarse")
        self.assertTrue(next(model.spectrum_encoder.parameters()).requires_grad)
        self.assertFalse(next(model.phase_encoder.parameters()).requires_grad)
        model.set_trainable_stage("detail")
        self.assertFalse(next(model.spectrum_encoder.parameters()).requires_grad)
        self.assertTrue(next(model.phase_encoder.parameters()).requires_grad)
        model.set_trainable_stage("joint")
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )

    def test_factorized_training_stage_boundaries(self) -> None:
        section = {"coarse_pretrain_epochs": 2, "detail_pretrain_epochs": 3}
        self.assertEqual(autoencoder_training_stage(section, 0), "coarse")
        self.assertEqual(autoencoder_training_stage(section, 1), "coarse")
        self.assertEqual(autoencoder_training_stage(section, 2), "detail")
        self.assertEqual(autoencoder_training_stage(section, 4), "detail")
        self.assertEqual(autoencoder_training_stage(section, 5), "joint")

    def test_metric_cosine_is_stable_for_tiny_and_zero_rows(self) -> None:
        prediction = torch.tensor(
            [[1e-20, 2e-20, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
        )
        target = prediction.detach().clone()
        accuracy = cosine_accuracy(prediction, target)
        self.assertAlmostEqual(float(accuracy.detach()), 1.0, places=6)
        (1.0 - accuracy).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(
            float(cosine_accuracy(torch.ones(1, 3), torch.zeros(1, 3))),
            0.0,
        )

    def test_complex_coherence_loss_rewards_identical_fields(self) -> None:
        target = torch.randn(2, *self.shape.ad_shape)
        self.assertAlmostEqual(
            float(complex_coherence_loss(target, target)), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(complex_coherence_loss(-target, target)), 2.0, places=6
        )
        self.assertAlmostEqual(
            float(energy_weighted_complex_direction_loss(target, target, self.shape)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(energy_weighted_complex_direction_loss(-target, target, self.shape)),
            2.0,
            places=6,
        )

    def test_metric_aligned_score_loss_matches_official_formula(self) -> None:
        torch.manual_seed(8)
        target = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        prediction = target + 0.2 * torch.complex(
            torch.randn_like(target.real), torch.randn_like(target.real)
        )
        losses = metric_aligned_channel_losses(prediction, target, self.shape)
        pas = float(pas_accuracy(prediction, target, self.shape))
        pdp = float(pdp_accuracy(prediction, target))
        channel_nmse = float(nmse(prediction, target))
        self.assertAlmostEqual(
            float(losses["score"]),
            1.0 - official_score(pas, pdp, channel_nmse),
            places=6,
        )

    def test_soft_spectral_projection_preserves_total_power(self) -> None:
        torch.manual_seed(19)
        channel = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        proxy_count = 4
        pas_width = (proxy_count + 1) * self.shape.m_v * self.shape.m_h
        pas = torch.rand(2, pas_width)
        pdp = torch.rand(2, self.shape.n * self.shape.s)
        projected = soft_shape_projection(
            channel,
            torch.log1p(1000.0 * pas),
            torch.log1p(1000.0 * pdp),
            self.shape,
            alpha=0.3,
            proxy_count=proxy_count,
        )
        before = channel.abs().square().mean(dim=(1, 2, 3))
        after = projected.abs().square().mean(dim=(1, 2, 3))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.allclose(before, after, rtol=2e-5, atol=2e-5))


class SpatialAndContextTests(unittest.TestCase):
    def test_structured_prior_encoder_preserves_latent_grid(self) -> None:
        encoder = StructuredSpectralPriorEncoder(
            proxy_count=2,
            antenna_shape=(2, 3),
            pdp_shape=(2, 8),
            output_channels=4,
            output_shape=(1, 2, 4),
        )
        pas = torch.randn(3, 18, requires_grad=True)
        pdp = torch.randn(3, 16, requires_grad=True)
        output = encoder(pas, pdp, torch.rand(3))
        self.assertEqual(output.shape, (3, 4, 1, 2, 4))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(pas.grad).all())
        self.assertTrue(torch.isfinite(pdp.grad).all())

    def test_spectral_prior_compaction_keeps_shape_statistics(self) -> None:
        count = 3
        prior = {
            "pas_log": np.ones((count, 12), dtype=np.float32),
            "pdp_log": np.ones((count, 16), dtype=np.float32),
            "ue_log_energy": np.ones((count, 2), dtype=np.float32),
            "log_power": np.zeros(count, dtype=np.float32),
            "uncertainty": np.zeros(count, dtype=np.float32),
            "outage_probability": np.zeros(count, dtype=np.float32),
            "available": np.ones(count, dtype=np.float32),
        }
        compact = compact_spectral_prior(prior, proxy_count=2, pdp_bins=4)
        self.assertEqual(compact.shape, (count, 26))
        self.assertTrue(np.isfinite(compact).all())

    def test_nearest_observations_stay_inside_the_serving_cell(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.8, 0.0],
                [100.0, 0.0],
                [104.0, 0.0],
                [103.5, 0.0],
            ],
            dtype=np.float32,
        )
        cells = np.asarray([0, 0, 0, 1, 1, 1])
        nearest, distances = nearest_observed_indices(
            positions,
            cells,
            observed_indices=np.asarray([0, 1, 3, 4]),
            target_indices=np.asarray([2, 5]),
            neighbors=2,
        )
        self.assertEqual(nearest.tolist(), [[1, 0], [4, 3]])
        self.assertTrue(np.allclose(distances[:, 0], [0.2, 0.5], atol=1e-5))

    def test_hybrid_predictions_replace_only_requested_components(self) -> None:
        predicted = {
            "spectrum": np.full((2, 3), 1.0, dtype=np.float32),
            "phase": np.full((2, 4), 2.0, dtype=np.float32),
            "power": np.full(2, 3.0, dtype=np.float32),
            "outage_probability": np.full(2, 0.2, dtype=np.float32),
        }
        target = {
            "spectrum": np.full((2, 3), 10.0, dtype=np.float32),
            "phase": np.full((2, 4), 20.0, dtype=np.float32),
            "power": np.full(2, 30.0, dtype=np.float32),
            "outage": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        hybrid = hybrid_predictions(
            predicted,
            target,
            replacements=("phase", "power"),
            oracle_outage=True,
        )
        self.assertTrue(np.array_equal(hybrid["spectrum"], predicted["spectrum"]))
        self.assertTrue(np.array_equal(hybrid["phase"], target["phase"]))
        self.assertTrue(np.array_equal(hybrid["power"], target["power"]))
        self.assertTrue(np.array_equal(hybrid["outage_probability"], target["outage"]))

    def test_grid_offsets_and_sampling_coordinates(self) -> None:
        spec = GridSpec(10.0, 20.0, 3.0, height=2, width=4)
        centers = np.asarray([[11.5, 21.5], [20.5, 24.5]], dtype=np.float32)
        self.assertTrue(np.allclose(spec.offsets(centers), 0.0))
        expected = np.asarray([[-0.75, -0.5], [0.75, 0.5]], dtype=np.float32)
        self.assertTrue(np.allclose(spec.grid_sample_coordinates(centers), expected))

    def test_periodic_holdouts_split_every_cell(self) -> None:
        grid_x, grid_y = np.meshgrid(
            np.arange(0.0, 150.0, 5.0), np.arange(0.0, 80.0, 5.0)
        )
        xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        positions = np.concatenate([xy, np.full((len(xy), 1), 1.5)], axis=1)
        cells = (positions[:, 0] >= 75.0).astype(np.int64)
        masks = test_like_validation_masks(positions, cells, 5, 36.0, 12.0)
        self.assertEqual(masks.shape, (5, len(positions)))
        for mask in masks:
            for cell_id in (0, 1):
                count = int(np.sum(mask & (cells == cell_id)))
                self.assertGreater(count, 0)
                self.assertLess(count, int(np.sum(cells == cell_id)))

    def test_balanced_limit_preserves_two_cells(self) -> None:
        indices = np.arange(10, dtype=np.int64)
        cells = np.asarray([0] * 8 + [1] * 2)
        selected = balanced_limit(indices, 4, [cells], seed=3)
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(cells[selected].tolist()), {0, 1})

    def test_cell_pool_and_query_model_backpropagate(self) -> None:
        torch.manual_seed(4)
        pool = CellTokenPool(point_channels=4, token_channels=4, hidden_channels=8)
        point_features = torch.randn(3, 4, requires_grad=True)
        pooled, observed, log_count = pool(
            point_features, torch.tensor([0, 0, 7]), height=3, width=4
        )
        self.assertEqual(pooled.shape, (1, 4, 3, 4))
        self.assertEqual(int(observed.sum()), 2)
        self.assertGreater(float(log_count[0, 0, 0, 0]), 0.0)
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(point_features.grad).all())

        model = FullResolutionContextField(
            spectrum_shape=(4, 1, 2, 1),
            phase_shape=(2, 2, 4, 2),
            cell_count=2,
            static_context_channels=11,
            query_numeric_channels=9,
            map_token_channels=4,
            map_hidden_channels=8,
            context_base_channels=2,
            context_feature_channels=4,
            environment_base_channels=2,
            environment_feature_channels=2,
            environment_blocks=1,
            corridor_width=8,
            corridor_heads=2,
            corridor_layers=1,
            corridor_maximum_samples=8,
            station_embedding_channels=2,
            fourier_bands=2,
            global_width=16,
            global_blocks=1,
            router_width=8,
            router_top_k=3,
            pair_width=8,
            spectrum_token_channels=8,
            detail_token_channels=8,
            attention_heads=2,
            attention_chunk_size=2,
            refinement_blocks=1,
            axial_blocks=1,
            dropout=0.0,
            gradient_checkpointing=True,
        )
        outputs = model(
            cell_id=1,
            observed_spectrum=torch.randn(3, 4, 1, 2, 1),
            observed_phase=torch.randn(3, 2, 2, 4, 2),
            observed_power=torch.randn(3),
            observed_outage=torch.tensor([0.0, 0.0, 1.0]),
            point_features=torch.randn(3, 4),
            point_flat_indices=torch.tensor([0, 0, 7]),
            context_static=torch.randn(11, 7, 9),
            environment_bev=torch.randn(6, 13, 17),
            observed_context_coordinates=torch.tensor(
                [[-0.6, -0.5], [0.0, 0.0], [0.5, 0.4]]
            ),
            observed_environment_coordinates=torch.tensor(
                [[-0.5, -0.4], [0.1, 0.0], [0.4, 0.3]]
            ),
            observed_numeric=torch.randn(3, 9),
            observed_relative_xy=torch.randn(3, 2),
            query_context_coordinates=torch.tensor([[-0.5, -0.5], [0.2, 0.4]]),
            query_environment_coordinates=torch.tensor([[-0.4, -0.5], [0.3, 0.4]]),
            query_corridor_coordinates=torch.tensor(
                [
                    [[-0.9, -0.9], [-0.6, -0.6], [-0.4, -0.5]],
                    [[-0.9, -0.9], [-0.3, -0.2], [0.3, 0.4]],
                ]
            ),
            query_numeric=torch.randn(2, 9),
            query_relative_xy=torch.randn(2, 2),
        )
        self.assertEqual(outputs["spectrum"].shape, (2, 8))
        self.assertEqual(outputs["phase"].shape, (2, 32))
        self.assertEqual(outputs["power"].shape, (2,))
        self.assertTrue(torch.all(outputs["power_q10"] <= outputs["power"]))
        self.assertTrue(torch.all(outputs["power"] <= outputs["power_q90"]))
        self.assertEqual(outputs["outage_logit"].shape, (2,))
        self.assertEqual(outputs["router_entropy"].shape, (2,))
        self.assertEqual(outputs["router_top1_mass"].shape, (2,))
        self.assertEqual(outputs["router_effective_neighbors"].shape, (2,))
        self.assertEqual(outputs["router_distance"].shape, (2,))
        self.assertEqual(outputs["spectrum_warp"].shape, (2,))
        self.assertEqual(outputs["detail_warp"].shape, (2,))
        self.assertTrue(torch.all(outputs["router_top1_mass"] > 0.0))
        self.assertTrue(torch.all(outputs["router_effective_neighbors"] >= 1.0))
        self.assertTrue(
            torch.all(outputs["detail_token_effective_neighbors"] <= 2.0 + 1e-6)
        )
        model.set_diagnostic_ablation(disable_warp=True, route_bias_scale=0.25)
        ablated = model(
            cell_id=1,
            observed_spectrum=torch.randn(3, 4, 1, 2, 1),
            observed_phase=torch.randn(3, 2, 2, 4, 2),
            observed_power=torch.randn(3),
            observed_outage=torch.tensor([0.0, 0.0, 1.0]),
            point_features=torch.randn(3, 4),
            point_flat_indices=torch.tensor([0, 0, 7]),
            context_static=torch.randn(11, 7, 9),
            environment_bev=torch.randn(6, 13, 17),
            observed_context_coordinates=torch.tensor(
                [[-0.6, -0.5], [0.0, 0.0], [0.5, 0.4]]
            ),
            observed_environment_coordinates=torch.tensor(
                [[-0.5, -0.4], [0.1, 0.0], [0.4, 0.3]]
            ),
            observed_numeric=torch.randn(3, 9),
            observed_relative_xy=torch.randn(3, 2),
            query_context_coordinates=torch.tensor([[-0.5, -0.5], [0.2, 0.4]]),
            query_environment_coordinates=torch.tensor([[-0.4, -0.5], [0.3, 0.4]]),
            query_corridor_coordinates=torch.tensor(
                [
                    [[-0.9, -0.9], [-0.6, -0.6], [-0.4, -0.5]],
                    [[-0.9, -0.9], [-0.3, -0.2], [0.3, 0.4]],
                ]
            ),
            query_numeric=torch.randn(2, 9),
            query_relative_xy=torch.randn(2, 2),
        )
        self.assertEqual(float(ablated["spectrum_warp"].max()), 0.0)
        self.assertEqual(float(ablated["detail_warp"].max()), 0.0)
        sum(value.square().mean() for value in outputs.values()).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_spatial_mask_hides_the_complete_region_and_guard_band(self) -> None:
        repository = ContextRepository.__new__(ContextRepository)
        x, y = np.meshgrid(np.arange(0.0, 30.0, 2.0), np.arange(0.0, 20.0, 2.0))
        positions = np.stack(
            [x.ravel(), y.ravel(), np.full(x.size, 1.5)], axis=1
        ).astype(np.float32)
        repository.metadata = {
            "train_positions": positions,
            "outage": np.zeros(len(positions), dtype=bool),
        }
        repository.indices_by_cell = [np.arange(len(positions), dtype=np.int64)]
        repository.test_component_templates = [[]]
        sample = repository.sample_spatial_mask(
            np.random.default_rng(7),
            cell_id=0,
            minimum_meters=12.0,
            maximum_meters=12.0,
            minimum_targets=2,
            maximum_targets=3,
            test_template_probability=0.0,
            guard_min_meters=4.0,
            guard_max_meters=4.0,
        )
        observed = repository.context_indices(0, sample.hidden)
        self.assertLessEqual(len(sample.targets), 3)
        self.assertTrue(set(sample.targets).issubset(set(sample.hidden)))
        self.assertFalse(np.intersect1d(sample.hidden, observed).size)
        distance = np.linalg.norm(
            positions[sample.targets, None, :2] - positions[observed][None, :, :2],
            axis=2,
        )
        self.assertGreaterEqual(float(distance.min()), 4.0)


if __name__ == "__main__":
    unittest.main()
