from __future__ import annotations

import unittest

import numpy as np
import torch

from structured_context_field.angle_delay import (
    ChannelShape,
    angle_delay_to_channel,
    channel_to_angle_delay,
)
from structured_context_field.autoencoder import StructuredAngleDelayAutoencoder
from structured_context_field.context_model import CellTokenPool, StructuredContextField
from structured_context_field.data import balanced_limit
from structured_context_field.metrics import nmse, pas_accuracy, pdp_accuracy
from structured_context_field.spatial_grid import GridSpec, test_like_validation_masks


class AngleDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = ChannelShape(m=64, m_h=8, m_v=4, m_p=2, n=1, s=16)

    def test_transform_round_trip(self) -> None:
        torch.manual_seed(1)
        channel = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        restored = angle_delay_to_channel(channel_to_angle_delay(channel, self.shape), self.shape)
        self.assertTrue(torch.allclose(channel, restored, atol=2e-5, rtol=2e-5))

    def test_metrics_are_exact_for_identical_channels(self) -> None:
        torch.manual_seed(2)
        channel = torch.complex(torch.randn(2, 64, 1, 16), torch.randn(2, 64, 1, 16))
        self.assertAlmostEqual(float(pas_accuracy(channel, channel, self.shape)), 1.0, places=5)
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


class SpatialAndContextTests(unittest.TestCase):
    def test_grid_offsets_and_sampling_coordinates(self) -> None:
        spec = GridSpec(10.0, 20.0, 3.0, height=2, width=4)
        centers = np.asarray([[11.5, 21.5], [20.5, 24.5]], dtype=np.float32)
        self.assertTrue(np.allclose(spec.offsets(centers), 0.0))
        expected = np.asarray([[-0.75, -0.5], [0.75, 0.5]], dtype=np.float32)
        self.assertTrue(np.allclose(spec.grid_sample_coordinates(centers), expected))

    def test_periodic_holdouts_split_every_cell(self) -> None:
        grid_x, grid_y = np.meshgrid(np.arange(0.0, 150.0, 5.0), np.arange(0.0, 80.0, 5.0))
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
        pool = CellTokenPool(point_channels=14, token_channels=4, hidden_channels=8)
        point_features = torch.randn(3, 14, requires_grad=True)
        pooled, observed, log_count = pool(
            point_features, torch.tensor([0, 0, 7]), height=3, width=4
        )
        self.assertEqual(pooled.shape, (1, 4, 3, 4))
        self.assertEqual(int(observed.sum()), 2)
        self.assertGreater(float(log_count[0, 0, 0, 0]), 0.0)
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(point_features.grad).all())

        model = StructuredContextField(
            spectrum_latent_dim=6,
            phase_latent_dim=4,
            cell_count=2,
            static_context_channels=11,
            query_numeric_channels=9,
            token_channels=4,
            token_hidden_channels=8,
            context_base_channels=2,
            context_feature_channels=4,
            environment_feature_channels=2,
            station_embedding_channels=2,
            fourier_bands=2,
            query_width=16,
            query_blocks=1,
            adapter_width=4,
            dropout=0.0,
        )
        outputs = model(
            cell_id=1,
            point_features=torch.randn(3, 14),
            point_flat_indices=torch.tensor([0, 0, 7]),
            context_static=torch.randn(11, 7, 9),
            environment_bev=torch.randn(6, 13, 17),
            query_context_coordinates=torch.tensor([[-0.5, -0.5], [0.2, 0.4]]),
            query_environment_coordinates=torch.tensor([[-0.4, -0.5], [0.3, 0.4]]),
            query_numeric=torch.randn(2, 9),
            query_relative_xy=torch.randn(2, 2),
        )
        self.assertEqual(outputs["spectrum"].shape, (2, 6))
        self.assertEqual(outputs["phase"].shape, (2, 4))
        self.assertEqual(outputs["power"].shape, (2,))
        self.assertEqual(outputs["outage_logit"].shape, (2,))
        sum(value.square().mean() for value in outputs.values()).backward()
        self.assertIsNotNone(next(model.parameters()).grad)


if __name__ == "__main__":
    unittest.main()
