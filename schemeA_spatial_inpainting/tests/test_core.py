from __future__ import annotations

import unittest

import numpy as np
import torch

from spatial_inpainting.angle_delay import (
    ChannelShape,
    angle_delay_to_channel,
    channel_to_angle_delay,
    restore_power,
    split_power,
)
from spatial_inpainting.metrics import pas_accuracy, pdp_accuracy
from spatial_inpainting.spatial_grid import (
    GridSpec,
    assign_cells,
    build_bev_features,
    build_geometry_maps,
    infer_two_cell_rule,
)
from spatial_inpainting.unet import SpatialUNet, pad_to_multiple, unpad


class CoreTests(unittest.TestCase):
    @staticmethod
    def competition_shape() -> ChannelShape:
        return ChannelShape(m=256, m_h=16, m_v=8, m_p=2, n=4, s=192)

    def test_power_split_and_restore_round_trip(self) -> None:
        generator = torch.Generator().manual_seed(7)
        real = torch.randn((2, 256, 4, 192), generator=generator)
        imaginary = torch.randn((2, 256, 4, 192), generator=generator)
        channel = torch.complex(real, imaginary) * 1e-5
        normalized, log_power, outage = split_power(channel)
        restored = restore_power(normalized, log_power, outage)
        torch.testing.assert_close(restored, channel, rtol=2e-5, atol=1e-10)
        torch.testing.assert_close(
            normalized.abs().square().mean(dim=(1, 2, 3)),
            torch.ones(2),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_angle_delay_round_trip(self) -> None:
        shape = self.competition_shape()
        generator = torch.Generator().manual_seed(11)
        channel = torch.complex(
            torch.randn((1, *shape.raw_shape), generator=generator),
            torch.randn((1, *shape.raw_shape), generator=generator),
        )
        angle_delay = channel_to_angle_delay(channel, shape)
        restored = angle_delay_to_channel(angle_delay, shape)
        self.assertEqual(angle_delay.shape, (1, *shape.ad_shape))
        torch.testing.assert_close(restored, channel, rtol=2e-5, atol=2e-5)

    def test_two_cell_rule_uses_empty_spatial_gap(self) -> None:
        lower = np.stack([np.linspace(-5, 5, 20), np.linspace(-30, -20, 20)], axis=1)
        upper = np.stack([np.linspace(-4, 6, 20), np.linspace(20, 30, 20)], axis=1)
        positions = np.pad(np.concatenate([lower, upper]), ((0, 0), (0, 1)))
        rule = infer_two_cell_rule(positions)
        labels = assign_cells(positions, rule)
        self.assertEqual(rule["axis"], 1)
        np.testing.assert_array_equal(labels[:20], np.zeros(20, dtype=np.int64))
        np.testing.assert_array_equal(labels[20:], np.ones(20, dtype=np.int64))

    def test_grid_geometry_shapes(self) -> None:
        spec = GridSpec(minimum_x=-2.0, minimum_y=-3.0, resolution=1.0, height=5, width=7)
        geometry = build_geometry_maps(
            spec,
            base_station=np.asarray([0.0, 0.0, 10.0]),
            boresight_degrees=0.0,
            sector_half_angle_degrees=61.0,
            maximum_distance=20.0,
            cell_id=1,
            cell_count=2,
        )
        self.assertEqual(geometry["distance"].shape, (1, 5, 7))
        self.assertEqual(geometry["relative_angle"].shape, (1, 5, 7))
        self.assertEqual(geometry["valid"].shape, (1, 5, 7))
        self.assertEqual(geometry["identity"].shape, (2, 5, 7))
        self.assertTrue(np.all(geometry["identity"][0] == 0))
        self.assertTrue(np.all(geometry["identity"][1] == 1))

    def test_bev_channel_semantics(self) -> None:
        spec = GridSpec(minimum_x=0.0, minimum_y=0.0, resolution=1.0, height=1, width=4)
        points = np.asarray(
            [[0.5, 0.5, 2.0], [1.5, 0.5, 5.0], [2.5, 0.5, 15.0], [3.5, 0.5, 30.0]],
            dtype=np.float32,
        )
        bev = build_bev_features(points, spec)
        self.assertEqual(bev.shape, (6, 1, 4))
        np.testing.assert_allclose(bev[0], np.full((1, 4), np.log(2.0)))
        np.testing.assert_allclose(bev[1, 0], np.asarray([0.0, 3 / 28, 13 / 28, 1.0]))
        np.testing.assert_array_equal(bev[2:, 0], np.eye(4, dtype=np.float32))

    def test_unet_padding_and_output_shapes(self) -> None:
        model = SpatialUNet(input_channels=21, latent_dim=8, base_channels=2, dropout=0.0)
        value = torch.randn(1, 21, 17, 23)
        padded, original = pad_to_multiple(value)
        output = {key: unpad(result, original) for key, result in model(padded).items()}
        self.assertEqual(output["latent"].shape, (1, 8, 17, 23))
        self.assertEqual(output["power"].shape, (1, 1, 17, 23))
        self.assertEqual(output["outage_logit"].shape, (1, 1, 17, 23))

    def test_zero_prediction_metrics_are_finite(self) -> None:
        shape = self.competition_shape()
        target = torch.complex(
            torch.full((1, *shape.raw_shape), 1e-7),
            torch.full((1, *shape.raw_shape), -2e-7),
        )
        prediction = torch.zeros_like(target)
        pas = pas_accuracy(prediction, target, shape)
        pdp = pdp_accuracy(prediction, target)
        self.assertTrue(torch.isfinite(pas))
        self.assertTrue(torch.isfinite(pdp))
        self.assertEqual(float(pas), 0.0)
        self.assertEqual(float(pdp), 0.0)


if __name__ == "__main__":
    unittest.main()
