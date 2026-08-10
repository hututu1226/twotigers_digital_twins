from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.metrics import nmse, pas_accuracy, pdp_accuracy
from channel_ai.preprocessing import infer_two_cell_labels
from channel_ai.transforms import ChannelShape, angle_delay_to_channel, channel_to_angle_delay


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = ChannelShape(m=256, m_h=16, m_v=8, m_p=2, n=4, s=192)

    def test_angle_delay_round_trip(self) -> None:
        generator = torch.Generator().manual_seed(7)
        real = torch.randn((1, *self.shape.raw_shape), generator=generator)
        imag = torch.randn((1, *self.shape.raw_shape), generator=generator)
        channel = torch.complex(real, imag)
        reconstructed = angle_delay_to_channel(channel_to_angle_delay(channel, self.shape), self.shape)
        self.assertLess(float((channel - reconstructed).abs().max()), 2e-5)

    def test_identity_metrics(self) -> None:
        channel = torch.complex(
            torch.randn((1, *self.shape.raw_shape)),
            torch.randn((1, *self.shape.raw_shape)),
        )
        self.assertAlmostEqual(float(pas_accuracy(channel, channel, self.shape)), 1.0, places=5)
        self.assertAlmostEqual(float(pdp_accuracy(channel, channel)), 1.0, places=5)
        self.assertAlmostEqual(float(nmse(channel, channel)), 0.0, places=7)


class LabelTests(unittest.TestCase):
    def test_two_disconnected_cells(self) -> None:
        left = np.array([[0.0, -20.0, 1.5], [2.0, -18.0, 1.5], [-1.0, -17.0, 1.5]])
        right = np.array([[1.0, 20.0, 1.5], [3.0, 18.0, 1.5], [-2.0, 17.0, 1.5]])
        positions = np.concatenate([left, right])
        base_stations = np.array([[0.0, -10.0, 10.0], [0.0, 10.0, 10.0]])
        labels, details = infer_two_cell_labels(positions, base_stations)
        self.assertTrue(np.all(labels[:3] == 0))
        self.assertTrue(np.all(labels[3:] == 1))
        self.assertEqual(details["axis"], 1)


if __name__ == "__main__":
    unittest.main()

