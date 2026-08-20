from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_context_masks import ensure_preprocessing  # noqa: E402
from scheme_d.context_model import (  # noqa: E402
    GeometryWarpedLatentField,
    ObservationRouter,
)
from scheme_d.reporting import evaluation_metrics  # noqa: E402


class ContextMaskBootstrapTests(unittest.TestCase):
    @patch("analyze_context_masks.preprocess_dataset")
    def test_missing_metadata_triggers_preprocessing(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "preprocessed"
            artifact_dir.mkdir()
            (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
            config = {"preprocessing": {"artifact_dir": str(artifact_dir)}}

            rebuilt = ensure_preprocessing(config)

        self.assertTrue(rebuilt)
        preprocess.assert_called_once_with(config, force=True)

    @patch("analyze_context_masks.preprocess_dataset")
    def test_existing_metadata_is_reused(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "preprocessed"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.npz").write_bytes(b"metadata")
            config = {"preprocessing": {"artifact_dir": str(artifact_dir)}}

            rebuilt = ensure_preprocessing(config)

        self.assertFalse(rebuilt)
        preprocess.assert_not_called()


class TransportResidualAutomationTests(unittest.TestCase):
    def test_nested_and_flat_evaluation_reports_are_supported(self) -> None:
        metrics = {"pas": 0.5, "pdp": 0.7, "nmse": 1.1, "score": 0.59}
        self.assertIs(evaluation_metrics(metrics), metrics)
        self.assertEqual(evaluation_metrics({"metrics": metrics}), metrics)

    def test_router_uniform_floor_prevents_top1_collapse(self) -> None:
        router = ObservationRouter(
            context_channels=4,
            router_width=4,
            pair_width=4,
            top_k=4,
            dropout=0.0,
            temperature=2.0,
            uniform_mix=0.1,
            route_dropout=0.0,
        )
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
        route = router(
            observed_context=torch.zeros(4, 4),
            target_context=torch.zeros(2, 4),
            observed_relative_xy=torch.tensor(
                [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            ),
            target_relative_xy=torch.zeros(2, 2),
        )
        self.assertTrue(torch.allclose(route["weights"], torch.full((2, 4), 0.25)))
        self.assertTrue(
            torch.allclose(route["effective_neighbors"], torch.full((2,), 4.0))
        )

    def test_zero_residual_starts_from_transport_base(self) -> None:
        field = GeometryWarpedLatentField(
            latent_shape=(2, 2, 2, 2),
            observed_context_channels=4,
            target_context_channels=4,
            pair_channels=4,
            cell_count=1,
            token_channels=4,
            attention_heads=1,
            attention_chunk_size=4,
            refinement_blocks=0,
            axial_blocks=0,
            maximum_warp=(1.0, 1.0, 1.0),
            dropout=0.0,
            gradient_checkpointing=False,
            maximum_residual=0.75,
            route_bias_scale=0.1,
        )
        with torch.no_grad():
            for parameter in field.warp.parameters():
                parameter.zero_()
        observed = torch.stack(
            [torch.zeros(2, 2, 2, 2), torch.full((2, 2, 2, 2), 4.0)]
        )
        route = {
            "indices": torch.tensor([[0, 1]]),
            "weights": torch.tensor([[0.25, 0.75]]),
            "pair": torch.zeros(1, 2, 4),
        }
        prediction, _, _, base, residual, _ = field(
            observed,
            target_context=torch.zeros(1, 4),
            route=route,
            cell_id=0,
        )
        expected = torch.full((1, 2, 2, 2, 2), 3.0)
        self.assertTrue(torch.allclose(base, expected))
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        self.assertTrue(torch.allclose(prediction, expected))

    def test_formal_config_uses_external_ae_and_v3_losses(self) -> None:
        config = json.loads(
            (PROJECT_DIR / "configs" / "fold0_5090.json").read_text(encoding="utf-8")
        )
        context = config["context"]
        self.assertEqual(context["architecture"], "transport_residual_context_v3")
        self.assertEqual(context["router_minimum_effective_neighbors"], 8.0)
        self.assertIn("base_spectrum_latent", context["loss_weights"])
        self.assertIn("router_diversity", context["loss_weights"])
        self.assertNotIn("warp_magnitude", context["loss_weights"])
        self.assertTrue(
            config["encoding"]["autoencoder_checkpoint"].startswith(
                "../schemeC_full_resolution_context/"
            )
        )


if __name__ == "__main__":
    unittest.main()
