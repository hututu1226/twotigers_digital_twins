from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_completion import verify_ae  # noqa: E402


class AeAutomationVerificationTests(unittest.TestCase):
    def make_project(
        self, root: Path, gate_status: str, context_allowed: bool
    ) -> Path:
        stage = root / "artifacts" / "fold0" / "autoencoder"
        stage.mkdir(parents=True)
        (stage / "best.pt").write_bytes(b"checkpoint" * 256)
        (stage / "summary.json").write_text(
            json.dumps(
                {
                    "architecture": "factorized_residual_v4",
                    "spectrum_latent_dim": 24576,
                    "phase_latent_dim": 6144,
                }
            ),
            encoding="utf-8",
        )
        (stage / "evaluation.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "pas": 0.8,
                        "pdp": 0.8,
                        "nmse": 0.3,
                        "score": 0.8,
                        "samples": 530,
                    }
                }
            ),
            encoding="utf-8",
        )
        (stage / "ablation.json").write_text(
            json.dumps({"detail_gain": 0.12, "shuffle_drop": 0.08}),
            encoding="utf-8",
        )
        (stage / "quality_gate.json").write_text(
            json.dumps(
                {
                    "status": gate_status,
                    "measurements": {
                        "score": 0.8,
                        "detail_gain": 0.12,
                        "shuffle_drop": 0.08,
                    },
                    "context_training_allowed": context_allowed,
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_completed_pass_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_ae(
                self.make_project(Path(directory), "PASS", context_allowed=True)
            )
        self.assertEqual(result["quality_gate"]["status"], "PASS")
        self.assertEqual(result["total_latent_elements"], 30720)
        self.assertGreater(result["checkpoint"]["bytes"], 1024)

    def test_completed_gate_failure_is_still_valid_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_ae(
                self.make_project(Path(directory), "FAIL", context_allowed=False)
            )
        self.assertEqual(result["quality_gate"]["status"], "FAIL")

    def test_inconsistent_gate_continuation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = self.make_project(
                Path(directory), "FAIL", context_allowed=True
            )
            with self.assertRaisesRegex(ValueError, "continuation flag"):
                verify_ae(project)


if __name__ == "__main__":
    unittest.main()
