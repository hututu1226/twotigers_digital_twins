from __future__ import annotations

from copy import deepcopy


def adaptive_hybrid_config(
    base: dict,
    *,
    adaptive_prior: str,
    initial_checkpoint: str,
    output_dir: str,
) -> dict:
    """Change only the Teacher distribution and fine-tuning runtime controls."""
    config = deepcopy(base)
    config["spectral_teacher"]["oof_output_path"] = str(adaptive_prior)
    hybrid = config["hybrid"]
    hybrid["initial_checkpoint"] = str(initial_checkpoint)
    hybrid["output_dir"] = str(output_dir)
    hybrid["learning_rate"] = 5e-5
    hybrid["minimum_learning_rate"] = 2e-6
    hybrid["epochs"] = 420
    hybrid["early_stopping_patience"] = 80
    hybrid["maximum_training_hours"] = 1.25
    hybrid["minimum_delta"] = 1e-4
    hybrid["resume"] = False
    return config
