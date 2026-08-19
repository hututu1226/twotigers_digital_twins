from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _boosting_models(seed: int, positive_weight: float) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = [
        (
            "svc",
            make_pipeline(
                StandardScaler(),
                SVC(
                    C=3.0,
                    gamma="scale",
                    probability=True,
                    class_weight={0: 1.0, 1: float(positive_weight)},
                    random_state=int(seed),
                ),
            ),
        )
    ]
    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    scale_pos_weight=float(positive_weight),
                    random_state=int(seed),
                    n_jobs=4,
                ),
            )
        )
    except ImportError:
        models.append(
            (
                "hist_gradient_boosting_fallback",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=int(seed),
                ),
            )
        )
    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=300,
                    num_leaves=31,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight={0: 1.0, 1: float(positive_weight)},
                    random_state=int(seed),
                    n_jobs=4,
                    verbosity=-1,
                ),
            )
        )
    except ImportError:
        models.append(
            (
                "random_forest_fallback",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight={0: 1.0, 1: float(positive_weight)},
                    random_state=int(seed),
                    n_jobs=4,
                ),
            )
        )
    return models


@dataclass
class OutageEnsemble:
    seed: int = 2026
    positive_weight: float = 4.0
    false_kill_cost: float = 0.56
    models: list[tuple[str, object]] | None = None
    threshold: float = 0.5

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "OutageEnsemble":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if len(np.unique(labels)) < 2:
            raise ValueError("Outage training requires both zero and nonzero channels")
        self.models = _boosting_models(self.seed, self.positive_weight)
        for _, model in self.models:
            model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("OutageEnsemble has not been fitted")
        predictions = [
            np.asarray(model.predict_proba(features), dtype=np.float64)[:, 1]
            for _, model in self.models
        ]
        return np.mean(predictions, axis=0).astype(np.float32)

    def calibrate_threshold(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.bool_)
        best_threshold = 0.5
        best_cost = float("inf")
        for threshold in np.linspace(0.05, 0.995, 190):
            prediction = probabilities >= threshold
            false_kill = np.mean(prediction & ~labels)
            missed_outage = np.mean(~prediction & labels)
            cost = self.false_kill_cost * false_kill + (1.0 - self.false_kill_cost) * missed_outage
            if cost < best_cost:
                best_cost = float(cost)
                best_threshold = float(threshold)
        self.threshold = best_threshold
        return best_threshold

    def state_dict(self) -> dict:
        if not self.models:
            raise RuntimeError("OutageEnsemble has not been fitted")
        return {
            "seed": int(self.seed),
            "positive_weight": float(self.positive_weight),
            "false_kill_cost": float(self.false_kill_cost),
            "threshold": float(self.threshold),
            "models": self.models,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "OutageEnsemble":
        instance = cls(
            seed=int(state["seed"]),
            positive_weight=float(state["positive_weight"]),
            false_kill_cost=float(state["false_kill_cost"]),
        )
        instance.threshold = float(state["threshold"])
        instance.models = state["models"]
        return instance

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "OutageEnsemble":
        with Path(path).open("rb") as handle:
            return cls.from_state_dict(pickle.load(handle))


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.bool_)
    prediction = probabilities >= float(threshold)
    true_positive = int(np.sum(prediction & labels))
    false_positive = int(np.sum(prediction & ~labels))
    false_negative = int(np.sum(~prediction & labels))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(prediction == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "predicted_outages": int(prediction.sum()),
        "true_outages": int(labels.sum()),
    }
