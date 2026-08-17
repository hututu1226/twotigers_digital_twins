from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_manifest, load_metadata
from .spatial_grid import GridSpec, wrap_degrees


class ContextRepository:
    """Hold normalized structured latents and dual-resolution maps for one split."""

    query_numeric_channels = 9

    def __init__(self, config: dict, observed_indices: np.ndarray) -> None:
        self.config = config
        self.manifest = load_manifest(config)
        self.metadata = load_metadata(config)
        with np.load(config["encoding"]["output_path"]) as source:
            self.encoded = {key: source[key] for key in source.files}
        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        available = self.encoded.get(
            "available", np.ones(len(self.metadata["train_cells"]), dtype=bool)
        )
        self.observed_indices = self.observed_indices[available[self.observed_indices]]
        self.cell_count = int(self.manifest["setup"]["Q"])
        self.base_stations = np.asarray(self.manifest["setup"]["X"], dtype=np.float32)
        self.maximum_distance = float(config["preprocessing"]["maximum_distance"])
        self.context_specs: list[GridSpec] = []
        self.environment_specs: list[GridSpec] = []
        self.context_static: list[np.ndarray] = []
        self.environment_bev: list[np.ndarray] = []
        self.boresights: list[float] = []
        self._load_static_maps()
        self.spectrum_z = self._normalize_latent("spectrum")
        self.phase_z = self._normalize_latent("phase")
        self.spectrum_shape = self._latent_shape("spectrum", self.spectrum_z.shape[1])
        self.phase_shape = self._latent_shape("phase", self.phase_z.shape[1])
        self.power_z = self._normalize_power()
        outage = self.metadata["outage"]
        self.spectrum_z[outage] = 0.0
        self.phase_z[outage] = 0.0
        self.power_z[outage] = 0.0
        self.indices_by_cell = [
            self.observed_indices[
                self.metadata["train_cells"][self.observed_indices] == cell_id
            ]
            for cell_id in range(self.cell_count)
        ]

    @property
    def spectrum_latent_dim(self) -> int:
        return int(self.spectrum_z.shape[1])

    @property
    def phase_latent_dim(self) -> int:
        return int(self.phase_z.shape[1])

    @property
    def point_feature_channels(self) -> int:
        return 4

    @property
    def static_context_channels(self) -> int:
        return int(self.context_static[0].shape[0])

    def _load_static_maps(self) -> None:
        artifact_dir = Path(self.config["preprocessing"]["artifact_dir"])
        for entry in self.manifest["grids"]:
            self.context_specs.append(GridSpec.from_dict(entry["context_spec"]))
            self.environment_specs.append(GridSpec.from_dict(entry["environment_spec"]))
            self.boresights.append(float(entry["boresight_degrees"]))
            with np.load(artifact_dir / entry["context_static_path"]) as source:
                context = {key: source[key].astype(np.float32) for key in source.files}
            with np.load(artifact_dir / entry["environment_static_path"]) as source:
                environment = {key: source[key].astype(np.float32) for key in source.files}
            self.context_static.append(
                np.concatenate(
                    [
                        context["bev"],
                        context["valid"],
                        context["distance"],
                        context["relative_angle"],
                        context["identity"],
                    ],
                    axis=0,
                )
            )
            self.environment_bev.append(environment["bev"])

    def _normalize_latent(self, prefix: str) -> np.ndarray:
        values = self.encoded[f"{prefix}_latent"].astype(np.float32)
        mean = self.encoded[f"{prefix}_mean"].astype(np.float32)
        standard_deviation = self.encoded[f"{prefix}_std"].astype(np.float32)
        return ((values - mean) / standard_deviation).astype(np.float32)

    def _latent_shape(self, prefix: str, expected_elements: int) -> tuple[int, int, int, int]:
        key = f"{prefix}_shape"
        if key not in self.encoded:
            raise ValueError(
                f"Encoded file has no {key}; rerun scripts/encode_latents.py with Scheme C"
            )
        shape = tuple(int(value) for value in self.encoded[key].tolist())
        if len(shape) != 4 or int(np.prod(shape)) != int(expected_elements):
            raise ValueError(
                f"Invalid {key}={shape} for {expected_elements} encoded elements"
            )
        return shape

    def _normalize_power(self) -> np.ndarray:
        cells = self.metadata["train_cells"]
        values = (
            self.metadata["log_power"] - self.encoded["power_mean"][cells]
        ) / self.encoded["power_std"][cells]
        return values.astype(np.float32)

    def context_indices(self, cell_id: int, hidden_indices: np.ndarray | None = None) -> np.ndarray:
        indices = self.indices_by_cell[int(cell_id)]
        if hidden_indices is None or not len(hidden_indices):
            return indices
        return indices[~np.isin(indices, np.asarray(hidden_indices, dtype=np.int64))]

    def point_features(self, indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64)
        return np.concatenate(
            [
                self.power_z[selected, None],
                self.metadata["context_offsets"][selected],
                self.metadata["outage"][selected, None].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def structured_latents(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        return {
            "spectrum": self.spectrum_z[selected].reshape(-1, *self.spectrum_shape),
            "phase": self.phase_z[selected].reshape(-1, *self.phase_shape),
            "power": self.power_z[selected],
            "outage": self.metadata["outage"][selected].astype(np.float32),
        }

    def flat_indices(self, cell_id: int, indices: np.ndarray) -> np.ndarray:
        spec = self.context_specs[int(cell_id)]
        selected = np.asarray(indices, dtype=np.int64)
        return (
            self.metadata["context_rows"][selected].astype(np.int64) * spec.width
            + self.metadata["context_columns"][selected].astype(np.int64)
        )

    def target_values(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        return {
            "spectrum": self.spectrum_z[selected],
            "phase": self.phase_z[selected],
            "power": self.power_z[selected],
            "outage": self.metadata["outage"][selected].astype(np.float32),
        }

    def query_features(
        self,
        cell_id: int,
        indices: np.ndarray | None = None,
        test: bool = False,
        include_corridor: bool = True,
    ) -> dict[str, np.ndarray]:
        cell_id = int(cell_id)
        if test:
            all_positions = self.metadata["test_positions"]
            all_cells = self.metadata["test_cells"]
            all_offsets = self.metadata["test_context_offsets"]
        else:
            all_positions = self.metadata["train_positions"]
            all_cells = self.metadata["train_cells"]
            all_offsets = self.metadata["context_offsets"]
        selected = (
            np.flatnonzero(all_cells == cell_id).astype(np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if np.any(all_cells[selected] != cell_id):
            raise ValueError("Every query index must belong to the requested cell")
        positions = all_positions[selected].astype(np.float32)
        relative = positions - self.base_stations[cell_id]
        distance = np.linalg.norm(relative[:, :2], axis=1)
        absolute_angle = np.degrees(np.arctan2(relative[:, 1], relative[:, 0]))
        relative_angle = np.radians(
            wrap_degrees(absolute_angle - self.boresights[cell_id])
        )
        numeric = np.stack(
            [
                relative[:, 0] / self.maximum_distance,
                relative[:, 1] / self.maximum_distance,
                relative[:, 2] / 50.0,
                distance / self.maximum_distance,
                np.sin(relative_angle),
                np.cos(relative_angle),
                all_offsets[selected, 0],
                all_offsets[selected, 1],
                positions[:, 2] / 50.0,
            ],
            axis=1,
        ).astype(np.float32)
        result = {
            "indices": selected,
            "context_coordinates": self.context_specs[cell_id].grid_sample_coordinates(
                positions[:, :2]
            ),
            "environment_coordinates": self.environment_specs[
                cell_id
            ].grid_sample_coordinates(positions[:, :2]),
            "numeric": numeric,
            "relative_xy": (relative[:, :2] / self.maximum_distance).astype(np.float32),
        }
        if include_corridor:
            samples = max(2, int(self.config["context"].get("corridor_samples", 16)))
            fractions = np.linspace(0.0, 1.0, samples, dtype=np.float32)
            station_xy = self.base_stations[cell_id, :2]
            corridor_xy = (
                station_xy[None, None, :] * (1.0 - fractions[None, :, None])
                + positions[:, None, :2] * fractions[None, :, None]
            )
            result["corridor_coordinates"] = self.environment_specs[
                cell_id
            ].grid_sample_coordinates(corridor_xy.reshape(-1, 2)).reshape(
                len(selected), samples, 2
            )
        return result

    def sample_hole(
        self,
        rng: np.random.Generator,
        cell_id: int,
        minimum_meters: float,
        maximum_meters: float,
        minimum_targets: int,
        maximum_targets: int,
        outage_anchor_probability: float = 0.25,
    ) -> np.ndarray:
        candidates = self.indices_by_cell[int(cell_id)]
        positions = self.metadata["train_positions"][candidates, :2]
        selected = np.empty(0, dtype=np.int64)
        anchor = int(candidates[0])
        outage_candidates = candidates[self.metadata["outage"][candidates]]
        for _ in range(64):
            anchor_pool = (
                outage_candidates
                if len(outage_candidates) and rng.random() < outage_anchor_probability
                else candidates
            )
            anchor = int(anchor_pool[rng.integers(len(anchor_pool))])
            center = self.metadata["train_positions"][anchor, :2]
            width = rng.uniform(minimum_meters, maximum_meters)
            height = rng.uniform(minimum_meters, maximum_meters)
            delta = positions - center
            shape_kind = int(rng.integers(4))
            if shape_kind == 0:
                mask = (
                    (np.abs(delta[:, 0]) <= width / 2.0)
                    & (np.abs(delta[:, 1]) <= height / 2.0)
                )
            elif shape_kind == 1:
                normalized = (delta[:, 0] / (width / 2.0)) ** 2 + (
                    delta[:, 1] / (height / 2.0)
                ) ** 2
                mask = normalized <= 1.0
            elif shape_kind == 2:
                angle = rng.uniform(0.0, np.pi)
                along = delta[:, 0] * np.cos(angle) + delta[:, 1] * np.sin(angle)
                across = -delta[:, 0] * np.sin(angle) + delta[:, 1] * np.cos(angle)
                mask = (np.abs(along) <= width) & (np.abs(across) <= height / 4.0)
            else:
                second_center = center + rng.normal(0.0, maximum_meters / 4.0, size=2)
                second_delta = positions - second_center
                first = (delta[:, 0] / (width / 2.0)) ** 2 + (
                    delta[:, 1] / (height / 2.0)
                ) ** 2 <= 1.0
                second = (second_delta[:, 0] / (height / 2.0)) ** 2 + (
                    second_delta[:, 1] / (width / 2.0)
                ) ** 2 <= 1.0
                mask = first | second
            selected = candidates[mask]
            if len(selected) >= minimum_targets:
                break
        if not len(selected):
            selected = np.asarray([anchor], dtype=np.int64)
        if len(selected) > maximum_targets:
            selected = rng.choice(selected, size=maximum_targets, replace=False)
        return np.sort(selected.astype(np.int64))
