from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_manifest, load_metadata
from .spatial_grid import GridSpec, wrap_degrees


def compact_spectral_prior(
    prior: dict[str, np.ndarray],
    *,
    proxy_count: int = 24,
    pdp_bins: int = 16,
) -> np.ndarray:
    """Keep PAS/PDP structure while avoiding a 1,576-value query side input."""
    pas = np.asarray(prior["pas_log"], dtype=np.float32)
    groups = int(proxy_count) + 1
    if pas.ndim != 2 or pas.shape[1] % groups:
        raise ValueError(
            f"Invalid PAS prior shape {pas.shape} for {proxy_count} proxies"
        )
    beam_bins = pas.shape[1] // groups
    pas = pas.reshape(len(pas), groups, beam_bins)
    proxy = pas[:, :proxy_count]
    pas_features = np.concatenate(
        [proxy.mean(axis=1), proxy.std(axis=1), pas[:, -1]], axis=1
    )

    pdp = np.asarray(prior["pdp_log"], dtype=np.float32)
    ue = np.asarray(prior["ue_log_energy"], dtype=np.float32)
    if pdp.ndim != 2 or ue.ndim != 2 or pdp.shape[1] % ue.shape[1]:
        raise ValueError(f"Incompatible PDP/UE prior shapes {pdp.shape} and {ue.shape}")
    users = ue.shape[1]
    delay = pdp.shape[1] // users
    bins = min(max(1, int(pdp_bins)), delay)
    edges = np.linspace(0, delay, bins + 1, dtype=np.int64)
    pdp = pdp.reshape(len(pdp), users, delay)
    pdp_features = np.stack(
        [
            pdp[:, :, edges[index] : edges[index + 1]].mean(axis=2)
            for index in range(bins)
        ],
        axis=2,
    ).reshape(len(pdp), -1)
    available = np.asarray(
        prior.get("available", np.ones(len(pas), dtype=np.float32)), dtype=np.float32
    )
    return np.concatenate(
        [
            pas_features,
            pdp_features,
            ue,
            np.asarray(prior["log_power"], dtype=np.float32)[:, None],
            np.asarray(prior["uncertainty"], dtype=np.float32)[:, None],
            np.asarray(prior["outage_probability"], dtype=np.float32)[:, None],
            available[:, None],
        ],
        axis=1,
    ).astype(np.float32)


@dataclass(frozen=True)
class SpatialMaskSample:
    """A training query subset and every observation hidden around it."""

    targets: np.ndarray
    hidden: np.ndarray
    pattern: str
    guard_meters: float


class ContextRepository:
    """Hold normalized structured latents and dual-resolution maps for one split."""

    base_query_numeric_channels = 9

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
        self.geometry_channels = (
            int(self.metadata["train_geometry_features"].shape[1])
            if bool(config["context"].get("use_rf_geometry", True))
            and "train_geometry_features" in self.metadata
            else 0
        )
        self.train_prior_raw: dict[str, np.ndarray] | None = None
        self.test_prior_raw: dict[str, np.ndarray] | None = None
        self.train_prior, self.test_prior = self._load_spectral_prior_features()
        self.spectral_prior_channels = int(self.train_prior.shape[1])
        self.query_numeric_channels = (
            self.base_query_numeric_channels
            + self.geometry_channels
            + self.spectral_prior_channels
        )
        self.context_specs: list[GridSpec] = []
        self.environment_specs: list[GridSpec] = []
        self.context_static: list[np.ndarray] = []
        self.environment_bev: list[np.ndarray] = []
        self.boresights: list[float] = []
        self._load_static_maps()
        self.spectrum_z = self._normalize_latent("spectrum")
        self.phase_z = self._normalize_latent("phase")
        self.encoded.pop("spectrum_latent", None)
        self.encoded.pop("phase_latent", None)
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
        self.test_component_templates = self._build_test_component_templates()

    def _load_spectral_prior_features(self) -> tuple[np.ndarray, np.ndarray]:
        section = self.config["context"].get("spectral_prior", {})
        enabled = bool(section.get("enabled", False))
        train_count = len(self.metadata["train_positions"])
        test_count = len(self.metadata["test_positions"])
        if not enabled:
            return (
                np.empty((train_count, 0), dtype=np.float32),
                np.empty((test_count, 0), dtype=np.float32),
            )
        paths = (Path(section["oof_path"]), Path(section["test_path"]))
        if not all(path.is_file() for path in paths):
            if bool(section.get("required", True)):
                missing = [str(path) for path in paths if not path.is_file()]
                raise FileNotFoundError(
                    f"Scheme E spectral priors are missing: {missing}"
                )
            channels = int(section.get("fallback_channels", 168))
            return (
                np.zeros((train_count, channels), dtype=np.float32),
                np.zeros((test_count, channels), dtype=np.float32),
            )
        loaded: list[np.ndarray] = []
        raw_priors: list[dict[str, np.ndarray]] = []
        for path, expected in zip(paths, (train_count, test_count), strict=True):
            with np.load(path) as source:
                prior = {name: source[name] for name in source.files}
            raw_priors.append(prior)
            values = compact_spectral_prior(
                prior,
                proxy_count=int(section.get("proxy_count", 24)),
                pdp_bins=int(section.get("pdp_bins", 16)),
            )
            if len(values) != expected:
                raise ValueError(
                    f"Spectral prior {path} has {len(values)} rows, expected {expected}"
                )
            loaded.append(values)
        train, test = loaded
        self.train_prior_raw, self.test_prior_raw = raw_priors
        cells = self.metadata["train_cells"]
        test_cells = self.metadata["test_cells"]
        normalized_train = np.empty_like(train)
        normalized_test = np.empty_like(test)
        for cell_id in range(self.cell_count):
            selected = cells == cell_id
            mean = train[selected].mean(axis=0, dtype=np.float64).astype(np.float32)
            std = np.maximum(
                train[selected].std(axis=0, dtype=np.float64), 1e-3
            ).astype(np.float32)
            normalized_train[selected] = np.clip((train[selected] - mean) / std, -8, 8)
            test_selected = test_cells == cell_id
            normalized_test[test_selected] = np.clip(
                (test[test_selected] - mean) / std, -8, 8
            )
        return normalized_train, normalized_test

    def spectral_prior_values(
        self, indices: np.ndarray, *, test: bool = False
    ) -> dict[str, np.ndarray] | None:
        source = self.test_prior_raw if test else self.train_prior_raw
        if source is None:
            return None
        selected = np.asarray(indices, dtype=np.int64)
        return {
            name: np.asarray(source[name][selected], dtype=np.float32)
            for name in ("pas_log", "pdp_log", "uncertainty")
        }

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
                environment = {
                    key: source[key].astype(np.float32) for key in source.files
                }
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
        if mean.ndim == 2:
            cells = self.metadata["train_cells"]
            normalized = np.empty_like(values)
            for cell_id in range(len(mean)):
                selected = cells == cell_id
                normalized[selected] = (
                    values[selected] - mean[cell_id]
                ) / standard_deviation[cell_id]
            return normalized
        return ((values - mean) / standard_deviation).astype(np.float32)

    def _build_test_component_templates(self) -> list[list[np.ndarray]]:
        link_meters = float(
            self.config["preprocessing"].get("test_component_link_meters", 6.0)
        )
        templates: list[list[np.ndarray]] = []
        for cell_id in range(self.cell_count):
            selected = np.flatnonzero(self.metadata["test_cells"] == cell_id).astype(
                np.int64
            )
            positions = self.metadata["test_positions"][selected, :2].astype(np.float32)
            if not len(positions):
                templates.append([])
                continue
            distance = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=2
            )
            adjacency = distance <= link_meters
            remaining = set(range(len(positions)))
            cell_templates: list[np.ndarray] = []
            while remaining:
                seed = remaining.pop()
                component = [seed]
                frontier = [seed]
                while frontier:
                    current = frontier.pop()
                    linked = np.flatnonzero(adjacency[current]).tolist()
                    unseen = [index for index in linked if index in remaining]
                    for index in unseen:
                        remaining.remove(index)
                    frontier.extend(unseen)
                    component.extend(unseen)
                points = positions[np.asarray(component, dtype=np.int64)]
                cell_templates.append((points - points.mean(axis=0)).astype(np.float32))
            templates.append(cell_templates)
        return templates

    def _latent_shape(
        self, prefix: str, expected_elements: int
    ) -> tuple[int, int, int, int]:
        key = f"{prefix}_shape"
        if key not in self.encoded:
            raise ValueError(
                f"Encoded file has no {key}; rerun scripts/encode_latents.py with Scheme G"
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

    def context_indices(
        self, cell_id: int, hidden_indices: np.ndarray | None = None
    ) -> np.ndarray:
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
        return self.metadata["context_rows"][selected].astype(
            np.int64
        ) * spec.width + self.metadata["context_columns"][selected].astype(np.int64)

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
            all_geometry = self.metadata.get("test_geometry_features")
            all_prior = self.test_prior
        else:
            all_positions = self.metadata["train_positions"]
            all_cells = self.metadata["train_cells"]
            all_offsets = self.metadata["context_offsets"]
            all_geometry = self.metadata.get("train_geometry_features")
            all_prior = self.train_prior
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
        additions = [numeric]
        if self.geometry_channels:
            if all_geometry is None:
                raise ValueError(
                    "RF geometry is enabled but metadata has no geometry features"
                )
            additions.append(np.asarray(all_geometry[selected], dtype=np.float32))
        if self.spectral_prior_channels:
            additions.append(np.asarray(all_prior[selected], dtype=np.float32))
        numeric = np.concatenate(additions, axis=1)
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
            result["corridor_coordinates"] = (
                self.environment_specs[cell_id]
                .grid_sample_coordinates(corridor_xy.reshape(-1, 2))
                .reshape(len(selected), samples, 2)
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
        return self.sample_spatial_mask(
            rng,
            cell_id,
            minimum_meters,
            maximum_meters,
            minimum_targets,
            maximum_targets,
            outage_anchor_probability,
            test_template_probability=0.0,
            template_radius_meters=3.0,
            guard_min_meters=0.0,
            guard_max_meters=0.0,
        ).targets

    def sample_spatial_mask(
        self,
        rng: np.random.Generator,
        cell_id: int,
        minimum_meters: float,
        maximum_meters: float,
        minimum_targets: int,
        maximum_targets: int,
        outage_anchor_probability: float = 0.25,
        test_template_probability: float = 0.65,
        template_radius_meters: float = 3.0,
        guard_min_meters: float = 3.5,
        guard_max_meters: float = 8.5,
    ) -> SpatialMaskSample:
        """Hide a complete test-like region while supervising a bounded target subset."""
        candidates = self.indices_by_cell[int(cell_id)]
        positions = self.metadata["train_positions"][candidates, :2]
        selected = np.empty(0, dtype=np.int64)
        anchor = int(candidates[0])
        outage_candidates = candidates[self.metadata["outage"][candidates]]
        pattern = "fallback"
        for _ in range(64):
            anchor_pool = (
                outage_candidates
                if len(outage_candidates) and rng.random() < outage_anchor_probability
                else candidates
            )
            anchor = int(anchor_pool[rng.integers(len(anchor_pool))])
            center = self.metadata["train_positions"][anchor, :2]
            templates = self.test_component_templates[int(cell_id)]
            use_template = bool(templates) and rng.random() < test_template_probability
            if use_template:
                template = templates[int(rng.integers(len(templates)))]
                angle = rng.uniform(0.0, 2.0 * np.pi)
                scale = rng.uniform(0.85, 1.15)
                rotation = np.asarray(
                    [
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)],
                    ],
                    dtype=np.float32,
                )
                transformed = (template @ rotation.T) * scale + center
                distance = np.linalg.norm(
                    positions[:, None, :] - transformed[None, :, :], axis=2
                )
                mask = distance.min(axis=1) <= template_radius_meters
                pattern = "test_component"
            else:
                width = rng.uniform(minimum_meters, maximum_meters)
                height = rng.uniform(minimum_meters, maximum_meters)
                delta = positions - center
                shape_kind = int(rng.integers(4))
                if shape_kind == 0:
                    mask = (np.abs(delta[:, 0]) <= width / 2.0) & (
                        np.abs(delta[:, 1]) <= height / 2.0
                    )
                    pattern = "rectangle"
                elif shape_kind == 1:
                    normalized = (delta[:, 0] / (width / 2.0)) ** 2 + (
                        delta[:, 1] / (height / 2.0)
                    ) ** 2
                    mask = normalized <= 1.0
                    pattern = "ellipse"
                elif shape_kind == 2:
                    angle = rng.uniform(0.0, np.pi)
                    along = delta[:, 0] * np.cos(angle) + delta[:, 1] * np.sin(angle)
                    across = -delta[:, 0] * np.sin(angle) + delta[:, 1] * np.cos(angle)
                    mask = (np.abs(along) <= width) & (np.abs(across) <= height / 4.0)
                    pattern = "corridor"
                else:
                    second_center = center + rng.normal(
                        0.0, maximum_meters / 4.0, size=2
                    )
                    second_delta = positions - second_center
                    first = (delta[:, 0] / (width / 2.0)) ** 2 + (
                        delta[:, 1] / (height / 2.0)
                    ) ** 2 <= 1.0
                    second = (second_delta[:, 0] / (height / 2.0)) ** 2 + (
                        second_delta[:, 1] / (width / 2.0)
                    ) ** 2 <= 1.0
                    mask = first | second
                    pattern = "compound"
            selected = candidates[mask]
            if len(selected) >= minimum_targets:
                break
        if not len(selected):
            selected = np.asarray([anchor], dtype=np.int64)
        hidden = selected.astype(np.int64, copy=True)
        targets = hidden
        if len(targets) > maximum_targets:
            targets = rng.choice(targets, size=maximum_targets, replace=False)
            if anchor in hidden and anchor not in targets:
                targets[0] = anchor
        guard_low = max(0.0, float(guard_min_meters))
        guard_high = max(guard_low, float(guard_max_meters))
        guard = float(rng.uniform(guard_low, guard_high)) if guard_high > 0 else 0.0
        if guard > 0.0 and len(targets):
            target_positions = self.metadata["train_positions"][targets, :2]
            nearest_target = np.linalg.norm(
                positions[:, None, :] - target_positions[None, :, :], axis=2
            ).min(axis=1)
            hidden = np.union1d(hidden, candidates[nearest_target < guard])
        return SpatialMaskSample(
            targets=np.sort(targets.astype(np.int64)),
            hidden=np.sort(hidden.astype(np.int64)),
            pattern=pattern,
            guard_meters=guard,
        )
