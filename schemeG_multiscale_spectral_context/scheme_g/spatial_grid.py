from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    minimum_x: float
    minimum_y: float
    resolution: float
    height: int
    width: int

    @classmethod
    def from_dict(cls, value: dict) -> "GridSpec":
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def maximum_x(self) -> float:
        return self.minimum_x + self.width * self.resolution

    @property
    def maximum_y(self) -> float:
        return self.minimum_y + self.height * self.resolution

    def indices(
        self, xy: np.ndarray, clip: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=np.float64)
        columns = np.floor((points[:, 0] - self.minimum_x) / self.resolution).astype(
            np.int64
        )
        rows = np.floor((points[:, 1] - self.minimum_y) / self.resolution).astype(
            np.int64
        )
        if clip:
            rows = np.clip(rows, 0, self.height - 1)
            columns = np.clip(columns, 0, self.width - 1)
        elif np.any(
            (rows < 0) | (rows >= self.height) | (columns < 0) | (columns >= self.width)
        ):
            raise ValueError("One or more points lie outside the grid")
        return rows, columns

    def center_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x = (
            self.minimum_x
            + (np.arange(self.width, dtype=np.float32) + 0.5) * self.resolution
        )
        y = (
            self.minimum_y
            + (np.arange(self.height, dtype=np.float32) + 0.5) * self.resolution
        )
        return np.meshgrid(x, y)

    def offsets(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        rows, columns = self.indices(points, clip=True)
        centers_x = (
            self.minimum_x + (columns.astype(np.float64) + 0.5) * self.resolution
        )
        centers_y = self.minimum_y + (rows.astype(np.float64) + 0.5) * self.resolution
        scale = max(0.5 * self.resolution, 1e-6)
        return np.stack(
            [(points[:, 0] - centers_x) / scale, (points[:, 1] - centers_y) / scale],
            axis=1,
        ).astype(np.float32)

    def grid_sample_coordinates(self, xy: np.ndarray) -> np.ndarray:
        """Return align_corners=False coordinates in x/y order."""
        points = np.asarray(xy, dtype=np.float64)
        x = 2.0 * (points[:, 0] - self.minimum_x) / (self.width * self.resolution) - 1.0
        y = (
            2.0 * (points[:, 1] - self.minimum_y) / (self.height * self.resolution)
            - 1.0
        )
        return np.stack([x, y], axis=1).astype(np.float32)


def load_setup(data_root: str | Path) -> dict:
    setup_path = Path(data_root) / "Round2_Setup.json"
    with setup_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_ascii_ply_xyz(path: str | Path) -> np.ndarray:
    source = Path(path)
    with source.open("r", encoding="ascii") as handle:
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {source}")
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property ") and vertex_count is not None:
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {source}")
        try:
            indices = [properties.index(axis) for axis in ("x", "y", "z")]
        except ValueError as error:
            raise ValueError(f"PLY must contain x/y/z properties: {source}") from error
        values = np.loadtxt(handle, dtype=np.float32, max_rows=vertex_count)
    if values.ndim == 1:
        values = values[None, :]
    return values[:, indices]


def infer_two_cell_rule(train_positions: np.ndarray) -> dict:
    """Find the axis and largest empty interval that split the samples most evenly."""
    points = np.asarray(train_positions, dtype=np.float64)
    candidates: list[tuple[float, float, int, float, int, int]] = []
    for axis in (0, 1):
        order = np.sort(points[:, axis])
        gaps = np.diff(order)
        for index in np.argsort(gaps)[-32:]:
            left_count = int(index + 1)
            right_count = int(len(points) - left_count)
            balance = min(left_count, right_count) / max(left_count, right_count)
            score = float(gaps[index]) * balance
            threshold = float((order[index] + order[index + 1]) / 2.0)
            candidates.append(
                (score, float(gaps[index]), axis, threshold, left_count, right_count)
            )
    score, gap, axis, threshold, left_count, right_count = max(candidates)
    return {
        "axis": int(axis),
        "threshold": threshold,
        "gap": gap,
        "score": score,
        "lower_cell": 0,
        "upper_cell": 1,
        "counts": [left_count, right_count],
    }


def assign_cells(positions: np.ndarray, rule: dict) -> np.ndarray:
    axis = int(rule["axis"])
    threshold = float(rule["threshold"])
    lower = int(rule.get("lower_cell", 0))
    upper = int(rule.get("upper_cell", 1))
    return np.where(np.asarray(positions)[:, axis] <= threshold, lower, upper).astype(
        np.int64
    )


def wrap_degrees(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def infer_boresight(positions: np.ndarray, base_station: np.ndarray) -> float:
    delta = np.asarray(positions)[:, :2] - np.asarray(base_station)[:2]
    angles = np.mod(np.degrees(np.arctan2(delta[:, 1], delta[:, 0])), 360.0)
    ordered = np.sort(angles)
    wrapped = np.concatenate([ordered, ordered[:1] + 360.0])
    largest_gap_index = int(np.argmax(np.diff(wrapped)))
    arc_start = wrapped[largest_gap_index + 1]
    arc_stop = wrapped[largest_gap_index] + 360.0
    return float(wrap_degrees((arc_start + arc_stop) / 2.0))


def make_grid_spec(positions: np.ndarray, resolution: float, margin: float) -> GridSpec:
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    minimum = np.floor((xy.min(axis=0) - margin) / resolution) * resolution
    maximum = np.ceil((xy.max(axis=0) + margin) / resolution) * resolution
    width, height = np.maximum(
        1, np.ceil((maximum - minimum) / resolution).astype(np.int64)
    )
    return GridSpec(
        minimum_x=float(minimum[0]),
        minimum_y=float(minimum[1]),
        resolution=float(resolution),
        height=int(height),
        width=int(width),
    )


def build_geometry_maps(
    spec: GridSpec,
    base_station: np.ndarray,
    boresight_degrees: float,
    sector_half_angle_degrees: float,
    maximum_distance: float,
    cell_id: int,
    cell_count: int,
) -> dict[str, np.ndarray]:
    x, y = spec.center_mesh()
    dx = x - float(base_station[0])
    dy = y - float(base_station[1])
    distance = np.sqrt(dx * dx + dy * dy)
    absolute_angle = np.degrees(np.arctan2(dy, dx))
    relative_angle = wrap_degrees(absolute_angle - boresight_degrees)
    valid = (np.abs(relative_angle) <= sector_half_angle_degrees) & (
        distance <= maximum_distance
    )
    identity = np.zeros((cell_count, spec.height, spec.width), dtype=np.float32)
    identity[cell_id] = 1.0
    return {
        "distance": np.clip(distance / maximum_distance, 0.0, 1.5).astype(np.float32)[
            None
        ],
        "relative_angle": np.clip(
            relative_angle / max(sector_half_angle_degrees, 1e-6), -2.0, 2.0
        ).astype(np.float32)[None],
        "valid": valid.astype(np.float32)[None],
        "identity": identity,
    }


def build_bev_features(points_xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Build log-density, maximum height, and four height-bin occupancy channels."""
    points = np.asarray(points_xyz, dtype=np.float32)
    columns = np.floor((points[:, 0] - spec.minimum_x) / spec.resolution).astype(
        np.int64
    )
    rows = np.floor((points[:, 1] - spec.minimum_y) / spec.resolution).astype(np.int64)
    inside = (
        (rows >= 0) & (rows < spec.height) & (columns >= 0) & (columns < spec.width)
    )
    rows, columns, heights = rows[inside], columns[inside], points[inside, 2]
    flat = rows * spec.width + columns
    size = spec.height * spec.width
    count = np.bincount(flat, minlength=size).astype(np.float32)
    point_cloud_minimum = float(points[:, 2].min())
    point_cloud_span = max(float(points[:, 2].max() - point_cloud_minimum), 1.0)
    maximum_height = np.full(size, point_cloud_minimum, dtype=np.float32)
    np.maximum.at(maximum_height, flat, heights)
    density = np.log1p(count)
    density /= max(float(density.max()), 1.0)
    maximum_height = (maximum_height - point_cloud_minimum) / point_cloud_span
    height_bins: list[np.ndarray] = []
    for lower, upper in ((-np.inf, 3.0), (3.0, 10.0), (10.0, 25.0), (25.0, np.inf)):
        occupied = np.zeros(size, dtype=np.float32)
        selected = (heights >= lower) & (heights < upper)
        np.maximum.at(occupied, flat[selected], 1.0)
        height_bins.append(occupied)
    features = np.stack([density, maximum_height, *height_bins], axis=0)
    return features.reshape(6, spec.height, spec.width).astype(np.float32)


def test_like_validation_masks(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    fold_count: int,
    tile_meters: float,
    hole_meters: float,
) -> np.ndarray:
    """Build periodic square holdouts whose support distances resemble the public test geometry."""
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    if not 0.0 < hole_meters < tile_meters:
        raise ValueError("validation hole must be positive and smaller than its tile")
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    masks = np.zeros((fold_count, len(xy)), dtype=bool)
    for fold in range(fold_count):
        phase_x = fold * tile_meters / fold_count
        phase_y = ((2 * fold) % fold_count) * tile_meters / fold_count
        local_x = np.mod(xy[:, 0] - phase_x, tile_meters)
        local_y = np.mod(xy[:, 1] - phase_y, tile_meters)
        masks[fold] = (local_x < hole_meters) & (local_y < hole_meters)
        for cell_id in np.unique(cell_ids):
            cell_mask = cell_ids == cell_id
            selected = masks[fold] & cell_mask
            if not np.any(selected) or np.all(selected == cell_mask):
                raise ValueError(f"Fold {fold} does not split cell {int(cell_id)}")
    return masks


def validation_support_summary(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    validation_masks: np.ndarray,
) -> list[dict]:
    output: list[dict] = []
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    for fold, validation in enumerate(validation_masks):
        distances: list[np.ndarray] = []
        cell_counts: list[int] = []
        for cell_id in np.unique(cell_ids):
            query = xy[validation & (cell_ids == cell_id)]
            reference = xy[~validation & (cell_ids == cell_id)]
            square = ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            distances.append(np.sqrt(square.min(axis=1)))
            cell_counts.append(int(len(query)))
        values = np.concatenate(distances)
        output.append(
            {
                "fold": fold,
                "samples": int(validation.sum()),
                "cell_counts": cell_counts,
                "minimum": float(values.min()),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "maximum": float(values.max()),
            }
        )
    return output


def grid_collision_statistics(
    rows: np.ndarray, columns: np.ndarray, cell_ids: np.ndarray
) -> list[dict]:
    output = []
    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        coordinates = np.stack([rows[mask], columns[mask]], axis=1)
        _, counts = np.unique(coordinates, axis=0, return_counts=True)
        output.append(
            {
                "cell_id": int(cell_id),
                "sample_count": int(mask.sum()),
                "occupied_cells": int(len(counts)),
                "collision_samples": int((counts[counts > 1] - 1).sum()),
                "maximum_samples_per_cell": int(counts.max(initial=0)),
            }
        )
    return output


def nearest_neighbor_summary(positions: np.ndarray, cell_ids: np.ndarray) -> dict:
    distances: list[np.ndarray] = []
    for cell_id in np.unique(cell_ids):
        points = np.asarray(positions[cell_ids == cell_id, :2], dtype=np.float64)
        square = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(square, np.inf)
        distances.append(np.sqrt(square.min(axis=1)))
    values = np.concatenate(distances)
    return {
        "minimum": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "maximum": float(values.max()),
    }


def test_support_summary(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    component_link_meters: float,
) -> dict:
    nearest_distances: list[np.ndarray] = []
    components_by_cell: list[dict] = []
    for cell_id in np.unique(train_cells):
        train_xy = np.asarray(
            train_positions[train_cells == cell_id, :2], dtype=np.float64
        )
        test_xy = np.asarray(
            test_positions[test_cells == cell_id, :2], dtype=np.float64
        )
        cross_square = ((test_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2)
        nearest_distances.append(np.sqrt(cross_square.min(axis=1)))

        pair_square = ((test_xy[:, None, :] - test_xy[None, :, :]) ** 2).sum(axis=2)
        adjacent = pair_square <= component_link_meters**2
        visited = np.zeros(len(test_xy), dtype=bool)
        components: list[dict] = []
        for start in range(len(test_xy)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            members: list[int] = []
            while stack:
                current = stack.pop()
                members.append(current)
                neighbors = np.flatnonzero(adjacent[current] & ~visited)
                visited[neighbors] = True
                stack.extend(neighbors.tolist())
            points = test_xy[members]
            span = points.max(axis=0) - points.min(axis=0)
            components.append(
                {
                    "sample_count": len(members),
                    "width_meters": float(span[0]),
                    "height_meters": float(span[1]),
                    "minimum_xy": points.min(axis=0).tolist(),
                    "maximum_xy": points.max(axis=0).tolist(),
                }
            )
        components.sort(key=lambda value: value["sample_count"], reverse=True)
        components_by_cell.append(
            {
                "cell_id": int(cell_id),
                "component_count": len(components),
                "largest_components": components[:10],
            }
        )
    nearest = np.concatenate(nearest_distances)
    return {
        "nearest_same_cell_train_meters": {
            "minimum": float(nearest.min()),
            "p25": float(np.percentile(nearest, 25)),
            "median": float(np.median(nearest)),
            "p75": float(np.percentile(nearest, 75)),
            "maximum": float(nearest.max()),
        },
        "component_link_meters": float(component_link_meters),
        "components_by_cell": components_by_cell,
    }
