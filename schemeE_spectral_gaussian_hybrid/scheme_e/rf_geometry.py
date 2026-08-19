from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - only used by the minimal fallback environment
    cKDTree = None


LOCAL_RADII_METERS = (2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class GaussianField:
    centers: np.ndarray
    normals: np.ndarray
    tangent_scale: np.ndarray
    normal_scale: np.ndarray
    area: np.ndarray

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            centers=self.centers.astype(np.float32),
            normals=self.normals.astype(np.float32),
            tangent_scale=self.tangent_scale.astype(np.float32),
            normal_scale=self.normal_scale.astype(np.float32),
            area=self.area.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianField":
        with np.load(path) as source:
            return cls(
                centers=source["centers"].astype(np.float32),
                normals=source["normals"].astype(np.float32),
                tangent_scale=source["tangent_scale"].astype(np.float32),
                normal_scale=source["normal_scale"].astype(np.float32),
                area=source["area"].astype(np.float32),
            )


def read_ascii_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    with source.open("r", encoding="ascii") as handle:
        vertex_count: int | None = None
        face_count: int | None = None
        vertex_properties: list[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {source}")
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                continue
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
                continue
            if fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
                in_vertex = False
                continue
            if fields[:1] == ["property"] and in_vertex:
                vertex_properties.append(fields[-1])
            if fields[:1] == ["end_header"]:
                break
        if vertex_count is None or face_count is None:
            raise ValueError(f"PLY must contain vertex and face elements: {source}")
        try:
            xyz_columns = [vertex_properties.index(name) for name in ("x", "y", "z")]
        except ValueError as error:
            raise ValueError(f"PLY vertex properties must contain x/y/z: {source}") from error
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            values = handle.readline().split()
            vertices[index] = [float(values[column]) for column in xyz_columns]
        triangles: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            values = [int(value) for value in handle.readline().split()]
            if not values:
                continue
            count = values[0]
            indices = values[1 : count + 1]
            if count < 3:
                continue
            anchor = indices[0]
            triangles.extend((anchor, indices[i], indices[i + 1]) for i in range(1, count - 1))
    faces = np.asarray(triangles, dtype=np.int64)
    if not len(faces):
        raise ValueError(f"PLY has no triangulatable faces: {source}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"PLY face index is outside the vertex array: {source}")
    return vertices, faces


def _spread_bits_10(value: np.ndarray) -> np.ndarray:
    result = value.astype(np.uint64) & np.uint64(0x3FF)
    result = (result | (result << np.uint64(16))) & np.uint64(0x030000FF)
    result = (result | (result << np.uint64(8))) & np.uint64(0x0300F00F)
    result = (result | (result << np.uint64(4))) & np.uint64(0x030C30C3)
    result = (result | (result << np.uint64(2))) & np.uint64(0x09249249)
    return result


def _morton_order(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    span = np.maximum(np.ptp(points, axis=0), 1e-6)
    quantized = np.clip(np.rint((points - minimum) / span * 1023.0), 0, 1023).astype(np.uint64)
    keys = (
        _spread_bits_10(quantized[:, 0])
        | (_spread_bits_10(quantized[:, 1]) << np.uint64(1))
        | (_spread_bits_10(quantized[:, 2]) << np.uint64(2))
    )
    return np.argsort(keys, kind="stable")


def build_rf_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_count: int = 10_070,
) -> GaussianField:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-10
    if not np.any(valid):
        raise ValueError("PLY mesh contains no non-degenerate triangles")
    triangles = triangles[valid]
    cross = cross[valid]
    area = 0.5 * double_area[valid]
    centroids = triangles.mean(axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    order = _morton_order(centroids)
    group_count = min(int(target_count), len(order))
    groups = np.array_split(order, group_count)

    output_centers = np.empty((group_count, 3), dtype=np.float32)
    output_normals = np.empty((group_count, 3), dtype=np.float32)
    output_tangent = np.empty((group_count, 2), dtype=np.float32)
    output_normal_scale = np.empty(group_count, dtype=np.float32)
    output_area = np.empty(group_count, dtype=np.float32)
    for output_index, group in enumerate(groups):
        weights = area[group]
        total = max(float(weights.sum()), 1e-12)
        center = np.sum(centroids[group] * weights[:, None], axis=0) / total
        normal = np.sum(normals[group] * weights[:, None], axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        centered = triangles[group].reshape(-1, 3) - center
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
        tangent = np.sqrt(eigenvalues[-2:][::-1])
        output_centers[output_index] = center
        output_normals[output_index] = normal
        output_tangent[output_index] = np.maximum(tangent, 0.05)
        output_normal_scale[output_index] = max(float(np.sqrt(eigenvalues[0])), 0.03)
        output_area[output_index] = total
    return GaussianField(
        centers=output_centers,
        normals=output_normals,
        tangent_scale=output_tangent,
        normal_scale=output_normal_scale,
        area=output_area,
    )


def feature_names() -> list[str]:
    names = [
        "ue_x",
        "ue_y",
        "ue_z",
        "bs_x",
        "bs_y",
        "bs_z",
        "delta_x",
        "delta_y",
        "delta_z",
        "distance_3d",
        "distance_2d",
        "inverse_distance",
        "azimuth_sin",
        "azimuth_cos",
        "elevation_sin",
        "cell_0",
        "cell_1",
    ]
    local_suffixes = (
        "log_count",
        "area_density",
        "mean_center_dz",
        "mean_height_dz",
        "height_std",
        "abs_normal_x",
        "abs_normal_y",
        "abs_normal_z",
        "normal_scale",
    )
    for radius in LOCAL_RADII_METERS:
        names.extend(f"local_{int(radius)}m_{suffix}" for suffix in local_suffixes)
    names.extend(
        [
            "corridor_density_mean",
            "corridor_density_max",
            "corridor_density_std",
            "corridor_clearance_min",
            "corridor_clearance_mean",
            "corridor_density_q1",
            "corridor_density_q2",
            "corridor_density_q3",
            "corridor_density_q4",
            "fresnel_scale_050",
            "fresnel_scale_100",
            "fresnel_scale_150",
            "fresnel_scale_200",
            "surface_facing_mean",
            "surface_facing_max",
            "corridor_density_t025",
            "corridor_density_t050",
            "corridor_density_t075",
        ]
    )
    if len(names) != 71:
        raise AssertionError(f"Expected 71 geometry features, got {len(names)}")
    return names


def _neighbors(tree: object | None, centers: np.ndarray, point: np.ndarray, radius: float) -> np.ndarray:
    if tree is not None:
        return np.asarray(tree.query_ball_point(point, radius), dtype=np.int64)
    distance = np.linalg.norm(centers - point[None], axis=1)
    return np.flatnonzero(distance <= radius)


def _local_features(field: GaussianField, tree: object | None, point: np.ndarray) -> list[float]:
    output: list[float] = []
    for radius in LOCAL_RADII_METERS:
        indices = _neighbors(tree, field.centers, point, radius)
        if not len(indices):
            output.extend([0.0] * 9)
            continue
        weights = np.maximum(field.area[indices].astype(np.float64), 1e-8)
        weights /= weights.sum()
        centers = field.centers[indices].astype(np.float64)
        dz = centers[:, 2] - float(point[2])
        normals = np.abs(field.normals[indices].astype(np.float64))
        output.extend(
            [
                float(np.log1p(len(indices))),
                float(field.area[indices].sum() / (np.pi * radius * radius)),
                float(np.sum(weights * dz) / radius),
                float(np.sum(weights * centers[:, 2]) - point[2]) / radius,
                float(np.sqrt(np.sum(weights * (dz - np.sum(weights * dz)) ** 2)) / radius),
                float(np.sum(weights * normals[:, 0])),
                float(np.sum(weights * normals[:, 1])),
                float(np.sum(weights * normals[:, 2])),
                float(np.sum(weights * field.normal_scale[indices]) / radius),
            ]
        )
    return output


def _corridor_profiles(
    field: GaussianField,
    tree: object | None,
    station: np.ndarray,
    point: np.ndarray,
    sample_count: int,
    base_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = point - station
    distance = max(float(np.linalg.norm(direction)), 1e-6)
    unit = direction / distance
    samples = np.linspace(0.05, 0.95, max(sample_count, 8), dtype=np.float64)
    density = np.zeros(len(samples), dtype=np.float64)
    clearance = np.full(len(samples), base_radius * 2.0, dtype=np.float64)
    facing = np.zeros(len(samples), dtype=np.float64)
    for index, fraction in enumerate(samples):
        center = station + fraction * direction
        fresnel = base_radius * np.sqrt(max(4.0 * fraction * (1.0 - fraction), 0.05))
        candidates = _neighbors(tree, field.centers, center, max(2.0 * fresnel, 1.0))
        if not len(candidates):
            continue
        delta = field.centers[candidates].astype(np.float64) - center[None]
        projection = delta @ unit
        perpendicular = delta - projection[:, None] * unit[None]
        radial = np.linalg.norm(perpendicular, axis=1)
        scale = np.maximum(
            fresnel + field.tangent_scale[candidates].mean(axis=1).astype(np.float64),
            0.25,
        )
        weight = np.exp(-0.5 * (radial / scale) ** 2) * field.area[candidates]
        density[index] = float(weight.sum() / max(np.pi * fresnel * fresnel, 1e-6))
        clearance[index] = float(radial.min())
        normal_alignment = np.abs(field.normals[candidates].astype(np.float64) @ unit)
        facing[index] = float(np.sum(weight * normal_alignment) / max(weight.sum(), 1e-12))
    return density, clearance, facing


def _corridor_features(
    field: GaussianField,
    tree: object | None,
    station: np.ndarray,
    point: np.ndarray,
    sample_count: int,
    base_radius: float,
) -> list[float]:
    density, clearance, facing = _corridor_profiles(
        field, tree, station, point, sample_count, base_radius
    )
    quarters = np.array_split(density, 4)
    scaled_density: list[float] = []
    for scale in (0.5, 1.0, 1.5, 2.0):
        profile, _, _ = _corridor_profiles(
            field, tree, station, point, max(8, sample_count // 2), base_radius * scale
        )
        scaled_density.append(float(np.mean(profile)))
    sample_axis = np.linspace(0.05, 0.95, len(density))
    at = lambda fraction: float(density[int(np.argmin(np.abs(sample_axis - fraction)))])
    return [
        float(np.mean(density)),
        float(np.max(density)),
        float(np.std(density)),
        float(np.min(clearance) / max(base_radius, 1e-6)),
        float(np.mean(clearance) / max(base_radius, 1e-6)),
        *(float(np.mean(part)) for part in quarters),
        *scaled_density,
        float(np.mean(facing)),
        float(np.max(facing)),
        at(0.25),
        at(0.50),
        at(0.75),
    ]


def extract_geometry_features(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    stations: np.ndarray,
    field: GaussianField,
    corridor_samples: int = 24,
    fresnel_radius_meters: float = 2.0,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    stations = np.asarray(stations, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [samples,3]")
    if len(cell_ids) != len(positions):
        raise ValueError("cell_ids and positions must have the same length")
    if np.any(cell_ids < 0) or np.any(cell_ids >= len(stations)):
        raise ValueError("cell_ids contain an invalid station index")
    tree = cKDTree(field.centers.astype(np.float64)) if cKDTree is not None else None
    output = np.empty((len(positions), len(feature_names())), dtype=np.float32)
    for index, (point, cell_id) in enumerate(zip(positions, cell_ids, strict=True)):
        station = stations[int(cell_id)]
        delta = point - station
        distance_2d = max(float(np.linalg.norm(delta[:2])), 1e-6)
        distance_3d = max(float(np.linalg.norm(delta)), 1e-6)
        azimuth = np.arctan2(delta[1], delta[0])
        global_features = [
            *point.tolist(),
            *station.tolist(),
            *delta.tolist(),
            distance_3d,
            distance_2d,
            1.0 / distance_3d,
            float(np.sin(azimuth)),
            float(np.cos(azimuth)),
            float(delta[2] / distance_3d),
            float(cell_id == 0),
            float(cell_id == 1),
        ]
        values = global_features
        values += _local_features(field, tree, point)
        values += _corridor_features(
            field,
            tree,
            station,
            point,
            int(corridor_samples),
            float(fresnel_radius_meters),
        )
        if len(values) != 71:
            raise AssertionError(f"Expected 71 features, got {len(values)}")
        output[index] = np.asarray(values, dtype=np.float32)
    return output
