from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BevMap:
    features: np.ndarray
    minimum_xy: np.ndarray
    resolution: float

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[0])

    @property
    def maximum_xy(self) -> np.ndarray:
        height, width = self.features.shape[1:]
        return self.minimum_xy + self.resolution * np.array([width - 1, height - 1])


def read_ply_vertices(path: str | Path) -> np.ndarray:
    ply_path = Path(path)
    with ply_path.open("r", encoding="ascii") as handle:
        vertex_count = None
        format_name = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            fields = line.strip().split()
            if fields[:1] == ["format"]:
                format_name = fields[1]
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if line.strip() == "end_header":
                break
        if format_name != "ascii" or vertex_count is None:
            raise ValueError("Only ASCII PLY files with a vertex element are supported")
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        for index in range(vertex_count):
            vertices[index] = [float(value) for value in handle.readline().split()[:3]]
    return vertices


def build_bev(vertices: np.ndarray, resolution: float) -> BevMap:
    minimum = vertices[:, :2].min(axis=0) - resolution
    maximum = vertices[:, :2].max(axis=0) + resolution
    width, height = np.ceil((maximum - minimum) / resolution).astype(int) + 1
    x_index = np.clip(((vertices[:, 0] - minimum[0]) / resolution).astype(int), 0, width - 1)
    y_index = np.clip(((vertices[:, 1] - minimum[1]) / resolution).astype(int), 0, height - 1)

    count = np.zeros((height, width), dtype=np.float32)
    max_height = np.full((height, width), vertices[:, 2].min(), dtype=np.float32)
    np.add.at(count, (y_index, x_index), 1.0)
    np.maximum.at(max_height, (y_index, x_index), vertices[:, 2])

    density = np.log1p(count)
    density /= max(float(density.max()), 1.0)
    z_min = float(vertices[:, 2].min())
    z_span = max(float(vertices[:, 2].max() - z_min), 1.0)
    height_norm = (max_height - z_min) / z_span

    bins = []
    for lower, upper in [(-np.inf, 3.0), (3.0, 10.0), (10.0, 25.0), (25.0, np.inf)]:
        occupied = np.zeros((height, width), dtype=np.float32)
        mask = (vertices[:, 2] >= lower) & (vertices[:, 2] < upper)
        np.maximum.at(occupied, (y_index[mask], x_index[mask]), 1.0)
        bins.append(occupied)
    features = np.stack([density, height_norm, *bins], axis=0).astype(np.float32)
    return BevMap(features=features, minimum_xy=minimum.astype(np.float32), resolution=float(resolution))


def bilinear_sample(bev: BevMap, points_xy: np.ndarray) -> np.ndarray:
    grid = (points_xy - bev.minimum_xy[None, :]) / bev.resolution
    x = np.clip(grid[:, 0], 0, bev.features.shape[2] - 1)
    y = np.clip(grid[:, 1], 0, bev.features.shape[1] - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, bev.features.shape[2] - 1)
    y1 = np.minimum(y0 + 1, bev.features.shape[1] - 1)
    wx = (x - x0)[None, :]
    wy = (y - y0)[None, :]
    f00 = bev.features[:, y0, x0]
    f10 = bev.features[:, y0, x1]
    f01 = bev.features[:, y1, x0]
    f11 = bev.features[:, y1, x1]
    sampled = f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy)
    sampled += f01 * (1 - wx) * wy + f11 * wx * wy
    return sampled.T.astype(np.float32)


def build_map_tokens(
    bev: BevMap,
    positions: np.ndarray,
    base_stations: np.ndarray,
    link_samples: int,
    local_grid: int,
    local_radius: float,
) -> np.ndarray:
    if local_grid < 1 or local_grid % 2 == 0:
        raise ValueError("local_grid must be a positive odd integer")
    offsets_1d = np.linspace(-local_radius, local_radius, local_grid, dtype=np.float32)
    local_offsets = np.array([(x, y) for y in offsets_1d for x in offsets_1d], dtype=np.float32)
    token_count = link_samples + len(local_offsets)
    feature_dim = bev.feature_dim + 6
    result = np.empty((len(positions), len(base_stations), token_count, feature_dim), dtype=np.float32)
    map_span = np.maximum(bev.maximum_xy - bev.minimum_xy, 1.0)
    progress = np.linspace(0.0, 1.0, link_samples, dtype=np.float32)

    for sample_index, position in enumerate(positions):
        ue_xy = position[:2].astype(np.float32)
        for bs_index, base_station in enumerate(base_stations):
            bs_xy = base_station[:2].astype(np.float32)
            link_points = bs_xy[None, :] + progress[:, None] * (ue_xy - bs_xy)[None, :]
            local_points = ue_xy[None, :] + local_offsets
            points = np.concatenate([link_points, local_points], axis=0)
            raw = bilinear_sample(bev, points)
            normalized_xy = (points - bev.minimum_xy[None, :]) / map_span[None, :]
            relative_ue = (points - ue_xy[None, :]) / map_span[None, :]
            token_progress = np.concatenate([progress, np.ones(len(local_points), dtype=np.float32)])[:, None]
            token_type = np.concatenate([
                np.zeros(link_samples, dtype=np.float32),
                np.ones(len(local_points), dtype=np.float32),
            ])[:, None]
            result[sample_index, bs_index] = np.concatenate(
                [raw, normalized_xy, relative_ue, token_progress, token_type], axis=1
            )
    return result

