from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np


def _write_mesh(path: Path) -> None:
    vertices = np.asarray(
        [
            [-30, -40, 0], [30, -40, 0], [30, 50, 0], [-30, 50, 0],
            [-8, -12, 0], [0, -12, 0], [0, -4, 0], [-8, -4, 0],
            [-8, -12, 8], [0, -12, 8], [0, -4, 8], [-8, -4, 8],
            [8, 18, 0], [17, 18, 0], [17, 28, 0], [8, 28, 0],
            [8, 18, 10], [17, 18, 10], [17, 28, 10], [8, 28, 10],
        ],
        dtype=np.float32,
    )
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 9), (4, 9, 8), (5, 6, 10), (5, 10, 9),
        (6, 7, 11), (6, 11, 10), (7, 4, 8), (7, 8, 11),
        (8, 9, 10), (8, 10, 11),
        (12, 13, 17), (12, 17, 16), (13, 14, 18), (13, 18, 17),
        (14, 15, 19), (14, 19, 18), (15, 12, 16), (15, 16, 19),
        (16, 17, 18), (16, 18, 19),
    ]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(" ".join(f"{float(value):.6f}" for value in vertex) + "\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic full-shape data for Scheme E smoke tests")
    parser.add_argument("--output", default=str(_bootstrap.PROJECT_ROOT / "artifacts" / "smoke_data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    setup_path = output / "Round2_Setup.json"
    if setup_path.exists() and not args.force:
        print(f"Smoke data already exists: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    station = np.asarray([[-10.0, -25.0, 12.0], [10.0, 25.0, 12.0]], dtype=np.float32)
    first = np.column_stack(
        [rng.uniform(-24, 4, 16), rng.uniform(-38, -10, 16), np.full(16, 1.5)]
    )
    second = np.column_stack(
        [rng.uniform(-3, 25, 16), rng.uniform(12, 42, 16), np.full(16, 1.5)]
    )
    train_positions = np.concatenate([first, second], axis=0).astype(np.float64)
    test_positions = np.asarray(
        [[-7.0, -20.0, 1.5], [-17.0, -29.0, 1.5], [8.0, 22.0, 1.5], [18.0, 33.0, 1.5]],
        dtype=np.float64,
    )
    shape = (len(train_positions), 256, 4, 192)
    real = rng.standard_normal(shape, dtype=np.float32)
    imaginary = rng.standard_normal(shape, dtype=np.float32)
    channels = (real + 1j * imaginary).astype(np.complex64)
    distance = np.concatenate(
        [
            np.linalg.norm(train_positions[:16] - station[0], axis=1),
            np.linalg.norm(train_positions[16:] - station[1], axis=1),
        ]
    )
    channels *= (1.0 / np.maximum(distance, 1.0))[:, None, None, None].astype(np.float32)
    channels[[1, 3, 5, 7, 17, 19, 21, 23]] = 0.0
    setup = {
        "P_Train": 32, "P_Test": 4,
        "M": 256, "M_H": 16, "M_V": 8, "M_P": 2,
        "N": 4, "N_H": 1, "N_V": 2, "N_P": 2,
        "S": 192, "Q": 2, "X": station.tolist(), "w": [0.4, 0.4, 0.2],
    }
    setup_path.write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    np.save(output / "Round2_Train_Pos.npy", train_positions)
    np.save(output / "Round2_Test_Pos.npy", test_positions)
    np.save(output / "Round2_Train_Channel.npy", channels)
    _write_mesh(output / "Round2_Map.ply")
    print(json.dumps({"status": "PASS", "output": str(output), "channel_shape": list(channels.shape)}, indent=2))


if __name__ == "__main__":
    main()
