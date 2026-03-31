from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.arguments import parse as parse_config
from lib.dataloader.pandaset_loader import _resolve_source_root, load_pandaset_raw


def _color_array(rgb: tuple[int, int, int], n: int) -> np.ndarray:
    arr = np.zeros((n, 3), dtype=np.uint8)
    arr[:, 0] = rgb[0]
    arr[:, 1] = rgb[1]
    arr[:, 2] = rgb[2]
    return arr


def _direction_to_rgb(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    norm = np.clip(norm, 1.0e-6, None)
    unit = direction / norm
    return ((unit + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype(np.uint8)


def _make_vertex_block(points: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    colors = _color_array(rgb, len(points))
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertex = np.empty(len(points), dtype=dtype)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    return vertex


def _make_edge_block(edge_pairs: list[tuple[int, int]]) -> np.ndarray:
    dtype = [("vertex1", "i4"), ("vertex2", "i4")]
    edge = np.empty(len(edge_pairs), dtype=dtype)
    if edge_pairs:
        edge["vertex1"] = [a for a, _ in edge_pairs]
        edge["vertex2"] = [b for _, b in edge_pairs]
    return edge


def _write_ply(path: Path, vertices: np.ndarray, edges: np.ndarray | None = None) -> None:
    elements = [PlyElement.describe(vertices, "vertex")]
    if edges is not None:
        elements.append(PlyElement.describe(edges, "edge"))
    PlyData(elements, text=True).write(str(path))


def _write_direction_ply(path: Path, points: np.ndarray, directions: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    colors = _direction_to_rgb(directions)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertex = np.empty(len(points), dtype=dtype)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    _write_ply(path, vertex)


def _save_ray_arrays(path: Path, ray_origins: np.ndarray, ray_directions: np.ndarray) -> Path:
    out_path = path.with_name(path.stem + "_ray_grid.npz")
    np.savez_compressed(
        out_path,
        ray_origins=np.asarray(ray_origins, dtype=np.float32),
        ray_directions=np.asarray(ray_directions, dtype=np.float32),
    )
    return out_path


def _write_direction_field_ply(path: Path, ray_origins: np.ndarray, ray_directions: np.ndarray) -> None:
    ray_origins = np.asarray(ray_origins, dtype=np.float32)
    ray_directions = np.asarray(ray_directions, dtype=np.float32)
    h, w, _ = ray_directions.shape
    grid_x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    grid_y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    vertices = []
    edges = []
    for r in range(h):
        for c in range(w):
            start = np.array([grid_x[c], grid_y[r], 0.0], dtype=np.float32)
            direction = ray_directions[r, c]
            direction = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
            end = start + direction * 0.08
            idx = len(vertices)
            vertices.append(start.tolist())
            vertices.append(end.tolist())
            edges.append((idx, idx + 1))
    vertex = _make_vertex_block(np.asarray(vertices, dtype=np.float32), (120, 180, 255))
    edge = _make_edge_block(edges)
    _write_ply(path, vertex, edge)


def _full_ray_indices(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PandaSet scene points and ray edges to a PLY.")
    parser.add_argument(
        "--config",
        default="/mnt/data16/xuzhiy/LiDAR-RT/configs/pandaset/static/1.yaml",
        help="Path to PandaSet config.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to export.")
    parser.add_argument(
        "--out",
        default="/mnt/data16/xuzhiy/LiDAR-RT/artifacts/pandaset_center_ray_debug.ply",
        help="Output ply path.",
    )
    parser.add_argument("--ray-length", type=float, default=25.0, help="Length of each ray segment.")
    parser.add_argument(
        "--ray-visualization",
        choices=["origins", "segments", "both", "field"],
        default="origins",
        help="How to write ray debug geometry. origins avoids the spherical shell from fixed-length segments.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Override PandaSet source root.",
    )
    parser.add_argument(
        "--scene-mode",
        choices=["raw", "inverse"],
        default="raw",
        help="Export raw d=0 points or inverse-projected range points.",
    )
    args = parser.parse_args()

    cfg = parse_config(args.config)
    if args.source_root is not None:
        cfg.source_dir = args.source_root

    lidar, _ = load_pandaset_raw(cfg.source_dir, cfg)
    frame = int(args.frame)
    rays_o, rays_d = lidar.get_range_rays(frame)
    if args.scene_mode == "inverse":
        pts, _ = lidar.inverse_projection(frame)
    else:
        # Raw d=0 points from the underlying PandaSet dataframe are the canonical scene cloud.
        from pandaset import DataSet

        dataset = DataSet(_resolve_source_root(cfg.source_dir))
        sequence = dataset[cfg.seq]
        sequence.load_lidar()
        df = sequence.lidar.data[frame]
        if "d" in df.columns:
            df = df.loc[df["d"] == int(getattr(cfg, "pandaset_lidar_sensor_id", 0))]
        pts = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    if isinstance(pts, np.ndarray):
        pts_np = pts.astype(np.float32)
    else:
        pts_np = pts.detach().cpu().numpy().astype(np.float32)
    rays_o_np = rays_o.detach().cpu().numpy().astype(np.float32)
    rays_d_np = rays_d.detach().cpu().numpy().astype(np.float32)
    ply_path = Path(args.out).expanduser().resolve()

    # Scene cloud in gray.
    scene_vertex = _make_vertex_block(pts_np, (180, 180, 180))
    scene_ply = ply_path.with_name(ply_path.stem + "_scene.ply")
    _write_ply(scene_ply, scene_vertex)

    h, w = rays_d_np.shape[:2]
    samples = _full_ray_indices(h, w)
    ray_origin_ply = ply_path.with_name(ply_path.stem + "_ray_origins.ply")
    ray_segment_ply = ply_path.with_name(ply_path.stem + "_ray_segments.ply")
    ray_field_ply = ply_path.with_name(ply_path.stem + "_ray_field.ply")

    # Export ray origins as points colored by direction so the viewer does not
    # collapse the full ray set into a spherical shell of endpoints.
    ray_origins_flat = rays_o_np.reshape(-1, 3)
    ray_dirs_flat = rays_d_np.reshape(-1, 3)
    ray_npz = _save_ray_arrays(ply_path, rays_o_np, rays_d_np)
    if args.ray_visualization in {"origins", "both"}:
        _write_direction_ply(ray_origin_ply, ray_origins_flat, ray_dirs_flat)

    if args.ray_visualization in {"segments", "both"}:
        ray_vertices = []
        ray_edges = []
        for r, c in samples:
            origin = rays_o_np[r, c]
            direction = rays_d_np[r, c]
            start_index = len(ray_vertices)
            ray_vertices.append(origin.tolist())
            ray_vertices.append((origin + direction * args.ray_length).tolist())
            ray_edges.append((start_index, start_index + 1))
        ray_vertex = _make_vertex_block(np.asarray(ray_vertices, dtype=np.float32), (255, 80, 0))
        edge_block = _make_edge_block(ray_edges)
        _write_ply(ray_segment_ply, ray_vertex, edge_block)

    if args.ray_visualization == "field":
        _write_direction_field_ply(ray_field_ply, rays_o_np, rays_d_np)

    print(f"Saved scene PLY to: {scene_ply}")
    if args.ray_visualization in {"origins", "both"}:
        print(f"Saved ray origins PLY to: {ray_origin_ply}")
        print(f"ray_origins={len(ray_origins_flat)}")
    if args.ray_visualization in {"segments", "both"}:
        print(f"Saved ray segments PLY to: {ray_segment_ply}")
        print(f"ray_vertices={len(ray_vertex)} ray_edges={len(edge_block)}")
    if args.ray_visualization == "field":
        print(f"Saved ray field PLY to: {ray_field_ply}")
    print(f"Saved ray grid NPZ to: {ray_npz}")
    print(f"frame={frame} h={h} w={w} samples={samples}")
    print(f"first_origin={rays_o_np[samples[0][0], samples[0][1]].tolist()}")
    print(f"first_direction={rays_d_np[samples[0][0], samples[0][1]].tolist()}")


if __name__ == "__main__":
    main()
