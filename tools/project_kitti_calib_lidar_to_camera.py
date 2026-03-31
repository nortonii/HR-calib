#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.dataloader.kitti_calib_loader import _parse_kitti_calib_file, _read_ply_binary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project KITTI-calibration LiDAR points onto the paired camera image.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/kitti-calibration"),
        help="KITTI-calibration dataset root.",
    )
    parser.add_argument("--scene", type=str, default="5-50-t", help="Scene name.")
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 5, 15, 25, 35, 45],
        help="Frame indices to export.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=5,
        help="Projected point radius in pixels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Maximum number of points to draw per frame.",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=None,
        help="Optional fixed minimum depth for coloring.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Optional fixed maximum depth for coloring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kitti_calib_lidar_overlay"),
        help="Directory to save overlay images.",
    )
    return parser.parse_args()


def colorize_depth(depths: np.ndarray, depth_min: float | None, depth_max: float | None) -> np.ndarray:
    if depths.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo = float(np.percentile(depths, 2.0)) if depth_min is None else float(depth_min)
    hi = float(np.percentile(depths, 98.0)) if depth_max is None else float(depth_max)
    hi = max(hi, lo + 1.0e-6)
    normalized = np.clip((depths - lo) / (hi - lo), 0.0, 1.0)
    colors = colormaps["turbo_r"](normalized)[:, :3]
    return (colors * 255.0).astype(np.uint8)


def draw_points(image: Image.Image, points: np.ndarray, colors: np.ndarray, radius: int) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for (u, v), color in zip(points, colors, strict=False):
        draw.ellipse(
            (float(u - radius), float(v - radius), float(u + radius), float(v + radius)),
            fill=tuple(int(c) for c in color),
        )
    return overlay


def main():
    args = parse_args()
    scene_dir = args.source / args.scene
    pose_file = scene_dir / "LiDAR_poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"Missing pose file: {pose_file}")

    seq_num = int(args.scene.split("-")[0])
    calib_path = args.source / "calibs" / f"{seq_num:02d}.txt"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {calib_path}")

    P0, Tr = _parse_kitti_calib_file(str(calib_path))
    K = P0[:, :3]
    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3] = Tr
    cam_to_lidar = np.linalg.inv(Tr4)
    lidar_poses = np.loadtxt(pose_file).reshape(-1, 4, 4)

    output_dir = args.output_dir / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in args.frames:
        ply_path = scene_dir / f"{frame:02d}.ply"
        img_path = scene_dir / f"{frame:02d}.png"
        if not ply_path.exists() or not img_path.exists():
            print(f"skip frame={frame:02d} missing {(ply_path if not ply_path.exists() else img_path)}")
            continue

        raw_points = _read_ply_binary(str(ply_path))
        xyz_lidar = raw_points[:, :3].astype(np.float64)

        lidar_to_world = lidar_poses[frame]
        points_world = (lidar_to_world[:3, :3] @ xyz_lidar.T).T + lidar_to_world[:3, 3]

        cam_to_world = lidar_to_world @ cam_to_lidar
        world_to_cam = np.linalg.inv(cam_to_world)
        points_camera = (world_to_cam[:3, :3] @ points_world.T).T + world_to_cam[:3, 3]

        valid = points_camera[:, 2] > 0.0
        points_camera = points_camera[valid]
        if points_camera.shape[0] == 0:
            print(f"frame={frame:02d} no visible points")
            continue

        pixels_h = (K @ points_camera.T).T
        uv = pixels_h[:, :2] / pixels_h[:, 2:3]

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        width, height = image.size

        mask = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < height)
        )
        uv = uv[mask]
        depths = points_camera[mask, 2].astype(np.float32)
        if uv.shape[0] == 0:
            print(f"frame={frame:02d} no in-image points")
            continue

        order = np.argsort(depths)[::-1]
        if args.max_points > 0 and order.size > args.max_points:
            order = order[: args.max_points]
        uv = uv[order]
        depths = depths[order]

        colors = colorize_depth(depths, args.depth_min, args.depth_max)
        overlay = draw_points(image, uv, colors, args.point_radius)

        out_path = output_dir / f"{frame:02d}_overlay_r{args.point_radius}.png"
        overlay.save(out_path)
        print(
            f"saved {out_path} | points={len(uv)} | depth_z=[{depths.min():.2f}, {depths.max():.2f}]",
        )


if __name__ == "__main__":
    main()
