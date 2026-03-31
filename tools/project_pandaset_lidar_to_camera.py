#!/usr/bin/env python3
import argparse
from pathlib import Path

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw
from pandaset import DataSet
from pandaset.geometry import projection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project PandaSet LiDAR points onto a camera image with depth coloring.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/mnt/data16/xuzhiy/Xsim_min_slim/datasets/pandaset/pandaset"),
        help="PandaSet root directory containing sequence folders.",
    )
    parser.add_argument("--seq", type=str, default="001", help="PandaSet sequence id.")
    parser.add_argument(
        "--camera",
        type=str,
        default="front_camera",
        help="Camera name, e.g. front_camera or front_left_camera.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 50, 60, 70],
        help="Frame indices to export.",
    )
    parser.add_argument(
        "--lidar-sensor-id",
        type=int,
        default=0,
        help="PandaSet LiDAR sensor id: 0 for mechanical 360 LiDAR, 1 for forward-facing LiDAR.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=4,
        help="Projected point radius in pixels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Maximum number of projected points to draw per frame.",
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
        default=Path("artifacts/pandaset_lidar_overlay"),
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
        x0 = float(u - radius)
        y0 = float(v - radius)
        x1 = float(u + radius)
        y1 = float(v + radius)
        draw.ellipse((x0, y0, x1, y1), fill=tuple(int(c) for c in color))
    return overlay


def main():
    args = parse_args()
    dataset = DataSet(str(args.source))
    if args.seq not in dataset.sequences():
        raise ValueError(f"Sequence {args.seq} not found under {args.source}")

    sequence = dataset[args.seq]
    sequence.load_lidar()
    sequence.load_camera()
    sequence.lidar.set_sensor(args.lidar_sensor_id)

    camera = sequence.camera[args.camera]
    output_dir = args.output_dir / args.seq / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in args.frames:
        lidar_df = sequence.lidar.data[frame]
        image = camera.data[frame]
        pose = camera.poses[frame]
        uv, points_camera, keep = projection(
            lidar_df[["x", "y", "z"]].to_numpy(dtype=np.float64),
            image,
            pose,
            camera.intrinsics,
            filter_outliers=True,
        )

        if uv.shape[0] == 0:
            print(f"frame={frame:03d} no projected points")
            continue

        depths = points_camera[:, 2].astype(np.float32)
        order = np.argsort(depths)[::-1]
        if args.max_points > 0 and order.size > args.max_points:
            order = order[: args.max_points]
        uv = uv[order]
        depths = depths[order]

        colors = colorize_depth(depths, args.depth_min, args.depth_max)
        overlay = draw_points(image, uv, colors, args.point_radius)

        out_path = output_dir / f"{frame:03d}_overlay_r{args.point_radius}.png"
        overlay.save(out_path)
        print(
            f"saved {out_path} | points={len(uv)} | depth_z=[{depths.min():.2f}, {depths.max():.2f}]",
        )


if __name__ == "__main__":
    main()
