#!/usr/bin/env python3
"""Project a depth map into the RGB image plane using a calibrated RGB-D extrinsic."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils.rgbd_calibration import (
    CameraModel,
    load_depth_map,
    load_rgb_image,
    make_depth_overlay,
    project_depth_to_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp depth map into RGB image coordinates.")
    parser.add_argument("--rgb", required=True, help="RGB image path.")
    parser.add_argument("--depth", required=True, help="Depth map path.")
    parser.add_argument("--extrinsic", required=True, help="extrinsic.yaml from tools/rgbd_calib.py.")
    parser.add_argument("--output", required=True, help="Output overlay image path.")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor applied after loading the depth map.")
    parser.add_argument("--depth_npz_key", default=None, help="Array key for .npz depth files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.extrinsic, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    rgb_camera = CameraModel(
        K=np.asarray(payload["rgb_camera"]["K"], dtype=float),
        dist=np.asarray(payload["rgb_camera"]["dist"], dtype=float),
    )
    depth_camera = CameraModel(
        K=np.asarray(payload["depth_camera"]["K"], dtype=float),
        dist=np.asarray(payload["depth_camera"]["dist"], dtype=float),
    )
    rotation_matrix = np.asarray(payload["T_rgb_d"]["rotation_matrix"], dtype=float)
    translation = np.asarray(payload["T_rgb_d"]["translation_xyz"], dtype=float)

    rgb_image = load_rgb_image(args.rgb)
    depth_map = load_depth_map(args.depth, depth_scale=args.depth_scale, npz_key=args.depth_npz_key)
    warped_depth = project_depth_to_rgb(
        depth_map=depth_map,
        depth_camera=depth_camera,
        rgb_camera=rgb_camera,
        rotation_matrix=rotation_matrix,
        translation=translation,
        rgb_shape=rgb_image.shape[:2],
    )
    overlay = make_depth_overlay(rgb_image, warped_depth)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[done] saved {output_path}")


if __name__ == "__main__":
    main()
