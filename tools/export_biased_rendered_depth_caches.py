#!/usr/bin/env python3
"""Export multiple biased rendered depth caches for robustness testing."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.arguments import parse
from lib.dataloader.kitti_calib_loader import load_kitti_calib_cameras
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene.cameras import Camera
from lib.scene.gaussian_model import GaussianModel


def _axis_angle_to_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(float(angle_deg))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def _load_lidar_world_poses(scene_dir: Path) -> list[np.ndarray]:
    pose_path = scene_dir / "LiDAR_poses.txt"
    poses = np.loadtxt(pose_path).reshape(-1, 4, 4)
    return [pose.astype(np.float64) for pose in poses]


def _camera_from_l2c(
    frame_id: int,
    reference_camera: Camera,
    lidar_world_pose: np.ndarray,
    lidar_to_camera: np.ndarray,
) -> Camera:
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    camera_to_world = lidar_world_pose @ camera_to_lidar
    rotation_world = camera_to_world[:3, :3].astype(np.float32)
    camera_center = camera_to_world[:3, 3].astype(np.float32)
    translation = (-rotation_world.T @ camera_center).astype(np.float32)
    return Camera(
        timestamp=frame_id,
        R=torch.from_numpy(rotation_world),
        T=torch.from_numpy(translation),
        w=reference_camera.image_width,
        h=reference_camera.image_height,
        FoVx=reference_camera.FoVx,
        FoVy=reference_camera.FoVy,
        K=getattr(reference_camera, "K", None),
    )


def _build_gaussian_assets(args, checkpoint_path: str) -> tuple[list[GaussianModel], int]:
    model_params, iteration = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
    dc_only_sh = bool(getattr(args.model, "dc_only_sh", False))
    gaussian_assets: list[GaussianModel] = []
    for capture in model_params:
        gaussians = GaussianModel(args.model.dimension, args.model.sh_degree, extent=0)
        if dc_only_sh:
            gaussians.set_dc_only_sh(True)
        gaussians.restore(capture, args.opt)
        gaussian_assets.append(gaussians)
    return gaussian_assets, int(iteration)


def _build_presets(l2c_payload: dict) -> dict[str, np.ndarray]:
    gt = np.asarray(l2c_payload["correct"], dtype=np.float64)
    presets: dict[str, np.ndarray] = {}
    for key in ("initial1", "initial2"):
        if key in l2c_payload:
            presets[key] = np.asarray(l2c_payload[key], dtype=np.float64)

    gt_R = gt[:3, :3]
    gt_t = gt[:3, 3]

    example = np.eye(4, dtype=np.float64)
    example[:3, :3] = _axis_angle_to_matrix(np.array([0.5774, 0.5774, 0.5774]), 9.9239) @ gt_R
    example[:3, 3] = gt_t + np.array([0.0718, 0.1314, 0.0960], dtype=np.float64)
    presets["calib_example"] = example

    mild = np.eye(4, dtype=np.float64)
    mild[:3, :3] = _axis_angle_to_matrix(np.array([1.0, 0.0, 1.0]), 5.0) @ gt_R
    mild[:3, 3] = gt_t + np.array([0.03, -0.02, 0.04], dtype=np.float64)
    presets["mild_bias"] = mild
    return presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export several biased rendered-depth caches")
    parser.add_argument("-dc", "--data_config_path", required=True)
    parser.add_argument("-ec", "--exp_config_path", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output_root", required=True)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument(
        "--presets",
        default="initial1,initial2,calib_example,mild_bias",
        help="Comma-separated preset names to export",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = parse(cli.exp_config_path)
    args = parse(cli.data_config_path, args)
    args.model_path = cli.model
    args.camera_scale = cli.camera_scale

    scene_name = getattr(args, "kitti_calib_scene", None)
    if scene_name is None:
        raise ValueError("kitti_calib_scene must be set")

    scene_dir = Path(args.source_dir) / scene_name
    lidar_to_camera_path = scene_dir / "LiDAR-to-camera.json"
    with open(lidar_to_camera_path, "r", encoding="utf-8") as handle:
        l2c_payload = json.load(handle)

    presets = _build_presets(l2c_payload)
    selected_names = [name.strip() for name in cli.presets.split(",") if name.strip()]
    for name in selected_names:
        if name not in presets:
            raise KeyError(f"Unknown preset '{name}'. Available: {sorted(presets)}")

    gt_l2c = np.asarray(l2c_payload["correct"], dtype=np.float64)
    lidar_world_poses = _load_lidar_world_poses(scene_dir)
    frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
    cam_cameras, _ = load_kitti_calib_cameras(
        args.source_dir,
        args,
        scene_name=scene_name,
        frame_ids=frame_ids,
        scale=cli.camera_scale,
    )
    gaussian_assets, iteration = _build_gaussian_assets(args, cli.model)

    output_root = Path(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    intrinsics_path = output_root / "camera_intrinsics_rgbd.yaml"
    if not intrinsics_path.exists() or cli.overwrite:
        payload = {
            "K": json.load(open(scene_dir / "camera-intrinsic.json", "r", encoding="utf-8")),
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with open(intrinsics_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    for name in selected_names:
        biased_l2c = presets[name]
        gt_rgb_d = gt_l2c @ np.linalg.inv(biased_l2c)
        preset_dir = output_root / name
        depth_dir = preset_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        skipped = 0

        for frame_id in sorted(cam_cameras):
            output_path = depth_dir / f"{frame_id:06d}.npy"
            if output_path.exists() and not cli.overwrite:
                skipped += 1
                continue
            virtual_camera = _camera_from_l2c(
                frame_id=frame_id,
                reference_camera=cam_cameras[frame_id],
                lidar_world_pose=lidar_world_poses[frame_id],
                lidar_to_camera=biased_l2c,
            ).cuda()
            render_pkg = render_camera(virtual_camera, gaussian_assets, args)
            depth = render_pkg["depth"].detach().float().cpu().numpy().astype(np.float32)
            np.save(output_path, depth)
            exported += 1

        manifest = {
            "name": name,
            "checkpoint": os.path.abspath(cli.model),
            "iteration": iteration,
            "source_dir": os.path.abspath(args.source_dir),
            "scene_name": scene_name,
            "camera_scale": float(cli.camera_scale),
            "exported": exported,
            "skipped": skipped,
            "gt_lidar_to_camera": gt_l2c.tolist(),
            "biased_lidar_to_camera": biased_l2c.tolist(),
            "gt_T_rgb_d": gt_rgb_d.tolist(),
        }
        with open(preset_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"[done] {name}: exported={exported} skipped={skipped} dir={depth_dir}")


if __name__ == "__main__":
    main()
