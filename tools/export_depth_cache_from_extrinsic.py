#!/usr/bin/env python3
"""Export rendered depth caches using an explicit T_rgb_d extrinsic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.arguments import parse
from lib.dataloader.kitti_calib_loader import load_kitti_calib_cameras
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene.cameras import Camera
from lib.scene.gaussian_model import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rendered depth cache from an explicit extrinsic")
    parser.add_argument("-dc", "--data_config_path", required=True)
    parser.add_argument("-ec", "--exp_config_path", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-x", "--extrinsic", required=True, help="extrinsic.yaml containing T_rgb_d")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_extrinsic(path: Path) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pose = payload["T_rgb_d"]
    if "rotation_matrix" in pose:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    else:
        rvec = np.asarray(pose["rotation_vector"], dtype=np.float64).reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(pose["translation_xyz"], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


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


def _camera_from_rgb_camera(frame_id: int, reference_camera: Camera, T_rgb_d: np.ndarray) -> Camera:
    c2w_rgb = np.eye(4, dtype=np.float64)
    c2w_rgb[:3, :3] = np.asarray(reference_camera.R, dtype=np.float64)
    c2w_rgb[:3, 3] = np.asarray(reference_camera.camera_center, dtype=np.float64)
    c2w_depth = c2w_rgb @ T_rgb_d
    rotation_world = c2w_depth[:3, :3].astype(np.float32)
    camera_center = c2w_depth[:3, 3].astype(np.float32)
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
    frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
    cam_cameras, _ = load_kitti_calib_cameras(
        args.source_dir,
        args,
        scene_name=scene_name,
        frame_ids=frame_ids,
        scale=cli.camera_scale,
    )
    gaussian_assets, iteration = _build_gaussian_assets(args, cli.model)
    T_rgb_d = _load_extrinsic(Path(cli.extrinsic))

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_path = output_dir.parent / "camera_intrinsics_rgbd.yaml"
    if not intrinsics_path.exists() or cli.overwrite:
        payload = {
            "K": json.load(open(scene_dir / "camera-intrinsic.json", "r", encoding="utf-8")),
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with open(intrinsics_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    exported = 0
    skipped = 0
    for frame_id in sorted(cam_cameras):
        output_path = output_dir / f"{frame_id:06d}.npy"
        if output_path.exists() and not cli.overwrite:
            skipped += 1
            continue
        virtual_camera = _camera_from_rgb_camera(frame_id, cam_cameras[frame_id], T_rgb_d).cuda()
        render_pkg = render_camera(virtual_camera, gaussian_assets, args)
        depth = render_pkg["depth"].detach().float().cpu().numpy().astype(np.float32)
        np.save(output_path, depth)
        exported += 1

    manifest = {
        "checkpoint": os.path.abspath(cli.model),
        "iteration": iteration,
        "source_dir": os.path.abspath(args.source_dir),
        "scene_name": scene_name,
        "camera_scale": float(cli.camera_scale),
        "extrinsic": os.path.abspath(cli.extrinsic),
        "T_rgb_d": T_rgb_d.tolist(),
        "exported": exported,
        "skipped": skipped,
    }
    with open(output_dir.parent / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[done] exported={exported} skipped={skipped} dir={output_dir}")


if __name__ == "__main__":
    main()
