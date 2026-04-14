#!/usr/bin/env python3
"""Export per-frame rendered camera depth maps as cached .npy files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.arguments import parse
from lib.dataloader.kitti_calib_loader import load_kitti_calib_cameras
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene.gaussian_model import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached rendered depth maps from a trained 2DGS model")
    parser.add_argument("-dc", "--data_config_path", required=True, help="data config path")
    parser.add_argument("-ec", "--exp_config_path", required=True, help="experiment config path")
    parser.add_argument("-m", "--model", required=True, help="trained model checkpoint path")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to write .npy depth maps")
    parser.add_argument(
        "--camera_scale",
        type=float,
        default=1.0,
        help="camera image downscale factor used for loading cached camera poses/images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing cached depth files",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = parse(cli.exp_config_path)
    args = parse(cli.data_config_path, args)
    args.model_path = cli.model
    args.camera_scale = cli.camera_scale

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = str(args.data_type).lower().replace("-", "").replace("_", "")
    if "kitticalib" not in dtype:
        raise ValueError(f"Unsupported dataset type for this exporter: {args.data_type}")

    scene_name = getattr(args, "kitti_calib_scene", None)
    if scene_name is None:
        raise ValueError("kitti_calib_scene must be set in the data config")

    frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))

    print("[export-depth] loading camera cache")
    cam_cameras, _ = load_kitti_calib_cameras(
        args.source_dir,
        args,
        scene_name=scene_name,
        frame_ids=frame_ids,
        scale=cli.camera_scale,
    )

    print(f"[export-depth] restoring checkpoint: {cli.model}")
    model_params, iteration = torch.load(cli.model, weights_only=False, map_location="cuda")
    gaussian_assets = []
    dc_only_sh = bool(getattr(args.model, "dc_only_sh", False))
    for capture in model_params:
        gaussians = GaussianModel(args.model.dimension, args.model.sh_degree, extent=0)
        if dc_only_sh:
            gaussians.set_dc_only_sh(True)
        gaussians.restore(capture, args.opt)
        gaussian_assets.append(gaussians)

    exported = 0
    skipped = 0
    for frame_id in sorted(cam_cameras):
        output_path = output_dir / f"{frame_id:06d}.npy"
        if output_path.exists() and not cli.overwrite:
            skipped += 1
            continue

        camera = cam_cameras[frame_id].cuda()
        render_pkg = render_camera(camera, gaussian_assets, args)
        depth = render_pkg["depth"].detach().float().cpu().numpy().astype(np.float32)
        np.save(output_path, depth)
        exported += 1
        print(f"[export-depth] frame {frame_id:02d} -> {output_path.name}")

    manifest = {
        "checkpoint": os.path.abspath(cli.model),
        "iteration": int(iteration),
        "source_dir": os.path.abspath(args.source_dir),
        "scene_name": scene_name,
        "camera_scale": float(cli.camera_scale),
        "frame_ids": frame_ids,
        "exported": exported,
        "skipped": skipped,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        f"[done] exported={exported} skipped={skipped} "
        f"dir={output_dir}"
    )


if __name__ == "__main__":
    main()
