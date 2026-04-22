"""
Loader for the kitti-calibration dataset (derived from KITTI Odometry).
Point clouds are stored as binary .ply files with 4 float properties (x, y, z, intensity).
LiDAR_poses.txt stores 50 × 4×4 matrices mapping each frame to frame-0 space.
Sensor layout: Velodyne HDL-64E (same parameters as KITTI-360).
"""

import math
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from lib.scene import LiDARSensor
from lib.scene.cameras import Camera
from lib.utils.console_utils import *
from lib.utils.kitti_utils import (
    build_kitti_range_image_from_points,
    interpolate_sensor2world_columns,
    resolve_kitti_lidar_width,
)
from lib.utils.velodyne_utils import get_kitti_hdl64e_beam_inclinations_rad


def _read_ply_binary(ply_path):
    """Fast binary PLY reader for float32 point clouds (x, y, z, intensity)."""
    with open(ply_path, "rb") as f:
        num_vertices = None
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.float32)
    # Some PLY files may have extra properties; trim to num_vertices × 4
    if num_vertices is not None:
        data = data[: num_vertices * 4].reshape(-1, 4)
    else:
        data = data.reshape(-1, 4)
    return data  # (N, 4): x y z intensity


def _parse_kitti_calib_file(calib_path: str):
    """Parse KITTI calibration file, return P0 (3×4) and Tr (3×4) as float64 arrays."""
    calib = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            calib[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    P0 = calib['P0'].reshape(3, 4)
    Tr = calib['Tr'].reshape(3, 4)
    return P0, Tr


def load_kitti_calib_cameras(
    base_dir: str,
    args,
    scene_name: str,
    frame_ids,
    scale: int = 4,
) -> Tuple[Dict[int, Camera], Dict[int, torch.Tensor]]:
    """Load camera images and poses for a kitti-calibration scene.

    Camera extrinsics are derived from LiDAR poses + Tr (velo→cam) calibration,
    expressed in the LiDAR world frame (same frame as the Gaussian point cloud).

    C2W[i] = LiDAR_pose[i] @ inv(Tr_4x4)

    Intrinsics come from the KITTI calibration file (P0: fx, fy, cx, cy).
    Images are at {base_dir}/{scene_name}/{frame_id:02d}.png
    """
    import json as _json
    import glob as _glob

    scene_dir = os.path.join(base_dir, scene_name)
    pose_file = os.path.join(scene_dir, "LiDAR_poses.txt")
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"LiDAR_poses.txt not found: {pose_file}")

    # Sequence number from scene_name prefix (e.g. "5-50-t" → "05", "10-0-r" → "10")
    seq_num = int(scene_name.split('-')[0])
    calib_path = os.path.join(base_dir, "calibs", f"{seq_num:02d}.txt")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    P0, Tr = _parse_kitti_calib_file(calib_path)

    # Tr: velo→cam (3×4) → 4×4
    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3] = Tr
    # cam_to_lidar = inv(Tr4): maps camera-frame points to LiDAR frame
    cam_to_lidar = np.linalg.inv(Tr4)

    # LiDAR poses: frame_i → LiDAR world (4×4)
    lidar_poses = np.loadtxt(pose_file).reshape(-1, 4, 4)

    # Intrinsics from P0
    fx_orig = float(P0[0, 0])
    fy_orig = float(P0[1, 1])
    cx_orig = float(P0[0, 2])
    cy_orig = float(P0[1, 2])

    # Actual image dimensions from first frame on disk
    first_img_path = os.path.join(scene_dir, f"{frame_ids[0]:02d}.png")
    probe = cv2.imread(first_img_path)
    if probe is None:
        raise FileNotFoundError(f"Cannot probe image dimensions: {first_img_path}")
    H_orig, W_orig = probe.shape[:2]

    sx = 1.0 / scale
    W = int(W_orig * sx)
    H = int(H_orig * sx)
    fx_s, fy_s = fx_orig * sx, fy_orig * sx
    cx_s, cy_s = cx_orig * sx, cy_orig * sx
    FoVx = 2.0 * math.atan(W / (2.0 * fx_s))
    FoVy = 2.0 * math.atan(H / (2.0 * fy_s))
    K = np.array(
        [
            [fx_s, 0.0, cx_s],
            [0.0, fy_s, cy_s],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Cache: invalidate if dimensions or extrinsic method changed
    cache_dir = os.path.join(base_dir, f"cache_cam_{scene_name}_s{scale}")
    meta_file = os.path.join(cache_dir, "meta.json")
    os.makedirs(cache_dir, exist_ok=True)
    meta = {"W": W, "H": H, "convention": "lidar_pose_x_cam2lidar"}
    if os.path.exists(meta_file):
        old_meta = _json.load(open(meta_file))
        if old_meta != meta:
            print(red(f"[kitti-calib cam] Cache mismatch ({old_meta} vs {meta}), clearing cache"))
            for f in _glob.glob(os.path.join(cache_dir, "frame_*.pt")):
                os.remove(f)
    else:
        stale = _glob.glob(os.path.join(cache_dir, "frame_*.pt"))
        if stale:
            print(red(f"[kitti-calib cam] No meta.json, clearing {len(stale)} stale cache files"))
            for f in stale:
                os.remove(f)
    with open(meta_file, "w") as mf:
        _json.dump(meta, mf)

    cameras: Dict[int, Camera] = {}
    images: Dict[int, torch.Tensor] = {}

    for frame_id in frame_ids:
        if frame_id >= len(lidar_poses):
            print(red(f"[kitti-calib cam] frame {frame_id} out of LiDAR pose range, skipping"))
            continue

        cache_path = os.path.join(cache_dir, f"frame_{frame_id:02d}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            R, T, image_tensor = cached["R"], cached["T"], cached["image"]
        else:
            img_path = os.path.join(scene_dir, f"{frame_id:02d}.png")
            if not os.path.exists(img_path):
                print(red(f"[kitti-calib cam] Image not found: {img_path}, skipping"))
                continue

            # Camera pose in LiDAR world frame:
            # C2W = LiDAR_pose[i] @ cam_to_lidar
            C2W = lidar_poses[frame_id] @ cam_to_lidar
            R_wc = C2W[:3, :3].astype(np.float32)
            cam_center = C2W[:3, 3].astype(np.float32)
            R = torch.from_numpy(R_wc)
            T = torch.from_numpy(-R_wc.T @ cam_center)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(red(f"[kitti-calib cam] Failed to read: {img_path}"))
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if scale != 1:
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
            image_tensor = torch.from_numpy(img_rgb).float() / 255.0

            torch.save({"R": R, "T": T, "image": image_tensor}, cache_path)

        cam = Camera(timestamp=frame_id, R=R, T=T, w=W, h=H, FoVx=FoVx, FoVy=FoVy, K=K)
        cameras[frame_id] = cam
        images[frame_id] = image_tensor

    print(blue(f"[kitti-calib cam] Loaded {len(cameras)} camera frames "
               f"({W}×{H} @ 1/{scale} scale) for scene '{scene_name}'"))
    return cameras, images


def load_kitti_calib_raw(base_dir, args):
    frames = args.frame_length
    scene_name = getattr(args, "kitti_calib_scene", None)
    if scene_name is None:
        raise ValueError("kitti_calib_scene must be set in config (e.g. '5-50-t')")

    scene_dir = os.path.join(base_dir, scene_name)
    pose_file = os.path.join(scene_dir, "LiDAR_poses.txt")

    # poses[i] maps LiDAR frame i → frame-0 world space (4×4)
    raw_poses = np.loadtxt(pose_file).reshape(-1, 4, 4)
    num_poses = raw_poses.shape[0]

    frame_ids = [i for i in range(frames[0], frames[1] + 1) if i < num_poses]
    print(blue(f"[kitti-calib {scene_name}] Using {len(frame_ids)} frames in [{frames[0]}, {frames[1]}]"))

    # HDL-64E parameters (same nominal vertical layout as KITTI-360).
    beam_inclinations_top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(
        order="top_to_bottom"
    )
    beam_inclinations_bottom_to_top = get_kitti_hdl64e_beam_inclinations_rad(
        order="bottom_to_top"
    )
    W, H = resolve_kitti_lidar_width(args), len(beam_inclinations_top_to_bottom)
    max_depth = 80.0

    sensor2ego = np.eye(4, dtype=np.float64)

    lidar = LiDARSensor(
        sensor2ego=sensor2ego,
        name="velo",
        inclination_bounds=beam_inclinations_bottom_to_top.tolist(),
        data_type=args.data_type,
    )

    use_column_origin = getattr(args, "lidar_origin_mode", "center") == "column_interp"
    prev_ego2world = None

    for frame in frame_ids:
        ego2world = raw_poses[frame]  # (4, 4)
        ply_path = os.path.join(scene_dir, f"{frame:02d}.ply")
        pts = _read_ply_binary(ply_path)
        xyzs = pts[:, :3]
        intensities = pts[:, 3]

        ray_origin = None
        sensor2world_for_rays = ego2world
        if use_column_origin:
            sensor2world_for_rays = interpolate_sensor2world_columns(
                prev_sensor2world=prev_ego2world,
                current_sensor2world=ego2world,
                width=W,
            )
            ray_origin = sensor2world_for_rays[:, :3, 3]
        prev_ego2world = ego2world

        range_map, intensity_map, ray_direction = build_kitti_range_image_from_points(
            xyzs=xyzs,
            intensities=intensities,
            beam_inclinations_top_to_bottom=beam_inclinations_top_to_bottom,
            width=W,
            sensor2world=sensor2world_for_rays,
            ray_origin=ray_origin,
            max_depth=max_depth,
        )

        range_image_r1 = np.stack([range_map, intensity_map], axis=-1)
        range_image_r2 = np.zeros_like(range_image_r1)

        lidar.add_frame(
            frame,
            ego2world,
            range_image_r1,
            range_image_r2,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
        )

    return lidar, {}  # no bounding boxes
