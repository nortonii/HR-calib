# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import argparse
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_USE_CUDA_DSA"] = "1"
import torch
import torch.nn.functional as F
import yaml
from lib import dataloader
from lib.arguments import parse
from lib.dataloader import waymo_loader
from lib.dataloader.kitti_loader import load_kitti360_cameras
from lib.dataloader.kitti_calib_loader import (
    _parse_kitti_calib_file,
    _read_ply_binary,
    load_kitti_calib_cameras,
)
from lib.gaussian_renderer import raytracing
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene import Scene
from lib.scene.camera_pose_correction import CameraPoseCorrection
from lib.scene.unet import UNet
from lib.utils.console_utils import *
from lib.utils.image_utils import mse, psnr
from lib.utils.loss_utils import (
    BinaryCrossEntropyLoss,
    BinaryFocalLoss,
    l1_loss,
    l2_loss,
    phase_loss,
    ssim,
)
from lib.utils.vismatch_pose_utils import VismatchPoseEstimator
from lib.utils.record_utils import make_recorder
from ruamel.yaml import YAML
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def set_seed(seed):
    """
    Useless function, result still have a 1e-7 difference.
    Need to test problem in optix.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def _depth_to_colormap(depth_map, color_map, dmin, dmax):
    depth_map = depth_map.detach().float()
    if depth_map.dim() == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    denom = max(float(dmax - dmin), 1.0e-6)
    norm = ((depth_map - dmin) / denom).clamp(0.0, 1.0)
    img = (norm.cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img, color_map)


def _pose_checkpoint_path(model_path):
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(model_dir, model_name + "_pose.pth")


def _build_edge_distance_map(image_rgb, canny_threshold1, canny_threshold2, distance_clip):
    image_np = image_rgb.detach().cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(canny_threshold1), int(canny_threshold2))
    if np.count_nonzero(edges) == 0:
        distance = np.full(gray.shape, float(distance_clip), dtype=np.float32)
    else:
        distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        distance = np.clip(distance, 0.0, float(distance_clip)).astype(np.float32)
    return torch.from_numpy(distance)


def _select_pose_sdf_points(points_world, sensor_center, max_points):
    points_world = points_world.detach().float().cpu()
    sensor_center = torch.as_tensor(sensor_center, dtype=points_world.dtype).view(1, 3)
    valid = torch.norm(points_world - sensor_center, dim=-1) > 1.0e-3
    points_world = points_world[valid]
    if points_world.shape[0] == 0:
        return points_world.contiguous()
    if max_points <= 0 or points_world.shape[0] <= max_points:
        return points_world.contiguous()
    ranges = torch.norm(points_world - sensor_center, dim=-1)
    keep = torch.topk(ranges, k=max_points, largest=False).indices
    return points_world[keep].contiguous()


def _prepare_pose_sdf_targets(
    args,
    scene,
    cam_images,
    frame_ids,
    max_points,
    canny_threshold1,
    canny_threshold2,
    distance_clip,
):
    sdf_maps = {}
    points_world = {}
    projection_meta = {"mode": "generic_world", "intrinsics": None}
    lidar_type = str(getattr(scene.train_lidar, "data_type", "")).lower().replace("-", "").replace("_", "")
    if "kitticalib" in lidar_type:
        seq_num = int(getattr(args, "kitti_calib_scene").split("-")[0])
        calib_path = os.path.join(args.source_dir, "calibs", f"{seq_num:02d}.txt")
        P0, _ = _parse_kitti_calib_file(calib_path)
        scale = float(getattr(args, "camera_scale", 4))
        P0 = P0.copy()
        P0[0, :] /= scale
        P0[1, :] /= scale
        projection_meta = {
            "mode": "kitti_calib_raw",
            "intrinsics": torch.from_numpy(P0[:, :3]).float(),
        }
    for frame_id in frame_ids:
        if frame_id not in cam_images:
            continue
        sdf_maps[frame_id] = _build_edge_distance_map(
            cam_images[frame_id], canny_threshold1, canny_threshold2, distance_clip
        )
        if "kitticalib" in lidar_type:
            ply_path = os.path.join(
                args.source_dir,
                getattr(args, "kitti_calib_scene"),
                f"{int(frame_id):02d}.ply",
            )
            raw_points = _read_ply_binary(ply_path)
            lidar_points_world = torch.from_numpy(raw_points[:, :3]).float()
            point_center = torch.zeros(3, dtype=lidar_points_world.dtype)
        else:
            depth_map = scene.train_lidar.get_depth(frame_id)
            valid_mask = scene.train_lidar.get_mask(frame_id)
            lidar_points_world = scene.train_lidar.inverse_projection_with_range(
                frame_id, depth_map, valid_mask
            ).detach().cpu()
            point_center = scene.train_lidar.sensor_center[frame_id]
        points_world[frame_id] = _select_pose_sdf_points(
            lidar_points_world,
            point_center,
            max_points,
        )
    return sdf_maps, points_world, projection_meta


def _pose_sdf_alignment_loss(camera, sdf_map, points_world, distance_clip, depth_weight_power):
    if points_world is None or points_world.shape[0] == 0:
        return torch.zeros((), device=camera.R.device, dtype=camera.R.dtype)

    device = camera.R.device
    dtype = camera.R.dtype
    points_world = points_world.to(device=device, dtype=dtype)
    points_camera = points_world @ camera.R + camera.T
    depths = points_camera[:, 2]

    fx = camera.image_width / (2.0 * torch.tan(torch.tensor(camera.FoVx * 0.5, device=device, dtype=dtype)))
    fy = camera.image_height / (2.0 * torch.tan(torch.tensor(camera.FoVy * 0.5, device=device, dtype=dtype)))
    cx = (camera.image_width - 1.0) * 0.5
    cy = (camera.image_height - 1.0) * 0.5

    uv = torch.zeros((points_world.shape[0], 2), device=device, dtype=dtype)
    uv[:, 0] = fx * (points_camera[:, 0] / depths.clamp_min(1.0e-8)) + cx
    uv[:, 1] = fy * (points_camera[:, 1] / depths.clamp_min(1.0e-8)) + cy

    valid = (
        (depths > 1.0e-6)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.image_width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.image_height)
    )
    if not torch.any(valid):
        return torch.zeros((), device=device, dtype=dtype)

    uv = uv[valid]
    depths = depths[valid]
    sdf_map = sdf_map.to(device=device, dtype=dtype)

    grid_x = 2.0 * uv[:, 0] / max(camera.image_width - 1, 1) - 1.0
    grid_y = 2.0 * uv[:, 1] / max(camera.image_height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled_sdf = F.grid_sample(
        sdf_map.view(1, 1, camera.image_height, camera.image_width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)

    sampled_sdf = sampled_sdf / max(float(distance_clip), 1.0e-6)
    if depth_weight_power > 0.0:
        weights = depths.clamp_min(1.0e-3).pow(-float(depth_weight_power))
        weights = weights / weights.mean().clamp_min(1.0e-6)
        sampled_sdf = sampled_sdf * weights
    return sampled_sdf.mean()


def _build_lidar_depth_image_visibility_weight_map(
    lidar_sensor,
    frame_id,
    camera,
    visible_weight,
    occluded_weight,
    outside_weight,
    visibility_tolerance,
):
    depth_map = lidar_sensor.get_depth(frame_id)
    valid_mask = lidar_sensor.get_mask(frame_id)
    if not torch.is_tensor(depth_map):
        depth_map = torch.as_tensor(depth_map, dtype=torch.float32)
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    valid_mask = valid_mask.bool()
    weight_map = torch.full_like(depth_map, float(outside_weight), dtype=torch.float32)
    if not torch.any(valid_mask):
        return weight_map

    camera = camera.cuda()
    points_world = lidar_sensor.range2point(frame_id, depth_map).reshape(-1, 3)
    flat_valid = valid_mask.reshape(-1)
    points_world = points_world[flat_valid].to(device="cuda", dtype=torch.float32)
    points_camera = points_world @ camera.R + camera.T
    camera_depths = points_camera[:, 2]

    fx = camera.image_width / (
        2.0 * torch.tan(torch.tensor(camera.FoVx * 0.5, device=points_world.device, dtype=points_world.dtype))
    )
    fy = camera.image_height / (
        2.0 * torch.tan(torch.tensor(camera.FoVy * 0.5, device=points_world.device, dtype=points_world.dtype))
    )
    cx = (camera.image_width - 1.0) * 0.5
    cy = (camera.image_height - 1.0) * 0.5

    uv = torch.zeros((points_world.shape[0], 2), device=points_world.device, dtype=points_world.dtype)
    uv[:, 0] = fx * (points_camera[:, 0] / camera_depths.clamp_min(1.0e-8)) + cx
    uv[:, 1] = fy * (points_camera[:, 1] / camera_depths.clamp_min(1.0e-8)) + cy
    in_image = (
        (camera_depths > 1.0e-6)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.image_width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.image_height)
    )
    if not torch.any(in_image):
        return weight_map

    uv_in = uv[in_image].to(torch.long)
    depth_in = camera_depths[in_image]
    flat_idx = uv_in[:, 1] * int(camera.image_width) + uv_in[:, 0]
    min_depth = torch.full(
        (int(camera.image_width) * int(camera.image_height),),
        float("inf"),
        device=points_world.device,
        dtype=points_world.dtype,
    )
    min_depth.scatter_reduce_(0, flat_idx, depth_in, reduce="amin", include_self=True)
    visible_in = depth_in <= (min_depth[flat_idx] + float(visibility_tolerance))

    projected_weights = torch.full(
        (points_world.shape[0],),
        float(outside_weight),
        device=points_world.device,
        dtype=torch.float32,
    )
    projected_weights[in_image] = float(occluded_weight)
    projected_weights[in_image] = torch.where(
        visible_in,
        torch.full_like(depth_in, float(visible_weight)),
        projected_weights[in_image],
    )

    weight_map_flat = weight_map.reshape(-1)
    weight_map_flat[flat_valid] = projected_weights.detach().cpu()
    return weight_map


def _pose_sdf_alignment_loss_kitti_calib(
    pose_correction,
    frame_id,
    sdf_map,
    points_lidar,
    intrinsics,
    image_width,
    image_height,
    distance_clip,
    depth_weight_power,
):
    if points_lidar is None or points_lidar.shape[0] == 0:
        return torch.zeros((), device=pose_correction.delta_translations.device, dtype=pose_correction.delta_translations.dtype)

    device = pose_correction.delta_translations.device
    dtype = pose_correction.delta_translations.dtype
    points_lidar = points_lidar.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    extrinsic_rotation, extrinsic_translation = pose_correction.corrected_lidar_to_camera(
        frame_id, device=device
    )
    points_camera = points_lidar @ extrinsic_rotation.transpose(0, 1) + extrinsic_translation
    depths = points_camera[:, 2]
    safe_depths = torch.where(
        depths > 1.0e-6, depths, torch.ones_like(depths)
    )
    pixels_h = points_camera @ intrinsics.transpose(0, 1)
    uv = pixels_h[:, :2] / safe_depths.unsqueeze(-1)

    dx = F.relu(-uv[:, 0]) + F.relu(uv[:, 0] - (image_width - 1.0))
    dy = F.relu(-uv[:, 1]) + F.relu(uv[:, 1] - (image_height - 1.0))
    image_diag = max(float(np.hypot(image_width, image_height)), 1.0)
    boundary_penalty = torch.sqrt(dx * dx + dy * dy + 1.0e-12) / image_diag
    depth_penalty = F.relu(0.1 - depths) / 0.1
    sampled_sdf = boundary_penalty + depth_penalty

    inside = (
        (depths > 1.0e-6)
        & (dx == 0.0)
        & (dy == 0.0)
    )
    if torch.any(inside):
        uv_inside = uv[inside]
        sdf_map = sdf_map.to(device=device, dtype=dtype)
        grid_x = 2.0 * uv_inside[:, 0] / max(image_width - 1, 1) - 1.0
        grid_y = 2.0 * uv_inside[:, 1] / max(image_height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
        sampled_inside = F.grid_sample(
            sdf_map.view(1, 1, image_height, image_width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(-1)
        sampled_sdf[inside] = sampled_inside / max(float(distance_clip), 1.0e-6)

    if depth_weight_power > 0.0:
        weights = torch.where(
            depths > 1.0e-6,
            depths.clamp_min(1.0e-3).pow(-float(depth_weight_power)),
            torch.ones_like(depths),
        )
        weights = weights / weights.mean().clamp_min(1.0e-6)
        sampled_sdf = sampled_sdf * weights
    return sampled_sdf.mean()


def _save_pose_correction(model_dir, model_name, pose_correction, pose_optimizer, training_state=None):
    if pose_correction is None:
        return
    state = {
        "pose_correction": pose_correction.state_dict(),
    }
    if pose_optimizer is not None:
        state["pose_optimizer"] = pose_optimizer.state_dict()
    if training_state is not None:
        state["training_state"] = training_state
    torch.save(state, os.path.join(model_dir, model_name + "_pose.pth"))


def _apply_gaussian_training_constraints(scene, args):
    freeze_centers = bool(getattr(args.model, "freeze_gaussian_centers", False))
    if freeze_centers:
        for gaussians in scene.gaussians_assets:
            gaussians.freeze_positions(True)
        print(blue("[Gaussian] Freezing Gaussian centers: xyz positions will remain fixed during training."))


def training(args):
    first_iter = 0

    color = cv2.COLORMAP_JET

    scene = dataloader.load_scene(args.source_dir, args, test=False)
    gaussians_assets = scene.gaussians_assets
    scene.training_setup(args.opt)

    _dtype = getattr(args, "data_type", "").lower()
    if "kitticalib" in _dtype or "kitti_calib" in _dtype or "kitticalibration" in _dtype:
        dataset_prefix = "kitti_calib"
    elif "kitti" in _dtype:
        dataset_prefix = "kitti360"
    elif "pandaset" in _dtype or "panda" in _dtype:
        dataset_prefix = "pandaset"
    elif "waymo" in _dtype:
        dataset_prefix = "waymo"
    else:
        dataset_prefix = _dtype or "unknown"

    # ── Camera supervision setup ────────────────────────────────────────
    cam_cameras, cam_images = {}, {}
    camera_id = getattr(args, "camera_id", -1)
    camera_scale = getattr(args, "camera_scale", 4)
    lambda_rgb = getattr(args.opt, "lambda_rgb", 0.0)
    lambda_rgb_dssim = getattr(args.opt, "lambda_rgb_dssim", 0.2)  # SSIM weight
    lambda_rgb_phase = getattr(args.opt, "lambda_rgb_phase", 0.0)
    camera_geometry_grad_scale = float(getattr(args.opt, "camera_geometry_grad_scale", 1.0))
    lambda_depth_l1 = float(getattr(args.opt, "lambda_depth_l1", 0.0))
    enable_lidar_supervision = bool(getattr(args.opt, "enable_lidar_supervision", True))
    enable_densification = bool(getattr(args.opt, "enable_densification", True))
    use_lidar_depth_image_visibility_weights = bool(
        getattr(args.opt, "lidar_depth_use_image_visibility_weights", False)
    )
    lidar_depth_visible_weight = float(getattr(args.opt, "lidar_depth_visible_weight", 2.0))
    lidar_depth_occluded_weight = float(getattr(args.opt, "lidar_depth_occluded_weight", 0.5))
    lidar_depth_outside_weight = float(getattr(args.opt, "lidar_depth_outside_weight", 1.0))
    lidar_depth_visibility_tolerance = float(
        getattr(args.opt, "lidar_depth_visibility_tolerance", 0.25)
    )
    model_cfg = getattr(args, "model", None)
    pose_cfg = getattr(model_cfg, "pose_correction", None) if model_cfg is not None else None
    pose_sdf_weight = float(getattr(pose_cfg, "sdf_loss_weight", 0.0)) if pose_cfg is not None else 0.0
    is_kitti_family = "kitti" in _dtype.lower().replace("-", "").replace("_", "")
    camera_requested = (camera_id >= 0) or is_kitti_family
    need_camera_data = camera_requested and (
        (lambda_rgb > 0.0)
        or (pose_sdf_weight > 0.0)
        or use_lidar_depth_image_visibility_weights
    )
    use_camera_supervision = camera_requested and ((lambda_rgb > 0.0) or (pose_sdf_weight > 0.0))
    if lambda_rgb > 0.0:
        print(
            blue(
                "[Camera] RGB supervision uses a separate camera RGB SH branch; "
                "LiDAR [intensity, hit_prob, drop_prob] features remain unchanged."
            )
        )
    use_pose_correction = use_camera_supervision and bool(
        getattr(model_cfg, "use_pose_correction", False)
    )
    lidar_depth_loss_weight_maps = {}
    pose_sdf_maps = {}
    pose_sdf_points = {}
    pose_sdf_projection_meta = {"mode": "generic_world", "intrinsics": None}
    pose_sdf_bootstrap_iterations = 0
    pose_sdf_distance_clip = 64.0
    pose_sdf_depth_weight_power = 1.0
    if need_camera_data:
        if "waymo" in _dtype.lower():
            print(blue(f"[Camera] Loading Waymo camera {camera_id} at 1/{camera_scale} scale ..."))
            cam_cameras, cam_images = waymo_loader.load_waymo_cameras(
                args.source_dir, args, camera_id=camera_id, scale=camera_scale
            )
            first_img = next(v for v in cam_images.values())
            print(blue(f"[Camera] Loaded {len(cam_cameras)} camera frames. "
                       f"Resolution: {first_img.shape[:2]}"))
        elif "kitticalib" in _dtype.lower().replace("-", "").replace("_", ""):
            scene_name = getattr(args, "kitti_calib_scene", None)
            if scene_name is None:
                print(red("[Camera] kitti-calib camera supervision requires kitti_calib_scene. Disabling."))
                use_camera_supervision = False
            else:
                print(blue(f"[Camera] Loading kitti-calib camera for scene '{scene_name}' "
                           f"at 1/{camera_scale} scale ..."))
                frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
                cam_cameras, cam_images = load_kitti_calib_cameras(
                    args.source_dir, args,
                    scene_name=scene_name,
                    frame_ids=frame_ids,
                    scale=camera_scale,
                )
        elif "kitti" in _dtype.lower():
            kitti_seq = getattr(args, "kitti_seq", None)
            if kitti_seq is None:
                print(red("[Camera] KITTI camera supervision requires kitti_seq to be set. Disabling."))
                use_camera_supervision = False
            else:
                print(blue(f"[Camera] Loading KITTI-360 camera (image_00) for seq {kitti_seq} "
                           f"at 1/{camera_scale} scale ..."))
                frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
                cam_cameras, cam_images = load_kitti360_cameras(
                    args.source_dir, args,
                    seq_num=int(kitti_seq),
                    frame_ids=frame_ids,
                    scale=camera_scale,
                )
        else:
            print(red(f"[Camera] Camera supervision not implemented for dataset type '{_dtype}'. Disabling."))
            use_camera_supervision = False
            need_camera_data = False

    if use_lidar_depth_image_visibility_weights and cam_cameras:
        print(
            blue(
                "[LiDAR] Image-visibility weighted depth loss enabled "
                f"(visible={lidar_depth_visible_weight:.2f}, "
                f"occluded={lidar_depth_occluded_weight:.2f}, "
                f"outside={lidar_depth_outside_weight:.2f}, "
                f"tol={lidar_depth_visibility_tolerance:.2f}m)"
            )
        )
        for frame_id, camera in cam_cameras.items():
            lidar_depth_loss_weight_maps[frame_id] = _build_lidar_depth_image_visibility_weight_map(
                scene.train_lidar,
                frame_id,
                camera,
                lidar_depth_visible_weight,
                lidar_depth_occluded_weight,
                lidar_depth_outside_weight,
                lidar_depth_visibility_tolerance,
            )

    pose_correction = None
    pose_optimizer = None
    pose_matcher = None
    pose_accum_steps = 1
    pose_accum_counter = 0
    pose_stage_translation = False
    pose_translation_enabled = True
    pose_rotation_unlock_deg = 0.0
    pose_rotation_unlock_patience = 1
    pose_rotation_unlock_count = 0
    if use_pose_correction and cam_cameras:
        pose_update_method = (
            str(getattr(pose_cfg, "update_method", "gradient")).lower()
            if pose_cfg is not None else "gradient"
        )
        if pose_update_method not in {"gradient", "vismatch"}:
            raise ValueError(
                f"Unsupported pose_correction.update_method '{pose_update_method}'. "
                "Expected 'gradient' or 'vismatch'."
            )
        lidar_poses = None
        if "kitticalib" in _dtype.lower().replace("-", "").replace("_", ""):
            lidar_poses = scene.train_lidar.sensor2world
        pose_correction = CameraPoseCorrection(cam_cameras, pose_cfg, lidar_poses=lidar_poses).cuda()
        pose_correction.use_gt_translation = bool(getattr(pose_cfg, "use_gt_translation", False))
        pose_mode = str(getattr(pose_cfg, "mode", "frame")).lower() if pose_cfg is not None else "frame"
        pose_accum_steps = max(1, int(getattr(pose_cfg, "accumulate_steps", 1)))
        if pose_update_method == "gradient":
            pose_optimizer = torch.optim.Adam(
                [
                    {
                        "params": [pose_correction.delta_translations],
                        "lr": float(getattr(pose_cfg, "translation_lr", 5.0e-4)),
                        "name": "camera_translation",
                    },
                    {
                        "params": [pose_correction.delta_rotations_6d],
                        "lr": float(getattr(pose_cfg, "rotation_lr", 2.0e-4)),
                        "name": "camera_rotation",
                    },
                ],
                eps=1e-15,
            )
        else:
            if not pose_correction.use_shared_lidar_extrinsic:
                raise RuntimeError(
                    "pose_correction.update_method=vismatch currently requires "
                    "shared-extrinsic mode (e.g. kitti_calib with pose_correction.mode=all)."
                )
            pose_matcher = VismatchPoseEstimator(pose_cfg, device="cuda")
        pose_stage_translation = bool(getattr(pose_cfg, "stage_translation", False))
        pose_translation_enabled = not pose_stage_translation
        if pose_correction.use_gt_translation:
            pose_translation_enabled = False
        pose_rotation_unlock_deg = float(
            getattr(pose_cfg, "translation_warmup_rotation_error_deg", 3.0)
        )
        pose_rotation_unlock_patience = max(
            1, int(getattr(pose_cfg, "translation_warmup_patience", 5))
        )
        if pose_stage_translation and pose_optimizer is not None:
            for group in pose_optimizer.param_groups:
                if group.get("name") == "camera_translation":
                    group["lr"] = 0.0
        if pose_correction.use_gt_translation and pose_optimizer is not None:
            for group in pose_optimizer.param_groups:
                if group.get("name") == "camera_translation":
                    group["lr"] = 0.0
        pose_count = int(pose_correction.delta_translations.shape[0])
        print(blue(
            f"[Camera] Learnable relative pose enabled for {len(cam_cameras)} frames "
            f"with {pose_count} shared pose parameter set(s) "
            f"(mode={pose_mode}, update={pose_update_method}, accumulate_steps={pose_accum_steps})."
        ))
        if pose_stage_translation and pose_optimizer is not None:
            print(
                blue(
                    "[Camera] Translation warmup enabled: rotation-only updates until "
                    f"shared rotation error <= {pose_rotation_unlock_deg:.2f} deg for "
                    f"{pose_rotation_unlock_patience} pose step(s)."
                )
            )
        if pose_correction.use_gt_translation:
            print(blue("[Camera] Using ground-truth shared translation; only rotation is optimized."))
        if pose_sdf_weight > 0.0:
            pose_sdf_bootstrap_iterations = int(getattr(pose_cfg, "sdf_bootstrap_iterations", 0))
            pose_sdf_distance_clip = float(getattr(pose_cfg, "sdf_distance_clip", 64.0))
            pose_sdf_depth_weight_power = float(getattr(pose_cfg, "sdf_depth_weight_power", 1.0))
            pose_sdf_maps, pose_sdf_points, pose_sdf_projection_meta = _prepare_pose_sdf_targets(
                args,
                scene,
                cam_images,
                cam_cameras.keys(),
                int(getattr(pose_cfg, "sdf_max_points", 20000)),
                int(getattr(pose_cfg, "sdf_canny_threshold1", 100)),
                int(getattr(pose_cfg, "sdf_canny_threshold2", 200)),
                pose_sdf_distance_clip,
            )
            window_msg = "all iterations" if pose_sdf_bootstrap_iterations <= 0 else f"first {pose_sdf_bootstrap_iterations} iterations"
            print(
                blue(
                    f"[Camera] Pose SDF bootstrap enabled for {len(pose_sdf_maps)} frames "
                    f"({window_msg}, weight={pose_sdf_weight}, max_points={int(getattr(pose_cfg, 'sdf_max_points', 20000))})."
                )
            )

    log = {
        "depth_rmse": [],
        "points_num": [],
        "clone_sum": [],
        "split_sum": [],
        "prune_scale_sum": [],
        "prune_opacity_sum": [],
    }
    scene_id = str(args.scene_id) if isinstance(args.scene_id, int) else args.scene_id
    output_dir = os.path.join(
        args.model_dir, args.task_name, args.exp_name, "scene_" + scene_id
    )
    record_dir = os.path.join(output_dir, "records")
    recorder = make_recorder(args, record_dir)
    print(
        blue(
            f"Task: {args.task_name}, Experiment: {args.exp_name}, Scene: {args.scene_id}"
        )
    )
    print("Output dir: ", output_dir)

    if args.model_path:
        (model_params, first_iter) = torch.load(args.model_path)
        scene.restore(model_params, args.opt)
        if bool(getattr(args.model, "dc_only_sh", False)):
            print(blue("[Gaussian] DC-only SH mode enabled: higher-order SH coefficients are frozen and ignored."))
        if pose_correction is not None:
            pose_checkpoint_path = _pose_checkpoint_path(args.model_path)
            if os.path.exists(pose_checkpoint_path):
                pose_checkpoint = torch.load(pose_checkpoint_path)
                pose_correction.load_state_dict(pose_checkpoint["pose_correction"])
                if pose_optimizer is not None and "pose_optimizer" in pose_checkpoint:
                    pose_optimizer.load_state_dict(pose_checkpoint["pose_optimizer"])
                training_state = pose_checkpoint.get("training_state", {})
                pose_translation_enabled = bool(
                    training_state.get("pose_translation_enabled", pose_translation_enabled)
                )
                pose_rotation_unlock_count = int(
                    training_state.get("pose_rotation_unlock_count", pose_rotation_unlock_count)
                )
                if pose_stage_translation and not pose_translation_enabled and pose_optimizer is not None:
                    for group in pose_optimizer.param_groups:
                        if group.get("name") == "camera_translation":
                            group["lr"] = 0.0
                print(blue(f"[Camera] Loaded pose correction checkpoint: {pose_checkpoint_path}"))
            else:
                print(red(f"[Camera] Pose correction checkpoint not found: {pose_checkpoint_path}"))
        log_path = os.path.join(output_dir, "logs/log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as json_file:
                log = json.load(json_file)
    print("Continuing from iteration ", first_iter)
    _apply_gaussian_training_constraints(scene, args)
    if not enable_lidar_supervision:
        print(blue("[LiDAR] LiDAR supervision disabled: skipping LiDAR ray tracing and LiDAR depth loss during training."))
    if not enable_densification:
        print(blue("[Gaussian] Densification disabled: clone/split/prune and opacity reset are frozen."))
    if camera_geometry_grad_scale != 1.0:
        print(
            blue(
                "[Camera] Scaling camera-loss gradients on Gaussian geometry "
                f"(xyz/scale/rotation/opacity) by {camera_geometry_grad_scale:.3f}."
            )
        )

    # bg_color = [1, 1, 1] if args.model.white_background else [0, 0, 0]
    background = torch.tensor(
        [0, 0, 1], device="cuda"
    ).float()  # background (intensity, hit prob, drop prob)

    BFLoss = BinaryFocalLoss()
    BCELoss = BinaryCrossEntropyLoss()
    frame_stack = []

    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        initial=first_iter, total=args.opt.iterations, desc="Training progress"
    )
    first_iter += 1

    end = time.time()
    frame_s, frame_e = args.frame_length[0], args.frame_length[1]
    render_cams = []
    best_mix_metric = 0
    depth_supervision_interval = max(1, int(getattr(args.opt, "depth_supervision_interval", 1)))
    for iteration in range(first_iter, args.opt.iterations + 1):
        if args.only_refine:
            break
        recorder.step += 1

        scene.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            scene.oneupSHdegree()

        # Pick a random frame
        if not frame_stack:
            frame_stack = list(scene.train_lidar.train_frames)
            random.shuffle(frame_stack)
        frame = frame_stack.pop()
        data_time = time.time() - end

        # Render
        if args.pipe.debug_from and (iteration - 1) == args.pipe.debug_from:
            args.pipe.debug = True

        densify_grad_source = str(getattr(args.opt, "densify_grad_source", "lidar")).lower()
        do_depth_supervision = enable_lidar_supervision and (lambda_depth_l1 > 0.0) and (
            (depth_supervision_interval <= 1) or (((iteration - 1) % depth_supervision_interval) == 0)
        )
        densify_mean_grads = None
        densify_update_filter = None
        depth = None
        gt_depth = None
        static_mask = None

        if do_depth_supervision:
            render_pkg = raytracing(
                frame, gaussians_assets, scene.train_lidar, background, args, depth_only=True
            )
            batch_time = time.time() - end
            depth = render_pkg["depth"]
            intensity = render_pkg["intensity"]
            raydrop_prob = render_pkg["raydrop"]
            means3d = render_pkg["means3D"]
            acc_wet = render_pkg["accum_gaussian_weight"]

            if densify_grad_source != "camera":
                densify_mean_grads = means3d.grad
                densify_update_filter = acc_wet > 0

            gt_mask = scene.train_lidar.get_mask(frame).cuda()
            dynamic_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
            static_mask = gt_mask & ~dynamic_mask

            depth = depth.squeeze(-1)
            gt_depth = scene.train_lidar.get_depth(frame).cuda()
            if use_lidar_depth_image_visibility_weights and frame in lidar_depth_loss_weight_maps:
                depth_weights = lidar_depth_loss_weight_maps[frame].cuda()
                supervised_mask = static_mask & (depth_weights > 0.0)
                if torch.any(supervised_mask):
                    depth_residual = torch.abs(depth[supervised_mask] - gt_depth[supervised_mask])
                    supervised_weights = depth_weights[supervised_mask]
                    loss_depth = lambda_depth_l1 * (
                        (depth_residual * supervised_weights).sum()
                        / supervised_weights.sum().clamp_min(1.0e-8)
                    )
                    visible_mask = supervised_mask & (depth_weights == float(lidar_depth_visible_weight))
                    lidar_depth_visible_ratio = (
                        visible_mask.float().sum()
                        / supervised_mask.float().sum().clamp_min(1.0)
                    )
                else:
                    loss_depth = torch.tensor(0.0, device="cuda")
            else:
                loss_depth = lambda_depth_l1 * l1_loss(
                    depth[static_mask], gt_depth[static_mask]
                )
        else:
            batch_time = time.time() - end
            loss_depth = torch.tensor(0.0, device="cuda")

        # === Intensity loss (disabled) ===
        loss_intensity = torch.tensor(0.0, device="cuda")

        # === Raydrop loss (disabled) ===
        loss_raydrop = torch.tensor(0.0, device="cuda")

        # === regularization loss ===
        loss_reg = 0
        for gaussians in gaussians_assets:
            loss_reg += args.opt.lambda_reg * gaussians.box_reg_loss()

        # === Camera RGB supervision (standard 3DGS rasterization) ===
        # Backward separately BEFORE the LiDAR backward to avoid CUDA kernel conflicts
        # between the OptiX tracer and the tile rasterizer sharing the same parameters.
        loss_rgb = torch.tensor(0.0, device="cuda")
        loss_rgb_phase = torch.tensor(0.0, device="cuda")
        loss_pose_sdf = torch.tensor(0.0, device="cuda")
        cam_psnr = torch.tensor(0.0, device="cuda")
        pose_reg = torch.tensor(0.0, device="cuda")
        pose_delta_trans_error = torch.tensor(0.0, device="cuda")
        pose_delta_rot_error_deg = torch.tensor(0.0, device="cuda")
        pose_delta_rot_euler_mean_deg = torch.tensor(0.0, device="cuda")
        pose_delta_trans_error_x = torch.tensor(0.0, device="cuda")
        pose_delta_trans_error_y = torch.tensor(0.0, device="cuda")
        pose_delta_trans_error_z = torch.tensor(0.0, device="cuda")
        pose_translation_grad_norm = torch.tensor(0.0, device="cuda")
        pose_rotation_grad_norm = torch.tensor(0.0, device="cuda")
        pose_translation_unlocked = torch.tensor(
            1.0 if pose_translation_enabled else 0.0, device="cuda"
        )
        pose_match_attempted = torch.tensor(0.0, device="cuda")
        pose_match_success = torch.tensor(0.0, device="cuda")
        pose_match_num_matches = torch.tensor(0.0, device="cuda")
        pose_match_num_raw_matches = torch.tensor(0.0, device="cuda")
        pose_match_num_h_inliers = torch.tensor(0.0, device="cuda")
        pose_match_num_depth_matches = torch.tensor(0.0, device="cuda")
        pose_match_num_pnp_inliers = torch.tensor(0.0, device="cuda")
        pose_match_reproj_error = torch.tensor(0.0, device="cuda")
        pose_match_success_ratio = torch.tensor(0.0, device="cuda")
        pose_match_best_frame = torch.tensor(-1.0, device="cuda")
        lidar_depth_visible_ratio = torch.tensor(0.0, device="cuda")
        visual_cam_depth_match = None
        visual_cam_depth_match_gt = None
        if pose_optimizer is not None and pose_accum_counter == 0:
            pose_optimizer.zero_grad(set_to_none=True)
        if use_camera_supervision and frame in cam_cameras:
            if pose_correction is not None:
                camera = cam_cameras[frame].cuda()
                corrected_camera = pose_correction.corrected_camera(
                    cam_cameras[frame], device="cuda"
                )
                gt_rgb = cam_images[frame].cuda()
                if pose_matcher is not None and ((iteration - 1) % pose_matcher.update_interval == 0):
                    match_results = []
                    for match_frame in sorted(cam_cameras.keys()):
                        match_camera = cam_cameras[match_frame].cuda()
                        match_corrected_camera = pose_correction.corrected_camera(
                            cam_cameras[match_frame], device="cuda"
                        )
                        match_gt_rgb = cam_images[match_frame].cuda()
                        with torch.no_grad():
                            match_render = render_camera(
                                match_camera,
                                gaussians_assets,
                                args,
                                cam_rotation=match_corrected_camera.R.detach(),
                                cam_translation=match_corrected_camera.T.detach(),
                                require_rgb=False,
                            )
                        pose_match_frame = pose_matcher.estimate_relative_pose(
                            match_render["depth"],
                            match_gt_rgb,
                            match_corrected_camera,
                        )
                        pose_match_frame["frame_id"] = int(match_frame)
                        match_results.append(pose_match_frame)
                        del match_render
                        del match_gt_rgb
                        del match_corrected_camera
                        del match_camera
                        if ((len(match_results) % 5) == 0):
                            torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    pose_match = pose_matcher.aggregate_relative_pose_estimates(match_results)
                    pose_match_num_matches = torch.tensor(
                        float(pose_match.get("mean_num_matches", 0.0)), device="cuda"
                    )
                    pose_match_num_raw_matches = torch.tensor(
                        float(pose_match.get("mean_num_raw_matches", 0.0)), device="cuda"
                    )
                    pose_match_num_h_inliers = torch.tensor(
                        float(pose_match.get("mean_num_h_inliers", 0.0)), device="cuda"
                    )
                    pose_match_num_depth_matches = torch.tensor(
                        float(pose_match.get("mean_num_depth_matches", 0.0)), device="cuda"
                    )
                    pose_match_num_pnp_inliers = torch.tensor(
                        float(pose_match.get("mean_num_pnp_inliers", 0.0)), device="cuda"
                    )
                    pose_match_reproj_error = torch.tensor(
                        float(pose_match.get("mean_reproj_error", 0.0)), device="cuda"
                    )
                    pose_match_attempted = torch.tensor(
                        float(pose_match.get("attempted_frames", 0.0)), device="cuda"
                    )
                    pose_match_success = torch.tensor(
                        float(pose_match.get("successful_frames", 0.0)), device="cuda"
                    )
                    pose_match_success_ratio = torch.tensor(
                        float(pose_match.get("success_ratio", 0.0)), device="cuda"
                    )
                    pose_match_best_frame = torch.tensor(
                        float(pose_match.get("best_frame_id", -1)), device="cuda"
                    )
                    visual_cam_depth_match = pose_match.get("best_depth_vis")
                    best_frame_id = int(pose_match.get("best_frame_id", -1))
                    if best_frame_id in cam_images:
                        visual_cam_depth_match_gt = cam_images[best_frame_id].numpy()
                    if pose_match.get("success", False):
                        pose_correction.apply_relative_camera_transform(
                            frame,
                            pose_match["relative_rotation"],
                            pose_match["relative_translation"],
                        )
                        corrected_camera = pose_correction.corrected_camera(
                            cam_cameras[frame], device="cuda"
                        )
                pose_rotation, pose_translation = pose_correction.corrected_rt(frame, device="cuda")
                pose_reg = pose_correction.regularization_loss(frame, pose_cfg)
                if pose_matcher is not None:
                    pose_rotation = pose_rotation.detach()
                    pose_translation = pose_translation.detach()
                    pose_reg = pose_reg.detach()
            else:
                camera = cam_cameras[frame].cuda()
                corrected_camera = camera
                pose_rotation, pose_translation = None, None
                gt_rgb = cam_images[frame].cuda()
            total_camera_loss = pose_reg
            if lambda_rgb > 0.0:
                cam_render = render_camera(
                    camera, gaussians_assets, args,
                    cam_rotation=pose_rotation, cam_translation=pose_translation,
                    require_rgb=True,
                )
                pred_rgb = cam_render["rgb"].clamp(0.0, 1.0)  # (H, W, 3)
                if cam_render["num_visible"] > 0:
                    pred_chw = pred_rgb.permute(2, 0, 1)
                    gt_chw = gt_rgb.permute(2, 0, 1)
                    Ll1 = l1_loss(pred_rgb, gt_rgb)
                    ssim_val = ssim(pred_chw, gt_chw)
                    loss_rgb = lambda_rgb * (
                        (1.0 - lambda_rgb_dssim) * Ll1 + lambda_rgb_dssim * (1.0 - ssim_val)
                    )
                    if lambda_rgb_phase > 0.0:
                        loss_rgb_phase = float(lambda_rgb_phase) * phase_loss(pred_chw, gt_chw)
                        loss_rgb = loss_rgb + loss_rgb_phase
                    cam_psnr = psnr(pred_chw, gt_chw).detach()
                    total_camera_loss = total_camera_loss + loss_rgb
                    if densify_grad_source == "camera":
                        densify_mean_grads = cam_render["screenspace_points"].grad
                        densify_update_filter = cam_render["visibility_filter"]
            use_pose_sdf = (
                pose_sdf_weight > 0.0
                and frame in pose_sdf_maps
                and pose_correction is not None
                and pose_matcher is None
                and (pose_sdf_bootstrap_iterations <= 0 or iteration <= pose_sdf_bootstrap_iterations)
            )
            if use_pose_sdf:
                if pose_sdf_projection_meta["mode"] == "kitti_calib_raw":
                    loss_pose_sdf = float(pose_sdf_weight) * _pose_sdf_alignment_loss_kitti_calib(
                        pose_correction,
                        frame,
                        pose_sdf_maps[frame],
                        pose_sdf_points.get(frame),
                        pose_sdf_projection_meta["intrinsics"],
                        corrected_camera.image_width,
                        corrected_camera.image_height,
                        pose_sdf_distance_clip,
                        pose_sdf_depth_weight_power,
                    )
                else:
                    loss_pose_sdf = float(pose_sdf_weight) * _pose_sdf_alignment_loss(
                        corrected_camera,
                        pose_sdf_maps[frame],
                        pose_sdf_points.get(frame),
                        pose_sdf_distance_clip,
                        pose_sdf_depth_weight_power,
                    )
                total_camera_loss = total_camera_loss + loss_pose_sdf
            if total_camera_loss.requires_grad:
                total_camera_loss.backward()
                if camera_geometry_grad_scale != 1.0:
                    # Scale only the Gaussian parameter grads coming from the camera loss.
                    # Pose-correction grads are left untouched because they have already
                    # been written to pose_correction.*.grad by backward().
                    for gaussians in gaussians_assets:
                        gaussians.scale_optimizer_gradients(camera_geometry_grad_scale)
                if pose_optimizer is not None:
                    if not pose_translation_enabled and pose_correction.delta_translations.grad is not None:
                        pose_correction.delta_translations.grad.zero_()
                    pose_correction.sanitize_gradients()
                    pose_accum_counter += 1
                    grad_divisor = float(pose_accum_counter)
                    if pose_correction.delta_translations.grad is not None:
                        pose_translation_grad_norm = (
                            pose_correction.delta_translations.grad.norm() / grad_divisor
                        ).detach()
                    if pose_correction.delta_rotations_6d.grad is not None:
                        pose_rotation_grad_norm = (
                            pose_correction.delta_rotations_6d.grad.norm() / grad_divisor
                        ).detach()
                    if pose_accum_counter >= pose_accum_steps or iteration == args.opt.iterations:
                        for group in pose_optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is not None:
                                    param.grad.div_(grad_divisor)
                        if not pose_translation_enabled:
                            for group in pose_optimizer.param_groups:
                                if group.get("name") == "camera_translation":
                                    group["lr"] = 0.0
                        pose_optimizer.step()
                        sanitized_pose_num = pose_correction.sanitize_parameters()
                        if sanitized_pose_num > 0:
                            progress_bar.write(
                                red(f"[ITER {iteration}] Sanitized {sanitized_pose_num} non-finite pose parameters.")
                            )
                        pose_optimizer.zero_grad(set_to_none=True)
                        pose_accum_counter = 0
                torch.cuda.synchronize()
                loss_rgb = loss_rgb.detach()
                loss_pose_sdf = loss_pose_sdf.detach()
                pose_reg = pose_reg.detach()
            if pose_correction is not None:
                pose_delta_error = pose_correction.delta_pose_error(frame, device="cuda")
                pose_delta_trans_error = pose_delta_error["translation_error_norm"]
                pose_delta_rot_error_deg = pose_delta_error["rotation_error_deg"]
                pose_delta_rot_euler_mean_deg = pose_delta_error["rotation_euler_error_deg"].mean()
                pose_delta_trans_error_x = pose_delta_error["translation_error"][0]
                pose_delta_trans_error_y = pose_delta_error["translation_error"][1]
                pose_delta_trans_error_z = pose_delta_error["translation_error"][2]
                if pose_optimizer is not None:
                    if pose_stage_translation and not pose_translation_enabled and not pose_correction.use_gt_translation:
                        if float(pose_delta_rot_error_deg.item()) <= pose_rotation_unlock_deg:
                            pose_rotation_unlock_count += 1
                        else:
                            pose_rotation_unlock_count = 0
                        if pose_rotation_unlock_count >= pose_rotation_unlock_patience:
                            pose_translation_enabled = True
                            pose_translation_unlocked = torch.tensor(1.0, device="cuda")
                            for group in pose_optimizer.param_groups:
                                if group.get("name") == "camera_translation":
                                    group["lr"] = float(getattr(pose_cfg, "translation_lr", 5.0e-4))
                            progress_bar.write(
                                blue(
                                    f"[ITER {iteration}] Unlocking pose translation after rotation "
                                    f"error reached {float(pose_delta_rot_error_deg.item()):.4f} deg."
                                )
                            )
                    pose_translation_unlocked = torch.tensor(
                        1.0 if pose_translation_enabled else 0.0, device="cuda"
                    )

        loss = loss_depth + loss_reg
        if loss.requires_grad:
            loss.backward()

        with torch.no_grad():
            densify_info = scene.optimize(
                args, iteration, densify_mean_grads, densify_update_filter, None, None
            )
            prune_nonfinite_num = densify_info[4] if len(densify_info) > 4 else 0

            points_num = 0
            for i in gaussians_assets:
                points_num += i.get_local_xyz.shape[0]
            if do_depth_supervision:
                depth_mse = mse(depth[static_mask], gt_depth[static_mask]).mean().item() ** 0.5
            else:
                depth_mse = log["depth_rmse"][-1] if log["depth_rmse"] else 0.0
            clone_sum = (
                densify_info[0] + log["clone_sum"][-1]
                if log["clone_sum"]
                else densify_info[0]
            )
            split_sum = (
                densify_info[1] + log["split_sum"][-1]
                if log["split_sum"]
                else densify_info[1]
            )
            prune_scale_sum = (
                densify_info[2] + log["prune_scale_sum"][-1]
                if log["prune_scale_sum"]
                else densify_info[2]
            )
            prune_opacity_sum = (
                densify_info[3] + log["prune_opacity_sum"][-1]
                if log["prune_opacity_sum"]
                else densify_info[3]
            )
            if prune_nonfinite_num > 0:
                progress_bar.write(
                    red(f"[ITER {iteration}] Pruned {prune_nonfinite_num} non-finite Gaussian primitives.")
                )
            log["depth_rmse"].append(depth_mse)
            log["points_num"].append(points_num)
            log["clone_sum"].append(clone_sum)
            log["split_sum"].append(split_sum)
            log["prune_scale_sum"].append(prune_scale_sum)
            log["prune_opacity_sum"].append(prune_opacity_sum)

            logged_loss_rgb = loss_rgb
            logged_cam_psnr = cam_psnr
            visual_cam_gt = None
            visual_cam_pred = None
            if iteration % args.visual_interval == 0 and use_camera_supervision and frame in cam_cameras:
                cam_vis = cam_cameras[frame].cuda()
                if pose_correction is not None:
                    vis_rotation, vis_translation = pose_correction.corrected_rt(frame, device="cuda")
                else:
                    vis_rotation, vis_translation = None, None
                gt_cam_tensor = cam_images[frame].cuda()
                cam_r = render_camera(
                    cam_vis, gaussians_assets, args,
                    cam_rotation=vis_rotation, cam_translation=vis_translation,
                    require_rgb=True,
                )
                pred_cam_tensor = cam_r["rgb"].clamp(0, 1)
                if cam_r["num_visible"] > 0:
                    pred_cam_chw = pred_cam_tensor.permute(2, 0, 1)
                    gt_cam_chw = gt_cam_tensor.permute(2, 0, 1)
                    Ll1_vis = l1_loss(pred_cam_tensor, gt_cam_tensor)
                    ssim_vis = ssim(pred_cam_chw, gt_cam_chw)
                    # Recompute the recorded camera metrics after optimize() so they
                    # stay aligned with the saved cam_compare image for this step.
                    logged_loss_rgb = lambda_rgb * (
                        (1.0 - lambda_rgb_dssim) * Ll1_vis + lambda_rgb_dssim * (1.0 - ssim_vis)
                    )
                    logged_cam_psnr = psnr(pred_cam_chw, gt_cam_chw)
                visual_cam_gt = cam_images[frame].numpy()
                visual_cam_pred = pred_cam_tensor.cpu().numpy()

            # prepare loss stats for tensorboard record
            loss_stats = {
                "all_loss": loss,
                "depth_loss": loss_depth,
                "rgb_loss": logged_loss_rgb,
                "rgb_phase_loss": loss_rgb_phase.detach(),
                "pose_sdf_loss": loss_pose_sdf,
                "cam_psnr": logged_cam_psnr,
                "pose_reg": pose_reg,
                "cam_pose_delta_trans_error": pose_delta_trans_error,
                "cam_pose_delta_trans_error_x": pose_delta_trans_error_x,
                "cam_pose_delta_trans_error_y": pose_delta_trans_error_y,
                "cam_pose_delta_trans_error_z": pose_delta_trans_error_z,
                "cam_pose_delta_rot_error_deg": pose_delta_rot_error_deg,
                "cam_pose_delta_rot_euler_mean_deg": pose_delta_rot_euler_mean_deg,
                "cam_pose_translation_grad_norm": pose_translation_grad_norm,
                "cam_pose_rotation_grad_norm": pose_rotation_grad_norm,
                "cam_pose_translation_unlocked": pose_translation_unlocked,
                "cam_pose_match_attempted": pose_match_attempted,
                "cam_pose_match_success": pose_match_success,
                "cam_pose_match_num_matches": pose_match_num_matches,
                "cam_pose_match_num_raw_matches": pose_match_num_raw_matches,
                "cam_pose_match_num_h_inliers": pose_match_num_h_inliers,
                "cam_pose_match_num_depth_matches": pose_match_num_depth_matches,
                "cam_pose_match_num_pnp_inliers": pose_match_num_pnp_inliers,
                "cam_pose_match_reproj_error": pose_match_reproj_error,
                "cam_pose_match_success_ratio": pose_match_success_ratio,
                "cam_pose_match_best_frame": pose_match_best_frame,
                "lidar_depth_visible_ratio": lidar_depth_visible_ratio,
                "ema_loss": 0.4 * loss + 0.6 * ema_loss_for_log,
                "points_num": torch.tensor(points_num).float(),
                "depth_rmse": torch.tensor(depth_mse).float(),
            }

            reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
            recorder.update_loss_stats(reduced_losses)

            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            recorder.record(f"{dataset_prefix}/train")

            if iteration % args.visual_interval == 0:
                render_pkg = raytracing(
                    frame_s, gaussians_assets, scene.train_lidar, background, args
                )
                rendered_depth = render_pkg["depth"]

                pred_depth = rendered_depth.squeeze(-1)

                gt_depth = scene.train_lidar.get_depth(frame_s).cuda().float()
                gt_mask = scene.train_lidar.get_mask(frame_s).cuda().bool()
                valid = gt_mask & torch.isfinite(gt_depth) & (gt_depth > 0.0)
                if torch.any(valid):
                    dmin = float(gt_depth[valid].min().item())
                    dmax = float(gt_depth[valid].max().item())
                else:
                    dmin = float(pred_depth.min().item())
                    dmax = float(pred_depth.max().item())
                gt_depth_vis = _depth_to_colormap(gt_depth, color, dmin, dmax)
                pred_depth_cmp = _depth_to_colormap(pred_depth, color, dmin, dmax)
                gt_pred_depth = np.concatenate([gt_depth_vis, pred_depth_cmp], axis=1)
                depth_cmp_dir = os.path.join(output_dir, "depth_compare")
                os.makedirs(depth_cmp_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(depth_cmp_dir, str(iteration) + "_gt_pred_depth.png"),
                    gt_pred_depth,
                )

                # Save camera RGB comparison if camera supervision is on
                if visual_cam_gt is not None and visual_cam_pred is not None:
                    side = np.concatenate(
                        [(visual_cam_gt * 255).astype(np.uint8),
                         (visual_cam_pred * 255).astype(np.uint8)], axis=1
                    )
                    cam_cmp_dir = os.path.join(output_dir, "cam_compare")
                    os.makedirs(cam_cmp_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(cam_cmp_dir, str(iteration) + "_gt_pred_cam.png"),
                        cv2.cvtColor(side, cv2.COLOR_RGB2BGR),
                    )
                if visual_cam_depth_match is not None and visual_cam_depth_match_gt is not None:
                    match_side = np.concatenate(
                        [
                            (visual_cam_depth_match * 255).astype(np.uint8),
                            (visual_cam_depth_match_gt * 255).astype(np.uint8),
                        ],
                        axis=1,
                    )
                    cam_match_dir = os.path.join(output_dir, "cam_match_compare")
                    os.makedirs(cam_match_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(cam_match_dir, str(iteration) + "_depth_rgb_match.png"),
                        cv2.cvtColor(match_side, cv2.COLOR_RGB2BGR),
                    )

            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log.item():.{5}f}",
                        # "L_all": f"{loss.item():.{5}f}",
                        # "L_depth": f"{loss_depth.item():.{5}f}",
                        # "L_intensity": f"{loss_intensity.item():.{5}f}",
                        # "L_raydrop": f"{loss_raydrop.item():.{5}f}",
                        "points": f"{points_num}",
                        "exp": args.exp_name,
                        "scene": args.scene_id,
                    }
                )
                progress_bar.update(10)
            if iteration == args.opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in args.saving_iterations:
                progress_bar.write("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, "model_it_" + str(iteration))
                _save_pose_correction(
                    scene.model_save_dir,
                    "model_it_" + str(iteration),
                    pose_correction,
                    pose_optimizer,
                    {
                        "pose_translation_enabled": pose_translation_enabled,
                        "pose_rotation_unlock_count": pose_rotation_unlock_count,
                    },
                )

            if iteration % args.testing_iterations == 0:
                if iteration >= args.saving_iterations[0] - 3000:
                    total_sq_err = 0.0
                    total_pts = 0
                    for frame in scene.train_lidar.eval_frames:
                        render_pkg = raytracing(
                            frame, gaussians_assets, scene.train_lidar, background, args
                        )
                        depth = render_pkg["depth"].detach()

                        gt_depth = scene.train_lidar.get_depth(frame).cuda()
                        gt_mask = scene.train_lidar.get_mask(frame).cuda()
                        dyn_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
                        static_mask = gt_mask & ~dyn_mask
                        diff = depth[..., 0][static_mask] - gt_depth[static_mask]
                        total_sq_err += (diff ** 2).sum().item()
                        total_pts += static_mask.sum().item()
                    global_rmse = (total_sq_err / max(total_pts, 1)) ** 0.5
                    mix_metric = -global_rmse  # lower RMSE = better
                    progress_bar.write(f"\n[ITER {iteration}] Global depth RMSE: {global_rmse:.4f} m (best: {-best_mix_metric:.4f} m)")
                    recorder.writer.add_scalar(f"{dataset_prefix}/eval/depth_rmse", global_rmse, iteration)
                    if mix_metric > best_mix_metric:
                        for file in os.listdir(scene.model_save_dir):
                            if file.endswith(".pth") and "ckpt_it_" in file:
                                os.remove(os.path.join(scene.model_save_dir, file))
                        best_mix_metric = mix_metric
                        scene.save(iteration, "ckpt_it_" + str(iteration) + "_good")
                        _save_pose_correction(
                            scene.model_save_dir,
                            "ckpt_it_" + str(iteration) + "_good",
                            pose_correction,
                            pose_optimizer,
                            {
                                "pose_translation_enabled": pose_translation_enabled,
                                "pose_rotation_unlock_count": pose_rotation_unlock_count,
                            },
                        )
                else:
                    previous_checkpoint_nopfix = os.path.join(
                        scene.model_save_dir,
                        "ckpt_it_" + str(iteration - args.testing_iterations) + ".pth",
                    )
                    if os.path.exists(previous_checkpoint_nopfix):
                        os.remove(previous_checkpoint_nopfix)
                    previous_pose_checkpoint = os.path.join(
                        scene.model_save_dir,
                        "ckpt_it_" + str(iteration - args.testing_iterations) + "_pose.pth",
                    )
                    if os.path.exists(previous_pose_checkpoint):
                        os.remove(previous_pose_checkpoint)

                    progress_bar.write(
                        "\n[ITER {}] Saving Checkpoint".format(iteration)
                    )
                    scene.save(iteration, "ckpt_it_" + str(iteration))
                    _save_pose_correction(
                        scene.model_save_dir,
                        "ckpt_it_" + str(iteration),
                        pose_correction,
                        pose_optimizer,
                        {
                            "pose_translation_enabled": pose_translation_enabled,
                            "pose_rotation_unlock_count": pose_rotation_unlock_count,
                        },
                    )

        logging(log, output_dir)

    if args.refine.use_refine:
        print(output_dir)
        in_channels = 9 if args.refine.use_spatial else 3
        unet = UNet(in_channels=in_channels, out_channels=1).cuda()
        unet_optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
        for epoch in tqdm(range(0, args.refine.epochs), desc="Refine raydrop"):
            for iter in range(0, args.refine.batch_size):
                if not frame_stack:
                    frame_stack = list(scene.train_lidar.train_frames)
                    random.shuffle(frame_stack)
                frame = frame_stack.pop()

                render_pkg = raytracing(
                    frame, gaussians_assets, scene.train_lidar, background, args
                )
                depth = render_pkg["depth"].detach()
                intensity = render_pkg["intensity"].detach()
                raydrop_prob = render_pkg["raydrop"].detach()

                H, W = depth.shape[0], depth.shape[1]
                input_depth = depth.reshape(1, H, W)
                input_intensity = intensity.reshape(1, H, W)
                input_raydrop = raydrop_prob.reshape(1, H, W)
                raydrop_prob = torch.cat(
                    [input_raydrop, input_intensity, input_depth], dim=0
                )
                if args.refine.use_spatial:
                    ray_o, ray_d = scene.train_lidar.get_range_rays(frame)
                    raydrop_prob = torch.cat(
                        [raydrop_prob, ray_o.permute(2, 0, 1), ray_d.permute(2, 0, 1)],
                        dim=0,
                    )
                raydrop_prob = raydrop_prob.unsqueeze(0)
                if args.refine.use_rot:
                    rot = torch.randint(0, W, (1,))
                    raydrop_prob = torch.cat(
                        [raydrop_prob[:, :, :, rot:], raydrop_prob[:, :, :, :rot]],
                        dim=-1,
                    )
                raydrop_prob = unet(raydrop_prob)

                raydrop_prob = raydrop_prob.reshape(-1, 1)

                gt_mask = scene.train_lidar.get_mask(frame).cuda()
                labels_idx = (
                    ~gt_mask
                )  # (1, h, w) notice: hit is true (1). apply ~ to make idx 0 represent hit
                if args.refine.use_rot:
                    labels_idx = torch.cat(
                        [labels_idx[:, rot:], labels_idx[:, :rot]], dim=-1
                    )
                labels = labels_idx.reshape(-1, 1)  # (h*w, 1)
                loss_raydrop = args.refine.lambda_raydrop_bce * BCELoss(
                    labels, preds=raydrop_prob
                )

                loss_raydrop.backward()

            unet_optimizer.step()
            unet_optimizer.zero_grad()

        torch.save(unet.state_dict(), os.path.join(output_dir, "models", "unet.pth"))


def logging(log, output_dir):
    indices = range(len(log["depth_rmse"]))

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = "tab:blue"
    ax1.set_ylabel("Depth MSE", color=color)
    ax1.plot(indices, log["depth_rmse"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "tab:red"

    ax2.set_ylabel("Points Num", color=color)
    clone_sum = np.array(log["clone_sum"])
    split_sum = np.array(log["split_sum"])
    prune_scale_sum = np.array(log["prune_scale_sum"])
    prune_opacity_sum = np.array(log["prune_opacity_sum"])

    plt.fill_between(indices, 0, clone_sum, label="clone_sum", color="blue", alpha=0.5)
    plt.fill_between(
        indices,
        clone_sum,
        clone_sum + split_sum,
        label="split_sum",
        color="green",
        alpha=0.5,
    )
    plt.fill_between(
        indices,
        clone_sum + split_sum,
        clone_sum + split_sum + prune_scale_sum,
        label="prune_scale_sum",
        color="red",
        alpha=0.5,
    )
    plt.fill_between(
        indices,
        clone_sum + split_sum + prune_scale_sum,
        clone_sum + split_sum + prune_scale_sum + prune_opacity_sum,
        label="prune_opacity_sum",
        color="yellow",
        alpha=0.5,
    )

    ax2.plot(indices, log["points_num"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "log.png"))
    plt.close()
    with open(os.path.join(log_dir, "log.json"), "w") as json_file:
        json.dump(log, json_file, indent=4)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="launch args")
    parser.add_argument("-dc", "--data_config_path", type=str, help="config path")
    parser.add_argument("-ec", "--exp_config_path", type=str, help="config path")
    parser.add_argument("-m", "--model", type=str, help="the path to a checkpoint")
    parser.add_argument(
        "-r",
        "--only_refine",
        action="store_true",
        help="skip the training. only refine the model. E.g. load a checkpoint and only refine the unet to fit the checkpoint",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="Camera ID for camera supervision (-1=disabled for Waymo/PandaSet). For Waymo: 1=FRONT, 2=FRONT_LEFT, "
             "3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT. For KITTI-360/kitti-calib: use 0. "
             "Requires lambda_rgb > 0 or pose_correction.sdf_loss_weight > 0 in exp config.",
    )
    parser.add_argument(
        "--camera_scale",
        type=int,
        default=4,
        help="Integer downscale factor for camera images used in supervision (default: 4 → ¼ resolution).",
    )
    launch_args = parser.parse_args()

    args = parse(launch_args.exp_config_path)
    args = parse(launch_args.data_config_path, args)
    args.model_path = launch_args.model
    args.only_refine = launch_args.only_refine
    args.camera_id = launch_args.camera_id
    args.camera_scale = launch_args.camera_scale

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.seed is not None:
        set_seed(args.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args)

    # All done
    print(blue("\nTraining complete."))
