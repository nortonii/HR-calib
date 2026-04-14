#!/usr/bin/env python3
"""Calibration training loop for LiDAR-camera extrinsic calibration via 3DGS.

Continuous training (no reset); noise-injection mode only.

Supported datasets (set via data_type in the data config YAML):
  - KITTICalib  (data/kitti-calibration)  requires: kitti_calib_scene
  - KITTI       (data/kitti360)            requires: kitti_seq; optional: camera_scale
  - Waymo       (data/waymo/...)           optional: waymo_camera_id (1=FRONT), camera_scale
  - PandaSet    (data/pandaset)            optional: pandaset_camera_name, camera_scale

Usage (KITTI-Calibration)
-----
python tools/calib.py \\
    -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir output/calib/my_exp

Usage (KITTI-360)
-----
python tools/calib.py \\
    -dc configs/kitti360/static/k3_cam.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir output/calib/kitti360_k3

Usage (Waymo)
-----
python tools/calib.py \\
    -dc configs/waymo/static/t0_cam.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir output/calib/waymo_t0

Usage (PandaSet)
-----
python tools/calib.py \\
    -dc configs/pandaset/static/1.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir output/calib/pandaset_1
"""

import argparse
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import dataloader
from lib.arguments import parse
from lib.dataloader.kitti_calib_loader import load_kitti_calib_cameras
from lib.dataloader.kitti_loader import load_kitti360_cameras
from lib.dataloader.pandaset_loader import load_pandaset_cameras
from lib.dataloader.waymo_loader import load_waymo_cameras
from lib.gaussian_renderer import raytracing
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene.camera_pose_correction import CameraPoseCorrection
from lib.utils.console_utils import blue, green, red, yellow
from lib.utils.graphics_utils import fov2focal
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss, ssim
from lib.utils.rgbd_calibration import (
    _format_matcher_image,
    CameraModel,
    build_frame_correspondence,
    build_matcher,
    depth_to_match_image,
    initialize_shared_extrinsic,
    match_cross_modal,
    match_cross_modal_dense,
    optimize_shared_extrinsic,
    sample_depth_values,
    sample_depth_values_vectorized,
    select_match_points,
)


# ─────────────────────────────────────────────────────────────
# Quaternion helpers  [w, x, y, z] convention
# ─────────────────────────────────────────────────────────────

def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    R = R.float()
    K = torch.stack([
        torch.stack([R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1], R[2,0]+R[0,2], R[2,1]-R[1,2]]),
        torch.stack([R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2], R[0,2]-R[2,0]]),
        torch.stack([R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], R[1,0]-R[0,1]]),
        torch.stack([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]]),
    ]) / 3.0
    _, v = torch.linalg.eigh(K)
    q_xyzw = v[:, -1]
    q = torch.stack([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return q if q[0] >= 0 else -q


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q.float(), dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)]),
        torch.stack([  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)]),
        torch.stack([  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]),
    ])


def axis_angle_to_quaternion(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = F.normalize(axis.float(), dim=0)
    half = angle_rad / 2.0
    return torch.cat([
        torch.tensor([math.cos(half)], dtype=torch.float32, device=axis.device),
        math.sin(half) * axis,
    ])


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> float:
    R_rel = R_pred @ R_gt.T
    cos_a = ((R_rel.diagonal().sum().clamp(-1, 3) - 1) / 2).clamp(-1, 1)
    return math.degrees(math.acos(cos_a.item()))


def _translation_error_m(pose_correction, gt_l2c_T: torch.Tensor) -> float:
    """L2 distance (metres) between current effective l2c translation and GT.

    Uses the decoupled parametrisation: T_eff = T_base + delta_T,
    consistent with corrected_rt() when use_gt_translation=False.
    """
    base_T = pose_correction.base_lidar_to_camera_translation[0].float()
    delta_T = pose_correction.delta_translations[0].float()
    eff_T = base_T + delta_T
    return (eff_T - gt_l2c_T).norm().item()


def _effective_R(pose_correction) -> torch.Tensor:
    """Current effective l2c rotation (delta ⊗ base) as float32 matrix."""
    dq = F.normalize(pose_correction.delta_rotations_quat[0].float(), dim=0)
    bq = F.normalize(pose_correction.base_lidar_to_camera_quat[0].float(), dim=0)
    eff_q = quaternion_multiply(dq, bq)
    return quaternion_to_matrix(eff_q)


def _camera_model_from_camera(camera) -> CameraModel:
    width = float(camera.image_width)
    height = float(camera.image_height)
    fx = float(fov2focal(float(camera.FoVx), width))
    fy = float(fov2focal(float(camera.FoVy), height))
    cx = width * 0.5
    cy = height * 0.5
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraModel(K=K, dist=np.zeros((0,), dtype=np.float64))


def _blend_relative_transform(
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    blend = float(np.clip(blend, 0.0, 1.0))
    rotation_vector, _ = cv2.Rodrigues(np.asarray(rotation_matrix, dtype=np.float64))
    blended_rotation, _ = cv2.Rodrigues(rotation_vector.reshape(3) * blend)
    blended_translation = np.asarray(translation, dtype=np.float64).reshape(3) * blend
    return blended_rotation.astype(np.float64), blended_translation.astype(np.float64)


def _render_camera_with_backend(camera, gaussian_assets, args, backend=None, **kwargs):
    if backend is None:
        return render_camera(camera, gaussian_assets, args, **kwargs)
    prev_backend = getattr(getattr(args, "model", None), "camera_render_backend", "rasterization")
    args.model.camera_render_backend = backend
    try:
        return render_camera(camera, gaussian_assets, args, **kwargs)
    finally:
        args.model.camera_render_backend = prev_backend


def _sample_image_colors(image_hwc: torch.Tensor, points_xy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    if image_hwc.ndim != 3 or image_hwc.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {tuple(image_hwc.shape)}")
    h, w = int(image_hwc.shape[0]), int(image_hwc.shape[1])
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=image_hwc.device).reshape(-1, 2)
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(max(w - 1, 0)))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(max(h - 1, 0)))
    )
    if not torch.any(valid):
        return image_hwc.new_zeros((0, 3)), valid
    points = points[valid]
    grid = torch.empty((1, points.shape[0], 1, 2), dtype=image_hwc.dtype, device=image_hwc.device)
    grid[..., 0] = (points[:, 0].view(1, -1, 1) / float(max(w - 1, 1))) * 2.0 - 1.0
    grid[..., 1] = (points[:, 1].view(1, -1, 1) / float(max(h - 1, 1))) * 2.0 - 1.0
    sampled = F.grid_sample(
        image_hwc.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1), valid


def _filter_points_by_support(
    query_points: np.ndarray,
    support_points: np.ndarray | None,
    radius_px: float,
) -> np.ndarray:
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    if query_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if support_points is None:
        return np.ones((query_points.shape[0],), dtype=bool)
    support_points = np.asarray(support_points, dtype=np.float32).reshape(-1, 2)
    if support_points.shape[0] == 0:
        return np.zeros((query_points.shape[0],), dtype=bool)
    distances = np.linalg.norm(query_points[:, None, :] - support_points[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    return nearest <= float(radius_px)


def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.max(initial=0.0) <= 1.5:
        array = np.clip(array * 255.0, 0.0, 255.0)
    else:
        array = np.clip(array, 0.0, 255.0)
    return array.astype(np.uint8)


def _get_temporal_support_points(
    matcher,
    frame: int,
    cam_images: dict,
    cache: dict[int, np.ndarray],
    max_offset: int,
) -> np.ndarray:
    if frame in cache:
        return cache[frame]
    current = _to_uint8_rgb(cam_images[frame])
    supports = []
    for offset in range(1, int(max_offset) + 1):
        for neighbor in (frame - offset, frame + offset):
            if neighbor not in cam_images:
                continue
            neighbor_img = _to_uint8_rgb(cam_images[neighbor])
            result = matcher(_format_matcher_image(current), _format_matcher_image(neighbor_img))
            pts0, _ = select_match_points(result)
            if pts0.shape[0] > 0:
                supports.append(np.asarray(pts0, dtype=np.float32))
    cache[frame] = np.concatenate(supports, axis=0) if supports else np.zeros((0, 2), dtype=np.float32)
    return cache[frame]


def _get_temporal_match_pairs(
    matcher,
    frame: int,
    cam_images: dict,
    cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    max_offset: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if frame in cache:
        return cache[frame]
    current = _to_uint8_rgb(cam_images[frame])
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for offset in range(1, int(max_offset) + 1):
        for neighbor in (frame - offset, frame + offset):
            if neighbor not in cam_images:
                continue
            neighbor_img = _to_uint8_rgb(cam_images[neighbor])
            result = matcher(_format_matcher_image(current), _format_matcher_image(neighbor_img))
            pts0, pts1 = select_match_points(result)
            if pts0.shape[0] > 0 and pts1.shape[0] > 0:
                pairs.append(
                    (
                        int(neighbor),
                        np.asarray(pts0, dtype=np.float32),
                        np.asarray(pts1, dtype=np.float32),
                    )
                )
    cache[frame] = pairs
    return pairs


def _compute_matcher_color_loss(
    matcher,
    gt_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    depth: torch.Tensor,
    matcher_min_matches: int,
    matcher_min_depth_matches: int,
    matcher_depth_min: float,
    matcher_depth_max: float,
    matcher_depth_percentile_low: float,
    matcher_depth_percentile_high: float,
    matcher_depth_use_inverse: bool,
    matcher_color_weight: float,
    temporal_support_points: np.ndarray | None = None,
    temporal_support_radius_px: float = 6.0,
    dense_mode: bool = False,
    dense_stride: int = 4,
    dense_cert_threshold: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    device = pred_rgb.device
    gt_rgb_np = _to_uint8_rgb(gt_rgb)
    depth_np = depth.detach().squeeze(-1).cpu().numpy().astype(np.float32)
    depth_vis = depth_to_match_image(
        depth_np,
        percentile_low=matcher_depth_percentile_low,
        percentile_high=matcher_depth_percentile_high,
        use_inverse=matcher_depth_use_inverse,
    )
    if dense_mode:
        rgb_points, depth_points, _certs = match_cross_modal_dense(
            matcher, gt_rgb_np, depth_vis,
            query_stride=dense_stride,
            cert_threshold=dense_cert_threshold,
        )
    else:
        rgb_points, depth_points, _ = match_cross_modal(matcher, gt_rgb_np, depth_vis)
    raw_matches = int(rgb_points.shape[0])
    if temporal_support_points is not None:
        support_mask = _filter_points_by_support(
            query_points=rgb_points,
            support_points=temporal_support_points,
            radius_px=temporal_support_radius_px,
        )
        rgb_points = rgb_points[support_mask]
        depth_points = depth_points[support_mask]
    if rgb_points.shape[0] < int(matcher_min_matches):
        return pred_rgb.new_zeros(()), {
            "status": "skipped",
            "reason": "too_few_matches",
            "matches": int(rgb_points.shape[0]),
            "raw_matches": raw_matches,
        }

    keep_indices, _ = sample_depth_values(
        depth_map=depth_np,
        points=np.asarray(depth_points, dtype=np.float64),
        min_depth=matcher_depth_min,
        max_depth=matcher_depth_max,
        search_radius=2,
    )
    if keep_indices.shape[0] < int(matcher_min_depth_matches):
        return pred_rgb.new_zeros(()), {
            "status": "skipped",
            "reason": "too_few_depth_matches",
            "matches": int(keep_indices.shape[0]),
            "raw_matches": raw_matches,
        }

    rgb_points = np.asarray(rgb_points[keep_indices], dtype=np.float32)
    depth_points = np.asarray(depth_points[keep_indices], dtype=np.float32)
    gt_samples, gt_valid = _sample_image_colors(gt_rgb.to(device=device), rgb_points)
    pred_samples, pred_valid = _sample_image_colors(pred_rgb, depth_points)
    valid = gt_valid & pred_valid
    if int(valid.sum().item()) < int(matcher_min_depth_matches):
        return pred_rgb.new_zeros(()), {
            "status": "skipped",
            "reason": "too_few_valid_samples",
            "matches": int(valid.sum().item()),
            "raw_matches": raw_matches,
        }

    gt_samples = gt_samples[valid[gt_valid]]
    pred_samples = pred_samples[valid[pred_valid]]
    loss = float(matcher_color_weight) * F.l1_loss(pred_samples, gt_samples)
    mae = torch.mean(torch.abs(pred_samples.detach() - gt_samples.detach())).item()
    return loss, {
        "status": "applied",
        "matches": int(pred_samples.shape[0]),
        "raw_matches": raw_matches,
        "supervised_ratio": float(pred_samples.shape[0]) / float(max(gt_rgb.shape[0] * gt_rgb.shape[1], 1)),
        "mae": float(mae),
    }


def _compute_adjacent_matcher_color_loss(
    matcher,
    frame: int,
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    args,
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    matcher_adjacent_max_offset: int,
    matcher_min_matches: int,
    adjacent_color_weight: float,
) -> tuple[torch.Tensor, dict]:
    match_pairs = _get_temporal_match_pairs(
        matcher=matcher,
        frame=int(frame),
        cam_images=cam_images,
        cache=temporal_pair_cache,
        max_offset=matcher_adjacent_max_offset,
    )
    if not match_pairs:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "no_adjacent_pairs"}

    total_loss = pred_rgb.new_zeros(())
    pair_count = 0
    total_matches = 0
    for neighbor, pts_cur, pts_nbr in match_pairs:
        if pts_cur.shape[0] < int(matcher_min_matches) or neighbor not in cam_cameras:
            continue
        neighbor_camera = cam_cameras[neighbor].cuda()
        nbr_R, nbr_T = pose_correction.corrected_rt(neighbor, device="cuda")
        neighbor_render = render_camera(
            neighbor_camera,
            [gaussians],
            args,
            cam_rotation=nbr_R.detach(),
            cam_translation=nbr_T.detach(),
            require_rgb=True,
        )
        if int(neighbor_render.get("num_visible", 0)) <= 0:
            continue
        neighbor_pred = neighbor_render["rgb"].clamp(0.0, 1.0)
        neighbor_gt = cam_images[neighbor].to(device=pred_rgb.device)

        cur_gt_samples, cur_gt_valid = _sample_image_colors(gt_rgb, pts_cur)
        cur_pred_samples, cur_pred_valid = _sample_image_colors(pred_rgb, pts_cur)
        nbr_gt_samples, nbr_gt_valid = _sample_image_colors(neighbor_gt, pts_nbr)
        nbr_pred_samples, nbr_pred_valid = _sample_image_colors(neighbor_pred, pts_nbr)

        valid = cur_gt_valid & cur_pred_valid & nbr_gt_valid & nbr_pred_valid
        valid_count = int(valid.sum().item())
        if valid_count < int(matcher_min_matches):
            continue

        cur_gt_samples = cur_gt_samples[valid[cur_gt_valid]]
        cur_pred_samples = cur_pred_samples[valid[cur_pred_valid]]
        nbr_gt_samples = nbr_gt_samples[valid[nbr_gt_valid]]
        nbr_pred_samples = nbr_pred_samples[valid[nbr_pred_valid]]
        pair_loss = 0.5 * (
            F.l1_loss(cur_pred_samples, nbr_gt_samples) +
            F.l1_loss(nbr_pred_samples, cur_gt_samples)
        )
        total_loss = total_loss + pair_loss
        pair_count += 1
        total_matches += valid_count

    if pair_count == 0:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "too_few_adjacent_matches"}

    total_loss = float(adjacent_color_weight) * (total_loss / float(pair_count))
    return total_loss, {
        "status": "applied",
        "pairs": int(pair_count),
        "matches": int(total_matches),
    }


@torch.no_grad()
def _precompute_cycle_match_cache(
    matcher,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    frame_ids: list[int],
    args,
    matcher_min_matches: int,
    matcher_min_depth_matches: int,
    matcher_depth_min: float,
    matcher_depth_max: float,
    matcher_depth_percentile_low: float,
    matcher_depth_percentile_high: float,
    matcher_depth_use_inverse: bool,
    temporal_support_getter=None,
    temporal_support_radius_px: float = 6.0,
    dense_mode: bool = False,
    dense_stride: int = 4,
    dense_cert_threshold: float = 0.02,
) -> dict:
    """Run cross-modal matching once per cycle for all train frames.

    Returns a dict keyed by frame_id with:
      - 'depth_pts': np.ndarray [N, 2]  pixel coords in the rendered image
      - 'gt_colors': torch.Tensor [N, 3] (CPU) GT RGB at matched rgb_pts
    """
    cache = {}
    for frame in frame_ids:
        if frame not in cam_cameras or frame not in cam_images:
            continue
        cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
        camera = cam_cameras[frame].cuda()
        gt_rgb = cam_images[frame].cuda()

        depth_render = _render_camera_with_backend(
            camera, [gaussians], args,
            backend="surfel_rasterization",
            cam_rotation=cam_R.detach(),
            cam_translation=cam_T.detach(),
            require_rgb=False,
        )
        if not int(depth_render.get("num_visible", 0)):
            continue

        depth_np = depth_render["depth"].squeeze(-1).cpu().numpy().astype(np.float32)
        depth_vis = depth_to_match_image(
            depth_np,
            percentile_low=matcher_depth_percentile_low,
            percentile_high=matcher_depth_percentile_high,
            use_inverse=matcher_depth_use_inverse,
        )
        gt_rgb_np = _to_uint8_rgb(gt_rgb)

        if dense_mode:
            rgb_pts, depth_pts, _ = match_cross_modal_dense(
                matcher, gt_rgb_np, depth_vis,
                query_stride=dense_stride,
                cert_threshold=dense_cert_threshold,
            )
        else:
            rgb_pts, depth_pts, _ = match_cross_modal(matcher, gt_rgb_np, depth_vis)

        if temporal_support_getter is not None:
            support = temporal_support_getter(frame)
            if support is not None and support.shape[0] > 0:
                mask = _filter_points_by_support(rgb_pts, support, temporal_support_radius_px)
                rgb_pts = rgb_pts[mask]
                depth_pts = depth_pts[mask]

        if rgb_pts.shape[0] < int(matcher_min_matches):
            continue

        # Use fast vectorized sampling — rendered 2DGS depth is dense, no search radius needed
        keep_indices, _ = sample_depth_values_vectorized(
            depth_map=depth_np,
            points=np.asarray(depth_pts, dtype=np.float32),
            min_depth=matcher_depth_min,
            max_depth=matcher_depth_max,
        )
        if keep_indices.shape[0] < int(matcher_min_depth_matches):
            continue

        rgb_pts_f = np.asarray(rgb_pts[keep_indices], dtype=np.float32)
        depth_pts_f = np.asarray(depth_pts[keep_indices], dtype=np.float32)

        gt_colors, gt_valid = _sample_image_colors(gt_rgb, rgb_pts_f)
        valid_mask = gt_valid.cpu().numpy().astype(bool)
        if int(valid_mask.sum()) < int(matcher_min_depth_matches):
            continue

        cache[frame] = {
            "depth_pts": depth_pts_f[valid_mask],
            "gt_colors": gt_colors[gt_valid].detach().cpu(),
        }
    if cache:
        total_pts = sum(v["depth_pts"].shape[0] for v in cache.values())
        frames_hit = len(cache)
        # Image size from any camera
        sample_cam = next(iter(cam_cameras.values()))
        h, w = int(sample_cam.image_height), int(sample_cam.image_width)
        pixels_per_frame = h * w
        avg_pts = total_pts / frames_hit
        avg_coverage = avg_pts / pixels_per_frame * 100.0
        print(f"[Cycle cache] {frames_hit} frames, avg {avg_pts:.0f} matches/frame "
              f"({avg_coverage:.1f}% pixel coverage, image {w}×{h})")
    return cache


def _apply_cross_modal_cache_loss(
    pred_rgb: torch.Tensor,
    cache_entry: dict,
    matcher_color_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Compute match color loss using precomputed (depth_pts, gt_colors) cache."""
    depth_pts = cache_entry["depth_pts"]
    gt_colors = cache_entry["gt_colors"].to(pred_rgb.device)

    pred_colors, pred_valid = _sample_image_colors(pred_rgb, depth_pts)
    gt_colors_valid = gt_colors[pred_valid]
    pred_colors_valid = pred_colors[pred_valid]

    n = int(pred_colors_valid.shape[0])
    if n < 5:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "too_few_valid_samples", "matches": n}

    loss = float(matcher_color_weight) * F.l1_loss(pred_colors_valid, gt_colors_valid)
    mae = torch.mean(torch.abs(pred_colors_valid.detach() - gt_colors_valid.detach())).item()
    return loss, {
        "status": "applied",
        "matches": n,
        "raw_matches": n,
        "supervised_ratio": float(n) / float(max(pred_rgb.shape[0] * pred_rgb.shape[1], 1)),
        "mae": float(mae),
    }


def _precompute_adjacent_gt_cache(
    temporal_pair_cache: dict,
    cam_images: dict,
) -> dict:
    """For each frame that already has temporal pairs cached, sample GT colors at
    the matched pixel locations on both sides.  Returns:
      {frame: [(neighbor, pts_cur, cur_gt_colors_cpu, pts_nbr, nbr_gt_colors_cpu), ...]}
    """
    adj_cache: dict[int, list] = {}
    for frame, pairs in temporal_pair_cache.items():
        if frame not in cam_images:
            continue
        cur_gt = cam_images[frame].cpu()
        entries = []
        for neighbor, pts_cur, pts_nbr in pairs:
            if neighbor not in cam_images:
                continue
            nbr_gt = cam_images[neighbor].cpu()
            cur_gt_colors, cur_valid = _sample_image_colors(cur_gt, pts_cur)
            nbr_gt_colors, nbr_valid = _sample_image_colors(nbr_gt, pts_nbr)
            valid = cur_valid & nbr_valid
            if not torch.any(valid):
                continue
            valid_np = valid.numpy().astype(bool)
            # pts_cur/pts_nbr are [N,2], valid is [N] — direct mask works
            # cur/nbr_gt_colors are pre-indexed to cur_valid/nbr_valid, so sub-index within those
            entries.append((
                int(neighbor),
                pts_cur[valid_np],
                cur_gt_colors[valid[cur_valid]].detach(),
                pts_nbr[valid_np],
                nbr_gt_colors[valid[nbr_valid]].detach(),
            ))
        if entries:
            adj_cache[frame] = entries
    return adj_cache


def _apply_adjacent_cache_loss(
    frame: int,
    pred_rgb: torch.Tensor,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    args,
    adj_gt_cache: dict,
    adjacent_color_weight: float,
    matcher_min_matches: int,
) -> tuple[torch.Tensor, dict]:
    """Adjacent color loss using precomputed GT color targets."""
    if frame not in adj_gt_cache:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "no_adjacent_cache"}

    total_loss = pred_rgb.new_zeros(())
    pair_count = 0
    total_matches = 0

    for neighbor, pts_cur, cur_gt_colors, pts_nbr, nbr_gt_colors in adj_gt_cache[frame]:
        if neighbor not in cam_cameras:
            continue
        nbr_R, nbr_T = pose_correction.corrected_rt(neighbor, device="cuda")
        nbr_render = render_camera(
            cam_cameras[neighbor].cuda(), [gaussians], args,
            cam_rotation=nbr_R.detach(),
            cam_translation=nbr_T.detach(),
            require_rgb=True,
        )
        if int(nbr_render.get("num_visible", 0)) <= 0:
            continue
        nbr_pred = nbr_render["rgb"].clamp(0.0, 1.0)

        cur_pred_colors, cur_pred_valid = _sample_image_colors(pred_rgb, pts_cur)
        nbr_pred_colors, nbr_pred_valid = _sample_image_colors(nbr_pred, pts_nbr)

        valid = cur_pred_valid & nbr_pred_valid
        n = int(valid.sum().item())
        if n < int(matcher_min_matches):
            continue

        # cur_pred vs nbr_gt  and  nbr_pred vs cur_gt
        # cur/nbr_gt_colors are [N,3] on CPU — index with full valid mask (moved to cpu)
        # cur/nbr_pred_colors are [cur/nbr_pred_valid.sum(),3] — sub-index within that
        valid_cpu = valid.cpu()
        cur_gt_dev = cur_gt_colors[valid_cpu].to(pred_rgb.device)
        nbr_gt_dev = nbr_gt_colors[valid_cpu].to(pred_rgb.device)
        cur_pred_v = cur_pred_colors[valid[cur_pred_valid]]
        nbr_pred_v = nbr_pred_colors[valid[nbr_pred_valid]]

        pair_loss = 0.5 * (F.l1_loss(cur_pred_v, nbr_gt_dev) + F.l1_loss(nbr_pred_v, cur_gt_dev))
        total_loss = total_loss + pair_loss
        pair_count += 1
        total_matches += n

    if pair_count == 0:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "too_few_adjacent_matches"}

    total_loss = float(adjacent_color_weight) * (total_loss / float(pair_count))
    return total_loss, {"status": "applied", "pairs": pair_count, "matches": total_matches}


@torch.no_grad()
def _run_matcher_pose_update(
    matcher,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    args,
    frame_ids: list[int],
    matcher_min_matches: int,
    matcher_min_depth_matches: int,
    matcher_min_pnp_inliers: int,
    matcher_pnp_reproj_error: float,
    matcher_pnp_iterations: int,
    matcher_depth_min: float,
    matcher_depth_max: float,
    matcher_depth_percentile_low: float,
    matcher_depth_percentile_high: float,
    matcher_depth_use_inverse: bool,
    matcher_update_blend: float,
    temporal_support_getter=None,
    temporal_support_radius_px: float = 6.0,
):
    if not frame_ids:
        return {"status": "skipped", "reason": "no camera frames"}

    rgb_camera_model = _camera_model_from_camera(cam_cameras[frame_ids[0]])
    frame_data_list = []
    for frame in frame_ids:
        if frame not in cam_cameras or frame not in cam_images:
            continue
        camera = cam_cameras[frame].cuda()
        gt_rgb = cam_images[frame].detach().cpu().numpy()
        if gt_rgb.max(initial=0.0) <= 1.5:
            gt_rgb = np.clip(gt_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            gt_rgb = np.clip(gt_rgb, 0.0, 255.0).astype(np.uint8)

        cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
        render_pkg = _render_camera_with_backend(
            camera,
            [gaussians],
            args,
            backend="surfel_rasterization",
            cam_rotation=cam_R,
            cam_translation=cam_T,
            require_rgb=False,
        )
        if int(render_pkg.get("num_visible", 0)) <= 0:
            continue

        depth = render_pkg["depth"].detach().squeeze(-1).cpu().numpy().astype(np.float32)
        depth_vis = depth_to_match_image(
            depth,
            percentile_low=matcher_depth_percentile_low,
            percentile_high=matcher_depth_percentile_high,
            use_inverse=matcher_depth_use_inverse,
        )
        rgb_points, depth_points, _ = match_cross_modal(matcher, gt_rgb, depth_vis)
        if temporal_support_getter is not None:
            support_points = temporal_support_getter(frame)
            support_mask = _filter_points_by_support(
                query_points=rgb_points,
                support_points=support_points,
                radius_px=temporal_support_radius_px,
            )
            rgb_points = rgb_points[support_mask]
            depth_points = depth_points[support_mask]
        if rgb_points.shape[0] < int(matcher_min_matches):
            continue

        frame_data = build_frame_correspondence(
            frame_name=f"{frame:06d}",
            rgb_path=f"frame:{frame}",
            depth_path=f"render:{frame}",
            rgb_points=rgb_points,
            depth_points=depth_points,
            depth_map=depth,
            depth_camera=rgb_camera_model,
            min_depth=matcher_depth_min,
            max_depth=matcher_depth_max,
            search_radius=2,
        )
        if frame_data is None or frame_data.points_3d.shape[0] < int(matcher_min_depth_matches):
            continue
        frame_data.frame_id = int(frame)
        frame_data.frame_index = len(frame_data_list)
        frame_data_list.append(frame_data)

    if not frame_data_list:
        return {"status": "skipped", "reason": "no valid frame correspondences"}

    try:
        initial_rvec, initial_tvec = initialize_shared_extrinsic(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera_model,
            reproj_error=matcher_pnp_reproj_error,
            iterations=matcher_pnp_iterations,
            min_inliers=matcher_min_pnp_inliers,
        )
        comparison = optimize_shared_extrinsic(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera_model,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    relative_rotation = comparison.optimized.rotation_matrix
    relative_translation = comparison.optimized.translation
    if float(matcher_update_blend) < 1.0:
        relative_rotation, relative_translation = _blend_relative_transform(
            relative_rotation,
            relative_translation,
            matcher_update_blend,
        )

    pose_correction.apply_relative_camera_transform(
        frame_ids[0],
        torch.as_tensor(relative_rotation, dtype=torch.float32, device="cuda"),
        torch.as_tensor(relative_translation, dtype=torch.float32, device="cuda"),
    )
    pose_correction.update_extrinsics()

    return {
        "status": "applied",
        "frames_used": int(comparison.optimized.frames_used),
        "matches_used": int(comparison.optimized.matches_used),
        "mean_reproj_px": float(comparison.optimized.mean_reprojection_error),
        "rotation_delta_deg": float(comparison.rotation_delta_deg),
        "translation_delta_m": float(comparison.translation_delta_m),
    }


# ─────────────────────────────────────────────────────────────
# Gaussian state save / restore
# ─────────────────────────────────────────────────────────────

_GAUSSIAN_ATTRS = [
    "_xyz", "_features_dc", "_features_rest",
    "_features_rgb_dc", "_features_rgb_rest",
    "_scaling", "_rotation", "_opacity", "_opacity_cam",
]


def save_gaussian_state(gaussians) -> dict:
    state = {"active_sh_degree": gaussians.active_sh_degree}
    for attr in _GAUSSIAN_ATTRS:
        p = getattr(gaussians, attr, None)
        if p is not None:
            state[attr] = p.data.detach().clone().cpu()
    return state


def restore_gaussian_state(gaussians, state: dict, args):
    """Restore Gaussian parameters AND rebuild Adam optimizer from scratch."""
    gaussians.active_sh_degree = state["active_sh_degree"]
    for attr in _GAUSSIAN_ATTRS:
        if attr not in state:
            continue
        p = getattr(gaussians, attr, None)
        if p is None:
            continue
        if p.shape == state[attr].shape:
            p.data.copy_(state[attr].to(p.device))
        else:
            new_p = torch.nn.Parameter(
                state[attr].to(p.device).requires_grad_(True)
            )
            setattr(gaussians, attr, new_p)

    n = gaussians.get_local_xyz.shape[0]
    gaussians.max_radii2D = torch.zeros(n, device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
    gaussians.denom = torch.zeros((n, 1), device="cuda")

    for attr in _GAUSSIAN_ATTRS:
        p = getattr(gaussians, attr, None)
        if p is not None:
            p.requires_grad_(True)

    gaussians.training_setup(args.opt)


def _add_label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    canvas = cv2.copyMakeBorder(canvas, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _draw_points(image: np.ndarray, points_xy: np.ndarray, color: tuple[int, int, int], radius: int = 2) -> np.ndarray:
    canvas = image.copy()
    for x, y in np.asarray(points_xy, dtype=np.float32).reshape(-1, 2):
        cv2.circle(
            canvas,
            (int(round(float(x))), int(round(float(y)))),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _make_sparse_target_canvas(shape_hw: tuple[int, int], points_xy: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for (x, y), color in zip(
        np.asarray(points_xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3),
    ):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(canvas, (xi, yi), 2, tuple(int(v) for v in color.tolist()), -1, lineType=cv2.LINE_AA)
    return canvas


def _select_visualization_frames(cam_cameras: dict, scene, num_frames: int = 3) -> list[int]:
    frame_ids = sorted(int(k) for k in cam_cameras.keys())
    if not frame_ids:
        return []
    preferred = []
    eval_frames = getattr(getattr(scene, "train_lidar", None), "eval_frames", [])
    for frame in eval_frames:
        frame = int(frame)
        if frame in cam_cameras:
            preferred.append(frame)
    chosen = preferred if preferred else frame_ids
    if len(chosen) <= num_frames:
        return chosen
    indices = np.linspace(0, len(chosen) - 1, num_frames).round().astype(int)
    return [chosen[i] for i in indices]


@torch.no_grad()
def _save_run_visualizations(
    out_dir: str,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    gt_l2c_R: torch.Tensor,
    gt_l2c_T: torch.Tensor,
    args,
    matcher_name: str = "matchanything-roma",
    matcher_max_num_keypoints: int = 2048,
    matcher_ransac_reproj_thresh: float = 3.0,
    matcher_min_matches: int = 20,
    matcher_min_depth_matches: int = 12,
    matcher_depth_min: float = 0.1,
    matcher_depth_max: float = 80.0,
    matcher_depth_percentile_low: float = 5.0,
    matcher_depth_percentile_high: float = 95.0,
    matcher_depth_use_inverse: bool = True,
    matcher_adjacent_support: bool = False,
    matcher_adjacent_max_offset: int = 1,
    matcher_support_radius_px: float = 6.0,
):
    viz_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    rot_err = _rotation_error_deg(_effective_R(pose_correction), gt_l2c_R)
    trans_err = _translation_error_m(pose_correction, gt_l2c_T)
    frame_ids = _select_visualization_frames(cam_cameras, scene, num_frames=3)
    if not frame_ids:
        return

    matcher = None
    temporal_support_cache: dict[int, np.ndarray] = {}
    if matcher_name == "matchanything-roma":
        try:
            matcher = build_matcher(
                device="cuda",
                max_num_keypoints=int(matcher_max_num_keypoints),
                ransac_reproj_thresh=float(matcher_ransac_reproj_thresh),
            )
        except Exception as exc:
            summary_lines = [f"matcher_visualization_skipped={exc}"]
            with open(os.path.join(viz_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines) + "\n")
            return

    avg_psnr_values = []
    panels = []
    summary_lines = [
        f"rot_err_deg={rot_err:.6f}",
        f"trans_err_m={trans_err:.6f}",
    ]

    for frame_all in sorted(cam_cameras.keys()):
        camera = cam_cameras[frame_all].cuda()
        cam_R, cam_T = pose_correction.corrected_rt(frame_all, device="cuda")
        render_pkg = render_camera(
            camera, [gaussians], args,
            cam_rotation=cam_R, cam_translation=cam_T, require_rgb=True,
        )
        if int(render_pkg.get("num_visible", 0)) <= 0:
            continue
        pred_rgb = render_pkg["rgb"].clamp(0.0, 1.0)
        gt_rgb = cam_images[frame_all].cuda()
        avg_psnr_values.append(psnr(pred_rgb.permute(2, 0, 1), gt_rgb.permute(2, 0, 1)).item())

    if avg_psnr_values:
        summary_lines.append(f"avg_psnr_db={float(np.mean(avg_psnr_values)):.6f}")

    for frame in frame_ids:
        camera = cam_cameras[frame].cuda()
        gt_rgb = cam_images[frame].cuda()
        cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
        depth_render = _render_camera_with_backend(
            camera,
            [gaussians],
            args,
            backend="surfel_rasterization",
            cam_rotation=cam_R.detach(),
            cam_translation=cam_T.detach(),
            require_rgb=False,
        )
        color_render = render_camera(
            camera,
            [gaussians],
            args,
            cam_rotation=cam_R.detach(),
            cam_translation=cam_T.detach(),
            require_rgb=True,
        )
        pred_rgb = color_render["rgb"].clamp(0.0, 1.0)
        frame_psnr = psnr(pred_rgb.permute(2, 0, 1), gt_rgb.permute(2, 0, 1)).item()
        gt_np = _to_uint8_rgb(gt_rgb)
        pred_np = _to_uint8_rgb(pred_rgb)
        depth_np = depth_render["depth"].detach().squeeze(-1).cpu().numpy().astype(np.float32)
        depth_vis = depth_to_match_image(
            depth_np,
            percentile_low=matcher_depth_percentile_low,
            percentile_high=matcher_depth_percentile_high,
            use_inverse=matcher_depth_use_inverse,
        )

        panel_stats = {
            "frame": int(frame),
            "psnr": float(frame_psnr),
            "raw_matches": 0,
            "support_matches": 0,
            "kept_matches": 0,
        }
        tiles = [
            _add_label(gt_np, "GT RGB"),
            _add_label(depth_vis, "Rendered depth"),
            _add_label(pred_np, f"Rendered RGB  PSNR={frame_psnr:.2f}dB"),
        ]

        if matcher is not None and int(depth_render.get("num_visible", 0)) > 0 and int(color_render.get("num_visible", 0)) > 0:
            rgb_points, depth_points, _ = match_cross_modal(matcher, gt_np, depth_vis)
            panel_stats["raw_matches"] = int(rgb_points.shape[0])
            if matcher_adjacent_support:
                support_points = _get_temporal_support_points(
                    matcher=matcher,
                    frame=int(frame),
                    cam_images=cam_images,
                    cache=temporal_support_cache,
                    max_offset=matcher_adjacent_max_offset,
                )
                support_mask = _filter_points_by_support(
                    query_points=rgb_points,
                    support_points=support_points,
                    radius_px=matcher_support_radius_px,
                )
                rgb_points = rgb_points[support_mask]
                depth_points = depth_points[support_mask]
            panel_stats["support_matches"] = int(rgb_points.shape[0])

            sparse_target = np.zeros_like(gt_np)
            if rgb_points.shape[0] >= int(matcher_min_matches):
                keep_indices, _ = sample_depth_values(
                    depth_map=depth_np,
                    points=np.asarray(depth_points, dtype=np.float64),
                    min_depth=matcher_depth_min,
                    max_depth=matcher_depth_max,
                    search_radius=2,
                )
                rgb_points = np.asarray(rgb_points[keep_indices], dtype=np.float32)
                depth_points = np.asarray(depth_points[keep_indices], dtype=np.float32)
                gt_samples, gt_valid = _sample_image_colors(gt_rgb, rgb_points)
                pred_samples, pred_valid = _sample_image_colors(pred_rgb, depth_points)
                valid = gt_valid & pred_valid
                if int(valid.sum().item()) >= int(matcher_min_depth_matches):
                    rgb_points = rgb_points[valid.cpu().numpy()[gt_valid.cpu().numpy()]]
                    depth_points = depth_points[valid.cpu().numpy()[pred_valid.cpu().numpy()]]
                    gt_samples = gt_samples[valid[gt_valid]]
                    sparse_target = _make_sparse_target_canvas(
                        gt_np.shape[:2],
                        depth_points,
                        _to_uint8_rgb(gt_samples),
                    )
                    panel_stats["kept_matches"] = int(depth_points.shape[0])
                    tiles[0] = _add_label(_draw_points(gt_np, rgb_points, (0, 255, 255), radius=2), f"GT + kept rgb pts ({len(rgb_points)})")
                    tiles[1] = _add_label(_draw_points(depth_vis, depth_points, (0, 255, 0), radius=2), f"Depth + kept depth pts ({len(depth_points)})")
            tiles.append(_add_label(sparse_target, "Sparse GT colors on depth pts"))

        panel = cv2.hconcat(tiles)
        panel = _add_label(
            panel,
            f"frame={frame}  rot={rot_err:.3f}deg  trans={trans_err:.3f}m  raw={panel_stats['raw_matches']}  support={panel_stats['support_matches']}  kept={panel_stats['kept_matches']}",
        )
        panel_path = os.path.join(viz_dir, f"frame_{int(frame):04d}_grid.png")
        cv2.imwrite(panel_path, panel)
        panels.append(panel)
        summary_lines.append(
            f"frame_{int(frame):04d}: psnr={panel_stats['psnr']:.6f}, raw={panel_stats['raw_matches']}, support={panel_stats['support_matches']}, kept={panel_stats['kept_matches']}"
        )

    if panels:
        overview = cv2.vconcat(panels)
        cv2.imwrite(os.path.join(viz_dir, "overview.png"), overview)

    with open(os.path.join(viz_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


# ─────────────────────────────────────────────────────────────
# Main calibration training loop (noise-injection, continuous)
# ─────────────────────────────────────────────────────────────

def run_noise_inject_calib(
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    gt_l2c_R: torch.Tensor,
    gt_l2c_T: torch.Tensor,
    args,
    total_cycles: int = 100,
    iters_per_cycle: int = 200,
    rotation_lr: float = 2e-3,
    translation_lr: float = 0.0015,
    freeze_gaussians: bool = False,
    freeze_xyz: bool = False,
    freeze_colors: bool = False,
    freeze_covariance: bool = False,
    translation_start_cycle: int = 0,
    warmup_cycles: int = 0,
    freeze_rotation: bool = False,
    freeze_translation: bool = False,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    lr_min: float = 1e-5,
    lambda_rgb: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_dssim: float = 0.2,
    initial_gaussian_state: dict = None,
    tb_writer=None,
    cycle_ckpt_dir: str = None,
    save_cycle_every: int = 5,
    resume_cycle_ckpt: str = None,
    reset_gaussians_every: int = 0,
    disable_depth_after_cycle: int = 0,
    rgb_only_updates_color: bool = False,
    matcher_pose_update: bool = False,
    matcher_update_interval: int = 1,
    matcher_update_blend: float = 1.0,
    matcher_name: str = "matchanything-roma",
    matcher_max_num_keypoints: int = 2048,
    matcher_min_matches: int = 20,
    matcher_min_depth_matches: int = 12,
    matcher_min_pnp_inliers: int = 8,
    matcher_ransac_reproj_thresh: float = 3.0,
    matcher_pnp_reproj_error: float = 4.0,
    matcher_pnp_iterations: int = 1000,
    matcher_depth_min: float = 0.1,
    matcher_depth_max: float = 80.0,
    matcher_depth_use_inverse: bool = True,
    matcher_depth_percentile_low: float = 5.0,
    matcher_depth_percentile_high: float = 95.0,
    lidar_updates_opacity_covariance_only: bool = False,
    matcher_color_supervision: bool = False,
    matcher_color_weight: float = 1.0,
    camera_rgb_pose_only: bool = False,
    color_warmup_cycles: int = 0,
    initialize_pose_from_matcher: bool = False,
    matcher_adjacent_support: bool = False,
    matcher_adjacent_max_offset: int = 1,
    matcher_support_radius_px: float = 6.0,
    post_warmup_rgb_reg_scale: float = 0.0,
    matcher_color_lr_scale: float = 1.0,
    matcher_adjacent_color_supervision: bool = False,
    matcher_adjacent_color_weight: float = 1.0,
    disable_depth_during_color_warmup: bool = False,
    matcher_dense_mode: bool = False,
    matcher_dense_stride: int = 4,
    matcher_dense_cert_threshold: float = 0.02,
    match_once_per_cycle: bool = True,
):
    """Continuous calibration training loop.

    When ``pose_correction.use_gt_translation`` is False, ``delta_translations``
    is added to the optimizer so both rotation and translation are calibrated.

    When ``freeze_xyz`` is True, Gaussian mean positions are frozen (requires_grad
    disabled, LR zeroed) so they cannot absorb translation errors.

    ``translation_start_cycle`` implements a two-stage strategy:
      Stage 1 (cycles 1..translation_start_cycle): rotation-only optimisation.
      Stage 2 (cycles translation_start_cycle+1..total): translation optimisation enabled,
        xyz+colors frozen.
    Default 0 = simultaneous (legacy behaviour).

    The pose Adam state is kept across the whole run.
    """
    background = torch.tensor([0, 0, 1], device="cuda").float()
    frame_ids_train = sorted(scene.train_lidar.train_frames)

    # ── Two-stage setup ────────────────────────────────────────
    two_stage = (
        translation_start_cycle > 0
        and not pose_correction.use_gt_translation
    )
    stage2_active = False  # flipped when cycle > translation_start_cycle

    _COLOR_ATTRS = ["_features_dc", "_features_rgb_dc", "_features_rest", "_features_rgb_rest"]
    _RGB_FROZEN_ATTRS = [attr for attr in _GAUSSIAN_ATTRS if attr not in _COLOR_ATTRS]
    matcher = None
    temporal_support_cache: dict[int, np.ndarray] = {}
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    cycle_match_cache: dict[int, dict] = {}   # precomputed cross-modal targets, refreshed each cycle
    adj_gt_cache: dict[int, list] = {}         # precomputed adjacent GT colors, built once

    if lidar_updates_opacity_covariance_only:
        freeze_xyz = True
        freeze_colors = True
        print(blue("[NoiseInject] LiDAR-only scene updates limited to opacity/covariance (xyz+colors frozen)"))

    def _apply_gaussian_freezes():
        """Freeze xyz and/or colors on the Gaussian + its Adam optimizer."""
        if freeze_gaussians:
            for attr in _GAUSSIAN_ATTRS:
                p = getattr(gaussians, attr, None)
                if p is not None:
                    p.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                pg["lr"] = 0.0
            print(blue("[NoiseInject] ALL Gaussian parameters FROZEN — rotation trains against fixed scene"))
            return
        if freeze_xyz and not two_stage:
            xyz_param = getattr(gaussians, "_xyz", None)
            if xyz_param is not None:
                xyz_param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                if any(p is xyz_param for p in pg["params"]):
                    pg["lr"] = 0.0
            print(blue("[NoiseInject] Gaussian xyz FROZEN — translation has exclusive depth gradient"))
        if freeze_colors and not two_stage:
            for attr in _COLOR_ATTRS:
                param = getattr(gaussians, attr, None)
                if param is not None:
                    param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                for attr in _COLOR_ATTRS:
                    cp = getattr(gaussians, attr, None)
                    if cp is not None and any(p is cp for p in pg["params"]):
                        pg["lr"] = 0.0
            print(blue("[NoiseInject] Gaussian colors FROZEN — SH cannot absorb translation gradient"))
        _COV_ATTRS = ["_scaling", "_rotation", "_opacity", "_opacity_cam"]
        if freeze_covariance and not two_stage:
            print(blue("[NoiseInject] Gaussian cov+opacity: LiDAR-only gradients (RGB grads will be zeroed)"))
        if rgb_only_updates_color and not two_stage:
            print(blue("[NoiseInject] Camera RGB updates COLORS only — xyz/cov/opacity RGB grads will be zeroed"))

    def _apply_matcher_color_lr_scale():
        if float(matcher_color_lr_scale) == 1.0:
            return
        for pg in gaussians.optimizer.param_groups:
            if pg.get("name") in {"f_rgb_dc", "f_rgb_rest"}:
                pg["lr"] *= float(matcher_color_lr_scale)
        print(
            blue(
                "[NoiseInject] Matcher color LR scale applied to camera RGB features — "
                f"scale={matcher_color_lr_scale:.2f}"
            )
        )

    _apply_gaussian_freezes()
    _apply_matcher_color_lr_scale()

    if matcher_pose_update or matcher_color_supervision or matcher_adjacent_color_supervision:
        if matcher_name != "matchanything-roma":
            raise ValueError(
                f"Matcher supervision currently expects matchanything-roma, got '{matcher_name}'."
            )
        matcher = build_matcher(
            device="cuda",
            max_num_keypoints=matcher_max_num_keypoints,
            ransac_reproj_thresh=matcher_ransac_reproj_thresh,
        )
        # Warm up the dense code path to trigger all CUDA kernel JIT compilations
        # before the training loop. Without this, the first cycle precompute is
        # blocked for 20+ minutes on Blackwell (cc12.0 PTX JIT for new code paths).
        if matcher_dense_mode:
            print("[Matcher] Warming up dense inference path (one-time CUDA kernel compilation)...")
            _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            try:
                from lib.utils.rgbd_calibration import match_cross_modal_dense as _mcd
                _mcd(matcher, _dummy, _dummy, query_stride=8, cert_threshold=0.5)
            except Exception:
                pass
            print("[Matcher] Dense warm-up complete.")
        if matcher_pose_update:
            print(
                blue(
                    "[NoiseInject] Matcher pose update enabled — pose is updated once per cycle "
                    f"(interval={matcher_update_interval}, blend={matcher_update_blend:.2f})"
                )
            )
            print(blue("[NoiseInject] Camera supervision is matcher-only; photometric RGB loss is disabled"))
        if matcher_color_supervision:
            dense_info = (
                f" [DENSE stride={matcher_dense_stride} cert≥{matcher_dense_cert_threshold}]"
                if matcher_dense_mode else " [sparse]"
            )
            print(blue(f"[NoiseInject] Matcher color supervision enabled — sparse color loss weight={matcher_color_weight:.3f}{dense_info}"))
        if matcher_adjacent_color_supervision:
            print(
                blue(
                    "[NoiseInject] Adjacent-frame matcher color supervision enabled — "
                    f"weight={matcher_adjacent_color_weight:.3f}"
                )
            )
    if camera_rgb_pose_only:
        print(blue("[NoiseInject] Full-image camera RGB loss updates POSE only"))
    if freeze_translation:
        print(blue("[NoiseInject] Translation FROZEN — keep initialized translation fixed during training"))
    if color_warmup_cycles > 0:
        print(blue(f"[NoiseInject] Color warmup enabled for first {color_warmup_cycles} cycles (RGB losses update colors only)"))
    if disable_depth_during_color_warmup:
        print(blue("[NoiseInject] LiDAR depth supervision disabled during color warmup"))
    if matcher_adjacent_support:
        print(blue(f"[NoiseInject] Adjacent-frame RGB support enabled (offset={matcher_adjacent_max_offset}, radius={matcher_support_radius_px:.1f}px)"))
    if post_warmup_rgb_reg_scale > 0:
        print(
            blue(
                "[NoiseInject] Post-warmup RGB regularizer enabled — "
                f"full-image L1+SSIM weight = supervised_ratio × {post_warmup_rgb_reg_scale:.4f}"
            )
        )

    def get_temporal_support(frame: int) -> np.ndarray | None:
        if not matcher_adjacent_support or matcher is None or frame not in cam_images:
            return None
        return _get_temporal_support_points(
            matcher=matcher,
            frame=int(frame),
            cam_images=cam_images,
            cache=temporal_support_cache,
            max_offset=matcher_adjacent_max_offset,
        )

    if initialize_pose_from_matcher:
        init_summary = _run_matcher_pose_update(
            matcher=matcher,
            gaussians=gaussians,
            pose_correction=pose_correction,
            cam_cameras=cam_cameras,
            cam_images=cam_images,
            args=args,
            frame_ids=sorted(cam_cameras.keys()),
            matcher_min_matches=matcher_min_matches,
            matcher_min_depth_matches=matcher_min_depth_matches,
            matcher_min_pnp_inliers=matcher_min_pnp_inliers,
            matcher_pnp_reproj_error=matcher_pnp_reproj_error,
            matcher_pnp_iterations=matcher_pnp_iterations,
            matcher_depth_min=matcher_depth_min,
            matcher_depth_max=matcher_depth_max,
            matcher_depth_percentile_low=matcher_depth_percentile_low,
            matcher_depth_percentile_high=matcher_depth_percentile_high,
            matcher_depth_use_inverse=matcher_depth_use_inverse,
            matcher_update_blend=1.0,
            temporal_support_getter=get_temporal_support,
            temporal_support_radius_px=matcher_support_radius_px,
        )
        if init_summary["status"] == "applied":
            print(
                blue(
                    "[NoiseInject] Matcher PnP init applied — "
                    f"frames={init_summary['frames_used']} matches={init_summary['matches_used']} "
                    f"dR={init_summary['rotation_delta_deg']:.4f}° dT={init_summary['translation_delta_m']:.4f}m"
                )
            )
        else:
            print(blue(f"[NoiseInject] Matcher PnP init skipped: {init_summary['reason']}"))

    # ── Pose optimizer: stage 1 ────────────────────────────────
    optimizer_param_groups = []
    if not freeze_rotation:
        optimizer_param_groups.append(
            {"params": [pose_correction.delta_rotations_quat], "lr": rotation_lr}
        )
    if not pose_correction.use_gt_translation and not two_stage and not freeze_translation:
        optimizer_param_groups.append(
            {"params": [pose_correction.delta_translations], "lr": translation_lr}
        )
    if not optimizer_param_groups:
        raise ValueError("No parameters to optimize: both rotation and translation are frozen.")
    pose_optimizer = torch.optim.Adam(optimizer_param_groups)

    # ── ReduceLROnPlateau for pose rotation ───────────────────
    pose_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pose_optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
        threshold=1e-3,
    ) if lr_patience > 0 else None

    init_R   = _effective_R(pose_correction)
    init_err = _rotation_error_deg(init_R, gt_l2c_R)
    init_T_err = _translation_error_m(pose_correction, gt_l2c_T)
    total_iters = total_cycles * iters_per_cycle
    print(blue(f"[NoiseInject] Init rotation error vs GT: {init_err:.4f}°"))
    print(blue(f"[NoiseInject] Init translation error vs GT: {init_T_err:.4f} m"))
    print(blue(f"[NoiseInject] total_iters={total_iters} ({total_cycles}×{iters_per_cycle}), "
               f"rotation_lr={rotation_lr}, translation_lr={translation_lr}, "
               f"use_gt_translation={pose_correction.use_gt_translation}"))
    if disable_depth_after_cycle > 0:
        print(blue(f"[NoiseInject] Depth supervision disabled after cycle {disable_depth_after_cycle}"))

    if two_stage:
        print(blue(f"[NoiseInject] TWO-STAGE mode: rotation-only until cycle {translation_start_cycle}, "
                   f"then freeze xyz+colors + optimise translation"))

    global_iter   = 0
    gaussian_iter = 0
    frame_stack   = []
    psnr_accum    = 0.0
    psnr_count    = 0
    loss_accum       = 0.0
    loss_depth_accum = 0.0
    loss_rgb_accum   = 0.0
    loss_match_color_accum = 0.0
    loss_match_adjacent_accum = 0.0
    # Best-T tracking: save the delta_T that achieved the lowest T_err
    best_T_err     = float("inf")
    best_delta_T   = pose_correction.delta_translations.data.clone()
    start_cycle    = 1

    # ── Resume from cycle checkpoint ──────────────────────────
    if resume_cycle_ckpt is not None:
        print(blue(f"[NoiseInject] Resuming from cycle checkpoint: {resume_cycle_ckpt}"))
        ckpt = torch.load(resume_cycle_ckpt, weights_only=False, map_location="cuda")
        restore_gaussian_state(gaussians, ckpt["gaussian_state"], args)
        if "pose_correction_state" in ckpt:
            # New-format checkpoint: full state including accumulated base_q
            pose_correction.load_state_dict(
                {k: v.to("cuda") for k, v in ckpt["pose_correction_state"].items()})
        else:
            # Legacy checkpoint: only delta_q and delta_T saved
            pose_correction.delta_rotations_quat.data.copy_(
                ckpt["delta_rotations_quat"].to("cuda"))
            pose_correction.delta_translations.data.copy_(
                ckpt["delta_translations"].to("cuda"))
        pose_correction.update_extrinsics()
        global_iter   = ckpt["global_iter"]
        gaussian_iter = ckpt["global_iter"]
        best_T_err    = ckpt["best_T_err"]
        best_delta_T  = ckpt["best_delta_T"].to("cuda")
        stage2_active = ckpt["stage2_active"]
        start_cycle   = ckpt["cycle"] + 1
        print(blue(f"[NoiseInject] Resumed: start_cycle={start_cycle}, "
                   f"stage2_active={stage2_active}, best_T_err={best_T_err:.4f}m"))
        # restore_gaussian_state resets requires_grad — re-apply freezes
        _apply_gaussian_freezes()
        _apply_matcher_color_lr_scale()

    for cycle in range(start_cycle, total_cycles + 1):
        # ── Stage transition: enable translation at translation_start_cycle ──
        if two_stage and not stage2_active and cycle > translation_start_cycle:
            stage2_active = True
            # Freeze Gaussian xyz so they cannot absorb translation errors
            xyz_param = getattr(gaussians, "_xyz", None)
            if xyz_param is not None:
                xyz_param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                if any(p is xyz_param for p in pg["params"]):
                    pg["lr"] = 0.0
            # Freeze view-dependent SH colour features
            _color_attrs = ["_features_rest", "_features_rgb_rest", "_features_dc", "_features_rgb_dc"]
            for attr in _color_attrs:
                param = getattr(gaussians, attr, None)
                if param is not None:
                    param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                for attr in _color_attrs:
                    cparam = getattr(gaussians, attr, None)
                    if cparam is not None and any(p is cparam for p in pg["params"]):
                        pg["lr"] = 0.0
            print(blue(f"[NoiseInject] Stage 2 activated at cycle {cycle}: xyz+colors FROZEN, translation optimizer added"))
            pose_optimizer.add_param_group(
                {"params": [pose_correction.delta_translations], "lr": translation_lr}
            )
            if pose_lr_scheduler is not None:
                pose_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    pose_optimizer, mode="min", factor=lr_factor,
                    patience=lr_patience, min_lr=lr_min, threshold=1e-3,
                )

        # ── Per-cycle match precomputation ────────────────────
        # During color warmup, Gaussian geometry is frozen so rendered depth is stable —
        # reuse cycle 1's cache for all subsequent warmup cycles (no re-matching needed).
        warmup_active_now = bool(color_warmup_cycles > 0 and cycle <= color_warmup_cycles)
        skip_precompute = (match_once_per_cycle and cycle_match_cache
                           and warmup_active_now)
        if skip_precompute:
            print(f"[Cycle cache] warmup cycle {cycle}: reusing cached matches (depth unchanged)")
        elif match_once_per_cycle and matcher is not None and (matcher_color_supervision or matcher_adjacent_color_supervision):
            cycle_match_cache = _precompute_cycle_match_cache(
                matcher=matcher,
                gaussians=gaussians,
                pose_correction=pose_correction,
                cam_cameras=cam_cameras,
                cam_images=cam_images,
                frame_ids=[f for f in frame_ids_train if f in cam_cameras],
                args=args,
                matcher_min_matches=matcher_min_matches,
                matcher_min_depth_matches=matcher_min_depth_matches,
                matcher_depth_min=matcher_depth_min,
                matcher_depth_max=matcher_depth_max,
                matcher_depth_percentile_low=matcher_depth_percentile_low,
                matcher_depth_percentile_high=matcher_depth_percentile_high,
                matcher_depth_use_inverse=matcher_depth_use_inverse,
                temporal_support_getter=get_temporal_support if matcher_adjacent_support else None,
                temporal_support_radius_px=matcher_support_radius_px,
                dense_mode=matcher_dense_mode,
                dense_stride=matcher_dense_stride,
                dense_cert_threshold=matcher_dense_cert_threshold,
            )
            # Build adjacent GT color cache lazily (GT images never change)
            if matcher_adjacent_color_supervision and temporal_pair_cache and not adj_gt_cache:
                adj_gt_cache = _precompute_adjacent_gt_cache(temporal_pair_cache, cam_images)

        for it in range(1, iters_per_cycle + 1):
            global_iter += 1
            gaussian_iter += 1
            color_warmup_active = bool(color_warmup_cycles > 0 and cycle <= color_warmup_cycles)
            if not freeze_gaussians:
                gaussians.update_learning_rate(gaussian_iter)

            if not frame_stack:
                frame_stack = list(frame_ids_train)
                random.shuffle(frame_stack)
            frame = frame_stack.pop()

            # ── LiDAR depth loss ───────────────────────────────
            # Skipped when Gaussians are frozen: the T_l2c extrinsic does not
            # appear in the LiDAR raytracing path, so this render provides no
            # pose gradient and only wastes compute.
            loss_depth = torch.tensor(0.0, device="cuda")
            depth_active = (
                not freeze_gaussians
                and (disable_depth_after_cycle <= 0 or cycle <= disable_depth_after_cycle)
                and not (disable_depth_during_color_warmup and color_warmup_active)
            )
            if depth_active:
                render_pkg = raytracing(
                    frame, [gaussians], scene.train_lidar, background, args, depth_only=True
                )
                depth       = render_pkg["depth"].squeeze(-1)
                gt_mask     = scene.train_lidar.get_mask(frame).cuda()
                dyn_mask    = scene.train_lidar.get_dynamic_mask(frame).cuda()
                static_mask = gt_mask & ~dyn_mask
                gt_depth    = scene.train_lidar.get_depth(frame).cuda()
                loss_depth  = lambda_depth * l1_loss(depth[static_mask], gt_depth[static_mask])

            # ── Camera RGB loss ───────────────────────────────
            loss_rgb = torch.tensor(0.0, device="cuda")
            loss_match_color = torch.tensor(0.0, device="cuda")
            loss_match_adjacent = torch.tensor(0.0, device="cuda")
            loss_rgb_reg = torch.tensor(0.0, device="cuda")
            matcher_color_summary = None
            base_rgb_photo_loss = None
            if frame in cam_cameras and not matcher_pose_update:
                cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
                camera  = cam_cameras[frame].cuda()
                gt_rgb  = cam_images[frame].cuda()
                cam_render = render_camera(
                    camera, [gaussians], args,
                    cam_rotation=cam_R, cam_translation=cam_T,
                    require_rgb=True,
                )
                if cam_render["num_visible"] > 0:
                    pred_rgb = cam_render["rgb"].clamp(0.0, 1.0)
                    pred_chw = pred_rgb.permute(2, 0, 1)
                    gt_chw   = gt_rgb.permute(2, 0, 1)
                    Ll1      = l1_loss(pred_rgb, gt_rgb)
                    ssim_val = ssim(pred_chw, gt_chw)
                    base_rgb_photo_loss = lambda_rgb * (
                        (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
                    )
                    loss_rgb = base_rgb_photo_loss
                    psnr_accum += psnr(pred_chw, gt_chw).item()
                    psnr_count += 1
                if matcher_color_supervision:
                    if match_once_per_cycle and frame in cycle_match_cache:
                        # Fast path: use precomputed match targets, only need a color render
                        color_render = render_camera(
                            camera,
                            [gaussians],
                            args,
                            cam_rotation=cam_R.detach(),
                            cam_translation=cam_T.detach(),
                            require_rgb=True,
                        )
                        if int(color_render.get("num_visible", 0)) > 0:
                            loss_match_color, matcher_color_summary = _apply_cross_modal_cache_loss(
                                pred_rgb=color_render["rgb"].clamp(0.0, 1.0),
                                cache_entry=cycle_match_cache[frame],
                                matcher_color_weight=matcher_color_weight,
                            )
                    else:
                        depth_render = _render_camera_with_backend(
                            camera,
                            [gaussians],
                            args,
                            backend="surfel_rasterization",
                            cam_rotation=cam_R.detach(),
                            cam_translation=cam_T.detach(),
                            require_rgb=False,
                        )
                        color_render = render_camera(
                            camera,
                            [gaussians],
                            args,
                            cam_rotation=cam_R.detach(),
                            cam_translation=cam_T.detach(),
                            require_rgb=True,
                        )
                        if int(depth_render.get("num_visible", 0)) > 0 and int(color_render.get("num_visible", 0)) > 0:
                            loss_match_color, matcher_color_summary = _compute_matcher_color_loss(
                                matcher=matcher,
                                gt_rgb=gt_rgb,
                                pred_rgb=color_render["rgb"].clamp(0.0, 1.0),
                                depth=depth_render["depth"],
                                matcher_min_matches=matcher_min_matches,
                                matcher_min_depth_matches=matcher_min_depth_matches,
                                matcher_depth_min=matcher_depth_min,
                                matcher_depth_max=matcher_depth_max,
                                matcher_depth_percentile_low=matcher_depth_percentile_low,
                                matcher_depth_percentile_high=matcher_depth_percentile_high,
                                matcher_depth_use_inverse=matcher_depth_use_inverse,
                                matcher_color_weight=matcher_color_weight,
                                temporal_support_points=get_temporal_support(frame),
                                temporal_support_radius_px=matcher_support_radius_px,
                                dense_mode=matcher_dense_mode,
                                dense_stride=matcher_dense_stride,
                                dense_cert_threshold=matcher_dense_cert_threshold,
                            )
                    if (
                        post_warmup_rgb_reg_scale > 0.0
                        and not color_warmup_active
                        and base_rgb_photo_loss is not None
                        and matcher_color_summary is not None
                        and matcher_color_summary.get("status") == "applied"
                    ):
                        supervised_ratio = float(matcher_color_summary.get("supervised_ratio", 0.0))
                        if supervised_ratio > 0.0:
                            loss_rgb_reg = base_rgb_photo_loss * (supervised_ratio * float(post_warmup_rgb_reg_scale))
                if matcher_adjacent_color_supervision and int(cam_render.get("num_visible", 0)) > 0:
                    if match_once_per_cycle and adj_gt_cache:
                        loss_match_adjacent, _ = _apply_adjacent_cache_loss(
                            frame=int(frame),
                            pred_rgb=pred_rgb,
                            gaussians=gaussians,
                            pose_correction=pose_correction,
                            cam_cameras=cam_cameras,
                            args=args,
                            adj_gt_cache=adj_gt_cache,
                            adjacent_color_weight=matcher_adjacent_color_weight,
                            matcher_min_matches=matcher_min_matches,
                        )
                    else:
                        loss_match_adjacent, _ = _compute_adjacent_matcher_color_loss(
                            matcher=matcher,
                            frame=int(frame),
                            pred_rgb=pred_rgb,
                            gt_rgb=gt_rgb,
                            gaussians=gaussians,
                            pose_correction=pose_correction,
                            cam_cameras=cam_cameras,
                            cam_images=cam_images,
                            args=args,
                            temporal_pair_cache=temporal_pair_cache,
                            matcher_adjacent_max_offset=matcher_adjacent_max_offset,
                            matcher_min_matches=matcher_min_matches,
                            adjacent_color_weight=matcher_adjacent_color_weight,
                        )

            total_loss = loss_depth + loss_rgb + loss_match_color + loss_match_adjacent + loss_rgb_reg
            if (freeze_covariance or rgb_only_updates_color or camera_rgb_pose_only or matcher_color_supervision or matcher_adjacent_color_supervision) and not freeze_gaussians:
                # Selectively gate RGB gradients on Gaussian parameters, then add LiDAR depth grads.
                rgb_zero_attrs = []
                retain_graph_for_rgb = bool(loss_match_color.requires_grad or loss_match_adjacent.requires_grad or loss_rgb_reg.requires_grad)
                if loss_rgb.requires_grad:
                    if color_warmup_active:
                        rgb_zero_attrs = [attr for attr in _GAUSSIAN_ATTRS if attr not in _COLOR_ATTRS]
                    elif camera_rgb_pose_only:
                        rgb_zero_attrs = list(_GAUSSIAN_ATTRS)
                    else:
                        if freeze_covariance:
                            rgb_zero_attrs.extend(["_scaling", "_rotation", "_opacity", "_opacity_cam"])
                        if rgb_only_updates_color:
                            rgb_zero_attrs.extend(_RGB_FROZEN_ATTRS)
                        rgb_zero_attrs = list(dict.fromkeys(rgb_zero_attrs))
                    loss_rgb.backward(retain_graph=retain_graph_for_rgb)
                    for attr in rgb_zero_attrs:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_match_color.requires_grad:
                    loss_match_color.backward(retain_graph=bool(loss_match_adjacent.requires_grad or loss_rgb_reg.requires_grad))
                    for attr in [attr for attr in _GAUSSIAN_ATTRS if attr not in _COLOR_ATTRS]:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_match_adjacent.requires_grad:
                    loss_match_adjacent.backward(retain_graph=bool(loss_rgb_reg.requires_grad))
                    for attr in [attr for attr in _GAUSSIAN_ATTRS if attr not in _COLOR_ATTRS]:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_rgb_reg.requires_grad:
                    loss_rgb_reg.backward()
                    for attr in [attr for attr in _GAUSSIAN_ATTRS if attr not in _COLOR_ATTRS]:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_depth.requires_grad:
                    loss_depth.backward()
            else:
                total_loss.backward()
            loss_accum       += total_loss.item()
            loss_depth_accum += loss_depth.item()
            loss_rgb_accum   += loss_rgb.item()
            loss_match_color_accum += loss_match_color.item()
            loss_match_adjacent_accum += loss_match_adjacent.item()

            if not freeze_gaussians:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if cycle > warmup_cycles and not matcher_pose_update and not color_warmup_active:
                pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

        # ── Fold delta into base once per cycle ────────────────
        matcher_update_summary = None
        if matcher_pose_update and cycle > warmup_cycles and cycle > color_warmup_cycles and matcher_update_interval > 0 and cycle % matcher_update_interval == 0:
            matcher_update_summary = _run_matcher_pose_update(
                matcher=matcher,
                gaussians=gaussians,
                pose_correction=pose_correction,
                cam_cameras=cam_cameras,
                cam_images=cam_images,
                args=args,
                frame_ids=sorted(cam_cameras.keys()),
                matcher_min_matches=matcher_min_matches,
                matcher_min_depth_matches=matcher_min_depth_matches,
                matcher_min_pnp_inliers=matcher_min_pnp_inliers,
                matcher_pnp_reproj_error=matcher_pnp_reproj_error,
                matcher_pnp_iterations=matcher_pnp_iterations,
                matcher_depth_min=matcher_depth_min,
                matcher_depth_max=matcher_depth_max,
                matcher_depth_percentile_low=matcher_depth_percentile_low,
                matcher_depth_percentile_high=matcher_depth_percentile_high,
                matcher_depth_use_inverse=matcher_depth_use_inverse,
                matcher_update_blend=matcher_update_blend,
                temporal_support_getter=get_temporal_support,
                temporal_support_radius_px=matcher_support_radius_px,
            )
        elif cycle > warmup_cycles and cycle > color_warmup_cycles:
            pose_correction.update_extrinsics()

        # ── Periodic Gaussian reset: keep pose, refresh scene ──
        if (
            reset_gaussians_every > 0
            and initial_gaussian_state is not None
            and cycle < total_cycles
            and cycle % reset_gaussians_every == 0
        ):
            restore_gaussian_state(gaussians, initial_gaussian_state, args)
            _apply_gaussian_freezes()
            gaussian_iter = 0
            frame_stack = []
            print(blue(f"[NoiseInject] Gaussian state RESET to init at cycle {cycle} (pose kept)"))

        # ── End-of-cycle logging ───────────────────────────────
        eff_R    = _effective_R(pose_correction)
        rot_err  = _rotation_error_deg(eff_R, gt_l2c_R)
        T_err    = _translation_error_m(pose_correction, gt_l2c_T)
        avg_psnr = psnr_accum / psnr_count if psnr_count > 0 else 0.0
        avg_loss  = loss_accum       / iters_per_cycle
        avg_depth = loss_depth_accum / iters_per_cycle
        avg_rgb   = loss_rgb_accum   / iters_per_cycle
        avg_match_color = loss_match_color_accum / iters_per_cycle
        avg_match_adjacent = loss_match_adjacent_accum / iters_per_cycle
        psnr_accum = 0.0
        psnr_count = 0
        loss_accum       = 0.0
        loss_depth_accum = 0.0
        loss_rgb_accum   = 0.0
        loss_match_color_accum = 0.0
        loss_match_adjacent_accum = 0.0

        # Track best translation seen so far
        if T_err < best_T_err:
            best_T_err = T_err
            best_delta_T = pose_correction.delta_translations.data.clone()

        # ── ReduceLROnPlateau step ────────────────────────────
        cur_rot_lr = pose_optimizer.param_groups[0]["lr"]
        if pose_lr_scheduler is not None and cycle > warmup_cycles:
            pose_lr_scheduler.step(rot_err)
            new_rot_lr = pose_optimizer.param_groups[0]["lr"]
            if new_rot_lr < cur_rot_lr:
                print(blue(f"  [LR] rotation_lr reduced: {cur_rot_lr:.2e} → {new_rot_lr:.2e}"))
            cur_rot_lr = new_rot_lr

        print(yellow(
            f"  Cycle {cycle:3d}/{total_cycles}  rot_err={rot_err:.4f}°  "
            f"T_err={T_err:.4f}m  "
            f"PSNR={avg_psnr:.2f} dB  loss={avg_loss:.5f}  "
            f"[d={avg_depth:.4f}  rgb={avg_rgb:.4f}  match_rgb={avg_match_color:.4f}  adj_rgb={avg_match_adjacent:.4f}]  lr={cur_rot_lr:.2e}"
        ))
        if matcher_update_summary is not None:
            if matcher_update_summary["status"] == "applied":
                print(
                    blue(
                        "    [matcher] "
                        f"frames={matcher_update_summary['frames_used']}  "
                        f"matches={matcher_update_summary['matches_used']}  "
                        f"reproj={matcher_update_summary['mean_reproj_px']:.3f}px  "
                        f"dR={matcher_update_summary['rotation_delta_deg']:.4f}°  "
                        f"dT={matcher_update_summary['translation_delta_m']:.4f}m"
                    )
                )
            else:
                print(blue(f"    [matcher] skipped: {matcher_update_summary['reason']}"))

        if tb_writer is not None:
            tb_writer.add_scalar("calib/rot_err_deg",  rot_err,  cycle)
            tb_writer.add_scalar("calib/trans_err_m",  T_err,    cycle)
            tb_writer.add_scalar("calib/psnr_db",      avg_psnr, cycle)
            tb_writer.add_scalar("calib/loss",         avg_loss, cycle)
            tb_writer.add_scalar("calib/loss_depth",   avg_depth, cycle)
            tb_writer.add_scalar("calib/loss_rgb",     avg_rgb,  cycle)
            tb_writer.add_scalar("calib/loss_match_color", avg_match_color, cycle)
            tb_writer.add_scalar("calib/loss_match_adjacent", avg_match_adjacent, cycle)
            if matcher_update_summary is not None and matcher_update_summary["status"] == "applied":
                tb_writer.add_scalar("calib/matcher_frames", matcher_update_summary["frames_used"], cycle)
                tb_writer.add_scalar("calib/matcher_matches", matcher_update_summary["matches_used"], cycle)
                tb_writer.add_scalar("calib/matcher_reproj_px", matcher_update_summary["mean_reproj_px"], cycle)
                tb_writer.add_scalar("calib/matcher_rotation_delta_deg", matcher_update_summary["rotation_delta_deg"], cycle)
                tb_writer.add_scalar("calib/matcher_translation_delta_m", matcher_update_summary["translation_delta_m"], cycle)

        # ── Cycle checkpoint ──────────────────────────────────
        if (cycle_ckpt_dir is not None
                and save_cycle_every > 0
                and cycle % save_cycle_every == 0):
            os.makedirs(cycle_ckpt_dir, exist_ok=True)
            ckpt_payload = {
                "cycle":                cycle,
                "global_iter":          global_iter,
                "best_T_err":           best_T_err,
                "best_delta_T":         best_delta_T.cpu(),
                "stage2_active":        stage2_active,
                "delta_rotations_quat": pose_correction.delta_rotations_quat.data.cpu(),
                "delta_translations":   pose_correction.delta_translations.data.cpu(),
                # Full pose_correction state (includes base_lidar_to_camera_quat which
                # accumulates the rotation via update_extrinsics(); delta_q alone is
                # always identity after each fold and cannot reconstruct the rotation).
                "pose_correction_state": {k: v.cpu() for k, v in
                                          pose_correction.state_dict().items()},
                "gaussian_state":       save_gaussian_state(gaussians),
            }
            ckpt_path = os.path.join(cycle_ckpt_dir, f"cycle_{cycle:04d}.pth")
            torch.save(ckpt_payload, ckpt_path)
            # Permanently preserve the last stage-1 checkpoint so stage-2
            # can always be re-run without repeating stage 1.
            if two_stage and cycle == translation_start_cycle:
                stage1_path = os.path.join(cycle_ckpt_dir, "stage1_final.pth")
                torch.save(ckpt_payload, stage1_path)
                print(blue(f"  [ckpt] stage-1 final checkpoint → {stage1_path}"))
            # Keep only the 3 most recent rolling checkpoints to save disk space
            existing = sorted(
                f for f in os.listdir(cycle_ckpt_dir)
                if f.startswith("cycle_") and f.endswith(".pth")
            )
            for old in existing[:-3]:
                try:
                    os.remove(os.path.join(cycle_ckpt_dir, old))
                except OSError:
                    pass
            print(blue(f"  [ckpt] saved cycle checkpoint → {ckpt_path}"))

    # Restore best translation
    with torch.no_grad():
        pose_correction.delta_translations.copy_(best_delta_T)

    final_R     = _effective_R(pose_correction)
    final_err   = _rotation_error_deg(final_R, gt_l2c_R)
    final_T_err = _translation_error_m(pose_correction, gt_l2c_T)
    print(green(f"\n[NoiseInject] Init  rot : {init_err:.4f}°   trans: {init_T_err:.4f} m"))
    print(green(f"[NoiseInject] Final rot : {final_err:.4f}°   trans: {final_T_err:.4f} m  (best T={best_T_err:.4f} m)"))
    print(green(f"[NoiseInject] Rot improvement : {init_err - final_err:+.4f}°"))
    print(green(f"[NoiseInject] Trans improvement: {init_T_err - final_T_err:+.4f} m"))
    return final_R


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiDAR-camera extrinsic calibration via 3DGS")
    parser.add_argument("-dc", "--data_config",  required=True)
    parser.add_argument("-ec", "--exp_config",   required=True)
    parser.add_argument("--checkpoint",          default=None,
                        help="Gaussian checkpoint (.pth) to use as base state")
    parser.add_argument("--downsample_ratio",    type=float, default=1.0,
                        help="Randomly keep this fraction of Gaussians after loading "
                             "checkpoint (e.g. 0.5 keeps 50%%). Default 1.0 = no downsampling. "
                             "Ignored if --voxel_size is set.")
    parser.add_argument("--voxel_size",          type=float, default=0.0,
                        help="Voxel size (meters) for spatial downsampling after loading "
                             "checkpoint. Keeps one Gaussian per voxel cell. 0.0 = disabled.")
    parser.add_argument("--init_rot_deg",        type=float, default=15.0,
                        help="Initial rotation error magnitude (degrees)")
    parser.add_argument("--init_rot_axis",       type=float, nargs=3, default=None,
                        help="Fixed rotation axis (x y z); random if omitted")
    parser.add_argument("--init_trans_xyz",      type=float, nargs=3, default=None,
                        help="Initial translation error (dx dy dz) in meters added to "
                             "the estimated extrinsic at start. E.g. --init_trans_xyz 0.07 0.13 0.10")
    parser.add_argument("--total_cycles",        type=int, default=100)
    parser.add_argument("--iters_per_cycle",     type=int, default=100)
    parser.add_argument("--rotation_lr",         type=float, default=2e-3)
    parser.add_argument("--freeze_gaussians",    action="store_true",
                        help="Freeze Gaussian parameters; only pose gets gradient")
    parser.add_argument("--freeze_xyz",          action="store_true",
                        help="Freeze Gaussian mean positions (_xyz) during calibration.")
    parser.add_argument("--freeze_colors",       action="store_true",
                        help="Freeze Gaussian SH color features during calibration.")
    parser.add_argument("--freeze_covariance",   action="store_true",
                        help="Freeze Gaussian scale and rotation (covariance) during calibration.")
    parser.add_argument("--translation_start_cycle", type=int, default=0,
                        help="Two-stage calibration: optimise rotation-only for this many cycles, "
                             "then freeze xyz+colors and add translation optimisation. "
                             "0 = simultaneous (default). Ignored when use_gt_translation=True.")
    parser.add_argument("--warmup_cycles",         type=int, default=0,
                        help="Freeze pose optimizer for this many cycles at the start. Default: 0.")
    parser.add_argument("--freeze_rotation",       action="store_true",
                        help="Freeze rotation, optimise translation only.")
    parser.add_argument("--freeze_translation",    action="store_true",
                        help="Freeze translation, keeping the initialized translation fixed while optimizing other variables.")
    parser.add_argument("--lr_patience",          type=int,   default=0,
                        help="ReduceLROnPlateau patience (cycles). 0 = disabled.")
    parser.add_argument("--lr_factor",            type=float, default=0.5,
                        help="Multiplicative factor for ReduceLROnPlateau. Default: 0.5.")
    parser.add_argument("--lr_min",               type=float, default=1e-5,
                        help="Minimum rotation_lr floor for ReduceLROnPlateau. Default: 1e-5.")
    parser.add_argument("--use_gt_translation",  action="store_true",
                        help="Lock translation to GT (skip translation optimisation).")
    parser.add_argument("--translation_lr",      type=float, default=0.0015,
                        help="Learning rate for delta_translations (default: 0.0015)")
    parser.add_argument("--resume_from",         default=None,
                        help="Path to best_rotation.npz from a previous run; "
                             "use its final_R as starting rotation (overrides --init_rot_deg)")
    parser.add_argument("--resume_cycle_ckpt",   default=None,
                        help="Path to a cycle_NNNN.pth checkpoint; fully resumes Gaussian + "
                             "pose state from that cycle (overrides --resume_from)")
    parser.add_argument("--save_cycle_every",    type=int, default=5,
                        help="Save a cycle checkpoint every N cycles (default: 5). "
                             "Only the 3 most recent are kept. Set 0 to disable.")
    parser.add_argument("--reset_gaussians_every", type=int, default=0,
                        help="Reset Gaussians to their initial state every N cycles "
                             "while keeping the current pose. 0 = disabled.")
    parser.add_argument("--disable_depth_after_cycle", type=int, default=0,
                        help="Disable LiDAR raytracing depth supervision after this cycle. "
                             "0 = keep depth supervision for all cycles.")
    parser.add_argument("--rgb_only_updates_color", action="store_true",
                        help="For camera RGB supervision, zero Gaussian grads on everything "
                             "except color features. Pose gradients are kept.")
    parser.add_argument("--matcher_pose_update", action="store_true",
                        help="Update pose once per cycle from matchanything-roma RGB-vs-rendered-depth calibration.")
    parser.add_argument("--matcher_color_supervision", action="store_true",
                        help="Use matchanything-roma correspondences to map GT RGB colors onto rendered depth pixels for sparse color supervision.")
    parser.add_argument("--matcher_color_weight", type=float, default=1.0,
                        help="Weight for sparse matcher color supervision.")
    parser.add_argument("--camera_rgb_pose_only", action="store_true",
                        help="Use full-image camera L1/SSIM to update pose only; Gaussian grads from this loss are zeroed.")
    parser.add_argument("--color_warmup_cycles", type=int, default=0,
                        help="For the first N cycles, RGB losses update only Gaussian color parameters and do not update pose.")
    parser.add_argument("--initialize_pose_from_matcher", action="store_true",
                        help="Before training, run a matcher-based shared PnP initialization and use it as the starting pose.")
    parser.add_argument("--matcher_adjacent_support", action="store_true",
                        help="Filter matcher correspondences by adjacent-frame RGB-RGB support.")
    parser.add_argument("--matcher_adjacent_max_offset", type=int, default=1,
                        help="Use adjacent RGB frames up to this temporal offset when building support.")
    parser.add_argument("--matcher_support_radius_px", type=float, default=6.0,
                        help="Max pixel distance between cross-modal matches and adjacent-frame support points.")
    parser.add_argument("--matcher_adjacent_color_supervision", action="store_true",
                        help="Add sparse adjacent-frame RGB-RGB matcher supervision on rendered colors.")
    parser.add_argument("--matcher_adjacent_color_weight", type=float, default=1.0,
                        help="Weight for adjacent-frame RGB-RGB matcher color supervision.")
    parser.add_argument("--disable_depth_during_color_warmup", action="store_true",
                        help="Disable LiDAR depth/raytracing supervision during color warmup cycles to reduce cost and interference.")
    parser.add_argument("--post_warmup_rgb_reg_scale", type=float, default=0.0,
                        help="After color warmup, add a full-image RGB L1+SSIM regularizer on colors only, weighted by supervised_ratio * this scale.")
    parser.add_argument("--matcher_update_interval", type=int, default=None,
                        help="Run matcher pose update every N cycles. Defaults to pose_correction.matcher_update_interval.")
    parser.add_argument("--matcher_update_blend", type=float, default=None,
                        help="Blend factor for each matcher pose update. Defaults to pose_correction.matcher_update_blend.")
    parser.add_argument("--matcher_name", default=None,
                        help="Matcher backend for cycle pose update. Currently only matchanything-roma is supported.")
    parser.add_argument("--matcher_max_num_keypoints", type=int, default=None)
    parser.add_argument("--matcher_min_matches", type=int, default=None)
    parser.add_argument("--matcher_dense_mode", action="store_true",
                        help="Use RoMa dense warp field for cross-modal matching instead of sparse keypoint sampling. "
                             "Gives ~10-100x more supervision pixels per frame.")
    parser.add_argument("--matcher_dense_stride", type=int, default=4,
                        help="Pixel stride for the dense query grid (default 4 = every 4th pixel). "
                             "Smaller = denser but slower per-iter matching.")
    parser.add_argument("--matcher_dense_cert_threshold", type=float, default=0.02,
                        help="Minimum RoMa certainty to keep a dense match (default 0.02).")
    parser.add_argument("--no_match_once_per_cycle", action="store_true",
                        help="Disable per-cycle match precomputation (re-run matcher every iteration).")
    parser.add_argument("--matcher_min_depth_matches", type=int, default=None)
    parser.add_argument("--matcher_min_pnp_inliers", type=int, default=None)
    parser.add_argument("--matcher_ransac_reproj_thresh", type=float, default=None)
    parser.add_argument("--matcher_pnp_reproj_error", type=float, default=None)
    parser.add_argument("--matcher_pnp_iterations", type=int, default=None)
    parser.add_argument("--matcher_depth_min", type=float, default=None)
    parser.add_argument("--matcher_depth_max", type=float, default=None)
    parser.add_argument("--matcher_depth_use_inverse", type=int, choices=[0, 1], default=None)
    parser.add_argument("--matcher_depth_percentile_low", type=float, default=None)
    parser.add_argument("--matcher_depth_percentile_high", type=float, default=None)
    parser.add_argument("--lidar_updates_opacity_covariance_only", action="store_true",
                        help="Freeze xyz+colors so LiDAR supervision only updates opacity/covariance related Gaussian params.")
    parser.add_argument("--lambda_depth",           type=float, default=None,
                        help="Override the depth loss weight (default from exp config, "
                             "usually 1.0). Set to 0 to disable depth supervision entirely.")
    parser.add_argument("--lambda_rgb",             type=float, default=None,
                        help="Override the full-image camera RGB loss weight. Set to 0 to disable dense RGB supervision.")
    parser.add_argument("--matcher_color_lr_scale", type=float, default=1.0,
                        help="Multiply optimizer learning rates for camera RGB feature params when using matcher color supervision.")
    parser.add_argument("--output_dir",          default=None)
    parser.add_argument("--gpu",                 type=int, default=None)
    cli = parser.parse_args()

    if cli.gpu is not None:
        torch.cuda.set_device(cli.gpu)

    args = parse(cli.exp_config)
    args = parse(cli.data_config, args)
    _dtype = str(getattr(args, "data_type", "")).lower()

    scene_id = getattr(args, "scene_id", "calib_scene")
    out_dir  = cli.output_dir or os.path.join("output", "calib", scene_id)
    os.makedirs(out_dir, exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # ── Scene ─────────────────────────────────────────────────
    print(blue("[Calib] Loading scene..."))
    scene = dataloader.load_scene(args.source_dir, args)
    gaussians = scene.gaussians_assets[0]

    # ── Camera data ───────────────────────────────────────────
    camera_scale = float(getattr(args, "camera_scale", 1))
    _dtype_norm = _dtype.replace("-", "").replace("_", "")
    if "kitticalib" in _dtype_norm:
        scene_name = getattr(args, "kitti_calib_scene", None)
        if scene_name is None:
            print(red("[Calib] kitti_calib_scene not set. Exiting."))
            sys.exit(1)
        frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
        cam_cameras, cam_images = load_kitti_calib_cameras(
            args.source_dir, args, scene_name=scene_name,
            frame_ids=frame_ids, scale=camera_scale,
        )
    elif "kitti" in _dtype_norm:
        kitti_seq = getattr(args, "kitti_seq", None)
        if kitti_seq is None:
            print(red("[Calib] kitti_seq not set. Exiting."))
            sys.exit(1)
        frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
        cam_cameras, cam_images = load_kitti360_cameras(
            args.source_dir, args, seq_num=int(kitti_seq),
            frame_ids=frame_ids, scale=camera_scale,
        )
    elif "waymo" in _dtype_norm:
        camera_id = int(getattr(args, "waymo_camera_id", 1))
        cam_cameras, cam_images = load_waymo_cameras(
            args.source_dir, args, camera_id=camera_id, scale=camera_scale,
        )
    elif "pandaset" in _dtype_norm or "panda" in _dtype_norm:
        camera_name = getattr(args, "pandaset_camera_name", "front_camera")
        cam_cameras, cam_images = load_pandaset_cameras(
            args.source_dir, args, camera_name=camera_name, scale=camera_scale,
        )
    else:
        print(red(f"[Calib] Dataset type '{_dtype}' not supported. "
                  f"Supported: KITTICalib, KITTI (KITTI-360), Waymo, PandaSet."))
        sys.exit(1)
    print(blue(f"[Calib] Loaded {len(cam_cameras)} camera frames."))

    lidar_world_poses = {k: v.float() for k, v in scene.train_lidar.sensor2world.items()}

    # ── Load Gaussian checkpoint ──────────────────────────────
    if cli.checkpoint:
        print(blue(f"[Calib] Loading checkpoint: {cli.checkpoint}"))
        model_params, _ = torch.load(cli.checkpoint, weights_only=False, map_location="cuda")
        scene.restore(model_params, args.opt)
        for attr in _GAUSSIAN_ATTRS:
            p = getattr(gaussians, attr, None)
            if p is not None and not p.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(p.data.cuda()))
        gaussians.training_setup(args.opt)
        print(blue(f"[Calib] Checkpoint loaded: {gaussians.get_local_xyz.shape[0]} Gaussians."))
    else:
        print(blue("[Calib] No checkpoint — using initial point-cloud state."))
        gaussians.training_setup(args.opt)

    # ── Optional Gaussian downsampling ───────────────────────
    if cli.voxel_size > 0.0:
        xyz = gaussians.get_local_xyz.detach()          # (N, 3)
        origin = xyz.min(dim=0).values
        ijk = ((xyz - origin) / cli.voxel_size).long()  # (N, 3)
        grid_size = ijk.max(dim=0).values + 1
        voxel_key = ijk[:, 0] * (grid_size[1] * grid_size[2]) + \
                    ijk[:, 1] * grid_size[2] + ijk[:, 2]
        sorted_order = voxel_key.argsort(stable=True)
        sorted_keys  = voxel_key[sorted_order]
        first_mask   = torch.cat([
            torch.tensor([True], device="cuda"),
            sorted_keys[1:] != sorted_keys[:-1]
        ])
        keep_idx = sorted_order[first_mask]
        keep_idx, _ = keep_idx.sort()
        n_total = xyz.shape[0]
        for attr in _GAUSSIAN_ATTRS:
            p = getattr(gaussians, attr, None)
            if p is not None:
                setattr(gaussians, attr, torch.nn.Parameter(
                    p.data[keep_idx].requires_grad_(True)))
        n_after = gaussians.get_local_xyz.shape[0]
        gaussians.max_radii2D        = torch.zeros(n_after, device="cuda")
        gaussians.xyz_gradient_accum = torch.zeros((n_after, 1), device="cuda")
        gaussians.denom              = torch.zeros((n_after, 1), device="cuda")
        gaussians.training_setup(args.opt)
        print(blue(f"[Calib] Voxel-downsampled {n_total} → {n_after} Gaussians "
                   f"(voxel_size={cli.voxel_size:.3f}m)."))
    elif cli.downsample_ratio < 1.0:
        n_total = gaussians.get_local_xyz.shape[0]
        n_keep  = max(1, int(n_total * cli.downsample_ratio))
        keep_idx = torch.randperm(n_total, device="cuda")[:n_keep]
        keep_idx, _ = keep_idx.sort()
        for attr in _GAUSSIAN_ATTRS:
            p = getattr(gaussians, attr, None)
            if p is not None:
                setattr(gaussians, attr, torch.nn.Parameter(
                    p.data[keep_idx].requires_grad_(True)))
        n_after = gaussians.get_local_xyz.shape[0]
        gaussians.max_radii2D        = torch.zeros(n_after, device="cuda")
        gaussians.xyz_gradient_accum = torch.zeros((n_after, 1), device="cuda")
        gaussians.denom              = torch.zeros((n_after, 1), device="cuda")
        gaussians.training_setup(args.opt)
        print(blue(f"[Calib] Random-downsampled {n_total} → {n_after} Gaussians "
                   f"(ratio={cli.downsample_ratio:.2f})."))

    # ── Save base state ───────────────────────────────────────
    base_state = save_gaussian_state(gaussians)
    print(blue(f"[Calib] Base state saved: {gaussians.get_local_xyz.shape[0]} Gaussians."))

    # ── CameraPoseCorrection ──────────────────────────────────
    model_cfg = getattr(args, "model", None)
    pose_cfg  = getattr(model_cfg, "pose_correction", None)
    if pose_cfg is None:
        raise ValueError("model.pose_correction config is required for calibration.")
    pose_correction = CameraPoseCorrection(
        cam_cameras, pose_cfg, lidar_poses=lidar_world_poses
    ).cuda()
    pose_correction.use_gt_translation = cli.use_gt_translation

    gt_l2c_R = pose_correction.gt_lidar_to_camera_rotation[0].float().cuda()
    gt_l2c_T = pose_correction.gt_lidar_to_camera_translation[0].float().cuda()
    print(blue(f"[Calib] GT l2c translation: {gt_l2c_T.cpu().numpy()}"))

    # ── Apply initial rotation bias ───────────────────────────
    gt_l2c_q = matrix_to_quaternion(gt_l2c_R)
    if cli.resume_from is not None:
        npz = np.load(cli.resume_from)
        resume_R = torch.tensor(npz["final_R"], dtype=torch.float32, device="cuda")
        init_q = matrix_to_quaternion(resume_R)
        resume_err = _rotation_error_deg(resume_R, gt_l2c_R)
        print(blue(f"[Calib] Resuming from: {cli.resume_from}  (err={resume_err:.4f}°)"))
    elif cli.init_rot_deg > 0.0:
        if cli.init_rot_axis is not None:
            axis = F.normalize(
                torch.tensor(cli.init_rot_axis, dtype=torch.float32, device="cuda"), dim=0
            )
        else:
            axis = F.normalize(torch.randn(3, dtype=torch.float32, device="cuda"), dim=0)
        init_delta_q = axis_angle_to_quaternion(axis, math.radians(cli.init_rot_deg))
        init_q = quaternion_multiply(init_delta_q, gt_l2c_q)
        print(blue(f"[Calib] Init perturbation: {cli.init_rot_deg:.1f}° along {axis.cpu().numpy().round(3)}"))
    else:
        init_q = gt_l2c_q.clone()
        print(blue("[Calib] Starting from GT rotation."))

    # Set base = init_q, delta = identity
    with torch.no_grad():
        if init_q[0] < 0:
            init_q = -init_q
        pose_correction.base_lidar_to_camera_quat.data[0].copy_(init_q)
        pose_correction.base_lidar_to_camera_rotation.data[0].copy_(
            quaternion_to_matrix(init_q)
        )
        pose_correction.delta_rotations_quat.data.fill_(0.0)
        pose_correction.delta_rotations_quat.data[:, 0] = 1.0

    # ── Apply initial translation bias ───────────────────────
    if cli.init_trans_xyz is not None:
        delta_t = torch.tensor(cli.init_trans_xyz, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pose_correction.base_lidar_to_camera_translation.data[0].copy_(
                gt_l2c_T + delta_t
            )
            pose_correction.delta_translations.data.zero_()
        init_t_err = delta_t.norm().item()
        print(blue(f"[Calib] Init translation perturbation: {cli.init_trans_xyz}  "
                   f"(err={init_t_err:.4f} m)"))

    # ── Run calibration ───────────────────────────────────────
    final_R = run_noise_inject_calib(
        gaussians=gaussians,
        pose_correction=pose_correction,
        cam_cameras=cam_cameras,
        cam_images=cam_images,
        scene=scene,
        gt_l2c_R=gt_l2c_R,
        gt_l2c_T=gt_l2c_T,
        args=args,
        total_cycles=cli.total_cycles,
        iters_per_cycle=cli.iters_per_cycle,
        rotation_lr=cli.rotation_lr,
        translation_lr=cli.translation_lr,
        freeze_gaussians=cli.freeze_gaussians,
        freeze_xyz=cli.freeze_xyz,
        freeze_colors=cli.freeze_colors,
        freeze_covariance=cli.freeze_covariance,
        translation_start_cycle=cli.translation_start_cycle,
        warmup_cycles=cli.warmup_cycles,
        freeze_rotation=cli.freeze_rotation,
        freeze_translation=cli.freeze_translation,
        lr_patience=cli.lr_patience,
        lr_factor=cli.lr_factor,
        lr_min=cli.lr_min,
        lambda_rgb=cli.lambda_rgb if cli.lambda_rgb is not None else 1.0,
        initial_gaussian_state=base_state,
        tb_writer=tb_writer,
        cycle_ckpt_dir=os.path.join(out_dir, "cycle_ckpts"),
        save_cycle_every=cli.save_cycle_every,
        resume_cycle_ckpt=cli.resume_cycle_ckpt,
        reset_gaussians_every=cli.reset_gaussians_every,
        disable_depth_after_cycle=cli.disable_depth_after_cycle,
        rgb_only_updates_color=cli.rgb_only_updates_color,
        lambda_depth=cli.lambda_depth if cli.lambda_depth is not None else 1.0,
        matcher_pose_update=cli.matcher_pose_update,
        matcher_color_supervision=cli.matcher_color_supervision,
        matcher_color_weight=cli.matcher_color_weight,
        camera_rgb_pose_only=cli.camera_rgb_pose_only,
        color_warmup_cycles=cli.color_warmup_cycles,
        initialize_pose_from_matcher=cli.initialize_pose_from_matcher,
        matcher_adjacent_support=cli.matcher_adjacent_support,
        matcher_adjacent_max_offset=cli.matcher_adjacent_max_offset,
        matcher_support_radius_px=cli.matcher_support_radius_px,
        matcher_adjacent_color_supervision=cli.matcher_adjacent_color_supervision,
        matcher_adjacent_color_weight=cli.matcher_adjacent_color_weight,
        disable_depth_during_color_warmup=cli.disable_depth_during_color_warmup,
        post_warmup_rgb_reg_scale=cli.post_warmup_rgb_reg_scale,
        matcher_color_lr_scale=cli.matcher_color_lr_scale,
        matcher_update_interval=cli.matcher_update_interval if cli.matcher_update_interval is not None else int(getattr(pose_cfg, "matcher_update_interval", 1)),
        matcher_update_blend=cli.matcher_update_blend if cli.matcher_update_blend is not None else float(getattr(pose_cfg, "matcher_update_blend", 1.0)),
        matcher_name=cli.matcher_name or str(getattr(pose_cfg, "matcher_name", "matchanything-roma")),
        matcher_max_num_keypoints=cli.matcher_max_num_keypoints if cli.matcher_max_num_keypoints is not None else int(getattr(pose_cfg, "matcher_max_num_keypoints", 2048)),
        matcher_min_matches=cli.matcher_min_matches if cli.matcher_min_matches is not None else int(getattr(pose_cfg, "matcher_min_matches", 20)),
        matcher_min_depth_matches=cli.matcher_min_depth_matches if cli.matcher_min_depth_matches is not None else int(getattr(pose_cfg, "matcher_min_depth_matches", 12)),
        matcher_min_pnp_inliers=cli.matcher_min_pnp_inliers if cli.matcher_min_pnp_inliers is not None else int(getattr(pose_cfg, "matcher_min_pnp_inliers", 8)),
        matcher_ransac_reproj_thresh=cli.matcher_ransac_reproj_thresh if cli.matcher_ransac_reproj_thresh is not None else float(getattr(pose_cfg, "matcher_ransac_reproj_thresh", 3.0)),
        matcher_pnp_reproj_error=cli.matcher_pnp_reproj_error if cli.matcher_pnp_reproj_error is not None else float(getattr(pose_cfg, "matcher_pnp_reproj_error", 4.0)),
        matcher_pnp_iterations=cli.matcher_pnp_iterations if cli.matcher_pnp_iterations is not None else int(getattr(pose_cfg, "matcher_pnp_iterations", 1000)),
        matcher_depth_min=cli.matcher_depth_min if cli.matcher_depth_min is not None else float(getattr(pose_cfg, "matcher_depth_min", 0.1)),
        matcher_depth_max=cli.matcher_depth_max if cli.matcher_depth_max is not None else float(getattr(pose_cfg, "matcher_depth_max", 80.0)),
        matcher_depth_use_inverse=bool(cli.matcher_depth_use_inverse) if cli.matcher_depth_use_inverse is not None else bool(getattr(pose_cfg, "matcher_depth_use_inverse", True)),
        matcher_depth_percentile_low=cli.matcher_depth_percentile_low if cli.matcher_depth_percentile_low is not None else float(getattr(pose_cfg, "matcher_depth_percentile_low", 5.0)),
        matcher_depth_percentile_high=cli.matcher_depth_percentile_high if cli.matcher_depth_percentile_high is not None else float(getattr(pose_cfg, "matcher_depth_percentile_high", 95.0)),
        lidar_updates_opacity_covariance_only=cli.lidar_updates_opacity_covariance_only,
        matcher_dense_mode=cli.matcher_dense_mode,
        matcher_dense_stride=cli.matcher_dense_stride,
        matcher_dense_cert_threshold=cli.matcher_dense_cert_threshold,
        match_once_per_cycle=not cli.no_match_once_per_cycle,
    )

    _save_run_visualizations(
        out_dir=out_dir,
        gaussians=gaussians,
        pose_correction=pose_correction,
        cam_cameras=cam_cameras,
        cam_images=cam_images,
        scene=scene,
        gt_l2c_R=gt_l2c_R,
        gt_l2c_T=gt_l2c_T,
        args=args,
        matcher_name=cli.matcher_name or str(getattr(pose_cfg, "matcher_name", "matchanything-roma")),
        matcher_max_num_keypoints=cli.matcher_max_num_keypoints if cli.matcher_max_num_keypoints is not None else int(getattr(pose_cfg, "matcher_max_num_keypoints", 2048)),
        matcher_ransac_reproj_thresh=cli.matcher_ransac_reproj_thresh if cli.matcher_ransac_reproj_thresh is not None else float(getattr(pose_cfg, "matcher_ransac_reproj_thresh", 3.0)),
        matcher_min_matches=cli.matcher_min_matches if cli.matcher_min_matches is not None else int(getattr(pose_cfg, "matcher_min_matches", 20)),
        matcher_min_depth_matches=cli.matcher_min_depth_matches if cli.matcher_min_depth_matches is not None else int(getattr(pose_cfg, "matcher_min_depth_matches", 12)),
        matcher_depth_min=cli.matcher_depth_min if cli.matcher_depth_min is not None else float(getattr(pose_cfg, "matcher_depth_min", 0.1)),
        matcher_depth_max=cli.matcher_depth_max if cli.matcher_depth_max is not None else float(getattr(pose_cfg, "matcher_depth_max", 80.0)),
        matcher_depth_percentile_low=cli.matcher_depth_percentile_low if cli.matcher_depth_percentile_low is not None else float(getattr(pose_cfg, "matcher_depth_percentile_low", 5.0)),
        matcher_depth_percentile_high=cli.matcher_depth_percentile_high if cli.matcher_depth_percentile_high is not None else float(getattr(pose_cfg, "matcher_depth_percentile_high", 95.0)),
        matcher_depth_use_inverse=bool(cli.matcher_depth_use_inverse) if cli.matcher_depth_use_inverse is not None else bool(getattr(pose_cfg, "matcher_depth_use_inverse", True)),
        matcher_adjacent_support=cli.matcher_adjacent_support,
        matcher_adjacent_max_offset=cli.matcher_adjacent_max_offset,
        matcher_support_radius_px=cli.matcher_support_radius_px,
    )
    print(green(f"[Calib] Visualizations saved to: {os.path.join(out_dir, 'visualizations')}"))

    # ── Save result ───────────────────────────────────────────
    out_path = os.path.join(out_dir, "best_rotation.npz")
    np.savez(out_path,
             final_R=final_R.detach().cpu().numpy(),
             gt_R=gt_l2c_R.detach().cpu().numpy(),
             init_rot_deg=cli.init_rot_deg)
    print(green(f"[Calib] Saved to: {out_path}"))


if __name__ == "__main__":
    main()
