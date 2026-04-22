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
    --output_dir /mnt/xzy/hr-tiny-output/calib/my_exp

Usage (KITTI-360)
-----
python tools/calib.py \\
    -dc configs/kitti360/static/k3_cam.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir /mnt/xzy/hr-tiny-output/calib/kitti360_k3

Usage (Waymo)
-----
python tools/calib.py \\
    -dc configs/waymo/static/t0_cam.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir /mnt/xzy/hr-tiny-output/calib/waymo_t0

Usage (PandaSet)
-----
python tools/calib.py \\
    -dc configs/pandaset/static/1.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --total_cycles 300 --iters_per_cycle 150 \\
    --rotation_lr 0.002 --warmup_cycles 1 \\
    --output_dir /mnt/xzy/hr-tiny-output/calib/pandaset_1
"""

import argparse
import math
import json
import os
import random
import shlex
import sys
import time

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
from lib.scene.cameras import Camera
from lib.utils.console_utils import blue, green, red, yellow
from lib.utils.graphics_utils import fov2focal
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss, ssim
from lib.utils.other_utils import depth2normal
from lib.utils.rgbd_calibration import (
    _format_matcher_image,
    CameraModel,
    TemporalDepthResidualData,
    TemporalPhotometricResidualData,
    TemporalResidualData,
    build_weighted_frame_data_list,
    build_frame_correspondence,
    build_matcher,
    depth_to_match_image,
    filter_frame_correspondence_by_reference_cloud,
    filter_frame_data_by_gt_reprojection,
    filter_frame_data_by_shared_ransac,
    filter_frame_data_by_pose_disagreement,
    filter_frame_data_by_single_frame_pnp_stability,
    filter_frame_data_by_reprojection_consensus,
    initialize_shared_extrinsic,
    learn_gt_point_weight_predictor,
    match_cross_modal,
    match_cross_modal_dense,
    optimize_shared_extrinsic,
    sample_depth_values,
    sample_depth_values_vectorized,
    select_match_points,
    stratify_frame_data_by_depth,
)


DEFAULT_OUTPUT_ROOT = os.environ.get("HR_TINY_OUTPUT_ROOT", "/mnt/xzy/hr-tiny-output")


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

    When shared extrinsics are enabled, this reports the effective LiDAR->camera
    translation currently used by pose_correction, including GT-translation mode.
    """
    if getattr(pose_correction, "use_shared_lidar_extrinsic", False):
        if bool(getattr(pose_correction, "use_gt_translation", False)):
            eff_T = pose_correction.gt_lidar_to_camera_translation[0].float()
        else:
            base_T = pose_correction.base_lidar_to_camera_translation[0].float()
            delta_T = pose_correction.delta_translations[0].float()
            eff_T = base_T + delta_T
    else:
        base_T = pose_correction.base_translations[0].float()
        delta_T = pose_correction.delta_translations[0].float()
        eff_T = base_T + delta_T
    return (eff_T - gt_l2c_T).norm().item()


def _effective_T(pose_correction) -> torch.Tensor:
    """Current effective l2c translation as float32 vector."""
    if getattr(pose_correction, "use_shared_lidar_extrinsic", False):
        if bool(getattr(pose_correction, "use_gt_translation", False)):
            return pose_correction.gt_lidar_to_camera_translation[0].float()
        base_T = pose_correction.base_lidar_to_camera_translation[0].float()
        delta_T = pose_correction.delta_translations[0].float()
        return base_T + delta_T
    base_T = pose_correction.base_translations[0].float()
    delta_T = pose_correction.delta_translations[0].float()
    return base_T + delta_T


def _effective_R(pose_correction) -> torch.Tensor:
    """Current effective l2c rotation (delta ⊗ base) as float32 matrix."""
    dq = F.normalize(pose_correction.delta_rotations_quat[0].float(), dim=0)
    bq = F.normalize(pose_correction.base_lidar_to_camera_quat[0].float(), dim=0)
    eff_q = quaternion_multiply(dq, bq)
    return quaternion_to_matrix(eff_q)


def _summarize_frame_correspondences(
    frame_data_list: list,
    rgb_camera: CameraModel,
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
    ref_rvec: np.ndarray | None = None,
    ref_tvec: np.ndarray | None = None,
) -> list[dict]:
    dist_coeffs = rgb_camera.dist if rgb_camera.dist.size > 0 else None
    summaries: list[dict] = []
    for frame_data in frame_data_list:
        depths = np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        summary = {
            "frame_id": int(frame_data.frame_id),
            "frame_name": str(frame_data.frame_name),
            "num_matches": int(frame_data.points_3d.shape[0]),
            "depth_median": float(np.median(depths)) if depths.size else float("nan"),
            "depth_p10": float(np.percentile(depths, 10.0)) if depths.size else float("nan"),
            "depth_p90": float(np.percentile(depths, 90.0)) if depths.size else float("nan"),
            "single_frame_inliers": int(getattr(frame_data, "pnp_inliers", 0)),
            "single_frame_reproj_px": float(getattr(frame_data, "pnp_reproj_error", float("nan"))),
        }
        for prefix, rvec, tvec in (
            ("gt", gt_rvec, gt_tvec),
            ("reference", ref_rvec, ref_tvec),
        ):
            if rvec is None or tvec is None or frame_data.points_3d.size == 0:
                continue
            projected, _ = cv2.projectPoints(
                frame_data.points_3d.astype(np.float64),
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                rgb_camera.K,
                dist_coeffs,
            )
            residual = projected.reshape(-1, 2) - frame_data.rgb_points
            reproj = np.sqrt(np.sum(residual**2, axis=1))
            summary[f"{prefix}_reproj_mean_px"] = float(np.mean(reproj))
            summary[f"{prefix}_reproj_median_px"] = float(np.median(reproj))
            summary[f"{prefix}_reproj_p90_px"] = float(np.percentile(reproj, 90.0))
        summaries.append(summary)
    return summaries


def _camera_model_from_camera(camera) -> CameraModel:
    if getattr(camera, "K", None) is not None:
        fx = float(camera.K[0, 0])
        fy = float(camera.K[1, 1])
        cx = float(camera.K[0, 2])
        cy = float(camera.K[1, 2])
    else:
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


def _camera_intrinsics_from_camera(camera) -> tuple[float, float, float, float]:
    if getattr(camera, "K", None) is not None:
        return (
            float(camera.K[0, 0]),
            float(camera.K[1, 1]),
            float(camera.K[0, 2]),
            float(camera.K[1, 2]),
        )
    return (
        float(fov2focal(float(camera.FoVx), float(camera.image_width))),
        float(fov2focal(float(camera.FoVy), float(camera.image_height))),
        float(camera.image_width) * 0.5,
        float(camera.image_height) * 0.5,
    )


def _gaussian_points_in_camera_frame(
    gaussians,
    frame: int,
    cam_rotation,
    cam_translation,
) -> np.ndarray:
    points = gaussians.get_world_xyz(frame)
    if torch.is_tensor(points):
        points_np = np.asarray(points.detach().cpu().numpy(), dtype=np.float32)
    else:
        points_np = np.asarray(points, dtype=np.float32)
    if points_np.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if torch.is_tensor(cam_rotation):
        cam_rotation_np = np.asarray(cam_rotation.detach().cpu().numpy(), dtype=np.float32)
    else:
        cam_rotation_np = np.asarray(cam_rotation, dtype=np.float32)
    if torch.is_tensor(cam_translation):
        cam_translation_np = np.asarray(cam_translation.detach().cpu().numpy(), dtype=np.float32).reshape(3)
    else:
        cam_translation_np = np.asarray(cam_translation, dtype=np.float32).reshape(3)
    return points_np.reshape(-1, 3) @ cam_rotation_np + cam_translation_np.reshape(1, 3)


def _make_render_intrinsics_override_camera(
    camera,
    *,
    fx_scale: float = 1.0,
    fy_scale: float = 1.0,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
):
    fx, fy, cx, cy = _camera_intrinsics_from_camera(camera)
    fx = float(fx) * float(fx_scale)
    fy = float(fy) * float(fy_scale)
    cx = float(cx) + float(cx_offset)
    cy = float(cy) + float(cy_offset)
    width = int(camera.image_width)
    height = int(camera.image_height)
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    fovx = 2.0 * math.atan(float(width) / (2.0 * fx))
    fovy = 2.0 * math.atan(float(height) / (2.0 * fy))
    data_device = None
    if torch.is_tensor(camera.R):
        data_device = str(camera.R.device)
    elif torch.is_tensor(camera.T):
        data_device = str(camera.T.device)
    else:
        data_device = "cuda"
    return Camera(
        timestamp=camera.timestamp,
        R=camera.R,
        T=camera.T,
        w=width,
        h=height,
        FoVx=fovx,
        FoVy=fovy,
        depth=camera.depth_map,
        intensity=camera.intensity_map,
        trans=camera.trans,
        scale=camera.scale,
        data_device=data_device,
        K=K,
    )


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
    if backend is not None and str(backend).strip().lower() in {"lidar_zbuffer", "kitti360_lidar_zbuffer"}:
        return _render_camera_with_lidar_zbuffer(camera, args, **kwargs)
    if backend is not None and str(backend).strip().lower() in {"lidar_scanline_zbuffer", "kitti360_lidar_scanline_zbuffer"}:
        return _render_camera_with_lidar_zbuffer(camera, args, use_scanlines=True, **kwargs)
    if backend is not None and str(backend).strip().lower() in {"point_zbuffer", "gaussian_zbuffer", "kitti360_point_zbuffer"}:
        return _render_camera_with_point_zbuffer(camera, gaussian_assets, **kwargs)
    if backend is None:
        return render_camera(camera, gaussian_assets, args, **kwargs)
    model = getattr(args, "model", None)
    if model is None:
        return render_camera(camera, gaussian_assets, args, **kwargs)
    prev_backend = getattr(model, "camera_render_backend", "rasterization")
    prev_mode = getattr(model, "training_render_mode", None)
    model.camera_render_backend = backend
    model.training_render_mode = ""
    try:
        return render_camera(camera, gaussian_assets, args, **kwargs)
    finally:
        model.camera_render_backend = prev_backend
        model.training_render_mode = prev_mode


def _render_camera_with_point_zbuffer(
    camera,
    gaussian_assets,
    cam_rotation=None,
    cam_translation=None,
    require_rgb=False,
    fill_kernel_size: int = 3,
    fill_iterations: int = 2,
):
    frame = int(camera.timestamp)
    if cam_rotation is None or cam_translation is None:
        cam_rotation = camera.R
        cam_translation = camera.T

    if torch.is_tensor(cam_rotation):
        cam_rotation_np = np.asarray(cam_rotation.detach().cpu().numpy(), dtype=np.float32)
    else:
        cam_rotation_np = np.asarray(cam_rotation, dtype=np.float32)
    if torch.is_tensor(cam_translation):
        cam_translation_np = np.asarray(cam_translation.detach().cpu().numpy(), dtype=np.float32).reshape(3)
    else:
        cam_translation_np = np.asarray(cam_translation, dtype=np.float32).reshape(3)

    world_points = []
    for pc in gaussian_assets:
        pts = pc.get_world_xyz(frame)
        if torch.is_tensor(pts):
            pts_np = np.asarray(pts.detach().cpu().numpy(), dtype=np.float32)
        else:
            pts_np = np.asarray(pts, dtype=np.float32)
        if pts_np.size > 0:
            world_points.append(pts_np.reshape(-1, 3))

    height = int(camera.image_height)
    width = int(camera.image_width)
    depth_map = np.zeros((height, width), dtype=np.float32)
    support_mask = np.zeros((height, width), dtype=np.float32)
    if not world_points:
        depth_tensor = torch.from_numpy(depth_map).to(
            device=camera.R.device if torch.is_tensor(camera.R) else "cpu",
            dtype=torch.float32,
        )
        render_pkg = {
            "depth": depth_tensor,
            "depth_expected": depth_tensor,
            "depth_median": depth_tensor,
            "depth_integrated": depth_tensor,
            "alpha": depth_tensor,
            "num_visible": 0,
            "render_backend": "point_zbuffer",
        }
        if require_rgb:
            render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=depth_tensor.device)
        return render_pkg

    points_world = np.concatenate(world_points, axis=0)
    points_cam = points_world @ cam_rotation_np + cam_translation_np.reshape(1, 3)
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1.0e-6)
    if not np.any(valid):
        depth_tensor = torch.from_numpy(depth_map).to(
            device=camera.R.device if torch.is_tensor(camera.R) else "cpu",
            dtype=torch.float32,
        )
        render_pkg = {
            "depth": depth_tensor,
            "depth_expected": depth_tensor,
            "depth_median": depth_tensor,
            "depth_integrated": depth_tensor,
            "alpha": depth_tensor,
            "num_visible": 0,
            "render_backend": "point_zbuffer",
        }
        if require_rgb:
            render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=depth_tensor.device)
        return render_pkg

    points_cam = points_cam[valid]
    z = z[valid]
    if getattr(camera, "K", None) is not None:
        if torch.is_tensor(camera.K):
            K = np.asarray(camera.K.detach().cpu().numpy(), dtype=np.float32)
        else:
            K = np.asarray(camera.K, dtype=np.float32)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    else:
        fx = float(width) / (2.0 * math.tan(float(camera.FoVx) * 0.5))
        fy = float(height) / (2.0 * math.tan(float(camera.FoVy) * 0.5))
        cx = float(width) * 0.5
        cy = float(height) * 0.5

    u = points_cam[:, 0] * fx / z + cx
    v = points_cam[:, 1] * fy / z + cy
    iu = np.rint(u).astype(np.int32)
    iv = np.rint(v).astype(np.int32)
    inside = (iu >= 0) & (iu < width) & (iv >= 0) & (iv < height)
    if np.any(inside):
        iu = iu[inside]
        iv = iv[inside]
        z = z[inside]
        lin = iv.astype(np.int64) * int(width) + iu.astype(np.int64)
        inv_z = 1.0 / np.maximum(z, 1.0e-6)
        order = np.argsort(lin, kind="stable")
        lin = lin[order]
        inv_z = inv_z[order]
        unique_lin, first_idx = np.unique(lin, return_index=True)
        max_inv_z = np.maximum.reduceat(inv_z, first_idx)
        inv_depth = np.zeros((height * width,), dtype=np.float32)
        inv_depth[unique_lin] = max_inv_z.astype(np.float32)
        support_mask.reshape(-1)[unique_lin] = 1.0

        kernel_size = max(int(fill_kernel_size), 0)
        if kernel_size > 1 and int(fill_iterations) > 0:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            inv_depth_2d = inv_depth.reshape(height, width)
            for _ in range(int(fill_iterations)):
                dilated = cv2.dilate(inv_depth_2d, kernel, iterations=1)
                inv_depth_2d = np.where(inv_depth_2d > 0.0, inv_depth_2d, dilated)
            inv_depth = inv_depth_2d.reshape(-1)

        positive = inv_depth > 0.0
        flat_depth = np.zeros_like(inv_depth)
        flat_depth[positive] = 1.0 / inv_depth[positive]
        depth_map = flat_depth.reshape(height, width)

    device = camera.R.device if torch.is_tensor(camera.R) else "cpu"
    depth_tensor = torch.from_numpy(depth_map).to(device=device, dtype=torch.float32)
    alpha_tensor = torch.from_numpy((depth_map > 0.0).astype(np.float32)).to(device=device, dtype=torch.float32)
    render_pkg = {
        "depth": depth_tensor,
        "depth_expected": depth_tensor,
        "depth_median": depth_tensor,
        "depth_integrated": depth_tensor,
        "alpha": alpha_tensor,
        "num_visible": int(np.count_nonzero(depth_map > 0.0)),
        "render_backend": "point_zbuffer",
    }
    if require_rgb:
        render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    return render_pkg


def _render_camera_with_lidar_zbuffer(
    camera,
    args,
    cam_rotation=None,
    cam_translation=None,
    require_rgb=False,
    fill_kernel_size: int = 3,
    fill_iterations: int = 2,
    use_scanlines: bool = False,
    scanline_thickness: int = 2,
    scanline_max_gap_px: float = 20.0,
    scanline_max_depth_ratio: float = 1.25,
):
    lidar_sensor = getattr(args, "_calib_lidar_sensor", None)
    if lidar_sensor is None:
        raise ValueError("lidar_zbuffer backend requires args._calib_lidar_sensor to be set.")

    frame = int(camera.timestamp)
    if cam_rotation is None or cam_translation is None:
        cam_rotation = camera.R
        cam_translation = camera.T
    if torch.is_tensor(cam_rotation):
        cam_rotation_np = np.asarray(cam_rotation.detach().cpu().numpy(), dtype=np.float32)
    else:
        cam_rotation_np = np.asarray(cam_rotation, dtype=np.float32)
    if torch.is_tensor(cam_translation):
        cam_translation_np = np.asarray(cam_translation.detach().cpu().numpy(), dtype=np.float32).reshape(3)
    else:
        cam_translation_np = np.asarray(cam_translation, dtype=np.float32).reshape(3)

    height = int(camera.image_height)
    width = int(camera.image_width)
    depth_map = np.zeros((height, width), dtype=np.float32)
    if height <= 0 or width <= 0:
        depth_tensor = torch.from_numpy(depth_map).to(
            device=camera.R.device if torch.is_tensor(camera.R) else "cpu",
            dtype=torch.float32,
        )
        render_pkg = {
            "depth": depth_tensor,
            "depth_expected": depth_tensor,
            "depth_median": depth_tensor,
            "depth_integrated": depth_tensor,
            "alpha": depth_tensor,
            "num_visible": 0,
            "render_backend": "lidar_zbuffer",
        }
        if require_rgb:
            render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=depth_tensor.device)
        return render_pkg

    if getattr(camera, "K", None) is not None:
        if torch.is_tensor(camera.K):
            K = np.asarray(camera.K.detach().cpu().numpy(), dtype=np.float32)
        else:
            K = np.asarray(camera.K, dtype=np.float32)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    else:
        fx = float(width) / (2.0 * math.tan(float(camera.FoVx) * 0.5))
        fy = float(height) / (2.0 * math.tan(float(camera.FoVy) * 0.5))
        cx = float(width) * 0.5
        cy = float(height) * 0.5

    if not use_scanlines:
        lidar_points_world, _ = lidar_sensor.inverse_projection(frame)
        points_world = np.asarray(lidar_points_world.detach().cpu().numpy(), dtype=np.float32).reshape(-1, 3)
        if points_world.size == 0:
            depth_tensor = torch.from_numpy(depth_map).to(
                device=camera.R.device if torch.is_tensor(camera.R) else "cpu",
                dtype=torch.float32,
            )
            render_pkg = {
                "depth": depth_tensor,
                "depth_expected": depth_tensor,
                "depth_median": depth_tensor,
                "depth_integrated": depth_tensor,
                "alpha": depth_tensor,
                "num_visible": 0,
                "render_backend": "lidar_zbuffer",
            }
            if require_rgb:
                render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=depth_tensor.device)
            return render_pkg

        points_cam = points_world @ cam_rotation_np + cam_translation_np.reshape(1, 3)
        z = points_cam[:, 2]
        valid = np.isfinite(points_cam).all(axis=1) & (z > 1.0e-6)
        points_cam = points_cam[valid]
        z = z[valid]
        if points_cam.size > 0:
            u = points_cam[:, 0] * fx / z + cx
            v = points_cam[:, 1] * fy / z + cy
            iu = np.rint(u).astype(np.int32)
            iv = np.rint(v).astype(np.int32)
            inside = (iu >= 0) & (iu < width) & (iv >= 0) & (iv < height)
            iu = iu[inside]
            iv = iv[inside]
            z = z[inside]
            if z.size > 0:
                lin = iv.astype(np.int64) * int(width) + iu.astype(np.int64)
                inv_z = 1.0 / np.maximum(z, 1.0e-6)
                order = np.argsort(lin, kind="stable")
                lin = lin[order]
                inv_z = inv_z[order]
                unique_lin, first_idx = np.unique(lin, return_index=True)
                max_inv_z = np.maximum.reduceat(inv_z, first_idx)
                inv_depth = np.zeros((height * width,), dtype=np.float32)
                inv_depth[unique_lin] = max_inv_z.astype(np.float32)
                kernel_size = max(int(fill_kernel_size), 0)
                if kernel_size > 1 and int(fill_iterations) > 0:
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    inv_depth_2d = inv_depth.reshape(height, width)
                    for _ in range(int(fill_iterations)):
                        dilated = cv2.dilate(inv_depth_2d, kernel, iterations=1)
                        inv_depth_2d = np.where(inv_depth_2d > 0.0, inv_depth_2d, dilated)
                    inv_depth = inv_depth_2d.reshape(-1)
                positive = inv_depth > 0.0
                flat_depth = np.zeros_like(inv_depth)
                flat_depth[positive] = 1.0 / inv_depth[positive]
                depth_map = flat_depth.reshape(height, width)

        device = camera.R.device if torch.is_tensor(camera.R) else "cpu"
        depth_tensor = torch.from_numpy(depth_map).to(device=device, dtype=torch.float32)
        alpha_tensor = torch.from_numpy((depth_map > 0.0).astype(np.float32)).to(device=device, dtype=torch.float32)
        render_pkg = {
            "depth": depth_tensor,
            "depth_expected": depth_tensor,
            "depth_median": depth_tensor,
            "depth_integrated": depth_tensor,
            "alpha": alpha_tensor,
            "num_visible": int(np.count_nonzero(depth_map > 0.0)),
            "render_backend": "lidar_zbuffer",
        }
        if require_rgb:
            render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
        return render_pkg

    inv_depth_2d = np.zeros((height, width), dtype=np.float32)

    def _accumulate_range_return(return_num: int) -> None:
        nonlocal inv_depth_2d
        range_map = lidar_sensor.get_depth(frame, return_num=return_num)
        if range_map is None:
            return
        if torch.is_tensor(range_map):
            range_np = np.asarray(range_map.detach().cpu().numpy(), dtype=np.float32)
        else:
            range_np = np.asarray(range_map, dtype=np.float32)
        if range_np.ndim != 2 or range_np.size == 0:
            return

        points_world = lidar_sensor.range2point(frame, range_np)
        points_world = np.asarray(points_world.detach().cpu().numpy(), dtype=np.float32)
        points_cam = points_world @ cam_rotation_np + cam_translation_np.reshape(1, 1, 3)
        z = points_cam[..., 2]
        valid = (
            np.isfinite(points_cam).all(axis=-1)
            & np.isfinite(range_np)
            & (range_np > 0.0)
            & (z > 1.0e-6)
        )
        if not np.any(valid):
            return

        u = points_cam[..., 0] * fx / z + cx
        v = points_cam[..., 1] * fy / z + cy
        iu = np.rint(u).astype(np.int32)
        iv = np.rint(v).astype(np.int32)
        inside = valid & (iu >= 0) & (iu < width) & (iv >= 0) & (iv < height)
        if np.any(inside):
            flat_lin = (iv[inside].astype(np.int64) * int(width)) + iu[inside].astype(np.int64)
            flat_inv_z = (1.0 / np.maximum(z[inside], 1.0e-6)).astype(np.float32)
            order = np.argsort(flat_lin, kind="stable")
            flat_lin = flat_lin[order]
            flat_inv_z = flat_inv_z[order]
            unique_lin, first_idx = np.unique(flat_lin, return_index=True)
            max_inv_z = np.maximum.reduceat(flat_inv_z, first_idx)
            flat_buffer = inv_depth_2d.reshape(-1)
            flat_buffer[unique_lin] = np.maximum(flat_buffer[unique_lin], max_inv_z.astype(np.float32))

        if use_scanlines:
            thickness = max(int(scanline_thickness), 1)
            max_gap = float(max(scanline_max_gap_px, 1.0))
            max_ratio = float(max(scanline_max_depth_ratio, 1.0))
            row_count = int(range_np.shape[0])
            for row in range(row_count):
                row_mask = inside[row]
                if np.count_nonzero(row_mask) < 2:
                    continue
                cols = np.flatnonzero(row_mask)
                row_u = u[row]
                row_v = v[row]
                row_z = z[row]
                for c0, c1 in zip(cols[:-1], cols[1:]):
                    if c1 - c0 > 2:
                        continue
                    z0 = float(row_z[c0])
                    z1 = float(row_z[c1])
                    z_min = max(min(z0, z1), 1.0e-6)
                    z_max = max(z0, z1)
                    if z_max / z_min > max_ratio:
                        continue
                    p0 = np.array([row_u[c0], row_v[c0]], dtype=np.float32)
                    p1 = np.array([row_u[c1], row_v[c1]], dtype=np.float32)
                    gap = float(np.linalg.norm(p1 - p0))
                    if not np.isfinite(gap) or gap <= 1.0 or gap > max_gap:
                        continue
                    tmp = np.zeros_like(inv_depth_2d)
                    inv_val = float(max(1.0 / max(z0, 1.0e-6), 1.0 / max(z1, 1.0e-6)))
                    cv2.line(
                        tmp,
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                        color=inv_val,
                        thickness=thickness,
                        lineType=cv2.LINE_AA,
                    )
                    inv_depth_2d = np.maximum(inv_depth_2d, tmp)

    _accumulate_range_return(return_num=1)
    _accumulate_range_return(return_num=2)

    kernel_size = max(int(fill_kernel_size), 0)
    if kernel_size > 1 and int(fill_iterations) > 0:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        for _ in range(int(fill_iterations)):
            dilated = cv2.dilate(inv_depth_2d, kernel, iterations=1)
            inv_depth_2d = np.where(inv_depth_2d > 0.0, inv_depth_2d, dilated)

    positive = inv_depth_2d > 0.0
    depth_map = np.zeros_like(inv_depth_2d)
    depth_map[positive] = 1.0 / inv_depth_2d[positive]

    device = camera.R.device if torch.is_tensor(camera.R) else "cpu"
    depth_tensor = torch.from_numpy(depth_map).to(device=device, dtype=torch.float32)
    alpha_tensor = torch.from_numpy((depth_map > 0.0).astype(np.float32)).to(device=device, dtype=torch.float32)
    render_pkg = {
        "depth": depth_tensor,
        "depth_expected": depth_tensor,
        "depth_median": depth_tensor,
        "depth_integrated": depth_tensor,
        "alpha": alpha_tensor,
        "num_visible": int(np.count_nonzero(depth_map > 0.0)),
        "render_backend": "lidar_zbuffer",
    }
    if require_rgb:
        render_pkg["rgb"] = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    return render_pkg


def _resolve_camera_aux_depth(
    render_pkg: dict,
    depth_mode: str = "median",
) -> torch.Tensor:
    mode = str(depth_mode).strip().lower()
    if mode == "median":
        return render_pkg.get("depth_median", render_pkg["depth"])
    if mode == "expected":
        return render_pkg.get("depth_expected", render_pkg["depth"])
    if mode == "depth":
        return render_pkg["depth"]
    raise ValueError(
        f"Unsupported camera auxiliary depth mode '{depth_mode}'. "
        "Expected one of: median, expected, depth."
    )


def _compute_lidar_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    accumulation: torch.Tensor | None = None,
    *,
    loss_mode: str = "l1",
    inverse_min_depth: float = 0.5,
    use_visibility_weights: bool = False,
    visible_weight: float = 2.0,
    occluded_weight: float = 0.5,
    outside_weight: float = 1.0,
    visibility_tolerance: float = 0.25,
) -> torch.Tensor:
    pred = pred_depth.reshape(-1).float()
    gt = gt_depth.reshape(-1).float()
    if pred.numel() == 0:
        return pred.new_zeros(())
    mode = str(loss_mode).strip().lower()
    if mode == "l1":
        err = torch.abs(pred - gt)
    elif mode in {"inverse", "inv_depth", "inverse_depth"}:
        min_depth = float(inverse_min_depth)
        pred_safe = torch.clamp(pred, min=min_depth)
        gt_safe = torch.clamp(gt, min=min_depth)
        err = torch.abs(pred_safe.reciprocal() - gt_safe.reciprocal())
    else:
        raise ValueError(
            f"Unsupported LiDAR depth loss mode '{loss_mode}'. Expected one of: l1, inverse_depth."
        )
    if not use_visibility_weights:
        return err.mean()

    valid_pred = pred > 1.0e-6
    if accumulation is not None:
        valid_pred = valid_pred & (accumulation.reshape(-1).float() > 1.0e-4)
    tol = float(visibility_tolerance)
    occluded = valid_pred & (pred > (gt + tol))
    visible = valid_pred & ~occluded
    outside = ~valid_pred

    weights = torch.full_like(err, float(outside_weight))
    weights[visible] = float(visible_weight)
    weights[occluded] = float(occluded_weight)
    weight_sum = torch.clamp(weights.sum(), min=1.0e-8)
    return (weights * err).sum() / weight_sum


def _four_neighbor_support_mask(mask: torch.Tensor) -> torch.Tensor:
    support = (
        mask[2:, 1:-1]
        & mask[:-2, 1:-1]
        & mask[1:-1, 2:]
        & mask[1:-1, :-2]
    )
    return torch.nn.functional.pad(
        support,
        (1, 1, 1, 1),
        mode="constant",
        value=False,
    )


def _compute_lidar_normal_loss(
    pred_normal_map: torch.Tensor,
    gt_normal_map: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    pred_normal_map = pred_normal_map.float()
    gt_normal_map = gt_normal_map.float()
    normal_mask = valid_mask.bool()
    if not torch.any(normal_mask):
        return pred_normal_map.new_zeros(())

    pred_normal = pred_normal_map[normal_mask]
    gt_normal = gt_normal_map[normal_mask]
    valid_normals = torch.isfinite(pred_normal).all(dim=-1) & torch.isfinite(gt_normal).all(dim=-1)
    if not torch.any(valid_normals):
        return pred_normal_map.new_zeros(())
    cosine = torch.sum(pred_normal[valid_normals] * gt_normal[valid_normals], dim=-1)
    cosine = cosine.clamp(-1.0, 1.0)
    return (1.0 - cosine).mean()


def _depth_to_camera_normal_map(
    depth_map: torch.Tensor,
    camera,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = depth_map.float()
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dim() != 2:
        raise ValueError(f"Expected 2D depth map, got shape {tuple(depth.shape)}")

    height, width = depth.shape
    valid_depth = torch.isfinite(depth) & (depth > 1.0e-6)
    normal_map = torch.zeros((height, width, 3), device=depth.device, dtype=depth.dtype)
    if height < 3 or width < 3:
        return normal_map, torch.zeros_like(valid_depth, dtype=torch.bool)

    fx, fy, cx, cy = _camera_intrinsics_from_camera(camera)
    ys = torch.arange(height, device=depth.device, dtype=depth.dtype)
    xs = torch.arange(width, device=depth.device, dtype=depth.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    z = depth
    x = (grid_x - float(cx)) * z / float(fx)
    y = (grid_y - float(cy)) * z / float(fy)
    points = torch.stack([x, y, z], dim=-1)

    dx = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    dy = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    normals = torch.cross(dx, dy, dim=-1)
    normals = F.normalize(normals, dim=-1)
    ray_dir = F.normalize(points[1:-1, 1:-1, :], dim=-1)
    sign = torch.sign(-torch.sum(normals * ray_dir, dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    normals = normals * sign
    normal_map[1:-1, 1:-1, :] = normals

    valid_normals = torch.zeros_like(valid_depth, dtype=torch.bool)
    valid_normals[1:-1, 1:-1] = (
        valid_depth[1:-1, 1:-1]
        & valid_depth[2:, 1:-1]
        & valid_depth[:-2, 1:-1]
        & valid_depth[1:-1, 2:]
        & valid_depth[1:-1, :-2]
        & torch.isfinite(normals).all(dim=-1)
    )
    return normal_map, valid_normals


def _compute_camera_depth_normal_consistency_loss(
    pred_depth_map: torch.Tensor,
    pred_normal_map: torch.Tensor,
    camera,
) -> torch.Tensor:
    depth_normal_map, valid_mask = _depth_to_camera_normal_map(pred_depth_map, camera)
    if not torch.any(valid_mask):
        return pred_depth_map.new_zeros(())
    pred_normal_map = pred_normal_map.float()
    pred_normals = pred_normal_map[valid_mask]
    depth_normals = depth_normal_map[valid_mask]
    valid = (
        torch.isfinite(pred_normals).all(dim=-1)
        & torch.isfinite(depth_normals).all(dim=-1)
    )
    if not torch.any(valid):
        return pred_depth_map.new_zeros(())
    cosine = torch.sum(pred_normals[valid] * depth_normals[valid], dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine.abs()).mean()


def _resolve_visualization_depth_backend(camera_aux_depth_render_backend: str) -> str:
    backend = str(camera_aux_depth_render_backend).strip().lower()
    if backend in {
        "lidar_zbuffer",
        "kitti360_lidar_zbuffer",
        "lidar_scanline_zbuffer",
        "kitti360_lidar_scanline_zbuffer",
    }:
        return "raytracing"
    return camera_aux_depth_render_backend


def _camera_rt_from_lidar_to_camera(
    pose_correction,
    frame_id: int,
    extrinsic_rotation: torch.Tensor,
    extrinsic_translation: torch.Tensor,
    device=None,
):
    if not getattr(pose_correction, "use_shared_lidar_extrinsic", False):
        return pose_correction.corrected_rt(frame_id, device=device)
    if device is None:
        device = (
            extrinsic_rotation.device
            if torch.is_tensor(extrinsic_rotation)
            else pose_correction.delta_translations.device
        )
    frame_index = pose_correction._frame_index(frame_id)
    extrinsic_rotation = extrinsic_rotation.to(device=device, dtype=torch.float32)
    extrinsic_translation = extrinsic_translation.to(device=device, dtype=torch.float32).reshape(3)
    lidar_world_rotation = pose_correction.lidar_world_rotations[frame_index].to(device=device, dtype=torch.float32)
    lidar_world_translation = pose_correction.lidar_world_translations[frame_index].to(device=device, dtype=torch.float32)

    camera_to_lidar_rotation = extrinsic_rotation.transpose(0, 1)
    camera_to_lidar_translation = -(camera_to_lidar_rotation @ extrinsic_translation)
    corrected_rotation = lidar_world_rotation @ camera_to_lidar_rotation
    corrected_center = (
        lidar_world_rotation @ camera_to_lidar_translation + lidar_world_translation
    )
    corrected_translation = -(corrected_rotation.T @ corrected_center)
    return corrected_rotation, corrected_translation


def _normalize_render_preset(value: str | None) -> str | None:
    if value is None:
        return None
    preset = str(value).strip().lower()
    aliases = {
        "hybrid_3dgrut": "hybrid_3dgrut",
        "hybrid-3dgrut": "hybrid_3dgrut",
        "3dgrut_hybrid": "hybrid_3dgrut",
        "3dgrut": "hybrid_3dgrut",
        "3dgut": "hybrid_3dgrut",
        "lidar_rt": "lidar_rt",
        "lidar-rt": "lidar_rt",
        "2dgs": "lidar_rt",
    }
    if preset not in aliases:
        raise ValueError(
            f"Unsupported render preset '{value}'. Expected one of: "
            "hybrid_3dgrut, 3dgrut, lidar_rt, lidar-rt, 2dgs."
        )
    return aliases[preset]


def _should_run_initial_pure_pnp(
    *,
    run_pending: bool,
    cycle: int,
    warmup_cycles: int,
    total_cycles: int,
) -> bool:
    if not run_pending:
        return False
    warmup_cycles = int(max(warmup_cycles, 0))
    total_cycles = int(max(total_cycles, 0))
    trigger_cycle = (warmup_cycles + 1) if total_cycles > warmup_cycles else total_cycles
    return int(cycle) == max(trigger_cycle, 1)


def _apply_render_preset(args, preset: str) -> tuple[str, str]:
    model = getattr(args, "model", None)
    if model is None:
        raise ValueError("args.model is required to apply a render preset.")
    if preset == "hybrid_3dgrut":
        model.training_render_mode = "hybrid_3dgrut"
        model.camera_render_backend = "3dgut"
        model.raytrace_backend = "3dgrt"
        return "3dgut_rasterization", "3dgut_rasterization"
    if preset == "lidar_rt":
        model.training_render_mode = ""
        model.camera_render_backend = "2dgs"
        model.raytrace_backend = "lidar_rt"
        return "raytracing", "raytracing"
    raise ValueError(f"Unhandled render preset '{preset}'.")


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


def _get_camera_supervision_mask(
    cam_cameras: dict,
    frame: int,
    device=None,
) -> torch.Tensor | None:
    camera = cam_cameras.get(int(frame))
    if camera is None:
        return None
    dynamic_mask = getattr(camera, "supervision_mask", None)
    if dynamic_mask is None:
        return None
    valid_mask = ~dynamic_mask.bool()
    if device is not None:
        valid_mask = valid_mask.to(device=device)
    return valid_mask


def _mask_image_for_matcher(
    image: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None,
) -> np.ndarray:
    rgb = _to_uint8_rgb(image)
    if valid_mask is None:
        return rgb
    if torch.is_tensor(valid_mask):
        mask = valid_mask.detach().cpu().numpy().astype(bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        return rgb
    out = rgb.copy()
    out[~mask] = 0
    return out


def _filter_points_by_image_mask(
    points_xy: np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0 or valid_mask is None:
        return np.ones((points.shape[0],), dtype=bool)
    if torch.is_tensor(valid_mask):
        mask = valid_mask.detach().cpu().numpy().astype(bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        return np.ones((points.shape[0],), dtype=bool)
    h, w = mask.shape
    xs = np.round(points[:, 0]).astype(np.int32)
    ys = np.round(points[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    keep = np.zeros((points.shape[0],), dtype=bool)
    if np.any(inside):
        keep[inside] = mask[ys[inside], xs[inside]]
    return keep


def _filter_points_by_support(
    query_points: np.ndarray,
    support_points: np.ndarray | None,
    radius_px: float,
) -> np.ndarray:
    """Return boolean mask of query points within radius_px of any support point.

    Uses KD-tree for O(N log M) complexity instead of O(N×M) brute-force.
    Critical when N (dense query points) is large (e.g. 450k at stride=1).
    """
    from scipy.spatial import KDTree
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    if query_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if support_points is None:
        return np.ones((query_points.shape[0],), dtype=bool)
    support_points = np.asarray(support_points, dtype=np.float32).reshape(-1, 2)
    if support_points.shape[0] == 0:
        return np.zeros((query_points.shape[0],), dtype=bool)
    tree = KDTree(support_points)
    nearest, _ = tree.query(query_points, k=1, workers=-1)
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


def _to_normalized_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    rgb = _to_uint8_rgb(image)
    rgb = rgb.astype(np.float32) / 255.0
    channel_mean = rgb.reshape(-1, 3).mean(axis=0, keepdims=True)
    channel_std = rgb.reshape(-1, 3).std(axis=0, keepdims=True)
    return (rgb - channel_mean.reshape(1, 1, 3)) / np.maximum(channel_std.reshape(1, 1, 3), 1.0e-6)


def _sample_map_at_points(
    image: np.ndarray,
    points_xy: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    array = np.asarray(image)
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        tail_shape = array.shape[2:] if array.ndim > 2 else ()
        return np.zeros((0,) + tail_shape, dtype=np.float64)
    coords = np.vstack(
        [
            np.clip(points[:, 1], 0.0, float(array.shape[0] - 1)),
            np.clip(points[:, 0], 0.0, float(array.shape[1] - 1)),
        ]
    )
    if array.ndim == 2:
        return map_coordinates(array, coords, order=order, mode="nearest").astype(np.float64)
    if array.ndim == 3:
        return np.stack(
            [
                map_coordinates(array[:, :, ch], coords, order=order, mode="nearest").astype(np.float64)
                for ch in range(array.shape[2])
            ],
            axis=1,
        )
    raise ValueError(f"Unsupported array shape for sampling: {array.shape}")


def _build_temporal_reliability_map(
    image: torch.Tensor | np.ndarray,
    gradient_scale: float,
) -> np.ndarray:
    rgb = _to_uint8_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    scale = float(np.percentile(grad_mag, 95.0)) if grad_mag.size > 0 else 0.0
    if not np.isfinite(scale) or scale <= 1.0e-6:
        return np.ones_like(gray, dtype=np.float32)
    grad_norm = np.clip(grad_mag / scale, 0.0, 1.0)
    reliability = 1.0 / (1.0 + float(gradient_scale) * grad_norm)
    return np.clip(reliability, 0.35, 1.0).astype(np.float32)


def _build_semidense_temporal_samples(
    source_depth: np.ndarray,
    source_c2w: np.ndarray,
    target_w2c: np.ndarray,
    source_depth_camera: CameraModel,
    target_depth_camera: CameraModel,
    target_depth: np.ndarray | None,
    support_points: np.ndarray | None,
    reliability_map: np.ndarray | None,
    stride: int,
    max_points: int,
    min_depth: float = 0.10,
    max_depth: float = 80.0,
    support_radius_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    depth_map = np.asarray(source_depth, dtype=np.float32)
    if depth_map.ndim != 2 or int(stride) <= 0 or int(max_points) <= 0:
        return None

    h, w = depth_map.shape
    ys = np.arange(max(int(stride) // 2, 0), h, int(stride), dtype=np.float64)
    xs = np.arange(max(int(stride) // 2, 0), w, int(stride), dtype=np.float64)
    if ys.size == 0 or xs.size == 0:
        return None

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    source_pixels = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float64)
    if support_points is not None and np.asarray(support_points).size > 0:
        radius_px = float(
            support_radius_px
            if support_radius_px is not None
            else max(float(stride) * 1.5, 6.0)
        )
        support_mask = _filter_points_by_support(source_pixels, support_points, radius_px)
        source_pixels = source_pixels[support_mask]
        if source_pixels.shape[0] == 0:
            return None

    ix = np.clip(np.round(source_pixels[:, 0]).astype(np.int32), 0, w - 1)
    iy = np.clip(np.round(source_pixels[:, 1]).astype(np.int32), 0, h - 1)
    sampled_depth = depth_map[iy, ix].astype(np.float64)
    valid = np.isfinite(sampled_depth) & (sampled_depth >= float(min_depth)) & (sampled_depth <= float(max_depth))
    if not np.any(valid):
        return None

    source_pixels = source_pixels[valid]
    sampled_depth = sampled_depth[valid]
    if reliability_map is not None:
        point_weights = _sample_map_at_points(reliability_map, source_pixels, order=1).reshape(-1)
    else:
        point_weights = np.ones((source_pixels.shape[0],), dtype=np.float64)

    priority = point_weights / np.maximum(sampled_depth, 1.0)
    if source_pixels.shape[0] > int(max_points):
        keep = np.argsort(priority)[-int(max_points):]
        source_pixels = source_pixels[keep]
        sampled_depth = sampled_depth[keep]
        point_weights = point_weights[keep]

    fx = float(source_depth_camera.K[0, 0])
    fy = float(source_depth_camera.K[1, 1])
    cx = float(source_depth_camera.K[0, 2])
    cy = float(source_depth_camera.K[1, 2])
    x = (source_pixels[:, 0] - cx) / fx * sampled_depth
    y = (source_pixels[:, 1] - cy) / fy * sampled_depth
    source_points_3d = np.stack([x, y, sampled_depth], axis=1).astype(np.float64)

    occlusion_keep = _filter_points_by_target_occlusion(
        source_points_3d=source_points_3d,
        source_c2w=source_c2w,
        target_w2c=target_w2c,
        target_depth=target_depth,
        target_depth_camera=target_depth_camera,
        min_depth=min_depth,
    )
    if not np.any(occlusion_keep):
        return None
    return (
        source_points_3d[occlusion_keep],
        source_pixels[occlusion_keep],
        np.clip(point_weights[occlusion_keep], 0.25, 4.0).astype(np.float64),
    )


def _camera_pose_matrices_from_corrected_rt(
    cam_R: torch.Tensor,
    cam_T: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    cam_R_np = cam_R.detach().cpu().numpy().astype(np.float64)
    cam_T_np = cam_T.detach().cpu().numpy().astype(np.float64)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = cam_R_np
    c2w[:3, 3] = -(cam_R_np @ cam_T_np)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = cam_R_np.T
    w2c[:3, 3] = cam_T_np
    return c2w, w2c


def _filter_points_by_target_occlusion(
    source_points_3d: np.ndarray,
    source_c2w: np.ndarray,
    target_w2c: np.ndarray,
    target_depth: np.ndarray | None,
    target_depth_camera: CameraModel,
    occlusion_margin_m: float = 0.30,
    occlusion_margin_ratio: float = 0.02,
    min_depth: float = 0.10,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    points_3d = np.asarray(source_points_3d, dtype=np.float64).reshape(-1, 3)
    if points_3d.shape[0] == 0 or target_depth is None:
        return np.ones((points_3d.shape[0],), dtype=bool)

    source_rgb = points_3d
    world_points = (source_c2w[:3, :3] @ source_rgb.T).T + source_c2w[:3, 3]
    target_rgb = (target_w2c[:3, :3] @ world_points.T).T + target_w2c[:3, 3]
    z = target_rgb[:, 2]
    valid = z > float(min_depth)
    if not np.any(valid):
        return np.zeros((points_3d.shape[0],), dtype=bool)

    projected, _ = cv2.projectPoints(
        target_rgb[valid].astype(np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        target_depth_camera.K,
        target_depth_camera.dist if target_depth_camera.dist.size > 0 else None,
    )
    projected = projected.reshape(-1, 2)
    inside = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(target_depth.shape[1] - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(target_depth.shape[0] - 1))
    )
    keep = np.zeros((points_3d.shape[0],), dtype=bool)
    if not np.any(inside):
        return keep

    valid_indices = np.flatnonzero(valid)
    inside_indices = np.flatnonzero(inside)
    coords = np.vstack(
        [
            projected[inside_indices, 1].astype(np.float64, copy=False),
            projected[inside_indices, 0].astype(np.float64, copy=False),
        ]
    )
    sampled_target_depth = map_coordinates(
        np.asarray(target_depth, dtype=np.float32),
        coords,
        order=1,
        mode="nearest",
    ).astype(np.float64)
    target_z = z[valid_indices[inside_indices]]
    depth_valid = sampled_target_depth > float(min_depth)
    depth_margin = np.maximum(float(occlusion_margin_m), sampled_target_depth * float(occlusion_margin_ratio))
    visible = depth_valid & (target_z <= sampled_target_depth + depth_margin)
    keep[valid_indices[inside_indices][visible]] = True
    return keep


def _build_temporal_photometric_residuals(
    frame_data_list: list,
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    cam_images: dict,
    pose_correction,
    rgb_camera: CameraModel,
    source_depth_maps: dict[int, np.ndarray] | None = None,
    source_depth_cameras: dict[int, CameraModel] | None = None,
    target_depth_maps: dict[int, np.ndarray] | None = None,
    target_depth_cameras: dict[int, CameraModel] | None = None,
    match_radius_px: float = 4.0,
    min_matches: int = 8,
    semidense_stride: int = 0,
    semidense_max_points: int = 0,
    gradient_scale: float = 0.0,
) -> tuple[list[TemporalPhotometricResidualData], dict[int, np.ndarray]]:
    from scipy.spatial import KDTree

    frame_data_by_id = {int(frame_data.frame_id): frame_data for frame_data in frame_data_list}
    image_cache: dict[int, np.ndarray] = {}
    reliability_cache: dict[int, np.ndarray] = {}
    pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    target_images: dict[int, np.ndarray] = {}
    photometric_residuals: list[TemporalPhotometricResidualData] = []

    for source_frame, pair_list in temporal_pair_cache.items():
        source_frame = int(source_frame)
        source_frame_data = frame_data_by_id.get(source_frame)
        if source_frame_data is None or source_frame not in cam_images:
            continue
        source_rgb_points = np.asarray(source_frame_data.rgb_points, dtype=np.float32).reshape(-1, 2)
        if source_rgb_points.shape[0] == 0:
            continue

        source_tree = KDTree(source_rgb_points)
        if source_frame not in image_cache:
            image_cache[source_frame] = _to_normalized_rgb(cam_images[source_frame])
        source_image = image_cache[source_frame]
        if source_frame not in reliability_cache:
            reliability_cache[source_frame] = _build_temporal_reliability_map(
                cam_images[source_frame],
                gradient_scale=gradient_scale,
            )
        source_reliability = reliability_cache[source_frame]

        if source_frame not in pose_cache:
            src_R, src_T = pose_correction.corrected_rt(source_frame, device="cpu")
            pose_cache[source_frame] = _camera_pose_matrices_from_corrected_rt(src_R, src_T)
        source_c2w = pose_cache[source_frame][0]

        for target_frame, pts_cur, _pts_nbr in pair_list:
            target_frame = int(target_frame)
            if target_frame not in cam_images or np.asarray(pts_cur).size == 0:
                continue

            nearest_dist, nearest_idx = source_tree.query(
                np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2),
                k=1,
                workers=-1,
            )
            keep_mask = nearest_dist <= float(match_radius_px)
            if int(np.count_nonzero(keep_mask)) < int(min_matches):
                continue
            matched_indices = nearest_idx[keep_mask].astype(np.int32)
            matched_distances = nearest_dist[keep_mask].astype(np.float64)
            point_indices, inverse = np.unique(matched_indices, return_inverse=True)
            if point_indices.shape[0] < int(min_matches):
                continue
            association_weights = np.zeros((point_indices.shape[0],), dtype=np.float64)
            if matched_distances.shape[0] > 0:
                proximity = 1.0 - np.clip(matched_distances / max(float(match_radius_px), 1.0e-6), 0.0, 1.0)
                np.maximum.at(association_weights, inverse, 0.25 + 0.75 * proximity)
            else:
                association_weights.fill(1.0)

            if target_frame not in image_cache:
                image_cache[target_frame] = _to_normalized_rgb(cam_images[target_frame])
            target_images[target_frame] = image_cache[target_frame]

            if target_frame not in pose_cache:
                tgt_R, tgt_T = pose_correction.corrected_rt(target_frame, device="cpu")
                pose_cache[target_frame] = _camera_pose_matrices_from_corrected_rt(tgt_R, tgt_T)
            target_w2c = pose_cache[target_frame][1]

            source_points_3d = np.asarray(source_frame_data.points_3d[point_indices], dtype=np.float64)
            target_depth_camera = None if target_depth_cameras is None else target_depth_cameras.get(target_frame)
            occlusion_keep = _filter_points_by_target_occlusion(
                source_points_3d=source_points_3d,
                source_c2w=source_c2w,
                target_w2c=target_w2c,
                target_depth=None if target_depth_maps is None else target_depth_maps.get(target_frame),
                target_depth_camera=target_depth_camera if target_depth_camera is not None else rgb_camera,
                min_depth=0.10,
            )
            if int(np.count_nonzero(occlusion_keep)) < int(min_matches):
                continue
            point_indices = point_indices[occlusion_keep]
            source_points_3d = source_points_3d[occlusion_keep]
            sparse_point_weights = association_weights[occlusion_keep]
            if source_frame_data.point_weights is not None:
                sparse_point_weights = sparse_point_weights * np.asarray(
                    source_frame_data.point_weights,
                    dtype=np.float64,
                ).reshape(-1)[point_indices]
            src_points_sel = source_rgb_points[point_indices]
            source_colors = _sample_map_at_points(source_image, src_points_sel, order=1)
            sparse_point_weights = sparse_point_weights * _sample_map_at_points(
                source_reliability,
                src_points_sel,
                order=1,
            ).reshape(-1)

            photometric_residuals.append(
                TemporalPhotometricResidualData(
                    source_frame_id=source_frame,
                    target_frame_id=target_frame,
                    source_points_3d=source_points_3d,
                    source_colors=source_colors,
                    source_c2w=source_c2w,
                    target_w2c=target_w2c,
                    weight=float(source_frame_data.temporal_weight),
                    point_weights=np.clip(sparse_point_weights, 0.25, 4.0).astype(np.float64),
                )
            )

            semidense_samples = _build_semidense_temporal_samples(
                source_depth=None if source_depth_maps is None else source_depth_maps.get(source_frame),
                source_c2w=source_c2w,
                target_w2c=target_w2c,
                source_depth_camera=(
                    source_depth_cameras.get(source_frame)
                    if source_depth_cameras is not None and source_frame in source_depth_cameras
                    else rgb_camera
                ),
                target_depth_camera=(
                    target_depth_cameras.get(target_frame)
                    if target_depth_cameras is not None and target_frame in target_depth_cameras
                    else rgb_camera
                ),
                target_depth=None if target_depth_maps is None else target_depth_maps.get(target_frame),
                support_points=np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2),
                reliability_map=source_reliability,
                stride=semidense_stride,
                max_points=semidense_max_points,
                min_depth=0.10,
                max_depth=80.0,
                support_radius_px=max(float(match_radius_px) * 2.0, float(semidense_stride) * 1.5),
            )
            if semidense_samples is None:
                continue
            semidense_points_3d, semidense_pixels, semidense_point_weights = semidense_samples
            if semidense_points_3d.shape[0] < int(min_matches):
                continue
            semidense_colors = _sample_map_at_points(source_image, semidense_pixels, order=1)
            photometric_residuals.append(
                TemporalPhotometricResidualData(
                    source_frame_id=source_frame,
                    target_frame_id=target_frame,
                    source_points_3d=semidense_points_3d,
                    source_colors=semidense_colors,
                    source_c2w=source_c2w,
                    target_w2c=target_w2c,
                    weight=float(source_frame_data.temporal_weight),
                    point_weights=semidense_point_weights,
                )
            )

    return photometric_residuals, target_images


def _build_temporal_geometric_residuals(
    frame_data_list: list,
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    pose_correction,
    rgb_camera: CameraModel,
    cam_images: dict | None = None,
    target_depth_maps: dict[int, np.ndarray] | None = None,
    target_depth_cameras: dict[int, CameraModel] | None = None,
    match_radius_px: float = 4.0,
    min_matches: int = 8,
    gradient_scale: float = 0.0,
) -> list[TemporalResidualData]:
    from scipy.spatial import KDTree

    frame_data_by_id = {int(frame_data.frame_id): frame_data for frame_data in frame_data_list}
    pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reliability_cache: dict[int, np.ndarray] = {}
    temporal_residuals: list[TemporalResidualData] = []

    for source_frame, pair_list in temporal_pair_cache.items():
        source_frame = int(source_frame)
        source_frame_data = frame_data_by_id.get(source_frame)
        if source_frame_data is None:
            continue
        source_rgb_points = np.asarray(source_frame_data.rgb_points, dtype=np.float32).reshape(-1, 2)
        if source_rgb_points.shape[0] == 0:
            continue
        source_tree = KDTree(source_rgb_points)
        if cam_images is not None and source_frame not in reliability_cache and source_frame in cam_images:
            reliability_cache[source_frame] = _build_temporal_reliability_map(
                cam_images[source_frame],
                gradient_scale=gradient_scale,
            )
        source_reliability = reliability_cache.get(source_frame)

        if source_frame not in pose_cache:
            src_R, src_T = pose_correction.corrected_rt(source_frame, device="cpu")
            pose_cache[source_frame] = _camera_pose_matrices_from_corrected_rt(src_R, src_T)
        source_c2w = pose_cache[source_frame][0]

        for target_frame, pts_cur, pts_nbr in pair_list:
            target_frame = int(target_frame)
            if np.asarray(pts_cur).size == 0 or np.asarray(pts_nbr).size == 0:
                continue
            nearest_dist, nearest_idx = source_tree.query(
                np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2),
                k=1,
                workers=-1,
            )
            keep_mask = nearest_dist <= float(match_radius_px)
            if int(np.count_nonzero(keep_mask)) < int(min_matches):
                continue
            point_indices = nearest_idx[keep_mask].astype(np.int32)
            if point_indices.shape[0] < int(min_matches):
                continue
            matched_distances = nearest_dist[keep_mask].astype(np.float64)
            point_weights = 0.25 + 0.75 * (
                1.0 - np.clip(matched_distances / max(float(match_radius_px), 1.0e-6), 0.0, 1.0)
            )
            if source_frame_data.point_weights is not None:
                point_weights = point_weights * np.asarray(
                    source_frame_data.point_weights,
                    dtype=np.float64,
                ).reshape(-1)[point_indices]

            if target_frame not in pose_cache:
                tgt_R, tgt_T = pose_correction.corrected_rt(target_frame, device="cpu")
                pose_cache[target_frame] = _camera_pose_matrices_from_corrected_rt(tgt_R, tgt_T)
            target_w2c = pose_cache[target_frame][1]
            source_points_3d = np.asarray(source_frame_data.points_3d[point_indices], dtype=np.float64)
            target_depth_camera = None if target_depth_cameras is None else target_depth_cameras.get(target_frame)
            occlusion_keep = _filter_points_by_target_occlusion(
                source_points_3d=source_points_3d,
                source_c2w=source_c2w,
                target_w2c=target_w2c,
                target_depth=None if target_depth_maps is None else target_depth_maps.get(target_frame),
                target_depth_camera=target_depth_camera if target_depth_camera is not None else rgb_camera,
                min_depth=0.10,
            )
            if int(np.count_nonzero(occlusion_keep)) < int(min_matches):
                continue

            sparse_pixels = source_rgb_points[point_indices][occlusion_keep]
            weighted_points = point_weights[occlusion_keep]
            if source_reliability is not None:
                weighted_points = weighted_points * _sample_map_at_points(
                    source_reliability,
                    sparse_pixels,
                    order=1,
                ).reshape(-1)
            temporal_residuals.append(
                TemporalResidualData(
                    source_frame_index=source_frame,
                    target_frame_index=target_frame,
                    source_points_3d=source_points_3d[occlusion_keep],
                    target_rgb_points=np.asarray(pts_nbr[keep_mask], dtype=np.float64)[occlusion_keep],
                    source_c2w=source_c2w,
                    target_w2c=target_w2c,
                    weight=float(source_frame_data.temporal_weight),
                    point_weights=np.clip(weighted_points, 0.25, 4.0).astype(np.float64),
                )
            )

    return temporal_residuals


def _build_temporal_depth_residuals(
    frame_data_list: list,
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    pose_correction,
    rgb_camera: CameraModel,
    source_depth_maps: dict[int, np.ndarray] | None = None,
    source_depth_cameras: dict[int, CameraModel] | None = None,
    cam_images: dict | None = None,
    target_depth_maps: dict[int, np.ndarray] | None = None,
    target_depth_cameras: dict[int, CameraModel] | None = None,
    match_radius_px: float = 4.0,
    min_matches: int = 8,
    semidense_stride: int = 0,
    semidense_max_points: int = 0,
    gradient_scale: float = 0.0,
) -> list[TemporalDepthResidualData]:
    from scipy.spatial import KDTree

    frame_data_by_id = {int(frame_data.frame_id): frame_data for frame_data in frame_data_list}
    pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reliability_cache: dict[int, np.ndarray] = {}
    depth_residuals: list[TemporalDepthResidualData] = []

    for source_frame, pair_list in temporal_pair_cache.items():
        source_frame = int(source_frame)
        source_frame_data = frame_data_by_id.get(source_frame)
        if source_frame_data is None:
            continue
        source_rgb_points = np.asarray(source_frame_data.rgb_points, dtype=np.float32).reshape(-1, 2)
        if source_rgb_points.shape[0] == 0:
            continue
        source_tree = KDTree(source_rgb_points)
        if cam_images is not None and source_frame not in reliability_cache and source_frame in cam_images:
            reliability_cache[source_frame] = _build_temporal_reliability_map(
                cam_images[source_frame],
                gradient_scale=gradient_scale,
            )
        source_reliability = reliability_cache.get(source_frame)

        if source_frame not in pose_cache:
            src_R, src_T = pose_correction.corrected_rt(source_frame, device="cpu")
            pose_cache[source_frame] = _camera_pose_matrices_from_corrected_rt(src_R, src_T)
        source_c2w = pose_cache[source_frame][0]

        for target_frame, pts_cur, _pts_nbr in pair_list:
            target_frame = int(target_frame)
            if np.asarray(pts_cur).size == 0:
                continue
            nearest_dist, nearest_idx = source_tree.query(
                np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2),
                k=1,
                workers=-1,
            )
            keep_mask = nearest_dist <= float(match_radius_px)
            if int(np.count_nonzero(keep_mask)) < int(min_matches):
                continue
            point_indices = nearest_idx[keep_mask].astype(np.int32)
            if point_indices.shape[0] < int(min_matches):
                continue
            matched_distances = nearest_dist[keep_mask].astype(np.float64)
            sparse_point_weights = 0.25 + 0.75 * (
                1.0 - np.clip(matched_distances / max(float(match_radius_px), 1.0e-6), 0.0, 1.0)
            )
            if source_frame_data.point_weights is not None:
                sparse_point_weights = sparse_point_weights * np.asarray(
                    source_frame_data.point_weights,
                    dtype=np.float64,
                ).reshape(-1)[point_indices]

            if target_frame not in pose_cache:
                tgt_R, tgt_T = pose_correction.corrected_rt(target_frame, device="cpu")
                pose_cache[target_frame] = _camera_pose_matrices_from_corrected_rt(tgt_R, tgt_T)
            target_w2c = pose_cache[target_frame][1]

            source_points_3d = np.asarray(source_frame_data.points_3d[point_indices], dtype=np.float64)
            target_depth_camera = None if target_depth_cameras is None else target_depth_cameras.get(target_frame)
            occlusion_keep = _filter_points_by_target_occlusion(
                source_points_3d=source_points_3d,
                source_c2w=source_c2w,
                target_w2c=target_w2c,
                target_depth=None if target_depth_maps is None else target_depth_maps.get(target_frame),
                target_depth_camera=target_depth_camera if target_depth_camera is not None else rgb_camera,
                min_depth=0.10,
            )
            if int(np.count_nonzero(occlusion_keep)) < int(min_matches):
                continue

            sparse_pixels = source_rgb_points[point_indices][occlusion_keep]
            sparse_point_weights = sparse_point_weights[occlusion_keep]
            if source_reliability is not None:
                sparse_point_weights = sparse_point_weights * _sample_map_at_points(
                    source_reliability,
                    sparse_pixels,
                    order=1,
                ).reshape(-1)
            depth_residuals.append(
                TemporalDepthResidualData(
                    source_frame_id=source_frame,
                    target_frame_id=target_frame,
                    source_points_3d=source_points_3d[occlusion_keep],
                    source_c2w=source_c2w,
                    target_w2c=target_w2c,
                    weight=float(source_frame_data.temporal_weight),
                    point_weights=np.clip(sparse_point_weights, 0.25, 4.0).astype(np.float64),
                )
            )

            semidense_samples = _build_semidense_temporal_samples(
                source_depth=None if source_depth_maps is None else source_depth_maps.get(source_frame),
                source_c2w=source_c2w,
                target_w2c=target_w2c,
                source_depth_camera=(
                    source_depth_cameras.get(source_frame)
                    if source_depth_cameras is not None and source_frame in source_depth_cameras
                    else rgb_camera
                ),
                target_depth_camera=(
                    target_depth_cameras.get(target_frame)
                    if target_depth_cameras is not None and target_frame in target_depth_cameras
                    else rgb_camera
                ),
                target_depth=None if target_depth_maps is None else target_depth_maps.get(target_frame),
                support_points=np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2),
                reliability_map=source_reliability,
                stride=semidense_stride,
                max_points=semidense_max_points,
                min_depth=0.10,
                max_depth=80.0,
                support_radius_px=max(float(match_radius_px) * 2.0, float(semidense_stride) * 1.5),
            )
            if semidense_samples is None:
                continue
            semidense_points_3d, _semidense_pixels, semidense_point_weights = semidense_samples
            if semidense_points_3d.shape[0] < int(min_matches):
                continue
            depth_residuals.append(
                TemporalDepthResidualData(
                    source_frame_id=source_frame,
                    target_frame_id=target_frame,
                    source_points_3d=semidense_points_3d,
                    source_c2w=source_c2w,
                    target_w2c=target_w2c,
                    weight=float(source_frame_data.temporal_weight),
                    point_weights=semidense_point_weights,
                )
            )

    return depth_residuals


def _get_temporal_support_points(
    matcher,
    frame: int,
    cam_images: dict,
    camera_masks: dict[int, torch.Tensor] | None,
    cache: dict[int, np.ndarray],
    max_offset: int,
) -> np.ndarray:
    if frame in cache:
        return cache[frame]
    current = _mask_image_for_matcher(
        cam_images[frame],
        None if camera_masks is None else camera_masks.get(int(frame)),
    )
    supports = []
    for offset in range(1, int(max_offset) + 1):
        for neighbor in (frame - offset, frame + offset):
            if neighbor not in cam_images:
                continue
            neighbor_img = _mask_image_for_matcher(
                cam_images[neighbor],
                None if camera_masks is None else camera_masks.get(int(neighbor)),
            )
            result = matcher(_format_matcher_image(current), _format_matcher_image(neighbor_img))
            pts0, _ = select_match_points(result)
            pts0 = np.asarray(pts0, dtype=np.float32)
            keep = _filter_points_by_image_mask(
                pts0,
                None if camera_masks is None else camera_masks.get(int(frame)),
            )
            pts0 = pts0[keep]
            if pts0.shape[0] > 0:
                supports.append(pts0)
    cache[frame] = np.concatenate(supports, axis=0) if supports else np.zeros((0, 2), dtype=np.float32)
    return cache[frame]


def _get_temporal_match_pairs(
    matcher,
    frame: int,
    cam_images: dict,
    camera_masks: dict[int, torch.Tensor] | None,
    cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    max_offset: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if frame in cache:
        return cache[frame]
    current = _mask_image_for_matcher(
        cam_images[frame],
        None if camera_masks is None else camera_masks.get(int(frame)),
    )
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for offset in range(1, int(max_offset) + 1):
        for neighbor in (frame - offset, frame + offset):
            if neighbor not in cam_images:
                continue
            neighbor_img = _mask_image_for_matcher(
                cam_images[neighbor],
                None if camera_masks is None else camera_masks.get(int(neighbor)),
            )
            result = matcher(_format_matcher_image(current), _format_matcher_image(neighbor_img))
            pts0, pts1 = select_match_points(result)
            pts0 = np.asarray(pts0, dtype=np.float32)
            pts1 = np.asarray(pts1, dtype=np.float32)
            keep = _filter_points_by_image_mask(
                pts0,
                None if camera_masks is None else camera_masks.get(int(frame)),
            )
            keep &= _filter_points_by_image_mask(
                pts1,
                None if camera_masks is None else camera_masks.get(int(neighbor)),
            )
            pts0 = pts0[keep]
            pts1 = pts1[keep]
            if pts0.shape[0] > 0 and pts1.shape[0] > 0:
                pairs.append(
                    (
                        int(neighbor),
                        pts0,
                        pts1,
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
    gt_valid_mask: torch.Tensor | None = None,
    temporal_support_points: np.ndarray | None = None,
    temporal_support_radius_px: float = 6.0,
    dense_mode: bool = False,
    dense_stride: int = 4,
    dense_cert_threshold: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    device = pred_rgb.device
    gt_rgb_np = _mask_image_for_matcher(gt_rgb, gt_valid_mask)
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
    if gt_valid_mask is not None:
        image_mask = _filter_points_by_image_mask(rgb_points, gt_valid_mask)
        rgb_points = rgb_points[image_mask]
        depth_points = depth_points[image_mask]
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
    camera_masks: dict[int, torch.Tensor] | None,
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
        camera_masks=camera_masks,
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
        cur_mask = None if camera_masks is None else camera_masks.get(int(frame))
        nbr_mask = None if camera_masks is None else camera_masks.get(int(neighbor))

        cur_gt_samples, cur_gt_valid = _sample_image_colors(gt_rgb, pts_cur)
        cur_pred_samples, cur_pred_valid = _sample_image_colors(pred_rgb, pts_cur)
        nbr_gt_samples, nbr_gt_valid = _sample_image_colors(neighbor_gt, pts_nbr)
        nbr_pred_samples, nbr_pred_valid = _sample_image_colors(neighbor_pred, pts_nbr)

        valid = cur_gt_valid & cur_pred_valid & nbr_gt_valid & nbr_pred_valid
        valid &= torch.from_numpy(_filter_points_by_image_mask(pts_cur, cur_mask)).to(valid.device)
        valid &= torch.from_numpy(_filter_points_by_image_mask(pts_nbr, nbr_mask)).to(valid.device)
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


def _compute_cross_frame_consistency_loss(
    pred_rgb: torch.Tensor,
    depth: torch.Tensor,
    cam_R: torch.Tensor,
    cam_T: torch.Tensor,
    nbr_gt_rgb: torch.Tensor,
    source_valid_mask: torch.Tensor | None,
    nbr_valid_mask: torch.Tensor | None,
    cam_R_nbr: torch.Tensor,
    cam_T_nbr: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    weight: float = 1.0,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    max_points: int = 20000,
) -> torch.Tensor:
    """HiGS-style cross-frame photometric consistency loss.

    Backprojects the current frame's rendered depth to world space, then reprojects
    into a neighbouring frame and compares sampled neighbour GT RGB against current
    rendered RGB.  Gradient flows through the projection matrices (cam_R / cam_T) to
    directly supervise the shared lidar-camera extrinsic; depth values and pred_rgb
    are detached to avoid confounding Gaussian-parameter gradients.

    Coordinate convention (matches render_camera_2dgs line 284):
        p_cam_row  = p_world_row  @ cam_R   + cam_T      (row-vector, world→camera)
        p_world_row = (p_cam_row  - cam_T)  @ cam_R.T    (backprojection)
    """
    H, W = depth.shape

    # Valid depth mask
    depth_det = depth.detach()
    depth_valid = (depth_det > min_depth) & (depth_det < max_depth)  # [H, W]
    if source_valid_mask is not None:
        depth_valid = depth_valid & source_valid_mask.to(device=depth.device)
    if not depth_valid.any():
        return pred_rgb.new_zeros(())

    # Pixel grid
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth.device),
        torch.arange(W, dtype=torch.float32, device=depth.device),
        indexing="ij",
    )
    u_valid = u_coords[depth_valid]  # [N]
    v_valid = v_coords[depth_valid]  # [N]
    d_valid = depth_det[depth_valid]  # [N]

    # Subsample for speed
    N = u_valid.shape[0]
    if N > max_points:
        idx = torch.randperm(N, device=depth.device)[:max_points]
        u_valid = u_valid[idx]
        v_valid = v_valid[idx]
        d_valid = d_valid[idx]
        if depth_valid.nonzero(as_tuple=False).shape[0] > max_points:
            # need to rebuild the flat index for pred_rgb sampling later
            flat_valid_idx = torch.where(depth_valid.flatten())[0][idx]
        else:
            flat_valid_idx = torch.where(depth_valid.flatten())[0][idx]
    else:
        flat_valid_idx = torch.where(depth_valid.flatten())[0]

    # Camera-space coords  [N, 3]
    x_cam = (u_valid - cx) / fx * d_valid
    y_cam = (v_valid - cy) / fy * d_valid
    P_cam = torch.stack([x_cam, y_cam, d_valid], dim=1)  # [N, 3]

    # Backproject to world: P_world = (P_cam - cam_T) @ cam_R.T   [N, 3]
    P_world = (P_cam - cam_T.unsqueeze(0)) @ cam_R.t()

    # Project into neighbour: P_cam_nbr = P_world @ cam_R_nbr + cam_T_nbr   [N, 3]
    P_cam_nbr = P_world @ cam_R_nbr + cam_T_nbr.unsqueeze(0)

    z_nbr = P_cam_nbr[:, 2]
    valid_nbr = z_nbr > min_depth  # [N]
    if not valid_nbr.any():
        return pred_rgb.new_zeros(())

    # Project to pixel coords in neighbour
    u_nbr = P_cam_nbr[valid_nbr, 0] / z_nbr[valid_nbr] * fx + cx  # [M]
    v_nbr = P_cam_nbr[valid_nbr, 1] / z_nbr[valid_nbr] * fy + cy  # [M]

    # Normalise to [-1, 1] for grid_sample
    grid_x = u_nbr / (W - 1) * 2.0 - 1.0  # [M]
    grid_y = v_nbr / (H - 1) * 2.0 - 1.0  # [M]
    in_bounds = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    if not in_bounds.any():
        return pred_rgb.new_zeros(())

    valid_nbr_flat = flat_valid_idx[valid_nbr][in_bounds]  # [M'] indices in [H*W]
    if nbr_valid_mask is not None:
        nbr_mask = nbr_valid_mask.to(device=depth.device)
        u_nbr_in = u_nbr[in_bounds]
        v_nbr_in = v_nbr[in_bounds]
        nbr_x = torch.round(u_nbr_in).long().clamp(0, int(W - 1))
        nbr_y = torch.round(v_nbr_in).long().clamp(0, int(H - 1))
        target_valid = nbr_mask[nbr_y, nbr_x]
        if not torch.any(target_valid):
            return pred_rgb.new_zeros(())
        grid_x = grid_x[in_bounds][target_valid]
        grid_y = grid_y[in_bounds][target_valid]
        valid_nbr_flat = valid_nbr_flat[target_valid]
    else:
        grid_x = grid_x[in_bounds]
        grid_y = grid_y[in_bounds]

    if grid_x.numel() == 0:
        return pred_rgb.new_zeros(())

    # Sample neighbour GT RGB at projected positions  [3, M']
    nbr_chw = nbr_gt_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)  # [1, 1, M', 2]
    sampled_nbr = F.grid_sample(
        nbr_chw.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(1)  # [3, M']

    # Corresponding rendered RGB from current frame (detached — no Gaussian grad)
    pred_rgb_flat = pred_rgb.reshape(-1, 3).detach()  # [H*W, 3]
    pred_at_source = pred_rgb_flat[valid_nbr_flat].t()  # [3, M']

    loss = weight * F.l1_loss(sampled_nbr, pred_at_source)
    return loss


def _precompute_flow_proj_cache(
    matcher,
    cam_images: dict,
    camera_masks: dict[int, torch.Tensor] | None = None,
    max_offset: int = 1,
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Precompute RGB-RGB MatchAnything correspondences for all adjacent frame pairs.

    Results are stored as pixel coordinates only (no colors) and reused every cycle.
    GT images never change, so this only needs to run once at startup.

    Returns:
        {frame_id: [(neighbor_id, pts_cur [N,2] float32, pts_nbr [N,2] float32), ...]}
    """
    cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    frames = sorted(cam_images.keys())
    print(f"[FlowProj] Precomputing RGB-RGB matches for {len(frames)} frames "
          f"(offset=±{max_offset})...", flush=True)
    _t0 = time.time()
    for i, frame in enumerate(frames):
        cur_img = _format_matcher_image(
            _mask_image_for_matcher(
                cam_images[frame],
                None if camera_masks is None else camera_masks.get(int(frame)),
            )
        )
        pairs = []
        for off in range(1, max_offset + 1):
            for nbr in (frame - off, frame + off):
                if nbr not in cam_images:
                    continue
                nbr_img = _format_matcher_image(
                    _mask_image_for_matcher(
                        cam_images[nbr],
                        None if camera_masks is None else camera_masks.get(int(nbr)),
                    )
                )
                result = matcher(cur_img, nbr_img)
                pts0, pts1 = select_match_points(result)
                pts0 = np.asarray(pts0, dtype=np.float32)
                pts1 = np.asarray(pts1, dtype=np.float32)
                keep = _filter_points_by_image_mask(
                    pts0,
                    None if camera_masks is None else camera_masks.get(int(frame)),
                )
                keep &= _filter_points_by_image_mask(
                    pts1,
                    None if camera_masks is None else camera_masks.get(int(nbr)),
                )
                pts0 = pts0[keep]
                pts1 = pts1[keep]
                if pts0.shape[0] > 0:
                    pairs.append((int(nbr), pts0, pts1))
        cache[frame] = pairs
        print(f"[FlowProj]   frame {i+1}/{len(frames)} (id={frame}) "
              f"→ {sum(len(p[1]) for p in pairs)} total matches", flush=True)
    print(f"[FlowProj] Precompute done in {time.time()-_t0:.1f}s", flush=True)
    return cache


def _compute_flow_projection_loss(
    flow_pairs: list[tuple[int, np.ndarray, np.ndarray]],
    depth: torch.Tensor,
    cam_R: torch.Tensor,
    cam_T: torch.Tensor,
    pose_correction,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    weight: float = 1.0,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> torch.Tensor:
    """Optical-flow-style reprojection loss using MatchAnything RGB-RGB correspondences.

    For each precomputed matched point pair (u_cur, u_nbr) across adjacent frames:
      1. Sample rendered depth at u_cur (detached — no Gaussian grad).
      2. Backproject u_cur to world space using current-frame pose.
      3. Reproject into neighbor frame using neighbor pose.
      4. Loss = mean pixel-distance between projected position and u_nbr.

    Gradient flows only through cam_R / cam_T (pose params), not through depth or
    Gaussian parameters.
    """
    H, W = depth.shape
    depth_det = depth.detach()
    total_loss = depth.new_zeros(())
    pair_count = 0

    for neighbor, pts_cur, pts_nbr in flow_pairs:
        if pts_cur.shape[0] == 0:
            continue

        cam_R_nbr, cam_T_nbr = pose_correction.corrected_rt(neighbor, device=depth.device)

        pts_t = torch.from_numpy(pts_cur).float().to(depth.device)  # [N, 2]

        # Sample depth at pts_cur positions
        gx = (pts_t[:, 0] / (W - 1)) * 2.0 - 1.0
        gy = (pts_t[:, 1] / (H - 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        d_samp = F.grid_sample(
            depth_det.unsqueeze(0).unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze()  # [N]

        valid = (d_samp > min_depth) & (d_samp < max_depth)
        if not valid.any():
            continue

        u = pts_t[valid, 0]
        v = pts_t[valid, 1]
        d = d_samp[valid]  # already detached via depth_det

        # Backproject to camera space then world space
        P_cam = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], dim=1)  # [M,3]
        P_world = (P_cam - cam_T.unsqueeze(0)) @ cam_R.t()  # grad: cam_R, cam_T

        # Project into neighbor frame
        P_cam_nbr = P_world @ cam_R_nbr + cam_T_nbr.unsqueeze(0)  # grad: cam_R_nbr, cam_T_nbr
        z_nbr = P_cam_nbr[:, 2]
        valid_z = z_nbr > min_depth
        if not valid_z.any():
            continue

        u_proj = P_cam_nbr[valid_z, 0] / z_nbr[valid_z] * fx + cx
        v_proj = P_cam_nbr[valid_z, 1] / z_nbr[valid_z] * fy + cy

        # Target from MatchAnything (fixed supervision)
        valid_np = valid.cpu().numpy()
        valid_z_np = valid_z.cpu().numpy()
        pts_nbr_sel = torch.from_numpy(
            pts_nbr[valid_np][valid_z_np]
        ).float().to(depth.device)  # [K, 2]

        in_bounds = (
            (u_proj >= 0) & (u_proj <= W - 1) &
            (v_proj >= 0) & (v_proj <= H - 1)
        )
        if not in_bounds.any():
            continue

        # Normalize by image diagonal so loss is scale-invariant
        diag = float((H ** 2 + W ** 2) ** 0.5)
        err_u = (u_proj[in_bounds] - pts_nbr_sel[in_bounds, 0]) / diag
        err_v = (v_proj[in_bounds] - pts_nbr_sel[in_bounds, 1]) / diag
        total_loss = total_loss + (err_u.abs() + err_v.abs()).mean()
        pair_count += 1

    if pair_count == 0:
        return depth.new_zeros(())
    return weight * total_loss / float(pair_count)


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
    camera_masks: dict[int, torch.Tensor] | None = None,
    temporal_support_getter=None,
    temporal_support_radius_px: float = 6.0,
    dense_mode: bool = False,
    dense_stride: int = 4,
    dense_cert_threshold: float = 0.02,
    dense_color_cert_threshold: float | None = None,
    depth_render_backend: str = "3dgut_rasterization",
    depth_render_mode: str = "median",
) -> dict:
    """Run cross-modal matching once per cycle for all train frames.

    Pipeline (3 phases to maximise CPU/GPU overlap):
      Phase 1 – Sequential GPU renders  → collect (depth_np, gt_rgb_cpu) per frame
      Phase 2 – Parallel CPU colourmap encoding  (ThreadPoolExecutor)
      Phase 3 – Sequential RoMa matcher inference  (single GPU, unavoidable bottleneck)
      Phase 4 – Parallel CPU postprocessing  (KDTree filter + depth sample + colour sample)

    Returns a dict keyed by frame_id with:
      - 'depth_pts':    np.ndarray [N, 2]  pixel coords in the rendered image
      - 'gt_colors':    torch.Tensor [N, 3] (CPU) GT RGB at matched rgb_pts
      - 'cert_weights': torch.Tensor [N] (CPU) per-point certainty (dense mode only)
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import map_coordinates

    n_cpu = min(os.cpu_count() or 4, 8)
    color_thresh = dense_color_cert_threshold if dense_color_cert_threshold is not None else dense_cert_threshold

    # ── Phase 1: sequential GPU render ──────────────────────────────────────
    render_items: list[tuple] = []  # (frame, depth_np, gt_rgb_np_uint8, gt_rgb_cpu_f32, valid_mask_cpu)
    n_frames = len(frame_ids)
    _t_render_start = time.time()
    for fi, frame in enumerate(frame_ids):
        if frame not in cam_cameras or frame not in cam_images:
            continue
        cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
        camera = cam_cameras[frame].cuda()
        gt_rgb = cam_images[frame].cuda()

        depth_render = _render_camera_with_backend(
            camera, [gaussians], args,
            backend=depth_render_backend,
            cam_rotation=cam_R.detach(),
            cam_translation=cam_T.detach(),
            require_rgb=False,
        )
        if not int(depth_render.get("num_visible", 0)):
            continue

        depth_tensor = _resolve_camera_aux_depth(depth_render, depth_mode=depth_render_mode)
        depth_np = depth_tensor.squeeze(-1).cpu().numpy().astype(np.float32)
        gt_valid_mask = None if camera_masks is None else camera_masks.get(int(frame))
        gt_rgb_np_uint8 = _mask_image_for_matcher(gt_rgb, gt_valid_mask)
        gt_rgb_cpu_f32  = gt_rgb.cpu()             # float32 HWC for colour sampling
        render_items.append((frame, depth_np, gt_rgb_np_uint8, gt_rgb_cpu_f32, gt_valid_mask))

    _t_render_end = time.time()
    print(f"[Cycle cache] Phase1 rendered {len(render_items)}/{n_frames} frames "
          f"in {_t_render_end-_t_render_start:.1f}s", flush=True)

    # ── Phase 2: parallel CPU depth encoding (grayscale×3, official MA convention) ──
    def _encode_depth_vis(item):
        frame, depth_np, gt_rgb_np_uint8, gt_rgb_cpu_f32, gt_valid_mask = item
        depth_vis = depth_to_match_image(
            depth_np,
            percentile_low=matcher_depth_percentile_low,
            percentile_high=matcher_depth_percentile_high,
            use_inverse=matcher_depth_use_inverse,
        )
        return frame, depth_np, depth_vis, gt_rgb_np_uint8, gt_rgb_cpu_f32, gt_valid_mask

    _t_enc_start = time.time()
    with ThreadPoolExecutor(max_workers=n_cpu) as pool:
        encoded_items = list(pool.map(_encode_depth_vis, render_items))
    _t_enc_end = time.time()
    print(f"[Cycle cache] Phase2 depth encode (grayscale×3) "
          f"({len(encoded_items)} frames, {n_cpu} threads) "
          f"in {_t_enc_end-_t_enc_start:.1f}s", flush=True)

    # ── Phase 3: sequential GPU matcher ──────────────────────────────────────
    matched_items: list[tuple] = []
    for fi, (frame, depth_np, depth_vis, gt_rgb_np_uint8, gt_rgb_cpu_f32, gt_valid_mask) in enumerate(encoded_items):
        print(f"[Cycle cache] frame {fi+1}/{len(encoded_items)} (id={frame})...", flush=True)
        _t0 = time.time()
        if dense_mode:
            rgb_pts, depth_pts, certs_np = match_cross_modal_dense(
                matcher, gt_rgb_np_uint8, depth_vis,
                query_stride=dense_stride,
                cert_threshold=color_thresh,
            )
        else:
            rgb_pts, depth_pts, _ = match_cross_modal(matcher, gt_rgb_np_uint8, depth_vis)
            certs_np = None
        _t1 = time.time()
        print(f"[Cycle cache]   match={_t1-_t0:.1f}s pts_raw={rgb_pts.shape[0]}", flush=True)
        matched_items.append((frame, depth_np, rgb_pts, depth_pts, certs_np, gt_rgb_cpu_f32, gt_valid_mask))

    # ── Phase 4: parallel CPU postprocessing ─────────────────────────────────
    def _postprocess_frame(item):
        frame, depth_np, rgb_pts, depth_pts, certs_np, gt_rgb_cpu_f32, gt_valid_mask = item

        # Temporal support filtering (KDTree, CPU)
        if temporal_support_getter is not None:
            support = temporal_support_getter(frame)
            if support is not None and support.shape[0] > 0:
                mask = _filter_points_by_support(rgb_pts, support, temporal_support_radius_px)
                rgb_pts   = rgb_pts[mask]
                depth_pts = depth_pts[mask]
                if certs_np is not None:
                    certs_np = certs_np[mask]

        image_keep = _filter_points_by_image_mask(rgb_pts, gt_valid_mask)
        rgb_pts = rgb_pts[image_keep]
        depth_pts = depth_pts[image_keep]
        if certs_np is not None:
            certs_np = certs_np[image_keep]

        if rgb_pts.shape[0] < int(matcher_min_matches):
            return frame, None

        keep_indices, _ = sample_depth_values_vectorized(
            depth_map=depth_np,
            points=np.asarray(depth_pts, dtype=np.float32),
            min_depth=matcher_depth_min,
            max_depth=matcher_depth_max,
        )
        if keep_indices.shape[0] < int(matcher_min_depth_matches):
            return frame, None

        rgb_pts_f   = np.asarray(rgb_pts[keep_indices],   dtype=np.float32)
        depth_pts_f = np.asarray(depth_pts[keep_indices], dtype=np.float32)
        certs_f     = np.asarray(certs_np[keep_indices],  dtype=np.float32) if certs_np is not None else None

        # CPU bilinear colour sampling via scipy (avoids CUDA in worker threads)
        gt_np = gt_rgb_cpu_f32.numpy()      # HWC float32
        H, W  = gt_np.shape[0], gt_np.shape[1]
        xs = rgb_pts_f[:, 0].clip(0.0, W - 1.0)
        ys = rgb_pts_f[:, 1].clip(0.0, H - 1.0)
        coords = np.array([ys, xs])  # [2, N] row-major for map_coordinates
        gt_colors_np = np.stack(
            [map_coordinates(gt_np[:, :, c], coords, order=1, mode="nearest") for c in range(3)],
            axis=-1,
        ).astype(np.float32)             # [N, 3]
        valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if int(valid_mask.sum()) < int(matcher_min_depth_matches):
            return frame, None

        depth_pts_v = depth_pts_f[valid_mask]
        # Sample depth values at depth_pts for PnP backprojection
        dxs = depth_pts_v[:, 0].clip(0.0, depth_np.shape[1] - 1.0)
        dys = depth_pts_v[:, 1].clip(0.0, depth_np.shape[0] - 1.0)
        depth_vals = map_coordinates(
            depth_np, np.array([dys, dxs]), order=1, mode="nearest"
        ).astype(np.float32)  # [N]

        entry = {
            "depth_pts":  depth_pts_v,
            "rgb_pts":    rgb_pts_f[valid_mask],        # 2D obs in camera image
            "depth_vals": depth_vals,                   # depth at depth_pts (for PnP)
            "gt_colors":  torch.as_tensor(gt_colors_np[valid_mask], dtype=torch.float32),
        }
        if certs_f is not None:
            entry["cert_weights"] = torch.as_tensor(certs_f[valid_mask], dtype=torch.float32)
        return frame, entry

    _t_post_start = time.time()
    with ThreadPoolExecutor(max_workers=n_cpu) as pool:
        post_results = list(pool.map(_postprocess_frame, matched_items))
    _t_post_end = time.time()
    print(f"[Cycle cache] Phase4 postprocess "
          f"({len(matched_items)} frames, {n_cpu} threads) "
          f"in {_t_post_end-_t_post_start:.1f}s", flush=True)

    cache = {frame: entry for frame, entry in post_results if entry is not None}
    coverage_rate = 0.0
    if cache:
        total_pts  = sum(v["depth_pts"].shape[0] for v in cache.values())
        frames_hit = len(cache)
        sample_cam = next(iter(cam_cameras.values()))
        h, w = int(sample_cam.image_height), int(sample_cam.image_width)
        pixels_per_frame = h * w
        avg_pts      = total_pts / frames_hit
        avg_coverage = avg_pts / pixels_per_frame * 100.0
        coverage_rate = avg_pts / pixels_per_frame
        print(f"[Cycle cache] {frames_hit} frames, avg {avg_pts:.0f} matches/frame "
              f"({avg_coverage:.1f}% pixel coverage, image {w}×{h})")
    return cache, coverage_rate


def _apply_cross_modal_cache_loss(
    pred_rgb: torch.Tensor,
    cache_entry: dict,
    matcher_color_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Compute match color loss using precomputed (depth_pts, gt_colors) cache.

    If 'cert_weights' is present in the cache entry, uses certainty-weighted
    mean absolute error instead of plain L1.  This allows using a lower cert
    threshold for more coverage while down-weighting uncertain matches.
    """
    depth_pts = cache_entry["depth_pts"]
    gt_colors = cache_entry["gt_colors"].to(pred_rgb.device)

    pred_colors, pred_valid = _sample_image_colors(pred_rgb, depth_pts)
    gt_colors_valid = gt_colors[pred_valid]
    pred_colors_valid = pred_colors[pred_valid]

    n = int(pred_colors_valid.shape[0])
    if n < 5:
        return pred_rgb.new_zeros(()), {"status": "skipped", "reason": "too_few_valid_samples", "matches": n}

    cert_w_raw = cache_entry.get("cert_weights")
    if cert_w_raw is not None:
        cert_w = cert_w_raw.to(pred_rgb.device)[pred_valid]  # [n]
        w_sum = cert_w.sum().clamp(min=1e-8)
        diff = torch.abs(pred_colors_valid - gt_colors_valid)  # [n, 3]
        loss = float(matcher_color_weight) * (cert_w.unsqueeze(1) * diff).sum() / w_sum
        mae = ((cert_w.unsqueeze(1) * diff).sum() / w_sum).item() / 3.0
    else:
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
    camera_masks: dict[int, torch.Tensor] | None = None,
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
            cur_keep = _filter_points_by_image_mask(
                pts_cur,
                None if camera_masks is None else camera_masks.get(int(frame)),
            )
            nbr_keep = _filter_points_by_image_mask(
                pts_nbr,
                None if camera_masks is None else camera_masks.get(int(neighbor)),
            )
            pair_keep = cur_keep & nbr_keep
            if not np.any(pair_keep):
                continue
            pts_cur = pts_cur[pair_keep]
            pts_nbr = pts_nbr[pair_keep]
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
    camera_masks: dict[int, torch.Tensor] | None,
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
    depth_render_backend: str = "3dgut_rasterization",
    depth_render_mode: str = "median",
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] | None = None,
    temporal_photometric_weight: float = 0.0,
    temporal_photometric_match_radius_px: float = 4.0,
    temporal_photometric_min_matches: int = 8,
    temporal_flow_weight: float = 0.0,
    temporal_flow_match_radius_px: float = 4.0,
    temporal_flow_min_matches: int = 8,
    temporal_depth_weight: float = 0.0,
    temporal_depth_match_radius_px: float = 4.0,
    temporal_depth_min_matches: int = 8,
    temporal_semidense_stride: int = 0,
    temporal_semidense_max_points: int = 0,
    temporal_gradient_scale: float = 0.0,
    init_depth_strat_bins: int = 0,
    init_depth_strat_max_points_per_bin: int = 0,
    far_depth_boost: float = 0.0,
    far_depth_start_percentile: float = 60.0,
    rgb_blur_kernel: int = 0,
    rgb_blur_sigma: float = 0.0,
    temporal_support_getter=None,
    temporal_support_radius_px: float = 6.0,
    pure_pnp_residual_filter_mad_scale: float = 0.0,
    pure_pnp_residual_filter_min_keep_ratio: float = 0.5,
    pure_pnp_residual_filter_min_keep_per_frame: int = 24,
    pure_pnp_residual_filter_max_reproj_error: float = 0.0,
    pure_pnp_single_frame_prefilter_min_inliers: int = 0,
    pure_pnp_single_frame_prefilter_min_inlier_ratio: float = 0.0,
    pure_pnp_frame_disagreement_mad_scale: float = 0.0,
    pure_pnp_frame_disagreement_min_keep_ratio: float = 0.7,
    pure_pnp_frame_disagreement_min_keep_frames: int = 12,
    pure_pnp_frame_disagreement_apply_max_dropped_frames: int = 0,
    pure_pnp_filter_shared_ransac_outliers: bool = False,
    pure_pnp_gt_reproj_filter_quantile: float = 0.0,
    pure_pnp_gt_reproj_filter_min_keep_per_frame: int = 24,
    pure_pnp_gt_soft_weight_mode: str = "none",
    pure_pnp_gt_soft_weight_translation_alpha: float = 0.5,
    pure_pnp_gt_pose_residual_weight: float = 0.0,
    pure_pnp_optimize_rotation: bool = True,
    pure_pnp_optimize_translation: bool = True,
    pure_pnp_solver_backend: str = "auto",
    matcher_lidar_nn_max_distance: float = 0.0,
    render_intrinsics_fx_scale: float = 1.0,
    render_intrinsics_fy_scale: float = 1.0,
    render_intrinsics_cx_offset: float = 0.0,
    render_intrinsics_cy_offset: float = 0.0,
    gt_rotation_matrix: np.ndarray | None = None,
    gt_translation: np.ndarray | None = None,
    match_diagnostics_path: str | None = None,
    gt_weight_analysis_json_path: str | None = None,
    gt_weight_analysis_npz_path: str | None = None,
):
    if not frame_ids:
        return {"status": "skipped", "reason": "no camera frames"}

    rgb_camera_model = _camera_model_from_camera(cam_cameras[frame_ids[0]])
    frame_data_list = []
    rendered_depth_maps: dict[int, np.ndarray] = {}
    rendered_depth_cameras: dict[int, CameraModel] = {}
    lidar_nn_total_points = 0
    lidar_nn_kept_points = 0
    n_pnp_frames = len([f for f in frame_ids if f in cam_cameras and f in cam_images])
    for fi, frame in enumerate(frame_ids):
        if frame not in cam_cameras or frame not in cam_images:
            continue
        if fi % 10 == 0:
            print(f"[PnP init] rendering frame {fi+1}/{n_pnp_frames} (id={frame})...", flush=True)
        camera = cam_cameras[frame].cuda()
        render_camera = _make_render_intrinsics_override_camera(
            camera,
            fx_scale=render_intrinsics_fx_scale,
            fy_scale=render_intrinsics_fy_scale,
            cx_offset=render_intrinsics_cx_offset,
            cy_offset=render_intrinsics_cy_offset,
        )
        render_camera_model = _camera_model_from_camera(render_camera)
        gt_rgb = cam_images[frame].detach().cpu().numpy()
        if gt_rgb.max(initial=0.0) <= 1.5:
            gt_rgb = np.clip(gt_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            gt_rgb = np.clip(gt_rgb, 0.0, 255.0).astype(np.uint8)
        gt_valid_mask = None if camera_masks is None else camera_masks.get(int(frame))
        blur_kernel = max(int(rgb_blur_kernel), 0)
        if blur_kernel > 0:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            gt_rgb = cv2.GaussianBlur(
                gt_rgb,
                (blur_kernel, blur_kernel),
                sigmaX=float(max(rgb_blur_sigma, 0.0)),
                sigmaY=float(max(rgb_blur_sigma, 0.0)),
                borderType=cv2.BORDER_REFLECT101,
            )

        cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
        render_pkg = _render_camera_with_backend(
            render_camera,
            [gaussians],
            args,
            backend=depth_render_backend,
            cam_rotation=cam_R,
            cam_translation=cam_T,
            require_rgb=False,
        )
        if int(render_pkg.get("num_visible", 0)) <= 0:
            continue

        depth_tensor = _resolve_camera_aux_depth(render_pkg, depth_mode=depth_render_mode)
        depth = depth_tensor.detach().squeeze(-1).cpu().numpy().astype(np.float32)
        rendered_depth_maps[int(frame)] = depth
        rendered_depth_cameras[int(frame)] = render_camera_model
        depth_vis = depth_to_match_image(
            depth,
            percentile_low=matcher_depth_percentile_low,
            percentile_high=matcher_depth_percentile_high,
            use_inverse=matcher_depth_use_inverse,
        )
        rgb_points, depth_points, _ = match_cross_modal(
            matcher,
            _mask_image_for_matcher(gt_rgb, gt_valid_mask),
            depth_vis,
        )
        if temporal_support_getter is not None:
            support_points = temporal_support_getter(frame)
            support_mask = _filter_points_by_support(
                query_points=rgb_points,
                support_points=support_points,
                radius_px=temporal_support_radius_px,
            )
            rgb_points = rgb_points[support_mask]
            depth_points = depth_points[support_mask]
        image_keep = _filter_points_by_image_mask(rgb_points, gt_valid_mask)
        rgb_points = rgb_points[image_keep]
        depth_points = depth_points[image_keep]
        if rgb_points.shape[0] < int(matcher_min_matches):
            continue

        frame_data = build_frame_correspondence(
            frame_name=f"{frame:06d}",
            rgb_path=f"frame:{frame}",
            depth_path=f"render:{frame}",
            rgb_points=rgb_points,
            depth_points=depth_points,
            depth_map=depth,
            depth_camera=render_camera_model,
            min_depth=matcher_depth_min,
            max_depth=matcher_depth_max,
            search_radius=2,
            far_depth_boost=far_depth_boost,
            far_depth_start_percentile=far_depth_start_percentile,
        )
        if frame_data is None or frame_data.points_3d.shape[0] < int(matcher_min_depth_matches):
            continue
        if float(matcher_lidar_nn_max_distance) > 0.0:
            lidar_reference_points = _gaussian_points_in_camera_frame(
                gaussians,
                int(frame),
                cam_R,
                cam_T,
            )
            frame_data_filtered, lidar_nn_diag = filter_frame_correspondence_by_reference_cloud(
                frame_data,
                reference_points=lidar_reference_points,
                max_distance=float(matcher_lidar_nn_max_distance),
            )
            lidar_nn_total_points += int(lidar_nn_diag.get("total_points", 0))
            lidar_nn_kept_points += int(lidar_nn_diag.get("kept_points", 0))
            frame_data = frame_data_filtered
            if frame_data is None or frame_data.points_3d.shape[0] < int(matcher_min_depth_matches):
                continue
        frame_data.frame_id = int(frame)
        frame_data.frame_index = len(frame_data_list)
        frame_data_list.append(frame_data)

    if not frame_data_list:
        return {"status": "skipped", "reason": "no valid frame correspondences"}

    init_frame_data_list = frame_data_list
    if int(init_depth_strat_bins) > 1 and int(init_depth_strat_max_points_per_bin) > 0:
        init_frame_data_list = stratify_frame_data_by_depth(
            frame_data_list=frame_data_list,
            num_bins=int(init_depth_strat_bins),
            max_points_per_bin=int(init_depth_strat_max_points_per_bin),
        )
        if not init_frame_data_list:
            return {"status": "skipped", "reason": "depth-stratified init rejected all correspondences"}

    photometric_residuals: list[TemporalPhotometricResidualData] = []
    photometric_target_images: dict[int, np.ndarray] = {}
    temporal_residuals: list[TemporalResidualData] = []
    depth_residuals: list[TemporalDepthResidualData] = []
    if temporal_photometric_weight > 0.0 and temporal_pair_cache:
        photometric_residuals, photometric_target_images = _build_temporal_photometric_residuals(
            frame_data_list=frame_data_list,
            temporal_pair_cache=temporal_pair_cache,
            cam_images=cam_images,
            pose_correction=pose_correction,
            rgb_camera=rgb_camera_model,
            source_depth_maps=rendered_depth_maps,
            source_depth_cameras=rendered_depth_cameras,
            target_depth_maps=rendered_depth_maps,
            target_depth_cameras=rendered_depth_cameras,
            match_radius_px=temporal_photometric_match_radius_px,
            min_matches=temporal_photometric_min_matches,
            semidense_stride=temporal_semidense_stride,
            semidense_max_points=temporal_semidense_max_points,
            gradient_scale=temporal_gradient_scale,
        )
    if temporal_flow_weight > 0.0 and temporal_pair_cache:
        temporal_residuals = _build_temporal_geometric_residuals(
            frame_data_list=frame_data_list,
            temporal_pair_cache=temporal_pair_cache,
            pose_correction=pose_correction,
            rgb_camera=rgb_camera_model,
            cam_images=cam_images,
            target_depth_maps=rendered_depth_maps,
            target_depth_cameras=rendered_depth_cameras,
            match_radius_px=temporal_flow_match_radius_px,
            min_matches=temporal_flow_min_matches,
            gradient_scale=temporal_gradient_scale,
        )
    if temporal_depth_weight > 0.0 and temporal_pair_cache:
        depth_residuals = _build_temporal_depth_residuals(
            frame_data_list=frame_data_list,
            temporal_pair_cache=temporal_pair_cache,
            pose_correction=pose_correction,
            rgb_camera=rgb_camera_model,
            source_depth_maps=rendered_depth_maps,
            source_depth_cameras=rendered_depth_cameras,
            cam_images=cam_images,
            target_depth_maps=rendered_depth_maps,
            target_depth_cameras=rendered_depth_cameras,
            match_radius_px=temporal_depth_match_radius_px,
            min_matches=temporal_depth_min_matches,
            semidense_stride=temporal_semidense_stride,
            semidense_max_points=temporal_semidense_max_points,
            gradient_scale=temporal_gradient_scale,
        )

    raw_matches_total = int(sum(frame.points_3d.shape[0] for frame in frame_data_list))
    frame_filter_diagnostics = None
    filter_diagnostics = None
    single_frame_prefilter_diagnostics = None
    shared_ransac_diagnostics = None
    gt_reproj_filter_diagnostics = None
    gt_weight_summary = None
    gt_weight_arrays = None
    filtered_matches_total = raw_matches_total
    gt_rvec = None
    gt_tvec = None
    if gt_rotation_matrix is not None and gt_translation is not None:
        current_l2c_R, current_l2c_T = pose_correction.corrected_lidar_to_camera(
            int(frame_ids[0]),
            device="cpu",
        )
        current_l2c_R_np = np.asarray(current_l2c_R.detach().cpu().numpy(), dtype=np.float64)
        current_l2c_T_np = np.asarray(current_l2c_T.detach().cpu().numpy(), dtype=np.float64).reshape(3)
        gt_l2c_R_np = np.asarray(gt_rotation_matrix, dtype=np.float64)
        gt_l2c_T_np = np.asarray(gt_translation, dtype=np.float64).reshape(3)
        gt_relative_R = gt_l2c_R_np @ current_l2c_R_np.T
        gt_relative_T = gt_l2c_T_np - (gt_relative_R @ current_l2c_T_np.reshape(3, 1)).reshape(3)
        gt_rvec, _ = cv2.Rodrigues(gt_relative_R)
        gt_rvec = gt_rvec.reshape(3)
        gt_tvec = gt_relative_T.reshape(3)

    try:
        initial_rvec, initial_tvec = initialize_shared_extrinsic(
            frame_data_list=init_frame_data_list,
            rgb_camera=rgb_camera_model,
            reproj_error=matcher_pnp_reproj_error,
            iterations=matcher_pnp_iterations,
            min_inliers=matcher_min_pnp_inliers,
            filter_frames=False,
        )
        active_frame_data_list = frame_data_list
        if pure_pnp_filter_shared_ransac_outliers:
            _, _, ransac_filtered_frames, shared_ransac_diagnostics = filter_frame_data_by_shared_ransac(
                frame_data_list=frame_data_list,
                rgb_camera=rgb_camera_model,
                reproj_error=matcher_pnp_reproj_error,
                iterations=matcher_pnp_iterations,
                min_inliers=matcher_min_pnp_inliers,
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
            )
            if ransac_filtered_frames:
                active_frame_data_list = ransac_filtered_frames
                filtered_matches_total = int(sum(frame.points_3d.shape[0] for frame in active_frame_data_list))
        if (
            pure_pnp_gt_reproj_filter_quantile > 0.0
            and gt_rvec is not None
            and gt_tvec is not None
        ):
            gt_filtered_frames, gt_reproj_filter_diagnostics = filter_frame_data_by_gt_reprojection(
                frame_data_list=active_frame_data_list,
                rgb_camera=rgb_camera_model,
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
                reference_rvec=initial_rvec,
                reference_tvec=initial_tvec,
                keep_quantile=pure_pnp_gt_reproj_filter_quantile,
                min_keep_per_frame=pure_pnp_gt_reproj_filter_min_keep_per_frame,
            )
            if gt_filtered_frames:
                active_frame_data_list = gt_filtered_frames
                filtered_matches_total = int(sum(frame.points_3d.shape[0] for frame in active_frame_data_list))
        if (
            (gt_weight_analysis_json_path is not None or pure_pnp_gt_soft_weight_mode != "none")
            and gt_rvec is not None
            and gt_tvec is not None
        ):
            gt_weight_summary, gt_weight_arrays = learn_gt_point_weight_predictor(
                frame_data_list=active_frame_data_list,
                rgb_camera=rgb_camera_model,
                init_rvec=initial_rvec,
                init_tvec=initial_tvec,
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
                gt_pose_residual_weight=pure_pnp_gt_pose_residual_weight,
                reproj_error=matcher_pnp_reproj_error,
                iterations=matcher_pnp_iterations,
                min_inliers=matcher_min_pnp_inliers,
                supervision_target=(
                    "gt_pose_balance"
                    if pure_pnp_gt_soft_weight_mode == "pose_balance"
                    else "gt_reproj"
                ),
                translation_sensitivity_alpha=pure_pnp_gt_soft_weight_translation_alpha,
            )
            if gt_weight_analysis_json_path is not None:
                os.makedirs(os.path.dirname(gt_weight_analysis_json_path), exist_ok=True)
                with open(gt_weight_analysis_json_path, "w", encoding="utf-8") as f:
                    json.dump(gt_weight_summary, f, indent=2)
            if gt_weight_analysis_npz_path is not None and gt_weight_arrays is not None:
                os.makedirs(os.path.dirname(gt_weight_analysis_npz_path), exist_ok=True)
                np.savez_compressed(gt_weight_analysis_npz_path, **gt_weight_arrays)
        if pure_pnp_gt_soft_weight_mode != "none" and gt_weight_arrays is not None:
            active_frame_data_list = build_weighted_frame_data_list(
                active_frame_data_list,
                gt_weight_arrays["learned_weight"],
            )
        if (
            pure_pnp_single_frame_prefilter_min_inliers > 0
            or pure_pnp_single_frame_prefilter_min_inlier_ratio > 0.0
        ):
            candidate_frames, single_frame_prefilter_diagnostics = filter_frame_data_by_single_frame_pnp_stability(
                frame_data_list=active_frame_data_list,
                rgb_camera=rgb_camera_model,
                reproj_error=matcher_pnp_reproj_error,
                iterations=matcher_pnp_iterations,
                pnp_min_inliers=matcher_min_pnp_inliers,
                keep_min_inliers=pure_pnp_single_frame_prefilter_min_inliers,
                keep_min_inlier_ratio=pure_pnp_single_frame_prefilter_min_inlier_ratio,
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
            )
            if candidate_frames:
                active_frame_data_list = candidate_frames
                filtered_matches_total = int(sum(frame.points_3d.shape[0] for frame in active_frame_data_list))
                filtered_init_frame_data_list = active_frame_data_list
                if int(init_depth_strat_bins) > 1 and int(init_depth_strat_max_points_per_bin) > 0:
                    filtered_init_frame_data_list = stratify_frame_data_by_depth(
                        frame_data_list=active_frame_data_list,
                        num_bins=int(init_depth_strat_bins),
                        max_points_per_bin=int(init_depth_strat_max_points_per_bin),
                    )
                if filtered_init_frame_data_list:
                    initial_rvec, initial_tvec = initialize_shared_extrinsic(
                        frame_data_list=filtered_init_frame_data_list,
                        rgb_camera=rgb_camera_model,
                        reproj_error=matcher_pnp_reproj_error,
                        iterations=matcher_pnp_iterations,
                        min_inliers=matcher_min_pnp_inliers,
                        filter_frames=False,
                    )
        apply_frame_filter = pure_pnp_frame_disagreement_mad_scale > 0.0
        if pure_pnp_frame_disagreement_mad_scale > 0.0:
            candidate_frames, frame_filter_diagnostics = filter_frame_data_by_pose_disagreement(
                frame_data_list=active_frame_data_list,
                rgb_camera=rgb_camera_model,
                shared_rvec=initial_rvec,
                shared_tvec=initial_tvec,
                reproj_error=matcher_pnp_reproj_error,
                iterations=matcher_pnp_iterations,
                min_inliers=matcher_min_pnp_inliers,
                mad_scale=pure_pnp_frame_disagreement_mad_scale,
                min_keep_ratio=pure_pnp_frame_disagreement_min_keep_ratio,
                min_keep_frames=pure_pnp_frame_disagreement_min_keep_frames,
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
            )
            dropped_frames = (
                0
                if not frame_filter_diagnostics
                else int(frame_filter_diagnostics.get("dropped_frames", 0))
            )
            if (
                pure_pnp_frame_disagreement_apply_max_dropped_frames > 0
                and dropped_frames > pure_pnp_frame_disagreement_apply_max_dropped_frames
            ):
                apply_frame_filter = False
                if frame_filter_diagnostics is not None:
                    frame_filter_diagnostics["applied"] = False
                    frame_filter_diagnostics["skip_reason"] = (
                        "too_many_dropped_frames"
                    )
                    frame_filter_diagnostics["apply_max_dropped_frames"] = int(
                        pure_pnp_frame_disagreement_apply_max_dropped_frames
                    )
            elif frame_filter_diagnostics is not None:
                frame_filter_diagnostics["applied"] = True
                frame_filter_diagnostics["apply_max_dropped_frames"] = int(
                    pure_pnp_frame_disagreement_apply_max_dropped_frames
                )
            if candidate_frames and apply_frame_filter:
                active_frame_data_list = candidate_frames
                filtered_matches_total = int(sum(frame.points_3d.shape[0] for frame in active_frame_data_list))
                filtered_init_frame_data_list = active_frame_data_list
                if int(init_depth_strat_bins) > 1 and int(init_depth_strat_max_points_per_bin) > 0:
                    filtered_init_frame_data_list = stratify_frame_data_by_depth(
                        frame_data_list=active_frame_data_list,
                        num_bins=int(init_depth_strat_bins),
                        max_points_per_bin=int(init_depth_strat_max_points_per_bin),
                    )
                if filtered_init_frame_data_list:
                    initial_rvec, initial_tvec = initialize_shared_extrinsic(
                        frame_data_list=filtered_init_frame_data_list,
                        rgb_camera=rgb_camera_model,
                        reproj_error=matcher_pnp_reproj_error,
                        iterations=matcher_pnp_iterations,
                        min_inliers=matcher_min_pnp_inliers,
                        filter_frames=False,
                    )
        if pure_pnp_residual_filter_mad_scale > 0.0 and not apply_frame_filter:
            candidate_frames, filter_diagnostics = filter_frame_data_by_reprojection_consensus(
                frame_data_list=active_frame_data_list,
                rgb_camera=rgb_camera_model,
                rvec=initial_rvec,
                tvec=initial_tvec,
                mad_scale=pure_pnp_residual_filter_mad_scale,
                min_keep_ratio=pure_pnp_residual_filter_min_keep_ratio,
                min_keep_per_frame=pure_pnp_residual_filter_min_keep_per_frame,
                max_reproj_error=(
                    pure_pnp_residual_filter_max_reproj_error
                    if pure_pnp_residual_filter_max_reproj_error > 0.0
                    else None
                ),
                gt_rvec=gt_rvec,
                gt_tvec=gt_tvec,
            )
            if candidate_frames:
                active_frame_data_list = candidate_frames
                filtered_matches_total = int(sum(frame.points_3d.shape[0] for frame in active_frame_data_list))
                filtered_init_frame_data_list = active_frame_data_list
                if int(init_depth_strat_bins) > 1 and int(init_depth_strat_max_points_per_bin) > 0:
                    filtered_init_frame_data_list = stratify_frame_data_by_depth(
                        frame_data_list=active_frame_data_list,
                        num_bins=int(init_depth_strat_bins),
                        max_points_per_bin=int(init_depth_strat_max_points_per_bin),
                    )
                if filtered_init_frame_data_list:
                    initial_rvec, initial_tvec = initialize_shared_extrinsic(
                        frame_data_list=filtered_init_frame_data_list,
                        rgb_camera=rgb_camera_model,
                        reproj_error=matcher_pnp_reproj_error,
                        iterations=matcher_pnp_iterations,
                        min_inliers=matcher_min_pnp_inliers,
                        filter_frames=False,
                    )
        comparison = optimize_shared_extrinsic(
            frame_data_list=active_frame_data_list,
            rgb_camera=rgb_camera_model,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
            temporal_residuals=temporal_residuals,
            temporal_residual_weight=temporal_flow_weight,
            photometric_residuals=photometric_residuals,
            photometric_target_images=photometric_target_images,
            photometric_residual_weight=temporal_photometric_weight,
            depth_residuals=depth_residuals,
            depth_target_maps=rendered_depth_maps,
            depth_target_cameras=rendered_depth_cameras,
            depth_residual_weight=temporal_depth_weight,
            optimize_rotation=pure_pnp_optimize_rotation,
            optimize_translation=pure_pnp_optimize_translation,
            solver_backend=pure_pnp_solver_backend,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            gt_pose_residual_weight=pure_pnp_gt_pose_residual_weight,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    if match_diagnostics_path is not None:
        os.makedirs(os.path.dirname(match_diagnostics_path), exist_ok=True)
        raw_frame_summaries = _summarize_frame_correspondences(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera_model,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            ref_rvec=initial_rvec,
            ref_tvec=initial_tvec,
        )
        active_frame_summaries = _summarize_frame_correspondences(
            frame_data_list=active_frame_data_list,
            rgb_camera=rgb_camera_model,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            ref_rvec=comparison.optimized.rotation_vector,
            ref_tvec=comparison.optimized.translation,
        )
        payload = {
            "raw_matches_total": raw_matches_total,
            "filtered_matches_total": filtered_matches_total,
            "frames_after_filter": int(comparison.optimized.frames_used),
            "matches_after_opt": int(comparison.optimized.matches_used),
            "mean_reproj_px": float(comparison.optimized.mean_reprojection_error),
            "rotation_delta_deg": float(comparison.rotation_delta_deg),
            "translation_delta_m": float(comparison.translation_delta_m),
            "single_frame_prefilter": single_frame_prefilter_diagnostics,
            "frame_disagreement_filter": frame_filter_diagnostics,
            "consensus_filter": filter_diagnostics,
            "shared_ransac_filter": shared_ransac_diagnostics,
            "gt_reprojection_filter": gt_reproj_filter_diagnostics,
            "gt_weight_learning": gt_weight_summary,
            "gt_soft_weight_mode": str(pure_pnp_gt_soft_weight_mode),
            "gt_soft_weight_translation_alpha": float(pure_pnp_gt_soft_weight_translation_alpha),
            "gt_pose_residual_weight": float(pure_pnp_gt_pose_residual_weight),
            "optimize_rotation": bool(pure_pnp_optimize_rotation),
            "optimize_translation": bool(pure_pnp_optimize_translation),
            "solver_backend": str(pure_pnp_solver_backend),
            "raw_frame_summaries": raw_frame_summaries,
            "active_frame_summaries": active_frame_summaries,
        }
        with open(match_diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

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
        "temporal_blocks": int(len(temporal_residuals)),
        "photometric_blocks": int(len(photometric_residuals)),
        "depth_blocks": int(len(depth_residuals)),
        "raw_matches_total": raw_matches_total,
        "filtered_matches_total": filtered_matches_total,
        "filter_dropped_matches": int(max(raw_matches_total - filtered_matches_total, 0)),
        "filter_kept_ratio": float(filtered_matches_total / max(raw_matches_total, 1)),
        "filter_gt_kept_median_px": (
            None
            if not filter_diagnostics
            else filter_diagnostics.get("global_kept_gt_reproj", {}).get("median")
        ),
        "filter_gt_dropped_median_px": (
            None
            if not filter_diagnostics
            else filter_diagnostics.get("global_dropped_gt_reproj", {}).get("median")
        ),
        "frame_filter_dropped_frames": (
            0
            if not frame_filter_diagnostics
            else int(frame_filter_diagnostics.get("dropped_frames", 0))
        ),
        "lidar_nn_total_points": int(lidar_nn_total_points),
        "lidar_nn_kept_points": int(lidar_nn_kept_points),
        "lidar_nn_dropped_points": int(max(lidar_nn_total_points - lidar_nn_kept_points, 0)),
        "lidar_nn_kept_ratio": float(lidar_nn_kept_points / max(lidar_nn_total_points, 1)),
        "gt_weight_top_gt_reproj_median_px": (
            None
            if not gt_weight_summary
            else gt_weight_summary.get("top_weight_summary", {}).get("gt_reproj_px", {}).get("median")
        ),
        "gt_soft_weight_mode": str(pure_pnp_gt_soft_weight_mode),
    }


# ─────────────────────────────────────────────────────────────
# Gaussian state save / restore
# ─────────────────────────────────────────────────────────────

_GAUSSIAN_ATTRS = [
    "_xyz", "_features_dc", "_features_rest",
    "_features_rgb_dc", "_features_rgb_rest",
    "_scaling", "_rotation", "_opacity",
]
_GEOMETRY_ATTRS = ["_xyz", "_scaling", "_rotation"]


def _collect_named_params(module, attrs: list[str]) -> list[tuple[str, torch.nn.Parameter]]:
    named_params: list[tuple[str, torch.nn.Parameter]] = []
    for attr in attrs:
        param = getattr(module, attr, None)
        if isinstance(param, torch.nn.Parameter) and param.requires_grad:
            named_params.append((attr, param))
    return named_params


def _compute_named_grads(
    loss: torch.Tensor,
    named_params: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor | None]:
    grads = {attr: None for attr, _ in named_params}
    if not torch.is_tensor(loss) or not bool(loss.requires_grad) or not named_params:
        return grads
    values = torch.autograd.grad(
        loss,
        [param for _, param in named_params],
        retain_graph=True,
        allow_unused=True,
    )
    for (attr, _), grad in zip(named_params, values):
        if grad is not None:
            grads[attr] = grad.detach().clone()
    return grads


def _accumulate_named_grads(
    accum: dict[str, torch.Tensor | None],
    grads: dict[str, torch.Tensor | None],
    blocked_attrs: set[str] | None = None,
) -> None:
    blocked = blocked_attrs or set()
    for attr, grad in grads.items():
        if attr in blocked or grad is None:
            continue
        current = accum.get(attr)
        accum[attr] = grad if current is None else (current + grad)


def _merge_lidar_priority_geom_grads(
    lidar_grads: dict[str, torch.Tensor | None],
    camera_grads: dict[str, torch.Tensor | None],
    camera_weight: float,
) -> dict[str, torch.Tensor | None]:
    merged: dict[str, torch.Tensor | None] = {}
    for attr in set(lidar_grads.keys()) | set(camera_grads.keys()):
        lidar_grad = lidar_grads.get(attr)
        camera_grad = camera_grads.get(attr)
        if lidar_grad is None and camera_grad is None:
            merged[attr] = None
            continue
        if lidar_grad is None:
            merged[attr] = float(camera_weight) * camera_grad
            continue
        if camera_grad is None:
            merged[attr] = lidar_grad
            continue
        projected_camera = camera_grad
        dot = torch.sum(lidar_grad * camera_grad)
        if bool(dot < 0):
            denom = torch.sum(lidar_grad * lidar_grad).clamp_min(1.0e-12)
            projected_camera = camera_grad - (dot / denom) * lidar_grad
        merged[attr] = lidar_grad + float(camera_weight) * projected_camera
    return merged


def _override_named_param_grads(
    named_params: list[tuple[str, torch.nn.Parameter]],
    grads: dict[str, torch.Tensor | None],
) -> None:
    for attr, param in named_params:
        grad = grads.get(attr)
        if grad is None:
            continue
        grad_value = grad.to(device=param.device, dtype=param.dtype)
        if param.grad is None:
            param.grad = grad_value.clone()
        else:
            param.grad.copy_(grad_value)


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


def _select_lidar_visualization_frames(scene, num_frames: int = 3) -> list[int]:
    lidar_sensor = getattr(scene, "train_lidar", None)
    if lidar_sensor is None:
        return []
    frame_ids = sorted(int(k) for k in getattr(lidar_sensor, "train_frames", []))
    if not frame_ids:
        frame_ids = sorted(int(k) for k in getattr(lidar_sensor, "sensor2world", {}).keys())
    if not frame_ids:
        return []
    if len(frame_ids) <= num_frames:
        return frame_ids
    indices = np.linspace(0, len(frame_ids) - 1, num_frames).round().astype(int)
    return [frame_ids[i] for i in indices]


def _depth_error_to_image(error_map: np.ndarray) -> np.ndarray:
    error = np.asarray(error_map, dtype=np.float32)
    valid = error > 0.0
    if not np.any(valid):
        return np.zeros((*error.shape, 3), dtype=np.uint8)
    vmax = max(float(np.percentile(error[valid], 95.0)), 1.0e-6)
    norm = np.clip(error / vmax, 0.0, 1.0)
    encoded = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(encoded, cv2.COLORMAP_INFERNO)


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
    matcher_minima_root: str | None = None,
    matcher_minima_ckpt: str | None = None,
    matcher_resize: int = 832,
    matcher_match_threshold: float = 0.2,
    matcher_render_intrinsics_fx_scale: float = 1.0,
    matcher_render_intrinsics_fy_scale: float = 1.0,
    matcher_render_intrinsics_cx_offset: float = 0.0,
    matcher_render_intrinsics_cy_offset: float = 0.0,
    matcher_lidar_nn_max_distance: float = 0.0,
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
    camera_aux_depth_render_backend: str = "3dgut_rasterization",
    camera_aux_depth_mode: str = "median",
):
    viz_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    viz_depth_backend = _resolve_visualization_depth_backend(
        camera_aux_depth_render_backend
    )

    rot_err = _rotation_error_deg(_effective_R(pose_correction), gt_l2c_R)
    trans_err = _translation_error_m(pose_correction, gt_l2c_T)
    frame_ids = _select_visualization_frames(cam_cameras, scene, num_frames=3)
    if not frame_ids:
        return

    matcher = None
    temporal_support_cache: dict[int, np.ndarray] = {}
    if matcher_name in {"matchanything-roma", "minima-roma"}:
        try:
            matcher = build_matcher(
                matcher_name=matcher_name,
                device="cuda",
                max_num_keypoints=int(matcher_max_num_keypoints),
                ransac_reproj_thresh=float(matcher_ransac_reproj_thresh),
                img_resize=int(matcher_resize),
                match_threshold=float(matcher_match_threshold),
                minima_root=matcher_minima_root,
                minima_ckpt=matcher_minima_ckpt,
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
            backend=viz_depth_backend,
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
        depth_tensor = _resolve_camera_aux_depth(depth_render, depth_mode=camera_aux_depth_mode)
        depth_np = depth_tensor.detach().squeeze(-1).cpu().numpy().astype(np.float32)
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
            gt_valid_mask = _get_camera_supervision_mask(cam_cameras, frame)
            rgb_points, depth_points, _ = match_cross_modal(
                matcher,
                _mask_image_for_matcher(gt_np, gt_valid_mask),
                depth_vis,
            )
            image_keep = _filter_points_by_image_mask(rgb_points, gt_valid_mask)
            rgb_points = rgb_points[image_keep]
            depth_points = depth_points[image_keep]
            panel_stats["raw_matches"] = int(rgb_points.shape[0])
            if matcher_adjacent_support:
                support_points = _get_temporal_support_points(
                    matcher=matcher,
                    frame=int(frame),
                    cam_images=cam_images,
                    camera_masks={int(k): _get_camera_supervision_mask(cam_cameras, k) for k in cam_images.keys()},
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
                    valid_np = valid.cpu().numpy()
                    rgb_points = rgb_points[valid_np]
                    depth_points = depth_points[valid_np]
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


@torch.no_grad()
def _save_lidar_supervision_depth_visualizations(
    out_dir: str,
    gaussians,
    scene,
    args,
    num_frames: int = 3,
):
    lidar_sensor = getattr(scene, "train_lidar", None)
    if lidar_sensor is None:
        return None
    frame_ids = _select_lidar_visualization_frames(scene, num_frames=num_frames)
    if not frame_ids:
        return None

    viz_dir = os.path.join(out_dir, "visualizations_lidar_supervision")
    os.makedirs(viz_dir, exist_ok=True)
    background = torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32)
    summary = []
    panels = []

    for frame in frame_ids:
        gt_mask = lidar_sensor.get_mask(frame).cuda()
        dyn_mask = lidar_sensor.get_dynamic_mask(frame).cuda()
        static_mask = gt_mask & ~dyn_mask
        if not torch.any(static_mask):
            continue

        gt_depth_map = lidar_sensor.get_depth(frame).cuda().float()
        rays_o_valid, rays_d_valid, _ = lidar_sensor.get_valid_depth_rays(frame)
        static_valid_mask = ~dyn_mask[gt_mask]
        if not torch.any(static_valid_mask):
            continue

        sparse_sensor = (
            rays_o_valid.unsqueeze(1),
            rays_d_valid.unsqueeze(1),
            lidar_sensor.sensor_center[frame].to(device="cuda", dtype=torch.float32),
        )
        sparse_render = raytracing(
            frame,
            [gaussians],
            sparse_sensor,
            background,
            args,
            depth_only=True,
        )
        sparse_pred_valid = sparse_render["depth"].reshape(-1)

        dense_render = raytracing(
            frame,
            [gaussians],
            lidar_sensor,
            background,
            args,
            depth_only=True,
        )
        dense_pred_map = dense_render["depth"]
        if dense_pred_map.ndim == 3:
            dense_pred_map = dense_pred_map.squeeze(-1)
        dense_pred_map = dense_pred_map.float()

        sparse_pred_map = torch.zeros_like(gt_depth_map)
        sparse_static_mask = torch.zeros_like(gt_mask, dtype=torch.bool)
        sparse_static_mask[gt_mask] = static_valid_mask
        sparse_pred_map[sparse_static_mask] = sparse_pred_valid[static_valid_mask]

        abs_err = torch.abs(
            sparse_pred_map[sparse_static_mask] - gt_depth_map[sparse_static_mask]
        )
        gt_depth_np = gt_depth_map.detach().cpu().numpy().astype(np.float32)
        sparse_pred_np = sparse_pred_map.detach().cpu().numpy().astype(np.float32)
        dense_pred_np = dense_pred_map.detach().cpu().numpy().astype(np.float32)
        static_mask_np = sparse_static_mask.detach().cpu().numpy().astype(bool)
        err_map_np = np.zeros_like(gt_depth_np, dtype=np.float32)
        err_map_np[static_mask_np] = abs_err.detach().cpu().numpy().astype(np.float32)

        gt_vis = depth_to_match_image(
            np.where(static_mask_np, gt_depth_np, 0.0),
            percentile_low=5.0,
            percentile_high=95.0,
            use_inverse=True,
        )
        pred_vis = depth_to_match_image(
            np.where(static_mask_np, dense_pred_np, 0.0),
            percentile_low=5.0,
            percentile_high=95.0,
            use_inverse=True,
        )
        err_vis = _depth_error_to_image(err_map_np)
        panel = np.concatenate(
            [
                _add_label(gt_vis, f"GT LiDAR depth  frame={int(frame)}"),
                _add_label(
                    pred_vis,
                    (
                        "Rendered LiDAR depth  "
                        f"gt_med={float(np.median(gt_depth_np[static_mask_np])):.2f}m  "
                        f"pred_med={float(np.median(sparse_pred_np[static_mask_np])):.2f}m"
                    ),
                ),
                _add_label(
                    err_vis,
                    (
                        "Abs error  "
                        f"med={float(np.median(err_map_np[static_mask_np])):.2f}m  "
                        f"p95={float(np.percentile(err_map_np[static_mask_np], 95.0)):.2f}m"
                    ),
                ),
            ],
            axis=1,
        )
        panels.append(panel)
        cv2.imwrite(os.path.join(viz_dir, f"frame_{int(frame):04d}.png"), panel)
        summary.append(
            {
                "frame": int(frame),
                "valid_pixels": int(static_mask_np.sum()),
                "gt_depth_median": float(np.median(gt_depth_np[static_mask_np])),
                "pred_depth_median": float(np.median(sparse_pred_np[static_mask_np])),
                "dense_pred_depth_median": float(np.median(dense_pred_np[static_mask_np])),
                "abs_err_mean": float(np.mean(err_map_np[static_mask_np])),
                "abs_err_median": float(np.median(err_map_np[static_mask_np])),
                "abs_err_p95": float(np.percentile(err_map_np[static_mask_np], 95.0)),
            }
        )

    if panels:
        overview = np.concatenate(panels, axis=0)
        cv2.imwrite(os.path.join(viz_dir, "overview.png"), overview)
    with open(os.path.join(viz_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return viz_dir


@torch.no_grad()
def _save_gt_camera_depth_visualizations(
    out_dir: str,
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    gt_l2c_R: torch.Tensor,
    gt_l2c_T: torch.Tensor,
    args,
    camera_aux_depth_render_backend: str = "3dgut_rasterization",
    camera_aux_depth_mode: str = "median",
    num_frames: int = 4,
):
    frame_ids = _select_visualization_frames(cam_cameras, scene, num_frames=num_frames)
    if not frame_ids:
        return None

    viz_dir = os.path.join(out_dir, "visualizations_gt_camera_depth")
    os.makedirs(viz_dir, exist_ok=True)
    viz_depth_backend = _resolve_visualization_depth_backend(
        camera_aux_depth_render_backend
    )
    summary = []
    panels = []

    for frame in frame_ids:
        camera = cam_cameras[frame].cuda()
        gt_rgb = cam_images[frame].cuda()
        cam_R, cam_T = _camera_rt_from_lidar_to_camera(
            pose_correction,
            frame,
            gt_l2c_R,
            gt_l2c_T,
            device="cuda",
        )
        depth_render = _render_camera_with_backend(
            camera,
            [gaussians],
            args,
            backend=viz_depth_backend,
            cam_rotation=cam_R.detach(),
            cam_translation=cam_T.detach(),
            require_rgb=False,
        )
        depth_tensor = _resolve_camera_aux_depth(
            depth_render,
            depth_mode=camera_aux_depth_mode,
        )
        depth_np = depth_tensor.detach().squeeze(-1).cpu().numpy().astype(np.float32)
        valid = depth_np > 0.0
        gt_rgb_np = _to_uint8_rgb(gt_rgb)
        depth_vis = depth_to_match_image(
            depth_np,
            percentile_low=5.0,
            percentile_high=95.0,
            use_inverse=True,
        )
        panel = np.concatenate(
            [
                _add_label(gt_rgb_np, f"GT RGB  frame={int(frame)}"),
                _add_label(
                    depth_vis,
                    (
                        "GT camera-view depth  "
                        f"med={float(np.median(depth_np[valid])):.2f}m"
                        if np.any(valid)
                        else "GT camera-view depth  empty"
                    ),
                ),
            ],
            axis=1,
        )
        panels.append(panel)
        cv2.imwrite(os.path.join(viz_dir, f"frame_{int(frame):04d}.png"), panel)
        summary.append(
            {
                "frame": int(frame),
                "valid_pixels": int(np.count_nonzero(valid)),
                "pred_depth_median": (
                    float(np.median(depth_np[valid]))
                    if np.any(valid)
                    else None
                ),
                "pred_depth_p95": (
                    float(np.percentile(depth_np[valid], 95.0))
                    if np.any(valid)
                    else None
                ),
            }
        )

    if panels:
        overview = np.concatenate(panels, axis=0)
        cv2.imwrite(os.path.join(viz_dir, "overview.png"), overview)
    with open(os.path.join(viz_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return viz_dir


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
    freeze_opacity: bool = False,
    translation_start_cycle: int = 0,
    stage2_freeze_colors: bool = True,
    warmup_cycles: int = 0,
    freeze_rotation: bool = False,
    freeze_translation: bool = False,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    lr_min: float = 1e-5,
    pose_lr_drop_cycle: int = 0,
    pose_lr_drop_factor: float = 0.1,
    translation_lr_drop_cycle: int = 0,
    translation_lr_drop_factor: float = 0.1,
    lambda_rgb: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_normal: float = 0.0,
    lambda_dssim: float = 0.2,
    lambda_reg: float = 0.0,
    geometry_conflict_policy: str = "none",
    geometry_conflict_camera_weight: float = 1.0,
    warmup_camera_depth_normal_weight: float = 0.0,
    warmup_disable_other_regularizers: bool = False,
    initial_gaussian_state: dict = None,
    tb_writer=None,
    cycle_ckpt_dir: str = None,
    save_cycle_every: int = 5,
    resume_cycle_ckpt: str = None,
    resume_cycle_ckpt_scene_only: bool = False,
    reset_gaussians_every: int = 0,
    disable_depth_after_cycle: int = 0,
    rgb_only_updates_color: bool = False,
    matcher_pose_update: bool = False,
    matcher_update_interval: int = 1,
    matcher_update_blend: float = 1.0,
    matcher_name: str = "matchanything-roma",
    matcher_minima_root: str | None = None,
    matcher_minima_ckpt: str | None = None,
    matcher_resize: int = 832,
    matcher_match_threshold: float = 0.2,
    matcher_render_intrinsics_fx_scale: float = 1.0,
    matcher_render_intrinsics_fy_scale: float = 1.0,
    matcher_render_intrinsics_cx_offset: float = 0.0,
    matcher_render_intrinsics_cy_offset: float = 0.0,
    matcher_lidar_nn_max_distance: float = 0.0,
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
    matcher_color_supervision: bool = False,
    matcher_color_weight: float = 1.0,
    camera_rgb_pose_only: bool = False,
    color_warmup_cycles: int = 0,
    initialize_pose_from_matcher: bool = False,
    post_init_supervision_blur_kernel: int = 0,
    post_init_supervision_blur_sigma: float = 0.0,
    post_init_supervision_blur_warmup_only: bool = False,
    freeze_gaussians_after_color_warmup: bool = False,
    freeze_gaussians_after_cycle: int = 0,
    pose_step_at_cycle_end: bool = False,
    pose_step_interval_iters: int = 0,
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
    matcher_dense_color_cert_threshold: float | None = None,
    match_once_per_cycle: bool = True,
    cross_frame_weight: float = 0.0,
    flow_proj_weight: float = 0.0,
    flow_proj_max_offset: int = 1,
    camera_aux_depth_render_backend: str = "3dgut_rasterization",
    camera_aux_depth_mode: str = "median",
    pure_pnp_iters: int = 0,
    pure_pnp_drop_first_n_frames: int = 0,
    pure_pnp_drop_last_n_frames: int = 0,
    pure_pnp_photo_weight: float = 0.0,
    pure_pnp_photo_match_radius_px: float = 4.0,
    pure_pnp_photo_min_matches: int = 8,
    pure_pnp_photo_max_offset: int = 1,
    pure_pnp_flow_weight: float = 0.0,
    pure_pnp_flow_match_radius_px: float = 4.0,
    pure_pnp_flow_min_matches: int = 8,
    pure_pnp_depth_weight: float = 0.0,
    pure_pnp_depth_match_radius_px: float = 4.0,
    pure_pnp_depth_min_matches: int = 8,
    pure_pnp_depth_render_backend: str = "3dgut_rasterization",
    pure_pnp_depth_mode: str = "median",
    pure_pnp_temporal_semidense_stride: int = 0,
    pure_pnp_temporal_semidense_max_points: int = 0,
    pure_pnp_temporal_gradient_scale: float = 0.0,
    pure_pnp_init_depth_strat_bins: int = 0,
    pure_pnp_init_depth_strat_max_points_per_bin: int = 0,
    pure_pnp_far_depth_boost: float = 0.0,
    pure_pnp_far_depth_start_percentile: float = 60.0,
    pure_pnp_post_init_rgb_blur_kernel: int = 0,
    pure_pnp_post_init_rgb_blur_sigma: float = 0.0,
    periodic_pure_pnp_interval_cycles: int = 0,
    disable_pose_grad_updates: bool = False,
    pure_pnp_history_path: str | None = None,
    pure_pnp_match_diagnostics_dir: str | None = None,
    pure_pnp_residual_filter_mad_scale: float = 0.0,
    pure_pnp_residual_filter_min_keep_ratio: float = 0.5,
    pure_pnp_residual_filter_min_keep_per_frame: int = 24,
    pure_pnp_residual_filter_max_reproj_error: float = 0.0,
    pure_pnp_residual_filter_start_step: int = 1,
    pure_pnp_single_frame_prefilter_min_inliers: int = 0,
    pure_pnp_single_frame_prefilter_min_inlier_ratio: float = 0.0,
    pure_pnp_frame_disagreement_mad_scale: float = 0.0,
    pure_pnp_frame_disagreement_min_keep_ratio: float = 0.7,
    pure_pnp_frame_disagreement_min_keep_frames: int = 12,
    pure_pnp_frame_disagreement_apply_max_dropped_frames: int = 0,
    pure_pnp_filter_shared_ransac_outliers: bool = False,
    pure_pnp_gt_reproj_filter_quantile: float = 0.0,
    pure_pnp_gt_reproj_filter_min_keep_per_frame: int = 24,
    pure_pnp_gt_soft_weight_mode: str = "none",
    pure_pnp_gt_soft_weight_translation_alpha: float = 0.5,
    pure_pnp_gt_pose_residual_weight: float = 0.0,
    pure_pnp_optimize_rotation: bool = True,
    pure_pnp_optimize_translation: bool = True,
    pure_pnp_solver_backend: str = "auto",
    pure_pnp_gt_weight_analysis_dir: str | None = None,
):
    """Continuous calibration training loop.

    When ``pose_correction.use_gt_translation`` is False, ``delta_translations``
    is added to the optimizer so both rotation and translation are calibrated.

    When ``freeze_xyz`` is True, Gaussian mean positions are frozen (requires_grad
    disabled, LR zeroed) so they cannot absorb translation errors.

    ``translation_start_cycle`` implements a two-stage strategy:
      Stage 1 (cycles 1..translation_start_cycle): rotation-only optimisation.
      Stage 2 (cycles translation_start_cycle+1..total): translation optimisation enabled,
        xyz frozen and colors optionally frozen.
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
    # Only xyz is frozen from match/RGB supervision; covariance and opacity are allowed to update
    _MATCH_FROZEN_ATTRS = ["_xyz"]
    matcher = None
    temporal_support_cache: dict[int, np.ndarray] = {}
    temporal_pair_cache: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    cycle_match_cache: dict[int, dict] = {}   # precomputed cross-modal targets, refreshed each cycle
    cycle_coverage_rate: float = 0.0          # avg match coverage fraction from last precompute
    adj_gt_cache: dict[int, list] = {}         # precomputed adjacent GT colors, built once
    flow_proj_cache: dict[int, list] = {}      # precomputed RGB-RGB match pairs for flow proj, built once
    pure_pnp_history: list[dict] = []

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
        _COV_ATTRS = ["_scaling", "_rotation"]
        if freeze_covariance and not two_stage:
            for attr in _COV_ATTRS:
                param = getattr(gaussians, attr, None)
                if param is not None:
                    param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                for attr in _COV_ATTRS:
                    cp = getattr(gaussians, attr, None)
                    if cp is not None and any(p is cp for p in pg["params"]):
                        pg["lr"] = 0.0
            print(blue("[NoiseInject] Gaussian covariance FROZEN — scaling/rotation cannot absorb pose error"))
        if freeze_opacity and not two_stage:
            opacity_param = getattr(gaussians, "_opacity", None)
            if opacity_param is not None:
                opacity_param.requires_grad_(False)
            for pg in gaussians.optimizer.param_groups:
                if opacity_param is not None and any(p is opacity_param for p in pg["params"]):
                    pg["lr"] = 0.0
            print(blue("[NoiseInject] Gaussian opacity FROZEN — opacity cannot absorb pose error"))
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

    def _blur_cam_images_dict(images: dict[int, torch.Tensor], kernel: int, sigma: float) -> dict[int, torch.Tensor]:
        kernel = max(int(kernel), 0)
        if kernel <= 0:
            return images
        if kernel % 2 == 0:
            kernel += 1
        sigma = float(max(sigma, 0.0))
        blurred: dict[int, torch.Tensor] = {}
        for frame_id, image in images.items():
            image_np = image.detach().cpu().numpy().astype(np.float32, copy=False)
            blurred_np = cv2.GaussianBlur(
                image_np,
                (kernel, kernel),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
            blurred[frame_id] = torch.from_numpy(np.ascontiguousarray(blurred_np)).float()
        return blurred

    def _flush_pure_pnp_history() -> None:
        if pure_pnp_history_path is None:
            return
        out_dir = os.path.dirname(pure_pnp_history_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(pure_pnp_history_path, "w", encoding="utf-8") as f:
            for row in pure_pnp_history:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _record_pure_pnp_step(
        phase: str,
        cycle_value: int,
        step_idx: int,
        total_steps: int,
        summary: dict,
    ) -> dict:
        eff_R = _effective_R(pose_correction)
        rot_err = _rotation_error_deg(eff_R, gt_l2c_R)
        trans_err = _translation_error_m(pose_correction, gt_l2c_T)
        record = {
            "phase": str(phase),
            "cycle": int(cycle_value),
            "step": int(step_idx),
            "total_steps": int(total_steps),
            "status": str(summary.get("status", "unknown")),
            "reason": str(summary.get("reason", "")),
            "rot_err_deg": float(rot_err),
            "trans_err_m": float(trans_err),
            "frames_used": int(summary.get("frames_used", 0)),
            "matches_used": int(summary.get("matches_used", 0)),
            "rotation_delta_deg": float(summary.get("rotation_delta_deg", 0.0)),
            "translation_delta_m": float(summary.get("translation_delta_m", 0.0)),
            "mean_reproj_px": float(summary.get("mean_reproj_px", 0.0)),
            "flow_blocks": int(summary.get("temporal_blocks", 0)),
            "depth_blocks": int(summary.get("depth_blocks", 0)),
            "photo_blocks": int(summary.get("photometric_blocks", 0)),
            "raw_matches_total": int(summary.get("raw_matches_total", summary.get("matches_used", 0))),
            "filtered_matches_total": int(summary.get("filtered_matches_total", summary.get("matches_used", 0))),
            "filter_dropped_matches": int(summary.get("filter_dropped_matches", 0)),
            "filter_kept_ratio": float(summary.get("filter_kept_ratio", 1.0)),
            "filter_gt_kept_median_px": summary.get("filter_gt_kept_median_px"),
            "filter_gt_dropped_median_px": summary.get("filter_gt_dropped_median_px"),
            "frame_filter_dropped_frames": int(summary.get("frame_filter_dropped_frames", 0)),
        }
        pure_pnp_history.append(record)
        _flush_pure_pnp_history()
        return record

    def _run_pure_pnp_phase(
        *,
        phase: str,
        cycle_value: int,
        num_steps: int,
    ) -> list[dict]:
        if num_steps <= 0:
            return []
        print(
            blue(
                f"[PurePnP] Phase={phase} cycle={cycle_value} "
                f"running {num_steps} iterative PnP steps..."
            )
        )
        pure_pnp_frame_ids = sorted(cam_cameras.keys())
        if pure_pnp_drop_first_n_frames > 0:
            pure_pnp_frame_ids = pure_pnp_frame_ids[int(pure_pnp_drop_first_n_frames):]
        if pure_pnp_drop_last_n_frames > 0:
            keep_count = max(0, len(pure_pnp_frame_ids) - int(pure_pnp_drop_last_n_frames))
            pure_pnp_frame_ids = pure_pnp_frame_ids[:keep_count]
        if not pure_pnp_frame_ids:
            raise ValueError(
                "pure_pnp frame trimming removed all frames; "
                "reduce --pure_pnp_drop_first_n_frames / --pure_pnp_drop_last_n_frames."
            )
        if pure_pnp_drop_first_n_frames > 0 or pure_pnp_drop_last_n_frames > 0:
            print(
                blue(
                    "[PurePnP] Frame trimming enabled — "
                    f"drop_first={pure_pnp_drop_first_n_frames}, "
                    f"drop_last={pure_pnp_drop_last_n_frames}, "
                    f"kept={len(pure_pnp_frame_ids)} frames "
                    f"(range {pure_pnp_frame_ids[0]}..{pure_pnp_frame_ids[-1]})"
                )
            )
        common = dict(
            matcher=matcher,
            gaussians=gaussians,
            pose_correction=pose_correction,
            cam_cameras=cam_cameras,
            cam_images=cam_images,
            camera_masks=camera_supervision_masks,
            args=args,
            frame_ids=pure_pnp_frame_ids,
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
            render_intrinsics_fx_scale=matcher_render_intrinsics_fx_scale,
            render_intrinsics_fy_scale=matcher_render_intrinsics_fy_scale,
            render_intrinsics_cx_offset=matcher_render_intrinsics_cx_offset,
            render_intrinsics_cy_offset=matcher_render_intrinsics_cy_offset,
            matcher_lidar_nn_max_distance=matcher_lidar_nn_max_distance,
            depth_render_backend=pure_pnp_depth_render_backend,
            depth_render_mode=pure_pnp_depth_mode,
            matcher_update_blend=1.0,
            temporal_pair_cache=temporal_pair_cache,
            temporal_photometric_weight=pure_pnp_photo_weight,
            temporal_photometric_match_radius_px=pure_pnp_photo_match_radius_px,
            temporal_photometric_min_matches=pure_pnp_photo_min_matches,
            temporal_flow_weight=pure_pnp_flow_weight,
            temporal_flow_match_radius_px=pure_pnp_flow_match_radius_px,
            temporal_flow_min_matches=pure_pnp_flow_min_matches,
            temporal_depth_weight=pure_pnp_depth_weight,
            temporal_depth_match_radius_px=pure_pnp_depth_match_radius_px,
            temporal_depth_min_matches=pure_pnp_depth_min_matches,
            temporal_semidense_stride=pure_pnp_temporal_semidense_stride,
            temporal_semidense_max_points=pure_pnp_temporal_semidense_max_points,
            temporal_gradient_scale=pure_pnp_temporal_gradient_scale,
            init_depth_strat_bins=pure_pnp_init_depth_strat_bins,
            init_depth_strat_max_points_per_bin=pure_pnp_init_depth_strat_max_points_per_bin,
            far_depth_boost=pure_pnp_far_depth_boost,
            far_depth_start_percentile=pure_pnp_far_depth_start_percentile,
            rgb_blur_kernel=pure_pnp_post_init_rgb_blur_kernel,
            rgb_blur_sigma=pure_pnp_post_init_rgb_blur_sigma,
            temporal_support_getter=get_temporal_support if matcher_adjacent_support else None,
            temporal_support_radius_px=matcher_support_radius_px,
            pure_pnp_residual_filter_mad_scale=pure_pnp_residual_filter_mad_scale,
            pure_pnp_residual_filter_min_keep_ratio=pure_pnp_residual_filter_min_keep_ratio,
            pure_pnp_residual_filter_min_keep_per_frame=pure_pnp_residual_filter_min_keep_per_frame,
            pure_pnp_residual_filter_max_reproj_error=pure_pnp_residual_filter_max_reproj_error,
            pure_pnp_single_frame_prefilter_min_inliers=pure_pnp_single_frame_prefilter_min_inliers,
            pure_pnp_single_frame_prefilter_min_inlier_ratio=pure_pnp_single_frame_prefilter_min_inlier_ratio,
            pure_pnp_frame_disagreement_mad_scale=pure_pnp_frame_disagreement_mad_scale,
            pure_pnp_frame_disagreement_min_keep_ratio=pure_pnp_frame_disagreement_min_keep_ratio,
            pure_pnp_frame_disagreement_min_keep_frames=pure_pnp_frame_disagreement_min_keep_frames,
            pure_pnp_frame_disagreement_apply_max_dropped_frames=pure_pnp_frame_disagreement_apply_max_dropped_frames,
            pure_pnp_filter_shared_ransac_outliers=pure_pnp_filter_shared_ransac_outliers,
            pure_pnp_gt_reproj_filter_quantile=pure_pnp_gt_reproj_filter_quantile,
            pure_pnp_gt_reproj_filter_min_keep_per_frame=pure_pnp_gt_reproj_filter_min_keep_per_frame,
            pure_pnp_gt_soft_weight_mode=pure_pnp_gt_soft_weight_mode,
            pure_pnp_gt_soft_weight_translation_alpha=pure_pnp_gt_soft_weight_translation_alpha,
            pure_pnp_gt_pose_residual_weight=pure_pnp_gt_pose_residual_weight,
            pure_pnp_optimize_rotation=pure_pnp_optimize_rotation,
            pure_pnp_optimize_translation=pure_pnp_optimize_translation,
            pure_pnp_solver_backend=pure_pnp_solver_backend,
            gt_rotation_matrix=gt_l2c_R.detach().cpu().numpy(),
            gt_translation=gt_l2c_T.detach().cpu().numpy(),
        )
        records: list[dict] = []
        for pnp_i in range(1, num_steps + 1):
            common["pure_pnp_residual_filter_mad_scale"] = (
                pure_pnp_residual_filter_mad_scale
                if int(pnp_i) >= int(max(pure_pnp_residual_filter_start_step, 1))
                else 0.0
            )
            if pure_pnp_match_diagnostics_dir is not None:
                common["match_diagnostics_path"] = os.path.join(
                    pure_pnp_match_diagnostics_dir,
                    f"{phase}_cycle{int(cycle_value):04d}_step{int(pnp_i):03d}.json",
                )
            if pure_pnp_gt_weight_analysis_dir is not None:
                common["gt_weight_analysis_json_path"] = os.path.join(
                    pure_pnp_gt_weight_analysis_dir,
                    f"{phase}_cycle{int(cycle_value):04d}_step{int(pnp_i):03d}.json",
                )
                common["gt_weight_analysis_npz_path"] = os.path.join(
                    pure_pnp_gt_weight_analysis_dir,
                    f"{phase}_cycle{int(cycle_value):04d}_step{int(pnp_i):03d}.npz",
                )
            summary = _run_matcher_pose_update(**common)
            record = _record_pure_pnp_step(
                phase=phase,
                cycle_value=cycle_value,
                step_idx=pnp_i,
                total_steps=num_steps,
                summary=summary,
            )
            records.append(record)
            if record["status"] == "applied":
                print(
                    blue(
                        f"[PurePnP] {phase} step {pnp_i}/{num_steps} "
                        f"rot_err={record['rot_err_deg']:.4f}°  "
                        f"T_err={record['trans_err_m']:.4f}m  "
                        f"frames={record['frames_used']} matches={record['matches_used']}  "
                        f"dR={record['rotation_delta_deg']:.4f}° dT={record['translation_delta_m']:.4f}m"
                    )
                )
            else:
                print(
                    blue(
                        f"[PurePnP] {phase} step {pnp_i}/{num_steps} skipped: "
                        f"{record['reason']}"
                    )
                )
        return records

    camera_supervision_masks = {
        int(frame): _get_camera_supervision_mask(cam_cameras, frame)
        for frame in cam_cameras.keys()
    }
    current_cam_images = cam_images
    blurred_cam_images: dict[int, torch.Tensor] | None = None

    _apply_gaussian_freezes()
    _apply_matcher_color_lr_scale()

    resume_ckpt = None
    if resume_cycle_ckpt is not None:
        print(blue(f"[NoiseInject] Preloading cycle checkpoint state: {resume_cycle_ckpt}"))
        resume_ckpt = torch.load(resume_cycle_ckpt, weights_only=False, map_location="cuda")
        restore_gaussian_state(gaussians, resume_ckpt["gaussian_state"], args)
        if resume_cycle_ckpt_scene_only:
            print(blue("[NoiseInject] Cycle checkpoint loaded in scene-only mode — pose/init state kept from current run"))
        else:
            if "pose_correction_state" in resume_ckpt:
                pose_correction.load_state_dict(
                    {k: v.to("cuda") for k, v in resume_ckpt["pose_correction_state"].items()}
                )
            else:
                pose_correction.delta_rotations_quat.data.copy_(
                    resume_ckpt["delta_rotations_quat"].to("cuda")
                )
                pose_correction.delta_translations.data.copy_(
                    resume_ckpt["delta_translations"].to("cuda")
                )
            pose_correction.update_extrinsics()
        _apply_gaussian_freezes()
        _apply_matcher_color_lr_scale()

    if matcher_pose_update or matcher_color_supervision or matcher_adjacent_color_supervision or flow_proj_weight > 0 or pure_pnp_iters > 0 or initialize_pose_from_matcher:
        matcher = build_matcher(
            matcher_name=matcher_name,
            device="cuda",
            max_num_keypoints=matcher_max_num_keypoints,
            ransac_reproj_thresh=matcher_ransac_reproj_thresh,
            img_resize=matcher_resize,
            match_threshold=matcher_match_threshold,
            minima_root=matcher_minima_root,
            minima_ckpt=matcher_minima_ckpt,
        )
        # Warm up the camera-side auxiliary depth renderer BEFORE matcher use.
        # This avoids backend-specific first-use stalls from blocking frame 0.
        print(
            f"[Renderer] Warming up camera auxiliary depth renderer "
            f"({camera_aux_depth_render_backend})..."
        )
        _warmup_frame = next(
            (f for f in frame_ids_train if f in cam_cameras and f in cam_images),
            None,
        )
        if _warmup_frame is not None:
            try:
                _wu_cam_R, _wu_cam_T = pose_correction.corrected_rt(_warmup_frame, device="cuda")
                _wu_camera = cam_cameras[_warmup_frame].cuda()
                with torch.no_grad():
                    _render_camera_with_backend(
                        _wu_camera,
                        [gaussians],
                        args,
                        backend=camera_aux_depth_render_backend,
                        cam_rotation=_wu_cam_R.detach(),
                        cam_translation=_wu_cam_T.detach(),
                    )
                torch.cuda.synchronize()
            except Exception as _e:
                print(f"[Renderer] Depth warmup exception (non-fatal): {_e}")
        print("[Renderer] Depth renderer warm-up complete.")

        # Warm up the dense code path to trigger all CUDA kernel JIT compilations
        # before the training loop. Without this, the first cycle precompute is
        # blocked for 20+ minutes on Blackwell (cc12.0 PTX JIT for new code paths).
        # IMPORTANT: Use the ACTUAL image size — different input shapes trigger
        # different CUDA kernel configs; a tiny 64×64 dummy won't cover real images.
        if matcher_dense_mode:
            print("[Matcher] Warming up dense inference path (one-time CUDA kernel compilation)...")
            _sample_cam = next(iter(cam_cameras.values()))
            _wu_h = int(_sample_cam.image_height)
            _wu_w = int(_sample_cam.image_width)
            _dummy = np.zeros((_wu_h, _wu_w, 3), dtype=np.uint8)
            try:
                from lib.utils.rgbd_calibration import match_cross_modal_dense as _mcd
                # Use a large stride and high cert to make warmup fast, but with correct image shape
                _mcd(matcher, _dummy, _dummy, query_stride=max(8, matcher_dense_stride), cert_threshold=0.99)
            except Exception as _e:
                print(f"[Matcher] Dense warm-up exception (non-fatal): {_e}")
            print(f"[Matcher] Dense warm-up complete ({_wu_w}×{_wu_h}).")
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
                f" [DENSE stride={matcher_dense_stride} cert≥{matcher_dense_cert_threshold}"
                + (f" | color cert≥{matcher_dense_color_cert_threshold} (weighted)" if matcher_dense_color_cert_threshold is not None else "")
                + "]"
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
        if pure_pnp_photo_weight > 0.0:
            print(
                blue(
                    "[PurePnP] Sparse temporal RGB residual enabled — "
                    f"weight={pure_pnp_photo_weight:.3f}, offset=±{pure_pnp_photo_max_offset}"
                )
            )
        if pure_pnp_flow_weight > 0.0:
            print(
                blue(
                    "[PurePnP] Temporal flow reprojection residual enabled — "
                    f"weight={pure_pnp_flow_weight:.3f}, offset=±{pure_pnp_photo_max_offset}"
                )
            )
        if pure_pnp_depth_weight > 0.0:
            print(
                blue(
                    "[PurePnP] Temporal depth consistency residual enabled — "
                    f"weight={pure_pnp_depth_weight:.3f}, offset=±{pure_pnp_photo_max_offset}, backend={pure_pnp_depth_render_backend}"
                )
            )
        if pure_pnp_temporal_semidense_stride > 0 and pure_pnp_temporal_semidense_max_points > 0:
            print(
                blue(
                    "[PurePnP] Semi-dense temporal projection enabled — "
                    f"stride={pure_pnp_temporal_semidense_stride}, max_points={pure_pnp_temporal_semidense_max_points}"
                )
            )
        if pure_pnp_temporal_gradient_scale > 0.0:
            print(
                blue(
                    "[PurePnP] Temporal reliability weighting enabled — "
                    f"gradient_scale={pure_pnp_temporal_gradient_scale:.2f}"
                )
            )
        if (
            pure_pnp_single_frame_prefilter_min_inliers > 0
            or pure_pnp_single_frame_prefilter_min_inlier_ratio > 0.0
        ):
            print(
                blue(
                    "[PurePnP] Single-frame PnP stability prefilter enabled — "
                    f"min_inliers={pure_pnp_single_frame_prefilter_min_inliers}, "
                    f"min_inlier_ratio={pure_pnp_single_frame_prefilter_min_inlier_ratio:.2f}"
                )
            )
        if pure_pnp_init_depth_strat_bins > 1 and pure_pnp_init_depth_strat_max_points_per_bin > 0:
            print(
                blue(
                    "[PurePnP] Depth-stratified init enabled — "
                    f"bins={pure_pnp_init_depth_strat_bins}, max_per_bin={pure_pnp_init_depth_strat_max_points_per_bin}"
                )
            )
        if pure_pnp_far_depth_boost > 0.0:
            print(
                blue(
                    "[PurePnP] Far-depth correspondence boost enabled — "
                    f"boost={pure_pnp_far_depth_boost:.2f}, start_pct={pure_pnp_far_depth_start_percentile:.1f}"
                )
            )
        if pure_pnp_post_init_rgb_blur_kernel > 0:
            print(
                blue(
                    "[PurePnP] Post-init RGB blur enabled — "
                    f"kernel={pure_pnp_post_init_rgb_blur_kernel}, sigma={pure_pnp_post_init_rgb_blur_sigma:.2f}"
                )
            )
        if pure_pnp_residual_filter_mad_scale > 0.0:
            print(
                blue(
                    "[PurePnP] Consensus residual trimming enabled — "
                    f"mad_scale={pure_pnp_residual_filter_mad_scale:.2f}, "
                    f"min_keep_ratio={pure_pnp_residual_filter_min_keep_ratio:.2f}, "
                    f"min_keep_per_frame={pure_pnp_residual_filter_min_keep_per_frame}, "
                    f"start_step={pure_pnp_residual_filter_start_step}"
                )
            )
        if pure_pnp_frame_disagreement_mad_scale > 0.0:
            print(
                blue(
                    "[PurePnP] Frame-disagreement filtering enabled — "
                    f"mad_scale={pure_pnp_frame_disagreement_mad_scale:.2f}, "
                    f"min_keep_ratio={pure_pnp_frame_disagreement_min_keep_ratio:.2f}, "
                    f"min_keep_frames={pure_pnp_frame_disagreement_min_keep_frames}, "
                    f"apply_max_dropped_frames={pure_pnp_frame_disagreement_apply_max_dropped_frames}"
                )
            )
        if flow_proj_weight > 0.0 or pure_pnp_photo_weight > 0.0 or pure_pnp_flow_weight > 0.0 or pure_pnp_depth_weight > 0.0:
            if flow_proj_weight > 0.0:
                print(blue(f"[FlowProj] Flow projection loss enabled — weight={flow_proj_weight:.3f}, offset=±{flow_proj_max_offset}"))
            else:
                print(blue(f"[FlowProj] Precomputing adjacent RGB-RGB pairs for pure PnP temporal residuals (offset=±{max(int(pure_pnp_photo_max_offset), 1)})"))
            flow_proj_cache = _precompute_flow_proj_cache(
                matcher=matcher,
                cam_images=cam_images,
                camera_masks=camera_supervision_masks,
                max_offset=max(int(flow_proj_max_offset), int(pure_pnp_photo_max_offset)),
            )
            # Reuse these RGB-RGB pairs as temporal_pair_cache so adj_gt_cache
            # and adjacent color supervision don't re-run the matcher.
            if matcher_adjacent_color_supervision or matcher_adjacent_support:
                temporal_pair_cache = flow_proj_cache
            elif (
                pure_pnp_photo_weight > 0.0
                or pure_pnp_flow_weight > 0.0
                or pure_pnp_depth_weight > 0.0
            ):
                temporal_pair_cache = flow_proj_cache
    if camera_rgb_pose_only:
        print(blue("[NoiseInject] Full-image camera RGB loss updates POSE only"))
    if freeze_translation:
        print(blue("[NoiseInject] Translation FROZEN — keep initialized translation fixed during training"))
    if warmup_cycles > 0:
        print(blue(f"[NoiseInject] LiDAR warmup enabled for first {warmup_cycles} cycles (camera-side supervision disabled, pose frozen)"))
    if color_warmup_cycles > 0:
        print(blue(f"[NoiseInject] Color warmup enabled for first {color_warmup_cycles} cycles (RGB losses update colors only)"))
    if disable_depth_during_color_warmup:
        print(blue("[NoiseInject] LiDAR depth supervision disabled during color warmup"))
    if geometry_conflict_policy != "none":
        print(
            blue(
                "[NoiseInject] Geometry conflict handling enabled — "
                f"policy={geometry_conflict_policy}, camera_weight={geometry_conflict_camera_weight:.2f}"
            )
        )
    if (
        matcher_pose_update
        or matcher_color_supervision
        or matcher_adjacent_color_supervision
        or flow_proj_weight > 0.0
        or pure_pnp_iters > 0
        or initialize_pose_from_matcher
    ):
        print(
            blue(
                "[Renderer] Camera auxiliary depth backend: "
                f"{camera_aux_depth_render_backend} | mode: {camera_aux_depth_mode}"
            )
        )
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
        if not matcher_adjacent_support or matcher is None or frame not in current_cam_images:
            return None
        return _get_temporal_support_points(
            matcher=matcher,
            frame=int(frame),
            cam_images=current_cam_images,
            camera_masks=camera_supervision_masks,
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
            camera_masks=camera_supervision_masks,
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
            render_intrinsics_fx_scale=matcher_render_intrinsics_fx_scale,
            render_intrinsics_fy_scale=matcher_render_intrinsics_fy_scale,
            render_intrinsics_cx_offset=matcher_render_intrinsics_cx_offset,
            render_intrinsics_cy_offset=matcher_render_intrinsics_cy_offset,
            matcher_lidar_nn_max_distance=matcher_lidar_nn_max_distance,
            matcher_update_blend=1.0,
            depth_render_backend=camera_aux_depth_render_backend,
            depth_render_mode=camera_aux_depth_mode,
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

    if post_init_supervision_blur_kernel > 0:
        blurred_cam_images = _blur_cam_images_dict(
            cam_images,
            kernel=int(post_init_supervision_blur_kernel),
            sigma=float(post_init_supervision_blur_sigma),
        )
        current_cam_images = blurred_cam_images
        print(
            blue(
                "[NoiseInject] Post-init supervision blur enabled — "
                f"kernel={post_init_supervision_blur_kernel}, sigma={post_init_supervision_blur_sigma:.2f}"
            )
        )
        if post_init_supervision_blur_warmup_only:
            print(
                blue(
                    "[NoiseInject] Post-init blur applies during color warmup only — "
                    "supervision switches back to original RGB after warmup"
                )
            )

    if periodic_pure_pnp_interval_cycles > 0 and pure_pnp_iters <= 0:
        raise ValueError(
            "periodic_pure_pnp_interval_cycles requires pure_pnp_iters > 0 "
            "so an initial pure PnP phase can be run before training."
        )
    if periodic_pure_pnp_interval_cycles > 0 and matcher_pose_update:
        raise ValueError(
            "periodic_pure_pnp_interval_cycles cannot be combined with matcher_pose_update; "
            "choose one pose-update schedule."
        )
    if periodic_pure_pnp_interval_cycles > 0:
        print(
            blue(
                "[PurePnP] Periodic schedule enabled — "
                f"initial_steps={pure_pnp_iters}, interval={periodic_pure_pnp_interval_cycles} cycles, "
                f"disable_pose_grad_updates={disable_pose_grad_updates}"
            )
        )
    if pure_pnp_iters > 0:
        print(blue(f"[PurePnP] Shared extrinsic solver backend: {pure_pnp_solver_backend}"))
    if pure_pnp_iters > 0 and (not pure_pnp_optimize_rotation or not pure_pnp_optimize_translation):
        mode_bits = []
        if pure_pnp_optimize_rotation:
            mode_bits.append("rotation")
        if pure_pnp_optimize_translation:
            mode_bits.append("translation")
        print(blue(f"[PurePnP] Shared extrinsic optimization mode: {'+'.join(mode_bits)}"))

    run_initial_pure_pnp_after_warmup = bool(pure_pnp_iters > 0)


    optimizer_param_groups = []
    if not disable_pose_grad_updates:
        if not freeze_rotation:
            optimizer_param_groups.append(
                {"params": [pose_correction.delta_rotations_quat], "lr": rotation_lr, "name": "rotation"}
            )
        if not pose_correction.use_gt_translation and not two_stage and not freeze_translation:
            optimizer_param_groups.append(
                {"params": [pose_correction.delta_translations], "lr": translation_lr, "name": "translation"}
            )
        if optimizer_param_groups:
            pose_optimizer = torch.optim.Adam(optimizer_param_groups)
            pose_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                pose_optimizer,
                mode="min",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=lr_min,
                threshold=1e-3,
            ) if lr_patience > 0 else None
        else:
            pose_optimizer = None
            pose_lr_scheduler = None
            print(blue("[NoiseInject] Pose optimizer disabled — both rotation and translation updates are frozen"))
    else:
        pose_optimizer = None
        pose_lr_scheduler = None
        print(blue("[NoiseInject] Pose gradient updates DISABLED — pose will only change via pure PnP"))

    init_R   = _effective_R(pose_correction)
    init_err = _rotation_error_deg(init_R, gt_l2c_R)
    init_T_err = _translation_error_m(pose_correction, gt_l2c_T)
    total_iters = total_cycles * iters_per_cycle
    print(blue(f"[NoiseInject] Init rotation error vs GT: {init_err:.4f}°"))
    print(blue(f"[NoiseInject] Init translation error vs GT: {init_T_err:.4f} m"))
    print(blue(f"[NoiseInject] total_iters={total_iters} ({total_cycles}×{iters_per_cycle}), "
               f"rotation_lr={rotation_lr}, translation_lr={translation_lr}, "
               f"use_gt_translation={pose_correction.use_gt_translation}"))
    if pose_step_at_cycle_end:
        print(blue("[NoiseInject] Pose optimizer accumulates gradients within each cycle and steps once at cycle end"))
    elif pose_step_interval_iters > 0:
        print(blue(f"[NoiseInject] Pose optimizer steps and folds extrinsics every {pose_step_interval_iters} iterations"))
    if freeze_gaussians_after_color_warmup:
        print(blue("[NoiseInject] Gaussian scene will be frozen after color warmup; later cycles optimize pose only"))
    if freeze_gaussians_after_cycle > 0:
        print(blue(f"[NoiseInject] Gaussian scene will be frozen after cycle {freeze_gaussians_after_cycle}; later cycles optimize pose only"))
    if pose_lr_drop_cycle > 0:
        print(blue(f"[NoiseInject] Late-stage pose LR drop enabled — cycle {pose_lr_drop_cycle} × {pose_lr_drop_factor:.3f}"))
    if translation_lr_drop_cycle > 0:
        print(blue(f"[NoiseInject] Late-stage translation LR drop enabled — cycle {translation_lr_drop_cycle} × {translation_lr_drop_factor:.3f}"))
    if disable_depth_after_cycle > 0:
        print(blue(f"[NoiseInject] Depth supervision disabled after cycle {disable_depth_after_cycle}"))

    if two_stage:
        stage2_scene_mode = "freeze xyz+colors" if stage2_freeze_colors else "freeze xyz only"
        print(blue(f"[NoiseInject] TWO-STAGE mode: rotation-only until cycle {translation_start_cycle}, "
                   f"then {stage2_scene_mode} + optimise translation"))

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
    loss_cross_frame_accum = 0.0
    loss_flow_proj_accum = 0.0
    loss_reg_accum = 0.0
    # Best-T tracking: save the delta_T that achieved the lowest T_err
    best_T_err     = float("inf")
    best_delta_T   = pose_correction.delta_translations.data.clone()
    start_cycle    = 1
    pose_lr_drop_applied = False
    translation_lr_drop_applied = False

    # ── Resume from cycle checkpoint ──────────────────────────
    if resume_ckpt is not None and not resume_cycle_ckpt_scene_only:
        global_iter   = resume_ckpt["global_iter"]
        gaussian_iter = resume_ckpt["global_iter"]
        best_T_err    = resume_ckpt["best_T_err"]
        best_delta_T  = resume_ckpt["best_delta_T"].to("cuda")
        stage2_active = resume_ckpt["stage2_active"]
        start_cycle   = resume_ckpt["cycle"] + 1
        print(blue(f"[NoiseInject] Resumed: start_cycle={start_cycle}, "
                   f"stage2_active={stage2_active}, best_T_err={best_T_err:.4f}m"))

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
            stage2_desc = "xyz FROZEN"
            if stage2_freeze_colors:
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
                stage2_desc = "xyz+colors FROZEN"
            if pose_optimizer is not None:
                print(blue(f"[NoiseInject] Stage 2 activated at cycle {cycle}: {stage2_desc}, translation optimizer added"))
                pose_optimizer.add_param_group(
                    {"params": [pose_correction.delta_translations], "lr": translation_lr, "name": "translation"}
                )
                if pose_lr_scheduler is not None:
                    pose_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        pose_optimizer, mode="min", factor=lr_factor,
                        patience=lr_patience, min_lr=lr_min, threshold=1e-3,
                    )
            else:
                print(blue(f"[NoiseInject] Stage 2 activated at cycle {cycle}: {stage2_desc}, pose gradients remain disabled"))

        if (
            freeze_gaussians_after_color_warmup
            and not freeze_gaussians
            and color_warmup_cycles > 0
            and cycle > color_warmup_cycles
        ):
            freeze_gaussians = True
            _apply_gaussian_freezes()
            gaussians.optimizer.zero_grad(set_to_none=True)
            print(blue(f"[NoiseInject] Gaussian scene frozen after color warmup at cycle {cycle} — pose-only refinement from here"))

        if (
            freeze_gaussians_after_cycle > 0
            and not freeze_gaussians
            and cycle > freeze_gaussians_after_cycle
        ):
            freeze_gaussians = True
            _apply_gaussian_freezes()
            gaussians.optimizer.zero_grad(set_to_none=True)
            print(blue(f"[NoiseInject] Gaussian scene frozen after cycle {freeze_gaussians_after_cycle} at cycle {cycle} — pose-only refinement from here"))

        if _should_run_initial_pure_pnp(
            run_pending=run_initial_pure_pnp_after_warmup,
            cycle=cycle,
            warmup_cycles=warmup_cycles,
            total_cycles=total_cycles,
        ):
            _run_pure_pnp_phase(
                phase="initial",
                cycle_value=max(int(warmup_cycles), 0),
                num_steps=pure_pnp_iters,
            )
            print(
                blue(
                    f"[PurePnP] Initial phase done after warmup. "
                    f"rot_err={_rotation_error_deg(_effective_R(pose_correction), gt_l2c_R):.4f}°"
                )
            )
            run_initial_pure_pnp_after_warmup = False
            if periodic_pure_pnp_interval_cycles <= 0:
                return _effective_R(pose_correction)

        # ── Per-cycle match precomputation ────────────────────
        lidar_warmup_active = bool(warmup_cycles > 0 and cycle <= warmup_cycles)
        warmup_active_now = bool(color_warmup_cycles > 0 and cycle <= color_warmup_cycles)
        if blurred_cam_images is not None:
            current_cam_images = (
                blurred_cam_images
                if (not post_init_supervision_blur_warmup_only or warmup_active_now)
                else cam_images
            )
            if post_init_supervision_blur_warmup_only and cycle == color_warmup_cycles + 1:
                print(blue("[NoiseInject] Color warmup finished — switching supervision back to original RGB"))
        pose_grad_accum_steps = 0
        pose_interval_accum_steps = 0
        if pose_step_at_cycle_end and pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
        if (
            match_once_per_cycle
            and matcher is not None
            and matcher_color_supervision
            and not lidar_warmup_active
        ):
            cycle_match_cache, cycle_coverage_rate = _precompute_cycle_match_cache(
                matcher=matcher,
                gaussians=gaussians,
                pose_correction=pose_correction,
                cam_cameras=cam_cameras,
                cam_images=current_cam_images,
                camera_masks=camera_supervision_masks,
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
                dense_color_cert_threshold=matcher_dense_color_cert_threshold,
                depth_render_backend=camera_aux_depth_render_backend,
                depth_render_mode=camera_aux_depth_mode,
            )
            if cycle_coverage_rate > 0.0:
                scale = 1.0 if warmup_active_now else 0.3
                print(f"[Cycle cache] GT RGB lambda = {cycle_coverage_rate:.3f} × {scale} = {cycle_coverage_rate*scale:.4f}")

        # Build adjacent GT color cache lazily (GT images never change, only needs to run once)
        if (
            matcher_adjacent_color_supervision
            and temporal_pair_cache
            and not adj_gt_cache
            and not lidar_warmup_active
        ):
            adj_gt_cache = _precompute_adjacent_gt_cache(
                temporal_pair_cache,
                current_cam_images,
                camera_masks=camera_supervision_masks,
            )

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
            loss_normal = torch.tensor(0.0, device="cuda")
            depth_active = (
                not freeze_gaussians
                and (disable_depth_after_cycle <= 0 or cycle <= disable_depth_after_cycle)
                and not (disable_depth_during_color_warmup and color_warmup_active)
            )
            if depth_active:
                gt_mask = scene.train_lidar.get_mask(frame).cuda()
                dyn_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
                if torch.any(gt_mask):
                    rays_o_valid, rays_d_valid, gt_depth_valid = scene.train_lidar.get_valid_depth_rays(frame)
                    static_valid_mask = ~dyn_mask[gt_mask]
                    if torch.any(static_valid_mask):
                        sparse_sensor = (
                            rays_o_valid.unsqueeze(1),
                            rays_d_valid.unsqueeze(1),
                            scene.train_lidar.sensor_center[frame].to(device="cuda", dtype=torch.float32),
                        )
                        render_pkg = raytracing(
                            frame,
                            [gaussians],
                            sparse_sensor,
                            background,
                            args,
                            depth_only=True,
                        )
                        pred_depth = render_pkg["depth"].reshape(-1)
                        pred_accum = render_pkg.get("accumulation")
                        if pred_accum is not None:
                            pred_accum = pred_accum.reshape(-1)
                        pred_normal = render_pkg.get("normal")
                        if pred_normal is not None:
                            pred_normal = pred_normal.reshape(-1, 3)
                        sparse_static_mask = torch.zeros_like(gt_mask, dtype=torch.bool)
                        sparse_static_mask[gt_mask] = static_valid_mask
                        sparse_pred_map = torch.zeros_like(scene.train_lidar.get_depth(frame).cuda().float())
                        sparse_pred_map[sparse_static_mask] = pred_depth[static_valid_mask]
                        loss_depth = lambda_depth * _compute_lidar_depth_loss(
                            pred_depth=pred_depth[static_valid_mask],
                            gt_depth=gt_depth_valid[static_valid_mask],
                            accumulation=(
                                None
                                if pred_accum is None
                                else pred_accum[static_valid_mask]
                            ),
                            loss_mode=str(
                                getattr(args.opt, "lidar_depth_loss_mode", "l1")
                            ),
                            inverse_min_depth=float(
                                getattr(args.opt, "lidar_depth_inverse_min_depth", 0.5)
                            ),
                            use_visibility_weights=bool(
                                getattr(args.opt, "lidar_depth_use_image_visibility_weights", False)
                            ),
                            visible_weight=float(
                                getattr(args.opt, "lidar_depth_visible_weight", 2.0)
                            ),
                            occluded_weight=float(
                                getattr(args.opt, "lidar_depth_occluded_weight", 0.5)
                            ),
                            outside_weight=float(
                                getattr(args.opt, "lidar_depth_outside_weight", 1.0)
                            ),
                            visibility_tolerance=float(
                                getattr(args.opt, "lidar_depth_visibility_tolerance", 0.25)
                            ),
                        )
                        if float(lambda_normal) > 0.0:
                            pred_support_mask = sparse_static_mask.clone()
                            if pred_accum is not None:
                                pred_support_mask[sparse_static_mask] = (
                                    pred_accum[static_valid_mask] > 1.0e-4
                                )
                            gt_normal_map, gt_normal_valid = scene.train_lidar.get_normal(frame)
                            gt_normal_map = gt_normal_map.cuda().float()
                            gt_normal_valid = gt_normal_valid.cuda()
                            pred_normal_map = torch.zeros(
                                (*gt_mask.shape, 3),
                                device=gt_mask.device,
                                dtype=torch.float32,
                            )
                            if pred_normal is not None:
                                pred_normal_map[sparse_static_mask] = pred_normal[static_valid_mask]
                            normal_valid_mask = (
                                _four_neighbor_support_mask(pred_support_mask)
                                & _four_neighbor_support_mask(~dyn_mask)
                                & gt_normal_valid
                            )
                            loss_normal = float(lambda_normal) * _compute_lidar_normal_loss(
                                pred_normal_map=pred_normal_map,
                                gt_normal_map=gt_normal_map,
                                valid_mask=normal_valid_mask,
                            )

            # ── Camera RGB loss ───────────────────────────────
            loss_rgb = torch.tensor(0.0, device="cuda")
            loss_match_color = torch.tensor(0.0, device="cuda")
            loss_match_adjacent = torch.tensor(0.0, device="cuda")
            loss_rgb_reg = torch.tensor(0.0, device="cuda")
            loss_reg = torch.tensor(0.0, device="cuda")
            loss_camera_depth_normal = torch.tensor(0.0, device="cuda")
            matcher_color_summary = None
            base_rgb_photo_loss = None
            if frame in cam_cameras and not lidar_warmup_active:
                cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
                camera  = cam_cameras[frame].cuda()
                gt_rgb  = current_cam_images[frame].cuda()
                valid_rgb_mask = camera_supervision_masks.get(int(frame))
                if valid_rgb_mask is not None:
                    valid_rgb_mask = valid_rgb_mask.to(device=gt_rgb.device)
                cam_render = render_camera(
                    camera, [gaussians], args,
                    cam_rotation=cam_R, cam_translation=cam_T,
                    require_rgb=True,
                )
                if cam_render["num_visible"] > 0:
                    pred_rgb = cam_render["rgb"].clamp(0.0, 1.0)
                    pred_chw = pred_rgb.permute(2, 0, 1)
                    gt_chw   = gt_rgb.permute(2, 0, 1)
                    if float(lambda_rgb) > 0.0 or post_warmup_rgb_reg_scale > 0:
                        if valid_rgb_mask is not None and torch.any(valid_rgb_mask):
                            mask3 = valid_rgb_mask.unsqueeze(-1).float()
                            masked_l1 = torch.abs(pred_rgb - gt_rgb)[valid_rgb_mask].mean()
                            masked_ssim = 1.0 - ssim(
                                pred_chw * mask3.permute(2, 0, 1),
                                gt_chw * mask3.permute(2, 0, 1),
                            )
                            base_rgb_photo_loss = (
                                (1.0 - float(lambda_dssim)) * masked_l1
                                + float(lambda_dssim) * masked_ssim
                            )
                        else:
                            base_rgb_photo_loss = (
                                (1.0 - float(lambda_dssim)) * l1_loss(pred_rgb, gt_rgb)
                                + float(lambda_dssim) * (1.0 - ssim(pred_chw, gt_chw))
                            )
                    if float(lambda_rgb) > 0.0 and base_rgb_photo_loss is not None:
                        loss_rgb = float(lambda_rgb) * base_rgb_photo_loss
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
                            backend=camera_aux_depth_render_backend,
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
                            aux_depth = _resolve_camera_aux_depth(
                                depth_render,
                                depth_mode=camera_aux_depth_mode,
                            )
                            loss_match_color, matcher_color_summary = _compute_matcher_color_loss(
                                matcher=matcher,
                                gt_rgb=gt_rgb,
                                pred_rgb=color_render["rgb"].clamp(0.0, 1.0),
                                depth=aux_depth,
                                matcher_min_matches=matcher_min_matches,
                                matcher_min_depth_matches=matcher_min_depth_matches,
                                matcher_depth_min=matcher_depth_min,
                                matcher_depth_max=matcher_depth_max,
                                matcher_depth_percentile_low=matcher_depth_percentile_low,
                                matcher_depth_percentile_high=matcher_depth_percentile_high,
                                matcher_depth_use_inverse=matcher_depth_use_inverse,
                                matcher_color_weight=matcher_color_weight,
                                gt_valid_mask=camera_supervision_masks.get(int(frame)),
                                temporal_support_points=get_temporal_support(frame),
                                temporal_support_radius_px=matcher_support_radius_px,
                                dense_mode=matcher_dense_mode,
                                dense_stride=matcher_dense_stride,
                                dense_cert_threshold=matcher_dense_cert_threshold,
                            )
                    if (
                        post_warmup_rgb_reg_scale > 0
                        and not color_warmup_active
                        and base_rgb_photo_loss is not None
                    ):
                        supervised_ratio = 1.0
                        if matcher_color_summary is not None:
                            supervised_ratio = float(matcher_color_summary.get("supervised_ratio", 1.0))
                        loss_rgb_reg = float(post_warmup_rgb_reg_scale) * supervised_ratio * base_rgb_photo_loss
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
                            cam_images=current_cam_images,
                            camera_masks=camera_supervision_masks,
                            args=args,
                            temporal_pair_cache=temporal_pair_cache,
                            matcher_adjacent_max_offset=matcher_adjacent_max_offset,
                            matcher_min_matches=matcher_min_matches,
                            adjacent_color_weight=matcher_adjacent_color_weight,
                        )

            if (
                lidar_warmup_active
                and float(warmup_camera_depth_normal_weight) > 0.0
                and frame in cam_cameras
            ):
                cam_R_reg, cam_T_reg = pose_correction.corrected_rt(frame, device="cuda")
                camera_reg = cam_cameras[frame].cuda()
                camera_depth_render = _render_camera_with_backend(
                    camera_reg,
                    [gaussians],
                    args,
                    backend="raytracing",
                    cam_rotation=cam_R_reg.detach(),
                    cam_translation=cam_T_reg.detach(),
                    require_rgb=False,
                )
                if int(camera_depth_render.get("num_visible", 0)) > 0:
                    camera_depth = _resolve_camera_aux_depth(
                        camera_depth_render,
                        depth_mode="depth",
                    )
                    camera_normal = camera_depth_render.get("normal")
                    if camera_normal is not None:
                        loss_camera_depth_normal = (
                            float(warmup_camera_depth_normal_weight)
                            * _compute_camera_depth_normal_consistency_loss(
                                pred_depth_map=camera_depth,
                                pred_normal_map=camera_normal,
                                camera=camera_reg,
                            )
                        )

            apply_other_regularizers = not (
                lidar_warmup_active and bool(warmup_disable_other_regularizers)
            )
            if apply_other_regularizers and float(lambda_reg) > 0.0:
                reg_term = gaussians.box_reg_loss()
                if not torch.is_tensor(reg_term):
                    reg_term = torch.tensor(float(reg_term), device=loss_depth.device, dtype=loss_depth.dtype)
                else:
                    reg_term = reg_term.to(device=loss_depth.device, dtype=loss_depth.dtype)
                loss_reg = float(lambda_reg) * reg_term
            if apply_other_regularizers and loss_normal.numel() > 0:
                loss_reg = loss_reg + loss_normal
            loss_reg = loss_reg + loss_camera_depth_normal

            total_loss = loss_depth + loss_rgb + loss_match_color + loss_match_adjacent + loss_rgb_reg + loss_reg

            # ── Cross-frame photometric consistency loss ──────────────────────────
            loss_cross_frame = torch.tensor(0.0, device="cuda")
            if cross_frame_weight > 0.0 and frame in cam_cameras and not color_warmup_active:
                cam_frames_sorted = sorted(cam_cameras.keys())
                _frame_pos = cam_frames_sorted.index(frame) if frame in cam_frames_sorted else -1
                _nbr_frame = None
                if _frame_pos >= 0:
                    for _off in (1, -1, 2, -2):
                        _cand_pos = _frame_pos + _off
                        if 0 <= _cand_pos < len(cam_frames_sorted):
                            _nbr_frame = cam_frames_sorted[_cand_pos]
                            break
                if _nbr_frame is not None and cam_render.get("num_visible", 0) > 0 and "depth" in cam_render:
                    cam_R_cf, cam_T_cf = pose_correction.corrected_rt(frame, device="cuda")
                    cam_R_nbr_cf, cam_T_nbr_cf = pose_correction.corrected_rt(_nbr_frame, device="cuda")
                    _cam = cam_cameras[frame]
                    _fx, _fy, _cx, _cy = _camera_intrinsics_from_camera(_cam)
                    loss_cross_frame = _compute_cross_frame_consistency_loss(
                        pred_rgb=cam_render["rgb"].clamp(0.0, 1.0),
                        depth=cam_render["depth"].squeeze(-1) if cam_render["depth"].dim() == 3 else cam_render["depth"],
                        cam_R=cam_R_cf,
                        cam_T=cam_T_cf,
                        nbr_gt_rgb=current_cam_images[_nbr_frame].cuda(),
                        source_valid_mask=camera_supervision_masks.get(int(frame)),
                        nbr_valid_mask=camera_supervision_masks.get(int(_nbr_frame)),
                        cam_R_nbr=cam_R_nbr_cf,
                        cam_T_nbr=cam_T_nbr_cf,
                        fx=_fx,
                        fy=_fy,
                        cx=_cx,
                        cy=_cy,
                        weight=cross_frame_weight,
                    )

            # ── Flow projection loss ──────────────────────────────────────────────
            loss_flow_proj = torch.tensor(0.0, device="cuda")
            if (flow_proj_weight > 0.0 and frame in flow_proj_cache
                    and frame in cam_cameras and not color_warmup_active):
                _cam_fp = cam_cameras[frame]
                _fx_fp, _fy_fp, _cx_fp, _cy_fp = _camera_intrinsics_from_camera(_cam_fp)
                # Get depth from surfel (2DGS) renderer — detached poses so no Gaussian grad.
                cam_R_fp, cam_T_fp = pose_correction.corrected_rt(frame, device="cuda")
                _fp_surfel = _render_camera_with_backend(
                    _cam_fp.cuda(), [gaussians], args,
                    backend=camera_aux_depth_render_backend,
                    cam_rotation=cam_R_fp.detach(),
                    cam_translation=cam_T_fp.detach(),
                    require_rgb=False,
                )
                if _fp_surfel.get("num_visible", 0) > 0 and "depth" in _fp_surfel:
                    _depth_fp = _resolve_camera_aux_depth(
                        _fp_surfel,
                        depth_mode=camera_aux_depth_mode,
                    )
                    if _depth_fp.dim() == 3:
                        _depth_fp = _depth_fp.squeeze(-1)
                    loss_flow_proj = _compute_flow_projection_loss(
                        flow_pairs=flow_proj_cache[frame],
                        depth=_depth_fp,
                        cam_R=cam_R_fp,
                        cam_T=cam_T_fp,
                        pose_correction=pose_correction,
                        fx=_fx_fp,
                        fy=_fy_fp,
                        cx=_cx_fp,
                        cy=_cy_fp,
                        weight=flow_proj_weight,
                    )
                merged_geometry_grads = None
                geometry_named_params = []
                # Selectively gate RGB gradients on Gaussian parameters, then add LiDAR depth grads.
                rgb_zero_attrs = []
                retain_graph_for_rgb = bool(loss_match_color.requires_grad or loss_match_adjacent.requires_grad or loss_rgb_reg.requires_grad)
                lidar_loss = loss_depth + loss_reg
                if geometry_conflict_policy == "lidar_priority" and not freeze_gaussians:
                    geometry_named_params = _collect_named_params(gaussians, _GEOMETRY_ATTRS)
                    if geometry_named_params:
                        lidar_geometry_grads = _compute_named_grads(lidar_loss, geometry_named_params)
                        camera_geometry_grads = {
                            attr: None for attr, _ in geometry_named_params
                        }
                        if loss_rgb.requires_grad:
                            rgb_blocked_attrs = set()
                            if color_warmup_active:
                                rgb_blocked_attrs.update(_MATCH_FROZEN_ATTRS)
                            elif camera_rgb_pose_only:
                                rgb_blocked_attrs.update(_GEOMETRY_ATTRS)
                            else:
                                if freeze_covariance:
                                    rgb_blocked_attrs.update(["_scaling", "_rotation"])
                                if rgb_only_updates_color:
                                    rgb_blocked_attrs.update(_RGB_FROZEN_ATTRS)
                            _accumulate_named_grads(
                                camera_geometry_grads,
                                _compute_named_grads(loss_rgb, geometry_named_params),
                                blocked_attrs=rgb_blocked_attrs,
                            )
                        if loss_match_color.requires_grad:
                            _accumulate_named_grads(
                                camera_geometry_grads,
                                _compute_named_grads(loss_match_color, geometry_named_params),
                                blocked_attrs=set(_MATCH_FROZEN_ATTRS),
                            )
                        if loss_match_adjacent.requires_grad:
                            _accumulate_named_grads(
                                camera_geometry_grads,
                                _compute_named_grads(loss_match_adjacent, geometry_named_params),
                                blocked_attrs=set(_MATCH_FROZEN_ATTRS),
                            )
                        if loss_rgb_reg.requires_grad:
                            _accumulate_named_grads(
                                camera_geometry_grads,
                                _compute_named_grads(loss_rgb_reg, geometry_named_params),
                                blocked_attrs=set(_MATCH_FROZEN_ATTRS),
                            )
                        merged_geometry_grads = _merge_lidar_priority_geom_grads(
                            lidar_grads=lidar_geometry_grads,
                            camera_grads=camera_geometry_grads,
                            camera_weight=geometry_conflict_camera_weight,
                        )
                if loss_rgb.requires_grad:
                    if color_warmup_active:
                        # During color warmup: RGB loss updates colors + covariance + opacity, but NOT xyz
                        rgb_zero_attrs = _MATCH_FROZEN_ATTRS
                    elif camera_rgb_pose_only:
                        rgb_zero_attrs = list(_GAUSSIAN_ATTRS)
                    else:
                        if freeze_covariance:
                            rgb_zero_attrs.extend(["_scaling", "_rotation"])
                        if freeze_opacity:
                            rgb_zero_attrs.append("_opacity")
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
                    for attr in _MATCH_FROZEN_ATTRS:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_match_adjacent.requires_grad:
                    loss_match_adjacent.backward(retain_graph=bool(loss_rgb_reg.requires_grad))
                    for attr in _MATCH_FROZEN_ATTRS:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if loss_rgb_reg.requires_grad:
                    loss_rgb_reg.backward()
                    for attr in _MATCH_FROZEN_ATTRS:
                        p = getattr(gaussians, attr, None)
                        if p is not None and p.grad is not None:
                            p.grad.zero_()
                if lidar_loss.requires_grad:
                    lidar_loss.backward()
                # Cross-frame loss: pred_rgb is detached → no Gaussian grads, only pose grads
                if loss_cross_frame.requires_grad:
                    loss_cross_frame.backward()
                # Flow projection loss: depth is detached → only pose grads
                if loss_flow_proj.requires_grad:
                    loss_flow_proj.backward()
                if merged_geometry_grads is not None:
                    _override_named_param_grads(geometry_named_params, merged_geometry_grads)
            else:
                (total_loss + loss_cross_frame + loss_flow_proj).backward()
            loss_accum       += total_loss.item() + loss_cross_frame.item() + loss_flow_proj.item()
            loss_depth_accum += loss_depth.item()
            loss_rgb_accum   += loss_rgb.item()
            loss_reg_accum   += float(loss_reg.detach())
            loss_match_color_accum += loss_match_color.item()
            loss_match_adjacent_accum += loss_match_adjacent.item()
            loss_cross_frame_accum += loss_cross_frame.item()
            loss_flow_proj_accum += loss_flow_proj.item()

            if not freeze_gaussians:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if pose_optimizer is None:
                pose_correction.zero_grad(set_to_none=True)
            elif pose_step_at_cycle_end:
                if cycle > warmup_cycles and not matcher_pose_update and not color_warmup_active:
                    pose_grad_accum_steps += 1
            else:
                if cycle > warmup_cycles and not matcher_pose_update and not color_warmup_active:
                    if pose_step_interval_iters > 0:
                        pose_interval_accum_steps += 1
                        if pose_interval_accum_steps >= pose_step_interval_iters:
                            pose_optimizer.step()
                            pose_correction.update_extrinsics()
                            pose_optimizer.zero_grad(set_to_none=True)
                            pose_interval_accum_steps = 0
                    else:
                        pose_optimizer.step()
                        pose_optimizer.zero_grad(set_to_none=True)
                else:
                    pose_optimizer.zero_grad(set_to_none=True)

        # ── Fold delta into base once per cycle ────────────────
        matcher_update_summary = None
        if pose_optimizer is None:
            pose_correction.zero_grad(set_to_none=True)
        elif pose_step_at_cycle_end:
            if cycle > warmup_cycles and not matcher_pose_update and cycle > color_warmup_cycles and pose_grad_accum_steps > 0:
                grad_scale = 1.0 / float(pose_grad_accum_steps)
                for pg in pose_optimizer.param_groups:
                    for p in pg["params"]:
                        if p.grad is not None:
                            p.grad.mul_(grad_scale)
                pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        elif pose_step_interval_iters > 0:
            if cycle > warmup_cycles and not matcher_pose_update and cycle > color_warmup_cycles and pose_interval_accum_steps > 0:
                pose_optimizer.step()
                pose_correction.update_extrinsics()
            pose_optimizer.zero_grad(set_to_none=True)
        if matcher_pose_update and cycle > warmup_cycles and cycle > color_warmup_cycles and matcher_update_interval > 0 and cycle % matcher_update_interval == 0:
            matcher_update_summary = _run_matcher_pose_update(
                matcher=matcher,
                gaussians=gaussians,
                pose_correction=pose_correction,
                cam_cameras=cam_cameras,
                cam_images=current_cam_images,
                camera_masks=camera_supervision_masks,
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
                render_intrinsics_fx_scale=matcher_render_intrinsics_fx_scale,
                render_intrinsics_fy_scale=matcher_render_intrinsics_fy_scale,
                render_intrinsics_cx_offset=matcher_render_intrinsics_cx_offset,
                render_intrinsics_cy_offset=matcher_render_intrinsics_cy_offset,
                matcher_lidar_nn_max_distance=matcher_lidar_nn_max_distance,
                matcher_update_blend=matcher_update_blend,
                temporal_support_getter=get_temporal_support,
                temporal_support_radius_px=matcher_support_radius_px,
            )
        elif (
            periodic_pure_pnp_interval_cycles > 0
            and cycle % periodic_pure_pnp_interval_cycles == 0
        ):
            periodic_records = _run_pure_pnp_phase(
                phase="periodic",
                cycle_value=cycle,
                num_steps=pure_pnp_iters,
            )
            if periodic_records:
                matcher_update_summary = periodic_records[-1]
        elif cycle > warmup_cycles and cycle > color_warmup_cycles and pose_step_interval_iters <= 0:
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
        avg_reg   = loss_reg_accum   / iters_per_cycle
        avg_match_color = loss_match_color_accum / iters_per_cycle
        avg_match_adjacent = loss_match_adjacent_accum / iters_per_cycle
        avg_cross_frame = loss_cross_frame_accum / iters_per_cycle
        avg_flow_proj = loss_flow_proj_accum / iters_per_cycle
        psnr_accum = 0.0
        psnr_count = 0
        loss_accum       = 0.0
        loss_depth_accum = 0.0
        loss_rgb_accum   = 0.0
        loss_reg_accum   = 0.0
        loss_match_color_accum = 0.0
        loss_match_adjacent_accum = 0.0
        loss_cross_frame_accum = 0.0
        loss_flow_proj_accum = 0.0

        # Track best translation seen so far
        if T_err < best_T_err:
            best_T_err = T_err
            best_delta_T = pose_correction.delta_translations.data.clone()

        if (
            pose_lr_drop_cycle > 0
            and not pose_lr_drop_applied
            and cycle >= pose_lr_drop_cycle
            and pose_optimizer is not None
        ):
            for pg in pose_optimizer.param_groups:
                pg["lr"] = max(float(lr_min), float(pg["lr"]) * float(pose_lr_drop_factor))
            pose_lr_drop_applied = True
            print(blue(f"  [LR] late-stage pose LR drop applied at cycle {cycle}: ×{pose_lr_drop_factor:.3f}"))

        if (
            translation_lr_drop_cycle > 0
            and not translation_lr_drop_applied
            and cycle >= translation_lr_drop_cycle
            and pose_optimizer is not None
        ):
            translation_lr_updated = False
            for pg in pose_optimizer.param_groups:
                if pg.get("name") != "translation":
                    continue
                pg["lr"] = max(float(lr_min), float(pg["lr"]) * float(translation_lr_drop_factor))
                translation_lr_updated = True
            if translation_lr_updated:
                translation_lr_drop_applied = True
                print(blue(f"  [LR] late-stage translation LR drop applied at cycle {cycle}: ×{translation_lr_drop_factor:.3f}"))

        # ── ReduceLROnPlateau step ────────────────────────────
        cur_rot_lr = pose_optimizer.param_groups[0]["lr"] if pose_optimizer is not None else 0.0
        if pose_lr_scheduler is not None and pose_optimizer is not None and cycle > warmup_cycles:
            pose_lr_scheduler.step(rot_err)
            new_rot_lr = pose_optimizer.param_groups[0]["lr"]
            if new_rot_lr < cur_rot_lr:
                print(blue(f"  [LR] rotation_lr reduced: {cur_rot_lr:.2e} → {new_rot_lr:.2e}"))
            cur_rot_lr = new_rot_lr

        _cf_str = f"  cf={avg_cross_frame:.4f}" if cross_frame_weight > 0.0 else ""
        _fp_str = f"  fp={avg_flow_proj:.5f}" if flow_proj_weight > 0.0 else ""
        print(yellow(
            f"  Cycle {cycle:3d}/{total_cycles}  rot_err={rot_err:.4f}°  "
            f"T_err={T_err:.4f}m  "
            f"PSNR={avg_psnr:.2f} dB  loss={avg_loss:.5f}  "
            f"[d={avg_depth:.4f}  reg={avg_reg:.4f}  rgb={avg_rgb:.4f}  match_rgb={avg_match_color:.4f}  adj_rgb={avg_match_adjacent:.4f}{_cf_str}{_fp_str}]  lr={cur_rot_lr:.2e}"
        ))
        if matcher_update_summary is not None:
            summary_tag = str(matcher_update_summary.get("phase", "matcher"))
            if matcher_update_summary["status"] == "applied":
                print(
                    blue(
                        f"    [{summary_tag}] "
                        f"frames={matcher_update_summary['frames_used']}  "
                        f"matches={matcher_update_summary['matches_used']}  "
                        f"reproj={matcher_update_summary['mean_reproj_px']:.3f}px  "
                        f"dR={matcher_update_summary['rotation_delta_deg']:.4f}°  "
                        f"dT={matcher_update_summary['translation_delta_m']:.4f}m"
                    )
                )
            else:
                print(blue(f"    [{summary_tag}] skipped: {matcher_update_summary['reason']}"))

        if tb_writer is not None:
            tb_writer.add_scalar("calib/rot_err_deg",  rot_err,  cycle)
            tb_writer.add_scalar("calib/trans_err_m",  T_err,    cycle)
            tb_writer.add_scalar("calib/psnr_db",      avg_psnr, cycle)
            tb_writer.add_scalar("calib/loss",         avg_loss, cycle)
            tb_writer.add_scalar("calib/loss_depth",   avg_depth, cycle)
            tb_writer.add_scalar("calib/loss_reg",     avg_reg, cycle)
            tb_writer.add_scalar("calib/loss_rgb",     avg_rgb,  cycle)
            tb_writer.add_scalar("calib/loss_match_color", avg_match_color, cycle)
            tb_writer.add_scalar("calib/loss_match_adjacent", avg_match_adjacent, cycle)
            if matcher_update_summary is not None and matcher_update_summary["status"] == "applied":
                tb_writer.add_scalar("calib/matcher_frames", matcher_update_summary["frames_used"], cycle)
                tb_writer.add_scalar("calib/matcher_matches", matcher_update_summary["matches_used"], cycle)
                tb_writer.add_scalar("calib/matcher_reproj_px", matcher_update_summary["mean_reproj_px"], cycle)
                tb_writer.add_scalar("calib/matcher_rotation_delta_deg", matcher_update_summary["rotation_delta_deg"], cycle)
                tb_writer.add_scalar("calib/matcher_translation_delta_m", matcher_update_summary["translation_delta_m"], cycle)
            tb_writer.flush()

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
                        help="Freeze Gaussian scale and rotation parameters during calibration.")
    parser.add_argument("--freeze_opacity",      action="store_true",
                        help="Freeze Gaussian opacity parameters during calibration.")
    parser.add_argument("--translation_start_cycle", type=int, default=0,
                        help="Two-stage calibration: optimise rotation-only for this many cycles, "
                             "then freeze xyz and add translation optimisation. "
                             "0 = simultaneous (default). Ignored when use_gt_translation=True.")
    parser.add_argument("--stage2_keep_colors_trainable", action="store_true",
                        help="When using --translation_start_cycle, keep Gaussian color features trainable during stage 2 instead of freezing them.")
    parser.add_argument("--warmup_cycles",         type=int, default=0,
                        help="Run pure LiDAR/depth warmup for this many cycles at the start: pose frozen, camera-side supervision disabled. Default: 0.")
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
    parser.add_argument("--pose_lr_drop_cycle",   type=int, default=0,
                        help="If >0, multiply pose learning rates once at this cycle for late-stage refinement.")
    parser.add_argument("--pose_lr_drop_factor",  type=float, default=0.1,
                        help="Multiplicative factor for the one-time late-stage pose LR drop.")
    parser.add_argument("--translation_lr_drop_cycle", type=int, default=0,
                        help="If >0, multiply only translation LR once at this cycle for late-stage refinement.")
    parser.add_argument("--translation_lr_drop_factor", type=float, default=0.1,
                        help="Multiplicative factor for the one-time late-stage translation LR drop.")
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
    parser.add_argument("--resume_cycle_ckpt_scene_only", action="store_true",
                        help="Restore only Gaussian scene state from --resume_cycle_ckpt, keeping pose/init state from the current run.")
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
                        help="Update pose once per cycle from RGB-vs-rendered-depth matcher calibration.")
    parser.add_argument("--pure_pnp_iters", type=int, default=0,
                        help="Run N pure iterative PnP steps (render depth → match → PnP). By default this exits immediately after the PnP phase; when --periodic_pure_pnp_interval_cycles > 0, these N steps become the initial pose update before training continues.")
    parser.add_argument("--pure_pnp_drop_first_n_frames", type=int, default=0,
                        help="For pure PnP only: drop this many earliest frames from the shared-extrinsic solve.")
    parser.add_argument("--pure_pnp_drop_last_n_frames", type=int, default=0,
                        help="For pure PnP only: drop this many latest frames from the shared-extrinsic solve.")
    parser.add_argument("--periodic_pure_pnp_interval_cycles", type=int, default=0,
                        help="If >0, after the initial pure PnP phase, rerun pure PnP every N training cycles instead of only once.")
    parser.add_argument("--disable_pose_grad_updates", action="store_true",
                        help="Disable pose gradient optimization during training. Pose is only updated by matcher / pure PnP schedules.")
    parser.add_argument("--matcher_color_supervision", action="store_true",
                        help="Use matcher correspondences to map GT RGB colors onto rendered depth pixels for sparse color supervision.")
    parser.add_argument("--matcher_color_weight", type=float, default=1.0,
                        help="Weight for sparse matcher color supervision.")
    parser.add_argument("--camera_rgb_pose_only", action="store_true",
                        help="Use full-image camera L1/SSIM to update pose only; Gaussian grads from this loss are zeroed.")
    parser.add_argument("--color_warmup_cycles", type=int, default=0,
                        help="For the first N cycles, RGB losses update only Gaussian color parameters and do not update pose.")
    parser.add_argument("--initialize_pose_from_matcher", action="store_true",
                        help="Before training, run a matcher-based shared PnP initialization and use it as the starting pose.")
    parser.add_argument("--post_init_supervision_blur_kernel", type=int, default=0,
                        help="After matcher/PnP initialization, apply this odd Gaussian blur kernel to GT camera images used for subsequent optimization supervision.")
    parser.add_argument("--post_init_supervision_blur_sigma", type=float, default=0.0,
                        help="Gaussian blur sigma for post-init optimization supervision images.")
    parser.add_argument("--post_init_supervision_blur_warmup_only", action="store_true",
                        help="Apply post-init supervision blur only during color warmup cycles, then switch back to original RGB supervision.")
    parser.add_argument("--freeze_gaussians_after_color_warmup", action="store_true",
                        help="Freeze the full Gaussian scene after color warmup finishes so later cycles optimize pose only.")
    parser.add_argument("--freeze_gaussians_after_cycle", type=int, default=0,
                        help="If >0, freeze the full Gaussian scene after this cycle so later cycles optimize pose only.")
    parser.add_argument("--pose_step_at_cycle_end", action="store_true",
                        help="Accumulate pose gradients within a cycle and update pose once at the end of that cycle.")
    parser.add_argument("--pose_step_interval_iters", type=int, default=0,
                        help="If >0, step pose optimizer and fold extrinsics every N iterations after warmup, similar to HiGS-Calib's periodic pose updates.")
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
                        help="Matcher backend for cycle pose update. Supports matchanything-roma and minima-roma.")
    parser.add_argument("--matcher_minima_root", type=str, default=None,
                        help="Optional MINIMA repo root for matcher_name=minima-roma.")
    parser.add_argument("--matcher_minima_ckpt", type=str, default=None,
                        help="Optional MINIMA RoMa checkpoint path for matcher_name=minima-roma.")
    parser.add_argument("--matcher_resize", type=int, default=None,
                        help="Matcher input resize before feature extraction.")
    parser.add_argument("--matcher_match_threshold", type=float, default=None,
                        help="MatchAnything coarse match threshold.")
    parser.add_argument("--matcher_render_intrinsics_fx_scale", type=float, default=None,
                        help="Render-only focal scale on fx for matcher/PurePnP depth rendering.")
    parser.add_argument("--matcher_render_intrinsics_fy_scale", type=float, default=None,
                        help="Render-only focal scale on fy for matcher/PurePnP depth rendering.")
    parser.add_argument("--matcher_render_intrinsics_cx_offset", type=float, default=None,
                        help="Render-only principal-point x offset in pixels for matcher/PurePnP depth rendering.")
    parser.add_argument("--matcher_render_intrinsics_cy_offset", type=float, default=None,
                        help="Render-only principal-point y offset in pixels for matcher/PurePnP depth rendering.")
    parser.add_argument("--matcher_lidar_nn_max_distance", type=float, default=None,
                        help="If >0, drop backprojected depth 3D points whose nearest LiDAR-point distance exceeds this threshold in meters.")
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
    parser.add_argument("--matcher_dense_color_cert_threshold", type=float, default=None,
                        help="Cert threshold for color supervision (default: same as --matcher_dense_cert_threshold). "
                             "Set lower (e.g. 0.001) or 0.0 for full coverage with cert-weighted loss.")
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
    parser.add_argument("--lambda_depth",           type=float, default=None,
                        help="Override the depth loss weight (default from exp config, "
                             "usually 1.0). Set to 0 to disable depth supervision entirely.")
    parser.add_argument("--lambda_normal",          type=float, default=None,
                        help="Override the LiDAR depth-normal consistency weight. Defaults to exp config lambda_normal.")
    parser.add_argument("--lambda_rgb",             type=float, default=None,
                        help="Override the full-image camera RGB loss weight. Set to 0 to disable dense RGB supervision.")
    parser.add_argument("--lambda_dssim",           type=float, default=None,
                        help="Override the RGB DSSIM mixing weight. Defaults to exp config lambda_rgb_dssim.")
    parser.add_argument("--lambda_reg",             type=float, default=None,
                        help="Override the Gaussian box regularization weight. Defaults to exp config lambda_reg.")
    parser.add_argument("--geometry_conflict_policy", type=str, default="none",
                        choices=["none", "lidar_priority"],
                        help="How to merge conflicting geometry gradients between LiDAR supervision and camera/RGB supervision.")
    parser.add_argument("--geometry_conflict_camera_weight", type=float, default=1.0,
                        help="Relative weight kept on camera geometry gradients after LiDAR-priority conflict projection.")
    parser.add_argument("--warmup_camera_depth_normal_weight", type=float, default=0.0,
                        help="During LiDAR warmup, add a camera-view depth/normal consistency regularizer from raytraced camera depth and rendered normals. 0 = disabled.")
    parser.add_argument("--warmup_disable_other_regularizers", action="store_true",
                        help="During LiDAR warmup, disable the existing box and LiDAR-normal regularizers so only LiDAR depth plus the camera-view depth/normal term remain.")
    parser.add_argument("--lidar_depth_use_image_visibility_weights", action="store_true",
                        help="Use lidar-rt-style visibility weighting for LiDAR depth supervision.")
    parser.add_argument("--lidar_depth_visible_weight", type=float, default=None,
                        help="Weight for visible LiDAR depth residuals when visibility weighting is enabled.")
    parser.add_argument("--lidar_depth_occluded_weight", type=float, default=None,
                        help="Weight for occluded/back-surface LiDAR depth residuals when visibility weighting is enabled.")
    parser.add_argument("--lidar_depth_outside_weight", type=float, default=None,
                        help="Weight for no-hit/outside LiDAR depth residuals when visibility weighting is enabled.")
    parser.add_argument("--lidar_depth_visibility_tolerance", type=float, default=None,
                        help="Absolute depth tolerance (meters) for visibility-weighted LiDAR depth supervision.")
    parser.add_argument("--lidar_depth_loss_mode", type=str, default=None, choices=["l1", "inverse_depth"],
                        help="LiDAR depth supervision residual: absolute depth (l1) or inverse depth.")
    parser.add_argument("--lidar_depth_inverse_min_depth", type=float, default=None,
                        help="Clamp predicted and GT depth to at least this many meters before inverse-depth supervision.")
    parser.add_argument("--matcher_color_lr_scale", type=float, default=1.0,
                        help="Multiply optimizer learning rates for camera RGB feature params when using matcher color supervision.")
    parser.add_argument("--cross_frame_weight",  type=float, default=0.0,
                        help="Weight for HiGS-style cross-frame photometric consistency loss. "
                             "Backprojects rendered depth, reprojects into adjacent frame and "
                             "compares sampled GT RGB vs current rendered RGB.  Gradient flows "
                             "through the projection matrices to supervise rotation. 0 = off.")
    parser.add_argument("--flow_proj_weight", type=float, default=0.0,
                        help="Weight for optical-flow-style reprojection loss. "
                             "Uses precomputed MatchAnything RGB-RGB correspondences to supervise "
                             "the geometric projection of 3D points across adjacent frames. "
                             "Gradient flows only through pose params (depth detached). 0 = off.")
    parser.add_argument("--flow_proj_max_offset", type=int, default=1,
                        help="Max frame offset for flow projection pairs (default 1 = ±1 neighbor).")
    parser.add_argument("--camera_aux_depth_render_backend", type=str, default=None,
                        choices=["3dgut_rasterization", "surfel_rasterization", "raytracing", "lidar_zbuffer", "lidar_scanline_zbuffer", "point_zbuffer"],
                        help="Backend used for camera-side auxiliary depth renders such as matcher precompute, "
                             "cross-modal matcher depth images, flow projection, and final visualizations. "
                             "Defaults to the backend implied by --render_preset.")
    parser.add_argument("--camera_aux_depth_mode", type=str, default="median",
                        choices=["median", "expected", "depth"],
                        help="Which depth map to consume from camera auxiliary renders. "
                             "'median' uses the current 3DGUT median depth, 'expected' restores the older expected-depth behavior, "
                             "and 'depth' uses the backend's default depth output directly.")
    parser.add_argument("--pure_pnp_photo_weight", type=float, default=0.0,
                        help="For pure PnP only: weight of sparse temporal photometric residuals built from depth-backed current-frame pixels projected into adjacent RGB frames.")
    parser.add_argument("--pure_pnp_photo_match_radius_px", type=float, default=4.0,
                        help="For pure PnP only: max distance from RGB-RGB temporal matches to accepted cross-modal source pixels.")
    parser.add_argument("--pure_pnp_photo_min_matches", type=int, default=8,
                        help="For pure PnP only: minimum supported pixels per photometric residual block.")
    parser.add_argument("--pure_pnp_photo_max_offset", type=int, default=1,
                        help="For pure PnP only: max temporal offset when precomputing adjacent RGB-RGB pairs for photometric residuals.")
    parser.add_argument("--pure_pnp_flow_weight", type=float, default=0.0,
                        help="For pure PnP only: weight of temporal geometric reprojection residuals built from RGB-RGB matched source pixels and depth-backed 3D points.")
    parser.add_argument("--pure_pnp_flow_match_radius_px", type=float, default=4.0,
                        help="For pure PnP only: max distance from RGB-RGB temporal matches to accepted cross-modal source pixels for the flow residual.")
    parser.add_argument("--pure_pnp_flow_min_matches", type=int, default=8,
                        help="For pure PnP only: minimum supported pixels per geometric temporal residual block.")
    parser.add_argument("--pure_pnp_depth_weight", type=float, default=0.0,
                        help="For pure PnP only: weight of temporal depth consistency residuals comparing projected point depth against target-frame rendered depth.")
    parser.add_argument("--pure_pnp_depth_match_radius_px", type=float, default=4.0,
                        help="For pure PnP only: max distance from RGB-RGB temporal matches to accepted cross-modal source pixels for the depth residual.")
    parser.add_argument("--pure_pnp_depth_min_matches", type=int, default=8,
                        help="For pure PnP only: minimum supported pixels per temporal depth residual block.")
    parser.add_argument("--pure_pnp_depth_render_backend", type=str, default=None,
                        choices=["3dgut_rasterization", "surfel_rasterization", "raytracing", "lidar_zbuffer", "lidar_scanline_zbuffer", "point_zbuffer"],
                        help="For pure PnP only: backend used to render depth for cross-modal matching and temporal depth/photo residuals. "
                             "Defaults to the backend implied by --render_preset.")
    parser.add_argument("--pure_pnp_depth_mode", type=str, default=None,
                        choices=["median", "expected", "depth"],
                        help="For pure PnP only: which depth map to consume from rendered camera depth. "
                             "Defaults to --camera_aux_depth_mode.")
    parser.add_argument("--pure_pnp_residual_filter_mad_scale", type=float, default=0.0,
                        help="For pure PnP only: if >0, trim high-residual correspondences after the first shared-consensus init using median+MAD thresholding.")
    parser.add_argument("--pure_pnp_residual_filter_min_keep_ratio", type=float, default=0.5,
                        help="For pure PnP only: minimum per-frame keep ratio enforced by consensus residual trimming.")
    parser.add_argument("--pure_pnp_residual_filter_min_keep_per_frame", type=int, default=24,
                        help="For pure PnP only: minimum number of correspondences retained per frame after residual trimming.")
    parser.add_argument("--pure_pnp_residual_filter_max_reproj_error", type=float, default=0.0,
                        help="For pure PnP only: optional absolute reprojection cap (px) applied after the MAD threshold; <=0 disables it.")
    parser.add_argument("--pure_pnp_residual_filter_start_step", type=int, default=1,
                        help="For pure PnP only: iterative PnP step index at which consensus residual trimming becomes active.")
    parser.add_argument("--pure_pnp_single_frame_prefilter_min_inliers", type=int, default=0,
                        help="For pure PnP only: if >0, drop whole frames whose single-frame PnP yields fewer inliers than this, then keep only each retained frame's single-frame RANSAC inliers.")
    parser.add_argument("--pure_pnp_single_frame_prefilter_min_inlier_ratio", type=float, default=0.0,
                        help="For pure PnP only: if >0, drop whole frames whose single-frame PnP inlier ratio is below this threshold before shared optimization.")
    parser.add_argument("--pure_pnp_frame_disagreement_mad_scale", type=float, default=0.0,
                        help="For pure PnP only: if >0, drop whole frames whose single-frame PnP disagrees with the shared init beyond a robust MAD threshold.")
    parser.add_argument("--pure_pnp_frame_disagreement_min_keep_ratio", type=float, default=0.7,
                        help="For pure PnP only: minimum frame keep ratio enforced by frame-disagreement filtering.")
    parser.add_argument("--pure_pnp_frame_disagreement_min_keep_frames", type=int, default=12,
                        help="For pure PnP only: minimum number of frames retained after frame-disagreement filtering.")
    parser.add_argument("--pure_pnp_frame_disagreement_apply_max_dropped_frames", type=int, default=0,
                        help="For pure PnP only: if >0 and frame-disagreement would drop more than this many frames, skip frame filtering and fall back to other enabled filters.")
    parser.add_argument("--pure_pnp_filter_shared_ransac_outliers", action="store_true",
                        help="For pure PnP only: drop the correspondences rejected by the initial global shared-RANSAC solve before the subsequent shared optimization.")
    parser.add_argument("--pure_pnp_gt_reproj_filter_quantile", type=float, default=0.0,
                        help="For pure PnP only: if >0 and GT is available, keep only the lowest GT-relative reprojection residual quantile of correspondences before shared optimization.")
    parser.add_argument("--pure_pnp_gt_reproj_filter_min_keep_per_frame", type=int, default=24,
                        help="For pure PnP only: minimum correspondences kept per frame when GT-relative reprojection filtering is enabled.")
    parser.add_argument("--pure_pnp_gt_soft_weight_mode", type=str, default="none",
                        choices=["none", "gt_reproj", "pose_balance"],
                        help="For pure PnP only: if GT is available, learn a GT-supervised soft point-weight model and apply it before shared optimization. "
                             "'gt_reproj' imitates low GT reprojection residual; 'pose_balance' also boosts translation-sensitive points to preserve translation constraints.")
    parser.add_argument("--pure_pnp_gt_soft_weight_translation_alpha", type=float, default=0.5,
                        help="For pure PnP only: exponent used by the GT pose-balance soft-weight target to boost translation-sensitive points. Higher values preserve more near/side translation signal.")
    parser.add_argument("--pure_pnp_gt_pose_residual_weight", type=float, default=0.0,
                        help="For pure PnP only: if >0 and GT extrinsic is available, add an explicit shared-pose residual term on the difference between the current relative extrinsic and the GT relative extrinsic.")
    parser.add_argument("--pure_pnp_solver_backend", type=str, default="auto",
                        choices=["auto", "opencv", "scipy"],
                        help="For pure PnP only: shared extrinsic solver backend. 'auto' prefers OpenCV solvePnPRefineLM when the requested residuals/modes are compatible, otherwise falls back to SciPy.")
    parser.add_argument("--pure_pnp_freeze_rotation", action="store_true",
                        help="For pure PnP only: keep rotation fixed and optimize translation only during the shared-extrinsic solve.")
    parser.add_argument("--pure_pnp_freeze_translation", action="store_true",
                        help="For pure PnP only: keep translation fixed and optimize rotation only during the shared-extrinsic solve.")
    parser.add_argument("--pure_pnp_export_match_diagnostics", action="store_true",
                        help="For pure PnP only: export per-step match diagnostics JSON, including GT-vs-filter summaries when GT is available.")
    parser.add_argument("--pure_pnp_export_gt_weight_analysis", action="store_true",
                        help="For pure PnP only: export GT-supervised learned point-weight analysis per step to JSON+NPZ.")
    parser.add_argument("--pure_pnp_temporal_semidense_stride", type=int, default=0,
                        help="For pure PnP only: if >0, also sample semi-dense temporal source pixels from rendered depth on this stride.")
    parser.add_argument("--pure_pnp_temporal_semidense_max_points", type=int, default=0,
                        help="For pure PnP only: max semi-dense temporal source pixels kept per directed frame pair.")
    parser.add_argument("--pure_pnp_temporal_gradient_scale", type=float, default=0.0,
                        help="For pure PnP only: strength of gradient-based reliability down-weighting for temporal residuals.")
    parser.add_argument("--pure_pnp_init_depth_strat_bins", type=int, default=0,
                        help="For pure PnP only: if >1, build the RANSAC initialization subset with this many per-frame depth bins.")
    parser.add_argument("--pure_pnp_init_depth_strat_max_points_per_bin", type=int, default=0,
                        help="For pure PnP only: max correspondences kept per depth bin in the stratified RANSAC init subset.")
    parser.add_argument("--pure_pnp_far_depth_boost", type=float, default=0.0,
                        help="For pure PnP only: extra weight given to farther cross-modal correspondences so distant thin structures contribute more.")
    parser.add_argument("--pure_pnp_far_depth_start_percentile", type=float, default=60.0,
                        help="For pure PnP only: start boosting correspondences deeper than this per-frame depth percentile.")
    parser.add_argument("--pure_pnp_post_init_rgb_blur_kernel", type=int, default=0,
                        help="For pure PnP only: odd Gaussian blur kernel applied to RGB images after matcher initialization and before iterative cross-modal matching.")
    parser.add_argument("--pure_pnp_post_init_rgb_blur_sigma", type=float, default=0.0,
                        help="For pure PnP only: Gaussian blur sigma for post-init RGB matching images.")
    parser.add_argument("--kitti_lidar_width", type=int, default=None,
                        help="KITTI/KITTI-360 range-image width used when rasterizing raw LiDAR packets.")
    parser.add_argument("--render_preset", type=str, default=None,
                        help="High-level render stack preset for the current calib pipeline. "
                             "Use 'hybrid_3dgrut' for LiDAR=3DGRT + Camera=3DGUT, or "
                             "'lidar_rt'/'2dgs' for LiDAR=legacy lidar-rt + Camera=2DGS.")
    parser.add_argument("--output_dir",          default=None)
    parser.add_argument("--gpu",                 type=int, default=None)
    cli = parser.parse_args()
    if cli.pure_pnp_freeze_rotation and cli.pure_pnp_freeze_translation:
        parser.error("PurePnP cannot freeze both rotation and translation at the same time.")

    if cli.gpu is not None:
        torch.cuda.set_device(cli.gpu)

    args = parse(cli.exp_config)
    args = parse(cli.data_config, args)
    render_preset = _normalize_render_preset(cli.render_preset)
    _dtype = str(getattr(args, "data_type", "")).lower()
    _dtype_norm = _dtype.replace("-", "").replace("_", "")
    default_camera_aux_depth_render_backend = "3dgut_rasterization"
    default_pure_pnp_depth_render_backend = "3dgut_rasterization"
    if "kitti" in _dtype_norm and "kitticalib" not in _dtype_norm:
        default_camera_aux_depth_render_backend = "lidar_zbuffer"
        default_pure_pnp_depth_render_backend = "lidar_zbuffer"
    if render_preset is not None:
        (
            default_camera_aux_depth_render_backend,
            default_pure_pnp_depth_render_backend,
        ) = _apply_render_preset(args, render_preset)
    resolved_camera_aux_depth_render_backend = (
        cli.camera_aux_depth_render_backend or default_camera_aux_depth_render_backend
    )
    resolved_pure_pnp_depth_render_backend = (
        cli.pure_pnp_depth_render_backend or default_pure_pnp_depth_render_backend
    )
    resolved_camera_aux_depth_mode = str(cli.camera_aux_depth_mode or "median")
    resolved_pure_pnp_depth_mode = str(
        cli.pure_pnp_depth_mode or resolved_camera_aux_depth_mode
    )

    scene_id = getattr(args, "scene_id", "calib_scene")
    out_dir  = cli.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, "calib", scene_id)
    os.makedirs(out_dir, exist_ok=True)
    command_path = os.path.join(out_dir, "command.sh")
    with open(command_path, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write(f"cd {shlex.quote(os.getcwd())}\n")
        handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")
    os.chmod(command_path, 0o755)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    print(
        blue(
            "[Calib] Render stack: "
            f"preset={render_preset or 'config'} | "
            f"training_render_mode={getattr(getattr(args, 'model', None), 'training_render_mode', '')!r} | "
            f"camera_render_backend={getattr(getattr(args, 'model', None), 'camera_render_backend', '')} | "
            f"raytrace_backend={getattr(getattr(args, 'model', None), 'raytrace_backend', '')} | "
            f"camera_aux_depth_render_backend={resolved_camera_aux_depth_render_backend} | "
            f"camera_aux_depth_mode={resolved_camera_aux_depth_mode} | "
            f"pure_pnp_depth_render_backend={resolved_pure_pnp_depth_render_backend} | "
            f"pure_pnp_depth_mode={resolved_pure_pnp_depth_mode}"
        )
    )

    # ── Scene ─────────────────────────────────────────────────
    print(blue("[Calib] Loading scene..."))
    scene = dataloader.load_scene(args.source_dir, args)
    gaussians = scene.gaussians_assets[0]
    setattr(args, "_calib_lidar_sensor", scene.train_lidar)

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
        resume_t = npz["final_t"] if "final_t" in npz.files else None
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
    if cli.resume_from is not None and resume_t is not None:
        with torch.no_grad():
            pose_correction.base_lidar_to_camera_translation.data[0].copy_(
                torch.tensor(resume_t, dtype=torch.float32, device="cuda")
            )
            pose_correction.delta_translations.data.zero_()
        resume_t_err = _translation_error_m(pose_correction, gt_l2c_T)
        print(blue(f"[Calib] Resuming translation from: {cli.resume_from}  (err={resume_t_err:.4f} m)"))
    elif cli.init_trans_xyz is not None:
        delta_t = torch.tensor(cli.init_trans_xyz, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pose_correction.base_lidar_to_camera_translation.data[0].copy_(
                gt_l2c_T + delta_t
            )
            pose_correction.delta_translations.data.zero_()
        init_t_err = delta_t.norm().item()
        print(blue(f"[Calib] Init translation perturbation: {cli.init_trans_xyz}  "
                   f"(err={init_t_err:.4f} m)"))

    if cli.lidar_depth_use_image_visibility_weights:
        args.opt.lidar_depth_use_image_visibility_weights = True
    if cli.lidar_depth_visible_weight is not None:
        args.opt.lidar_depth_visible_weight = float(cli.lidar_depth_visible_weight)
    if cli.lidar_depth_occluded_weight is not None:
        args.opt.lidar_depth_occluded_weight = float(cli.lidar_depth_occluded_weight)
    if cli.lidar_depth_outside_weight is not None:
        args.opt.lidar_depth_outside_weight = float(cli.lidar_depth_outside_weight)
    if cli.lidar_depth_visibility_tolerance is not None:
        args.opt.lidar_depth_visibility_tolerance = float(cli.lidar_depth_visibility_tolerance)
    if cli.lidar_depth_loss_mode is not None:
        args.opt.lidar_depth_loss_mode = str(cli.lidar_depth_loss_mode)
    if cli.lidar_depth_inverse_min_depth is not None:
        args.opt.lidar_depth_inverse_min_depth = float(cli.lidar_depth_inverse_min_depth)

    # ── Run calibration ───────────────────────────────────────
    try:
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
            freeze_opacity=cli.freeze_opacity,
            translation_start_cycle=cli.translation_start_cycle,
            stage2_freeze_colors=not cli.stage2_keep_colors_trainable,
            warmup_cycles=cli.warmup_cycles,
            freeze_rotation=cli.freeze_rotation,
            freeze_translation=cli.freeze_translation,
            lr_patience=cli.lr_patience,
            lr_factor=cli.lr_factor,
            lr_min=cli.lr_min,
            pose_lr_drop_cycle=cli.pose_lr_drop_cycle,
            pose_lr_drop_factor=cli.pose_lr_drop_factor,
            translation_lr_drop_cycle=cli.translation_lr_drop_cycle,
            translation_lr_drop_factor=cli.translation_lr_drop_factor,
            lambda_rgb=cli.lambda_rgb if cli.lambda_rgb is not None else float(getattr(args.opt, "lambda_rgb", 1.0)),
            lambda_dssim=cli.lambda_dssim if cli.lambda_dssim is not None else float(getattr(args.opt, "lambda_rgb_dssim", getattr(args.opt, "lambda_dssim", 0.2))),
            lambda_reg=cli.lambda_reg if cli.lambda_reg is not None else float(getattr(args.opt, "lambda_reg", 0.0)),
            geometry_conflict_policy=cli.geometry_conflict_policy,
            geometry_conflict_camera_weight=cli.geometry_conflict_camera_weight,
            lambda_normal=cli.lambda_normal if cli.lambda_normal is not None else float(getattr(args.opt, "lambda_normal", 0.0)),
            warmup_camera_depth_normal_weight=cli.warmup_camera_depth_normal_weight,
            warmup_disable_other_regularizers=cli.warmup_disable_other_regularizers,
            initial_gaussian_state=base_state,
            tb_writer=tb_writer,
            cycle_ckpt_dir=os.path.join(out_dir, "cycle_ckpts"),
            save_cycle_every=cli.save_cycle_every,
            resume_cycle_ckpt=cli.resume_cycle_ckpt,
            resume_cycle_ckpt_scene_only=cli.resume_cycle_ckpt_scene_only,
            reset_gaussians_every=cli.reset_gaussians_every,
            disable_depth_after_cycle=cli.disable_depth_after_cycle,
            rgb_only_updates_color=cli.rgb_only_updates_color,
            lambda_depth=cli.lambda_depth if cli.lambda_depth is not None else float(getattr(args.opt, "lambda_depth_l1", 1.0)),
            matcher_pose_update=cli.matcher_pose_update,
            matcher_color_supervision=cli.matcher_color_supervision,
            matcher_color_weight=cli.matcher_color_weight,
            camera_rgb_pose_only=cli.camera_rgb_pose_only,
            color_warmup_cycles=cli.color_warmup_cycles,
            initialize_pose_from_matcher=cli.initialize_pose_from_matcher,
            post_init_supervision_blur_kernel=cli.post_init_supervision_blur_kernel,
            post_init_supervision_blur_sigma=cli.post_init_supervision_blur_sigma,
            post_init_supervision_blur_warmup_only=cli.post_init_supervision_blur_warmup_only,
            freeze_gaussians_after_color_warmup=cli.freeze_gaussians_after_color_warmup,
            freeze_gaussians_after_cycle=cli.freeze_gaussians_after_cycle,
            pose_step_at_cycle_end=cli.pose_step_at_cycle_end,
            pose_step_interval_iters=cli.pose_step_interval_iters,
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
            matcher_minima_root=cli.matcher_minima_root or getattr(pose_cfg, "matcher_minima_root", None),
            matcher_minima_ckpt=cli.matcher_minima_ckpt or getattr(pose_cfg, "matcher_minima_ckpt", None),
            matcher_resize=cli.matcher_resize if cli.matcher_resize is not None else int(getattr(pose_cfg, "matcher_resize", 832)),
            matcher_match_threshold=cli.matcher_match_threshold if cli.matcher_match_threshold is not None else float(getattr(pose_cfg, "matcher_match_threshold", 0.2)),
            matcher_render_intrinsics_fx_scale=cli.matcher_render_intrinsics_fx_scale if cli.matcher_render_intrinsics_fx_scale is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_fx_scale", 1.0)),
            matcher_render_intrinsics_fy_scale=cli.matcher_render_intrinsics_fy_scale if cli.matcher_render_intrinsics_fy_scale is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_fy_scale", 1.0)),
            matcher_render_intrinsics_cx_offset=cli.matcher_render_intrinsics_cx_offset if cli.matcher_render_intrinsics_cx_offset is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_cx_offset", 0.0)),
            matcher_render_intrinsics_cy_offset=cli.matcher_render_intrinsics_cy_offset if cli.matcher_render_intrinsics_cy_offset is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_cy_offset", 0.0)),
            matcher_lidar_nn_max_distance=cli.matcher_lidar_nn_max_distance if cli.matcher_lidar_nn_max_distance is not None else float(getattr(pose_cfg, "matcher_lidar_nn_max_distance", 0.0)),
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
            matcher_dense_mode=cli.matcher_dense_mode,
            matcher_dense_stride=cli.matcher_dense_stride,
            matcher_dense_cert_threshold=cli.matcher_dense_cert_threshold,
            matcher_dense_color_cert_threshold=cli.matcher_dense_color_cert_threshold,
            match_once_per_cycle=not cli.no_match_once_per_cycle,
            cross_frame_weight=cli.cross_frame_weight,
            flow_proj_weight=cli.flow_proj_weight,
            flow_proj_max_offset=cli.flow_proj_max_offset,
            camera_aux_depth_render_backend=resolved_camera_aux_depth_render_backend,
            camera_aux_depth_mode=resolved_camera_aux_depth_mode,
            pure_pnp_iters=cli.pure_pnp_iters,
            pure_pnp_drop_first_n_frames=cli.pure_pnp_drop_first_n_frames,
            pure_pnp_drop_last_n_frames=cli.pure_pnp_drop_last_n_frames,
            pure_pnp_photo_weight=cli.pure_pnp_photo_weight,
            pure_pnp_photo_match_radius_px=cli.pure_pnp_photo_match_radius_px,
            pure_pnp_photo_min_matches=cli.pure_pnp_photo_min_matches,
            pure_pnp_photo_max_offset=cli.pure_pnp_photo_max_offset,
            pure_pnp_flow_weight=cli.pure_pnp_flow_weight,
            pure_pnp_flow_match_radius_px=cli.pure_pnp_flow_match_radius_px,
            pure_pnp_flow_min_matches=cli.pure_pnp_flow_min_matches,
            pure_pnp_depth_weight=cli.pure_pnp_depth_weight,
            pure_pnp_depth_match_radius_px=cli.pure_pnp_depth_match_radius_px,
            pure_pnp_depth_min_matches=cli.pure_pnp_depth_min_matches,
            pure_pnp_depth_render_backend=resolved_pure_pnp_depth_render_backend,
            pure_pnp_depth_mode=resolved_pure_pnp_depth_mode,
            pure_pnp_temporal_semidense_stride=cli.pure_pnp_temporal_semidense_stride,
            pure_pnp_temporal_semidense_max_points=cli.pure_pnp_temporal_semidense_max_points,
            pure_pnp_temporal_gradient_scale=cli.pure_pnp_temporal_gradient_scale,
            pure_pnp_init_depth_strat_bins=cli.pure_pnp_init_depth_strat_bins,
            pure_pnp_init_depth_strat_max_points_per_bin=cli.pure_pnp_init_depth_strat_max_points_per_bin,
            pure_pnp_far_depth_boost=cli.pure_pnp_far_depth_boost,
            pure_pnp_far_depth_start_percentile=cli.pure_pnp_far_depth_start_percentile,
            pure_pnp_post_init_rgb_blur_kernel=cli.pure_pnp_post_init_rgb_blur_kernel,
            pure_pnp_post_init_rgb_blur_sigma=cli.pure_pnp_post_init_rgb_blur_sigma,
            periodic_pure_pnp_interval_cycles=cli.periodic_pure_pnp_interval_cycles,
            disable_pose_grad_updates=cli.disable_pose_grad_updates,
            pure_pnp_history_path=os.path.join(out_dir, "pure_pnp_history.jsonl"),
            pure_pnp_match_diagnostics_dir=(
                os.path.join(out_dir, "pure_pnp_match_diagnostics")
                if cli.pure_pnp_export_match_diagnostics or cli.pure_pnp_residual_filter_mad_scale > 0.0
                else None
            ),
            pure_pnp_residual_filter_mad_scale=cli.pure_pnp_residual_filter_mad_scale,
            pure_pnp_residual_filter_min_keep_ratio=cli.pure_pnp_residual_filter_min_keep_ratio,
            pure_pnp_residual_filter_min_keep_per_frame=cli.pure_pnp_residual_filter_min_keep_per_frame,
            pure_pnp_residual_filter_max_reproj_error=cli.pure_pnp_residual_filter_max_reproj_error,
            pure_pnp_residual_filter_start_step=cli.pure_pnp_residual_filter_start_step,
            pure_pnp_single_frame_prefilter_min_inliers=cli.pure_pnp_single_frame_prefilter_min_inliers,
            pure_pnp_single_frame_prefilter_min_inlier_ratio=cli.pure_pnp_single_frame_prefilter_min_inlier_ratio,
            pure_pnp_frame_disagreement_mad_scale=cli.pure_pnp_frame_disagreement_mad_scale,
            pure_pnp_frame_disagreement_min_keep_ratio=cli.pure_pnp_frame_disagreement_min_keep_ratio,
            pure_pnp_frame_disagreement_min_keep_frames=cli.pure_pnp_frame_disagreement_min_keep_frames,
            pure_pnp_frame_disagreement_apply_max_dropped_frames=cli.pure_pnp_frame_disagreement_apply_max_dropped_frames,
            pure_pnp_filter_shared_ransac_outliers=cli.pure_pnp_filter_shared_ransac_outliers,
            pure_pnp_gt_reproj_filter_quantile=cli.pure_pnp_gt_reproj_filter_quantile,
            pure_pnp_gt_reproj_filter_min_keep_per_frame=cli.pure_pnp_gt_reproj_filter_min_keep_per_frame,
            pure_pnp_gt_soft_weight_mode=cli.pure_pnp_gt_soft_weight_mode,
            pure_pnp_gt_soft_weight_translation_alpha=cli.pure_pnp_gt_soft_weight_translation_alpha,
            pure_pnp_gt_pose_residual_weight=cli.pure_pnp_gt_pose_residual_weight,
            pure_pnp_optimize_rotation=not cli.pure_pnp_freeze_rotation,
            pure_pnp_optimize_translation=not cli.pure_pnp_freeze_translation,
            pure_pnp_solver_backend=cli.pure_pnp_solver_backend,
            pure_pnp_gt_weight_analysis_dir=(
                os.path.join(out_dir, "pure_pnp_gt_weight_analysis")
                if cli.pure_pnp_export_gt_weight_analysis
                else None
            ),
        )
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

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
        matcher_minima_root=cli.matcher_minima_root or getattr(pose_cfg, "matcher_minima_root", None),
        matcher_minima_ckpt=cli.matcher_minima_ckpt or getattr(pose_cfg, "matcher_minima_ckpt", None),
        matcher_resize=cli.matcher_resize if cli.matcher_resize is not None else int(getattr(pose_cfg, "matcher_resize", 832)),
        matcher_match_threshold=cli.matcher_match_threshold if cli.matcher_match_threshold is not None else float(getattr(pose_cfg, "matcher_match_threshold", 0.2)),
            matcher_render_intrinsics_fx_scale=cli.matcher_render_intrinsics_fx_scale if cli.matcher_render_intrinsics_fx_scale is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_fx_scale", 1.0)),
            matcher_render_intrinsics_fy_scale=cli.matcher_render_intrinsics_fy_scale if cli.matcher_render_intrinsics_fy_scale is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_fy_scale", 1.0)),
            matcher_render_intrinsics_cx_offset=cli.matcher_render_intrinsics_cx_offset if cli.matcher_render_intrinsics_cx_offset is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_cx_offset", 0.0)),
            matcher_render_intrinsics_cy_offset=cli.matcher_render_intrinsics_cy_offset if cli.matcher_render_intrinsics_cy_offset is not None else float(getattr(pose_cfg, "matcher_render_intrinsics_cy_offset", 0.0)),
            matcher_lidar_nn_max_distance=cli.matcher_lidar_nn_max_distance if cli.matcher_lidar_nn_max_distance is not None else float(getattr(pose_cfg, "matcher_lidar_nn_max_distance", 0.0)),
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
        camera_aux_depth_render_backend=resolved_camera_aux_depth_render_backend,
        camera_aux_depth_mode=resolved_camera_aux_depth_mode,
    )
    print(green(f"[Calib] Visualizations saved to: {os.path.join(out_dir, 'visualizations')}"))
    lidar_viz_dir = _save_lidar_supervision_depth_visualizations(
        out_dir=out_dir,
        gaussians=gaussians,
        scene=scene,
        args=args,
    )
    if lidar_viz_dir is not None:
        print(green(f"[Calib] LiDAR depth visualizations saved to: {lidar_viz_dir}"))
    gt_camera_viz_dir = _save_gt_camera_depth_visualizations(
        out_dir=out_dir,
        gaussians=gaussians,
        pose_correction=pose_correction,
        cam_cameras=cam_cameras,
        cam_images=cam_images,
        scene=scene,
        gt_l2c_R=gt_l2c_R,
        gt_l2c_T=gt_l2c_T,
        args=args,
        camera_aux_depth_render_backend=resolved_camera_aux_depth_render_backend,
        camera_aux_depth_mode=resolved_camera_aux_depth_mode,
    )
    if gt_camera_viz_dir is not None:
        print(green(f"[Calib] GT camera depth visualizations saved to: {gt_camera_viz_dir}"))

    # ── Save result ───────────────────────────────────────────
    if final_R is None:
        final_R = _effective_R(pose_correction)
    out_path = os.path.join(out_dir, "best_rotation.npz")
    final_T = _effective_T(pose_correction)
    np.savez(out_path,
             final_R=final_R.detach().cpu().numpy(),
             final_t=final_T.detach().cpu().numpy(),
             gt_R=gt_l2c_R.detach().cpu().numpy(),
             gt_t=gt_l2c_T.detach().cpu().numpy(),
             init_rot_deg=cli.init_rot_deg)
    print(green(f"[Calib] Saved to: {out_path}"))


if __name__ == "__main__":
    main()
