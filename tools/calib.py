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
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss, ssim


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

    _apply_gaussian_freezes()

    # ── Pose optimizer: stage 1 ────────────────────────────────
    optimizer_param_groups = []
    if not freeze_rotation:
        optimizer_param_groups.append(
            {"params": [pose_correction.delta_rotations_quat], "lr": rotation_lr}
        )
    if not pose_correction.use_gt_translation and not two_stage:
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

        for it in range(1, iters_per_cycle + 1):
            global_iter += 1
            gaussian_iter += 1
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
            if frame in cam_cameras:
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
                    loss_rgb = lambda_rgb * (
                        (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
                    )
                    psnr_accum += psnr(pred_chw, gt_chw).item()
                    psnr_count += 1

            total_loss = loss_depth + loss_rgb
            if freeze_covariance and not freeze_gaussians:
                # LiDAR-only covariance: RGB backward first, zero those grads, then depth
                _COV_OPA = ["_scaling", "_rotation", "_opacity", "_opacity_cam"]
                if loss_rgb.requires_grad:
                    loss_rgb.backward()  # separate graph from depth render, no retain needed
                    for attr in _COV_OPA:
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

            if not freeze_gaussians:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if cycle > warmup_cycles:
                pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

        # ── Fold delta into base once per cycle ────────────────
        if cycle > warmup_cycles:
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
        psnr_accum = 0.0
        psnr_count = 0
        loss_accum       = 0.0
        loss_depth_accum = 0.0
        loss_rgb_accum   = 0.0

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
            f"[d={avg_depth:.4f}  rgb={avg_rgb:.4f}]  lr={cur_rot_lr:.2e}"
        ))

        if tb_writer is not None:
            tb_writer.add_scalar("calib/rot_err_deg",  rot_err,  cycle)
            tb_writer.add_scalar("calib/trans_err_m",  T_err,    cycle)
            tb_writer.add_scalar("calib/psnr_db",      avg_psnr, cycle)
            tb_writer.add_scalar("calib/loss",         avg_loss, cycle)
            tb_writer.add_scalar("calib/loss_depth",   avg_depth, cycle)
            tb_writer.add_scalar("calib/loss_rgb",     avg_rgb,  cycle)

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
        lr_patience=cli.lr_patience,
        lr_factor=cli.lr_factor,
        lr_min=cli.lr_min,
        initial_gaussian_state=base_state,
        tb_writer=tb_writer,
        cycle_ckpt_dir=os.path.join(out_dir, "cycle_ckpts"),
        save_cycle_every=cli.save_cycle_every,
        resume_cycle_ckpt=cli.resume_cycle_ckpt,
        reset_gaussians_every=cli.reset_gaussians_every,
        disable_depth_after_cycle=cli.disable_depth_after_cycle,
    )

    # ── Save result ───────────────────────────────────────────
    out_path = os.path.join(out_dir, "best_rotation.npz")
    np.savez(out_path,
             final_R=final_R.detach().cpu().numpy(),
             gt_R=gt_l2c_R.detach().cpu().numpy(),
             init_rot_deg=cli.init_rot_deg)
    print(green(f"[Calib] Saved to: {out_path}"))


if __name__ == "__main__":
    main()
