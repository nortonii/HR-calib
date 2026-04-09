#!/usr/bin/env python3
"""Calibration training loop for LiDAR-camera extrinsic calibration via 3DGS.

Two modes
---------
  ``reset`` — full Gaussian parameter restore at start of every cycle.
  ``noise`` — continuous training (no reset); default mode.

Usage
-----
python tools/reset_prim_calib.py \\
    -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \\
    --init_trans_xyz 0.0718 0.1314 0.0960 \\
    --total_cycles 1200 --iters_per_cycle 50 \\
    --translation_start_cycle 600 \\
    --no_freeze_stage2 \\
    --rotation_lr 0.01 \\
    --warmup_cycles 4 \\
    --output_dir output/noise_inject_calib/my_exp
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
from lib.gaussian_renderer import raytracing
from lib.gaussian_renderer.camera_render import render_camera
from lib.scene.camera_pose_correction import CameraPoseCorrection
from lib.scene.cameras import Camera
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
# Main training loop with periodic Gaussian reset
# ─────────────────────────────────────────────────────────────

def run_reset_prim_calib(
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    base_state: dict,
    gt_l2c_R: torch.Tensor,
    args,
    total_cycles: int = 100,
    iters_per_cycle: int = 100,
    rotation_lr: float = 2e-3,
    freeze_gaussians: bool = False,
    lambda_rgb: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_dssim: float = 0.2,
    tb_writer=None,
):
    background = torch.tensor([0, 0, 1], device="cuda").float()
    frame_ids_train = sorted(scene.train_lidar.train_frames)

    # Pose optimizer — never reset across cycles
    pose_optimizer = torch.optim.Adam(
        [{"params": [pose_correction.delta_rotations_quat], "lr": rotation_lr}]
    )

    init_R = _effective_R(pose_correction)
    init_err = _rotation_error_deg(init_R, gt_l2c_R)
    print(blue(f"[ResetPrim] Init rotation error vs GT: {init_err:.4f}°"))
    print(blue(f"[ResetPrim] total_cycles={total_cycles}, iters_per_cycle={iters_per_cycle}, "
               f"rotation_lr={rotation_lr}, freeze_gaussians={freeze_gaussians}"))

    global_iter = 0
    frame_stack = []

    for cycle in range(1, total_cycles + 1):
        # ── Reset Gaussians at the start of each cycle ──────────
        restore_gaussian_state(gaussians, base_state, args)
        if freeze_gaussians:
            for attr in _GAUSSIAN_ATTRS:
                p = getattr(gaussians, attr, None)
                if p is not None:
                    p.requires_grad_(False)

        psnr_accum = 0.0
        psnr_count = 0
        loss_accum = 0.0

        for it in range(1, iters_per_cycle + 1):
            global_iter += 1
            if not freeze_gaussians:
                gaussians.update_learning_rate(it)

            if not frame_stack:
                frame_stack = list(frame_ids_train)
                random.shuffle(frame_stack)
            frame = frame_stack.pop()

            # ── LiDAR render ──────────────────────────────────
            # When Gaussians are frozen, skip LiDAR loss (no Gaussian grad needed).
            loss_depth = torch.tensor(0.0, device="cuda")
            if not freeze_gaussians:
                render_pkg = raytracing(
                    frame, [gaussians], scene.train_lidar, background, args, depth_only=True
                )
                depth    = render_pkg["depth"].squeeze(-1)
                gt_mask  = scene.train_lidar.get_mask(frame).cuda()
                dyn_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
                static_mask = gt_mask & ~dyn_mask
                gt_depth = scene.train_lidar.get_depth(frame).cuda()
                loss_depth = lambda_depth * l1_loss(depth[static_mask], gt_depth[static_mask])

            # ── Camera render (pose grad always active) ───────
            loss_rgb = torch.tensor(0.0, device="cuda")
            if frame in cam_cameras:
                cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
                camera = cam_cameras[frame].cuda()
                gt_rgb = cam_images[frame].cuda()

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
            total_loss.backward()
            loss_accum += total_loss.item()

            # Gaussian step (skip when frozen)
            if not freeze_gaussians:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Pose step + fold delta into base
            pose_optimizer.step()
            pose_correction.update_extrinsics()
            pose_optimizer.zero_grad(set_to_none=True)

        # ── End-of-cycle logging ───────────────────────────────
        eff_R    = _effective_R(pose_correction)
        rot_err  = _rotation_error_deg(eff_R, gt_l2c_R)
        avg_psnr = psnr_accum / psnr_count if psnr_count > 0 else 0.0
        avg_loss = loss_accum / iters_per_cycle

        print(yellow(
            f"  Cycle {cycle:3d}/{total_cycles}  rot_err={rot_err:.4f}°  "
            f"PSNR={avg_psnr:.2f} dB  loss={avg_loss:.5f}"
        ))

        if tb_writer is not None:
            tb_writer.add_scalar("reset_prim/rot_err_deg", rot_err,  cycle)
            tb_writer.add_scalar("reset_prim/psnr_db",     avg_psnr, cycle)
            tb_writer.add_scalar("reset_prim/loss",        avg_loss, cycle)

    final_R   = _effective_R(pose_correction)
    final_err = _rotation_error_deg(final_R, gt_l2c_R)
    print(green(f"\n[ResetPrim] Init error : {init_err:.4f}°"))
    print(green(f"[ResetPrim] Final error: {final_err:.4f}°"))
    print(green(f"[ResetPrim] Improvement: {init_err - final_err:+.4f}°"))
    return final_R


# ─────────────────────────────────────────────────────────────
# Noise-injection helpers
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
    lambda_reproj: float = 0.0,
    lambda_cross_view: float = 0.0,
    lambda_virtual_frame: float = 0.0,
    reset_gaussians_stage2: bool = False,
    no_freeze_stage2: bool = False,
    initial_gaussian_state: dict = None,
    dc_only: bool = False,
    tb_writer=None,
    cycle_ckpt_dir: str = None,
    save_cycle_every: int = 5,
    resume_cycle_ckpt: str = None,
    num_virtual_frames: int = 1,
    vf_neighbor_dist: int = 5,
    vf_perturb_rot_deg: float = 5.0,
    vf_perturb_trans_m: float = 0.3,
    vf_min_overlap: int = 200,
):
    """Continuous calibration training loop.

    When ``pose_correction.use_gt_translation`` is False, ``delta_translations``
    is added to the optimizer so both rotation and translation are calibrated.

    When ``freeze_xyz`` is True, Gaussian mean positions are frozen (requires_grad
    disabled, LR zeroed) so they cannot absorb translation errors.

    ``translation_start_cycle`` implements a two-stage strategy:
      Stage 1 (cycles 1..translation_start_cycle): rotation-only optimisation.
      Stage 2 (cycles translation_start_cycle+1..total): translation optimisation enabled.
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

    # ── DC-only mode: freeze high-order SH to prevent view-dependent colour ──
    # Higher-order SH coefficients allow Gaussians to learn view-dependent
    # colours.  During stage 1 they will adapt to look correct from the WRONG
    # camera pose, killing the T gradient in stage 2.  Forcing active_sh_degree=0
    # (DC only) blocks this null-space throughout calibration.  No data leakage.
    if dc_only:
        gaussians.active_sh_degree = 0
        _rest_attrs = ["_features_rest", "_features_rgb_rest"]
        for attr in _rest_attrs:
            p = getattr(gaussians, attr, None)
            if p is not None:
                p.requires_grad_(False)
        for pg in gaussians.optimizer.param_groups:
            for attr in _rest_attrs:
                rp = getattr(gaussians, attr, None)
                if rp is not None and any(p is rp for p in pg["params"]):
                    pg["lr"] = 0.0
        print(blue("[NoiseInject] DC-ONLY mode: active_sh_degree=0, "
                   "features_rest frozen — no view-dependent colour compensation"))

    _COLOR_ATTRS = ["_features_dc", "_features_rgb_dc", "_features_rest", "_features_rgb_rest"]

    def _apply_gaussian_freezes():
        """Freeze xyz and/or colors on the Gaussian + its Adam optimizer."""
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
            print(blue("[NoiseInject] Gaussian covariance FROZEN (scale+rot)"))

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
    # Monitors rot_err: if no improvement for `lr_patience` cycles, halve the LR.
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

    if two_stage:
        freeze_desc = "NO FREEZE (Gaussians continue training)" if no_freeze_stage2 else "freeze xyz+colors"
        print(blue(f"[NoiseInject] TWO-STAGE mode: rotation-only until cycle {translation_start_cycle}, "
                   f"then {freeze_desc} + optimise translation"))

    global_iter   = 0
    frame_stack   = []
    psnr_accum    = 0.0
    psnr_count    = 0
    loss_accum        = 0.0
    loss_depth_accum  = 0.0
    loss_rgb_accum    = 0.0
    loss_reproj_accum = 0.0
    loss_cross_accum  = 0.0
    loss_virt_accum   = 0.0
    # Best-T tracking: save the delta_T that achieved the lowest T_err
    best_T_err     = float("inf")
    best_delta_T   = pose_correction.delta_translations.data.clone()
    start_cycle    = 1

    # ── Resume from cycle checkpoint ──────────────────────────
    if resume_cycle_ckpt is not None:
        print(blue(f"[NoiseInject] Resuming from cycle checkpoint: {resume_cycle_ckpt}"))
        ckpt = torch.load(resume_cycle_ckpt, weights_only=False, map_location="cuda")
        restore_gaussian_state(gaussians, ckpt["gaussian_state"], args)
        pose_correction.delta_rotations_quat.data.copy_(
            ckpt["delta_rotations_quat"].to("cuda"))
        pose_correction.delta_translations.data.copy_(
            ckpt["delta_translations"].to("cuda"))
        pose_correction.update_extrinsics()
        global_iter   = ckpt["global_iter"]
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
            # ── Option A: Reset Gaussians to original checkpoint state ──
            # The Gaussians adapted their SH colors during stage 1 to look
            # correct even from the wrong camera pose.  By resetting to the
            # original (pre-calibration) state and freezing everything, the
            # RGB rendering loss becomes a clean photometric signal for T.
            if reset_gaussians_stage2 and initial_gaussian_state is not None:
                restore_gaussian_state(gaussians, initial_gaussian_state, args)
                # Freeze ALL Gaussian parameters — only delta_T is free
                for attr in _GAUSSIAN_ATTRS:
                    p = getattr(gaussians, attr, None)
                    if p is not None:
                        p.requires_grad_(False)
                for pg in gaussians.optimizer.param_groups:
                    pg["lr"] = 0.0
                print(blue(f"[NoiseInject] Stage 2 activated at cycle {cycle}: "
                           f"Gaussians RESET to original + ALL params FROZEN, "
                           f"translation optimizer added"))
            else:
                # ── Option B: Freeze xyz + colors only (original behaviour) ──
                # Freeze Gaussian xyz so they cannot absorb translation errors
                if not no_freeze_stage2:
                    xyz_param = getattr(gaussians, "_xyz", None)
                    if xyz_param is not None:
                        xyz_param.requires_grad_(False)
                    for pg in gaussians.optimizer.param_groups:
                        if any(p is xyz_param for p in pg["params"]):
                            pg["lr"] = 0.0
                    # Freeze view-dependent SH colour features: they can compensate for
                    # small camera shifts and kill the translation gradient signal.
                    _color_attrs = [
                        "_features_rest", "_features_rgb_rest",
                        "_features_dc", "_features_rgb_dc",
                    ]
                    for attr in _color_attrs:
                        param = getattr(gaussians, attr, None)
                        if param is not None:
                            param.requires_grad_(False)
                    for pg in gaussians.optimizer.param_groups:
                        for attr in _color_attrs:
                            cparam = getattr(gaussians, attr, None)
                            if cparam is not None and any(p is cparam for p in pg["params"]):
                                pg["lr"] = 0.0
                freeze_msg = "xyz+colors FROZEN" if not no_freeze_stage2 else "NO FREEZE (Gaussians continue training)"
                print(blue(f"[NoiseInject] Stage 2 activated at cycle {cycle}: "
                           f"{freeze_msg}, translation optimizer added"))
            # Add translation to pose optimizer (fresh Adam state)
            pose_optimizer.add_param_group(
                {"params": [pose_correction.delta_translations], "lr": translation_lr}
            )
            # Reset scheduler so patience restarts fresh for stage 2
            if pose_lr_scheduler is not None:
                pose_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    pose_optimizer, mode="min", factor=lr_factor,
                    patience=lr_patience, min_lr=lr_min, threshold=1e-3,
                )

        for it in range(1, iters_per_cycle + 1):
            global_iter += 1
            gaussians.update_learning_rate(global_iter)

            if not frame_stack:
                frame_stack = list(frame_ids_train)
                random.shuffle(frame_stack)
            frame = frame_stack.pop()

            # ── LiDAR depth loss ───────────────────────────────
            render_pkg = raytracing(
                frame, [gaussians], scene.train_lidar, background, args, depth_only=True
            )
            depth       = render_pkg["depth"].squeeze(-1)
            gt_mask     = scene.train_lidar.get_mask(frame).cuda()
            dyn_mask    = scene.train_lidar.get_dynamic_mask(frame).cuda()
            static_mask = gt_mask & ~dyn_mask
            gt_depth    = scene.train_lidar.get_depth(frame).cuda()
            loss_depth  = lambda_depth * l1_loss(depth[static_mask], gt_depth[static_mask])

            # ── Camera RGB + LiDAR reprojection losses ────────
            loss_rgb    = torch.tensor(0.0, device="cuda")
            loss_reproj = torch.tensor(0.0, device="cuda")
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

                    # ── LiDAR→Camera reprojection loss ────────────
                    # Projects GT LiDAR hit-points into camera using the
                    # current (differentiable) l2c extrinsic, then compares
                    # sampled rendered vs GT pixel colours.  The projection
                    # Jacobian d(u,v)/dT_l2c provides a direct geometric
                    # gradient for the translation that is absent from the
                    # standard pixel-space rendering loss.
                    # Only applied in stage 2 (when T is being optimised)
                    # because in stage 1 the wrong T produces misleading R
                    # gradients that interfere with rotation convergence.
                    if lambda_reproj > 0.0 and (not two_stage or stage2_active):
                        gt_mask_rp  = scene.train_lidar.get_mask(frame).cuda()
                        dyn_mask_rp = scene.train_lidar.get_dynamic_mask(frame).cuda()
                        valid_mask_rp = gt_mask_rp & ~dyn_mask_rp  # (H, W)

                        rays_o_rp, rays_d_rp = scene.train_lidar.get_range_rays(frame)
                        gt_depth_rp = scene.train_lidar.get_depth(frame).cuda()
                        P_world_rp  = rays_o_rp + rays_d_rp * gt_depth_rp.unsqueeze(-1)

                        # World → LiDAR sensor frame (constant, no gradient needed)
                        s2w_rp  = scene.train_lidar.sensor2world[frame].cuda()
                        R_l2w   = s2w_rp[:3, :3]
                        T_l2w   = s2w_rp[:3, 3]
                        P_flat  = P_world_rp.reshape(-1, 3).detach()
                        P_lidar = (P_flat - T_l2w) @ R_l2w  # (N, 3)

                        # Current l2c extrinsic – decoupled formula, grad wrt delta_T
                        pidx    = pose_correction._pose_index(frame)
                        dT_rp   = pose_correction.delta_translations[pidx]
                        bT_rp   = pose_correction.base_lidar_to_camera_translation[0].cuda()
                        T_l2c   = bT_rp + dT_rp                             # (3,) grad
                        dq_rp   = F.normalize(
                            pose_correction.delta_rotations_quat[pidx], dim=0
                        )
                        bq_rp   = pose_correction.base_lidar_to_camera_quat[0].cuda()
                        eff_q   = quaternion_multiply(dq_rp, bq_rp)
                        R_l2c   = quaternion_to_matrix(eff_q)                # (3,3) grad

                        # LiDAR → camera: grad flows to both R and T
                        P_cam   = P_lidar @ R_l2c.T + T_l2c                 # (N, 3)

                        valid_z  = P_cam[:, 2] > 0.1
                        valid_rp = valid_mask_rp.reshape(-1) & valid_z

                        if valid_rp.sum().item() > 100:
                            P_v    = P_cam[valid_rp]                        # (M, 3)
                            W_img  = float(cam_cameras[frame].image_width)
                            H_img  = float(cam_cameras[frame].image_height)
                            fx_rp  = W_img / (2.0 * math.tan(cam_cameras[frame].FoVx * 0.5))
                            fy_rp  = H_img / (2.0 * math.tan(cam_cameras[frame].FoVy * 0.5))
                            cx_rp, cy_rp = W_img / 2.0, H_img / 2.0

                            u_rp = fx_rp * P_v[:, 0] / P_v[:, 2] + cx_rp  # (M,)
                            v_rp = fy_rp * P_v[:, 1] / P_v[:, 2] + cy_rp  # (M,)

                            in_fov = (
                                (u_rp >= 0) & (u_rp < W_img) &
                                (v_rp >= 0) & (v_rp < H_img)
                            )
                            if in_fov.sum().item() > 50:
                                u_ib = u_rp[in_fov]
                                v_ib = v_rp[in_fov]
                                # Normalise to [-1, 1] for grid_sample
                                u_n  = (u_ib / (W_img - 1)) * 2.0 - 1.0
                                v_n  = (v_ib / (H_img - 1)) * 2.0 - 1.0
                                grid_rp = torch.stack(
                                    [u_n, v_n], dim=-1
                                )[None, None]  # (1,1,K,2)

                                pred_4d = (cam_render["rgb"].clamp(0, 1)
                                           .permute(2, 0, 1).unsqueeze(0))  # (1,3,H,W)
                                gt_4d   = gt_rgb.permute(2, 0, 1).unsqueeze(0)

                                # grid_rp depends on T through projection →
                                # provides geometric gradient for T_l2c
                                sp = F.grid_sample(
                                    pred_4d, grid_rp,
                                    mode='bilinear', align_corners=True,
                                    padding_mode='zeros',
                                )  # (1,3,1,K)
                                sg = F.grid_sample(
                                    gt_4d, grid_rp,
                                    mode='bilinear', align_corners=True,
                                    padding_mode='zeros',
                                )  # (1,3,1,K)
                                sp = sp.squeeze(0).squeeze(1).permute(1, 0)  # (K,3)
                                sg = sg.squeeze(0).squeeze(1).permute(1, 0)  # (K,3)
                                loss_reproj = lambda_reproj * l1_loss(
                                    sp, sg.detach()
                                )

            # ── A: Pure geometric cross-view consistency ──────────────────
            # GT_i(π_i(T,P)) vs GT_j(π_j(T,P)) — no Gaussian rendering.
            # Gaussians cannot compensate: gradient is purely from the
            # projection Jacobian ∂(u,v)/∂T applied to fixed GT images.
            loss_cross_view = torch.tensor(0.0, device="cuda")
            if lambda_cross_view > 0.0 and (not two_stage or stage2_active):
                frame_ids_list = list(frame_ids_train)
                j_cands = [fid for fid in frame_ids_list
                           if 1 <= abs(fid - frame) <= 3 and fid in cam_cameras]
                if j_cands and frame in cam_cameras:
                    j_cv = random.choice(j_cands)
                    gt_mask_cv    = scene.train_lidar.get_mask(frame).cuda()
                    dyn_mask_cv   = scene.train_lidar.get_dynamic_mask(frame).cuda()
                    valid_cv_flat = (gt_mask_cv & ~dyn_mask_cv).reshape(-1)

                    rays_o_cv, rays_d_cv = scene.train_lidar.get_range_rays(frame)
                    gt_depth_cv  = scene.train_lidar.get_depth(frame).cuda()
                    P_world_cv   = (rays_o_cv + rays_d_cv * gt_depth_cv.unsqueeze(-1)
                                    ).reshape(-1, 3).detach()
                    P_valid_cv   = P_world_cv[valid_cv_flat]

                    pidx_cv  = pose_correction._pose_index(frame)
                    dT_cv    = pose_correction.delta_translations[pidx_cv]
                    bT_cv    = pose_correction.base_lidar_to_camera_translation[0].cuda()
                    T_l2c_cv = bT_cv + dT_cv
                    dq_cv    = F.normalize(
                        pose_correction.delta_rotations_quat[pidx_cv], dim=0)
                    bq_cv    = pose_correction.base_lidar_to_camera_quat[0].cuda()
                    R_l2c_cv = quaternion_to_matrix(quaternion_multiply(dq_cv, bq_cv))

                    def _proj_pts(fid, P_w):
                        s2w  = scene.train_lidar.sensor2world[fid].cuda()
                        P_l  = (P_w - s2w[:3, 3]) @ s2w[:3, :3]
                        P_c  = P_l @ R_l2c_cv.T + T_l2c_cv
                        cf   = cam_cameras[fid]
                        W_f  = float(cf.image_width);  H_f = float(cf.image_height)
                        fx_f = W_f / (2.0 * math.tan(cf.FoVx * 0.5))
                        fy_f = H_f / (2.0 * math.tan(cf.FoVy * 0.5))
                        vz   = P_c[:, 2] > 0.1
                        u    = fx_f * P_c[:, 0] / P_c[:, 2].clamp(min=0.1) + W_f/2.0
                        v    = fy_f * P_c[:, 1] / P_c[:, 2].clamp(min=0.1) + H_f/2.0
                        fov  = vz & (u>=0)&(u<W_f)&(v>=0)&(v<H_f)
                        return u, v, W_f, H_f, fov

                    u_i, v_i, W_i, H_i, fov_i = _proj_pts(frame,  P_valid_cv)
                    u_j, v_j, W_j, H_j, fov_j = _proj_pts(j_cv,   P_valid_cv)
                    valid_both_cv = fov_i & fov_j

                    if valid_both_cv.sum().item() > 50:
                        def _sample_gt_cv(img, u, v, W, H, mask):
                            u_n = (u[mask] / (W-1)) * 2.0 - 1.0
                            v_n = (v[mask] / (H-1)) * 2.0 - 1.0
                            g   = torch.stack([u_n, v_n], -1)[None, None]
                            s   = F.grid_sample(
                                img.cuda().permute(2,0,1).unsqueeze(0),
                                g, mode='bilinear', align_corners=True,
                                padding_mode='zeros')
                            return s.squeeze(0).squeeze(1).permute(1, 0)

                        c_i_cv = _sample_gt_cv(
                            cam_images[frame], u_i, v_i, W_i, H_i, valid_both_cv)
                        c_j_cv = _sample_gt_cv(
                            cam_images[j_cv],  u_j, v_j, W_j, H_j, valid_both_cv)
                        # Both projections carry ∂(u,v)/∂T — symmetric, no detach
                        loss_cross_view = lambda_cross_view * l1_loss(c_i_cv, c_j_cv)

            # ── B: Virtual-frame loss ─────────────────────────────────────
            # Interpolate sensor pose between frame i and a random neighbor j
            # (within ±vf_neighbor_dist frames), then add a small random
            # perturbation for diversity.  Interpolated poses have guaranteed
            # overlap; perturbation adds viewpoint variety without risking
            # camera flip.  Gradient flows via real-frame projection ∂(u,v)/∂T.
            loss_virtual = torch.tensor(0.0, device="cuda")
            if lambda_virtual_frame > 0.0 and (not two_stage or stage2_active):
                if frame in cam_cameras:
                    frame_ids_list_v = list(frame_ids_train)
                    j_cands_v = [fid for fid in frame_ids_list_v
                                 if 1 <= abs(fid - frame) <= vf_neighbor_dist
                                 and fid in cam_cameras]
                if j_cands_v and frame in cam_cameras:
                    # ── Real-frame LiDAR projection (T with grad) ──────────
                    pidx_v   = pose_correction._pose_index(frame)
                    dT_v     = pose_correction.delta_translations[pidx_v]
                    bT_v     = pose_correction.base_lidar_to_camera_translation[0].cuda()
                    T_l2c_vs = bT_v + dT_v          # grad ← T
                    T_l2c_v  = T_l2c_vs.detach()
                    dq_v     = F.normalize(
                        pose_correction.delta_rotations_quat[pidx_v], dim=0)
                    bq_v     = pose_correction.base_lidar_to_camera_quat[0].cuda()
                    R_l2c_v  = quaternion_to_matrix(
                        quaternion_multiply(dq_v, bq_v)).detach()

                    gt_mask_vf    = scene.train_lidar.get_mask(frame).cuda()
                    dyn_mask_vf   = scene.train_lidar.get_dynamic_mask(frame).cuda()
                    valid_vf_flat = (gt_mask_vf & ~dyn_mask_vf).reshape(-1)
                    rays_o_vf, rays_d_vf = scene.train_lidar.get_range_rays(frame)
                    gt_dep_vf  = scene.train_lidar.get_depth(frame).cuda()
                    P_world_vf = (rays_o_vf + rays_d_vf * gt_dep_vf.unsqueeze(-1)
                                  ).reshape(-1, 3).detach()
                    P_valid_vf = P_world_vf[valid_vf_flat]

                    s2w_vf   = scene.train_lidar.sensor2world[frame].cuda()
                    P_l_vf   = (P_valid_vf - s2w_vf[:3, 3]) @ s2w_vf[:3, :3]
                    P_c_real = P_l_vf @ R_l2c_v.T + T_l2c_vs  # grad ← T

                    cam_r  = cam_cameras[frame]
                    W_vf   = float(cam_r.image_width); H_vf = float(cam_r.image_height)
                    fx_vf  = W_vf/(2.0*math.tan(cam_r.FoVx*0.5))
                    fy_vf  = H_vf/(2.0*math.tan(cam_r.FoVy*0.5))
                    vz_r   = P_c_real[:, 2] > 0.1
                    u_real = fx_vf*P_c_real[:,0]/P_c_real[:,2].clamp(min=0.1)+W_vf/2
                    v_real = fy_vf*P_c_real[:,1]/P_c_real[:,2].clamp(min=0.1)+H_vf/2
                    fov_r  = vz_r&(u_real>=0)&(u_real<W_vf)&(v_real>=0)&(v_real<H_vf)

                    gt_full = F.grid_sample(
                        cam_images[frame].cuda().permute(2,0,1).unsqueeze(0),
                        torch.stack([(u_real/(W_vf-1))*2-1,
                                     (v_real/(H_vf-1))*2-1], -1)[None, None],
                        mode='bilinear', align_corners=True, padding_mode='zeros',
                    ).squeeze(0).squeeze(1).permute(1, 0)  # [N_pts, 3]

                    vf_losses = []
                    for _ in range(num_virtual_frames):
                        j_v = random.choice(j_cands_v)
                        t_v = random.uniform(0.1, 0.9)

                        # SLERP rotation + LERP translation between S_i and S_j
                        S_i  = scene.train_lidar.sensor2world[frame].cuda().float()
                        S_j  = scene.train_lidar.sensor2world[j_v].cuda().float()
                        T_virt_s = (1-t_v)*S_i[:3, 3] + t_v*S_j[:3, 3]
                        q_is = matrix_to_quaternion(S_i[:3, :3])
                        q_js = matrix_to_quaternion(S_j[:3, :3])
                        dot_v = (q_is * q_js).sum().clamp(-1, 1)
                        if dot_v < 0:
                            q_js, dot_v = -q_js, -dot_v
                        if dot_v > 0.9995:
                            q_virt = F.normalize((1-t_v)*q_is + t_v*q_js, dim=0)
                        else:
                            th_v   = dot_v.acos()
                            q_virt = ((math.sin((1-t_v)*th_v.item())*q_is
                                       + math.sin(t_v*th_v.item())*q_js) / th_v.sin())
                        R_virt_s = quaternion_to_matrix(q_virt)

                        # Small random perturbation on top of interpolated pose
                        if vf_perturb_rot_deg > 0 or vf_perturb_trans_m > 0:
                            p_axis  = F.normalize(torch.randn(3, device="cuda"), dim=0)
                            p_angle = random.uniform(0, vf_perturb_rot_deg * math.pi / 180)
                            K_p = torch.zeros(3, 3, device="cuda")
                            K_p[0,1], K_p[1,0] = -p_axis[2],  p_axis[2]
                            K_p[0,2], K_p[2,0] =  p_axis[1], -p_axis[1]
                            K_p[1,2], K_p[2,1] = -p_axis[0],  p_axis[0]
                            R_p = (torch.eye(3, device="cuda")
                                   + math.sin(p_angle) * K_p
                                   + (1 - math.cos(p_angle)) * (K_p @ K_p))
                            t_p = (F.normalize(torch.randn(3, device="cuda"), dim=0)
                                   * random.uniform(0, vf_perturb_trans_m))
                            R_virt_s = R_p @ R_virt_s
                            T_virt_s = R_p @ T_virt_s + t_p

                        # Build virtual camera world-to-cam transform
                        R_c2w_v = R_virt_s @ R_l2c_v.T
                        T_c2w_v = R_virt_s @ (-(R_l2c_v.T @ T_l2c_v)) + T_virt_s
                        R_w2c_v = R_c2w_v.T
                        T_w2c_v = -R_c2w_v.T @ T_c2w_v

                        virt_cam = Camera(
                            timestamp=0,
                            R=R_w2c_v.detach().cpu(),
                            T=T_w2c_v.detach().cpu(),
                            w=cam_r.image_width, h=cam_r.image_height,
                            FoVx=cam_r.FoVx, FoVy=cam_r.FoVy,
                        ).cuda()
                        render_v = render_camera(virt_cam, [gaussians], args, require_rgb=True)
                        if render_v["num_visible"] == 0:
                            continue

                        # Project LiDAR points into virtual camera frame
                        P_in_virt_s = (P_valid_vf - T_virt_s) @ R_virt_s
                        P_c_virt    = P_in_virt_s @ R_l2c_v.T + T_l2c_v
                        vz_virt  = P_c_virt[:, 2] > 0.1
                        u_virt   = fx_vf*P_c_virt[:,0]/P_c_virt[:,2].clamp(min=0.1)+W_vf/2
                        v_virt   = fy_vf*P_c_virt[:,1]/P_c_virt[:,2].clamp(min=0.1)+H_vf/2
                        fov_virt = (vz_virt&(u_virt>=0)&(u_virt<W_vf)
                                    &(v_virt>=0)&(v_virt<H_vf))

                        valid_vf2 = fov_r & fov_virt
                        if valid_vf2.sum().item() < vf_min_overlap:
                            continue

                        u_vn = (u_virt[valid_vf2]/(W_vf-1))*2-1
                        v_vn = (v_virt[valid_vf2]/(H_vf-1))*2-1
                        pred_vf2 = F.grid_sample(
                            render_v["rgb"].clamp(0,1).permute(2,0,1).unsqueeze(0),
                            torch.stack([u_vn, v_vn], -1)[None, None],
                            mode='bilinear', align_corners=True, padding_mode='zeros',
                        ).squeeze(0).squeeze(1).permute(1, 0)

                        gt_vf2 = gt_full[valid_vf2]
                        vf_losses.append(l1_loss(pred_vf2, gt_vf2.detach()))

                    if vf_losses:
                        loss_virtual = lambda_virtual_frame * torch.stack(vf_losses).mean()


            total_loss = loss_depth + loss_rgb + loss_reproj + loss_cross_view + loss_virtual
            total_loss.backward()
            loss_accum        += total_loss.item()
            loss_depth_accum  += loss_depth.item()
            loss_rgb_accum    += loss_rgb.item()
            loss_reproj_accum += loss_reproj.item()
            loss_cross_accum  += loss_cross_view.item()
            loss_virt_accum   += loss_virtual.item()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if cycle > warmup_cycles:
                pose_optimizer.step()
                pose_correction.update_extrinsics()
            pose_optimizer.zero_grad(set_to_none=True)

        # ── End-of-cycle logging ───────────────────────────────
        eff_R    = _effective_R(pose_correction)
        rot_err  = _rotation_error_deg(eff_R, gt_l2c_R)
        T_err    = _translation_error_m(pose_correction, gt_l2c_T)
        avg_psnr = psnr_accum / psnr_count if psnr_count > 0 else 0.0
        avg_loss       = loss_accum        / iters_per_cycle
        avg_depth      = loss_depth_accum  / iters_per_cycle
        avg_rgb        = loss_rgb_accum    / iters_per_cycle
        avg_reproj     = loss_reproj_accum / iters_per_cycle
        avg_cross      = loss_cross_accum  / iters_per_cycle
        avg_virt       = loss_virt_accum   / iters_per_cycle
        psnr_accum = 0.0
        psnr_count = 0
        loss_accum        = 0.0
        loss_depth_accum  = 0.0
        loss_rgb_accum    = 0.0
        loss_reproj_accum = 0.0
        loss_cross_accum  = 0.0
        loss_virt_accum   = 0.0

        # Track best translation seen so far
        if T_err < best_T_err:
            best_T_err = T_err
            best_delta_T = pose_correction.delta_translations.data.clone()

        # Build optional loss-breakdown suffix
        breakdown = ""
        if lambda_reproj > 0 or lambda_cross_view > 0 or lambda_virtual_frame > 0:
            parts = [f"d={avg_depth:.4f}", f"rgb={avg_rgb:.4f}"]
            if lambda_reproj > 0:        parts.append(f"rp={avg_reproj:.4f}")
            if lambda_cross_view > 0:    parts.append(f"cv={avg_cross:.4f}")
            if lambda_virtual_frame > 0: parts.append(f"vf={avg_virt:.4f}")
            breakdown = "  [" + "  ".join(parts) + "]"

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
            f"PSNR={avg_psnr:.2f} dB  loss={avg_loss:.5f}  lr={cur_rot_lr:.2e}{breakdown}"
        ))

        if tb_writer is not None:
            tb_writer.add_scalar("noise_inject/rot_err_deg",   rot_err,      cycle)
            tb_writer.add_scalar("noise_inject/trans_err_m",   T_err,        cycle)
            tb_writer.add_scalar("noise_inject/psnr_db",       avg_psnr,     cycle)
            tb_writer.add_scalar("noise_inject/loss",          avg_loss,     cycle)
            tb_writer.add_scalar("noise_inject/loss_depth",    avg_depth,    cycle)
            tb_writer.add_scalar("noise_inject/loss_rgb",      avg_rgb,      cycle)
            tb_writer.add_scalar("noise_inject/loss_reproj",   avg_reproj,   cycle)
            tb_writer.add_scalar("noise_inject/loss_cross",    avg_cross,    cycle)
            tb_writer.add_scalar("noise_inject/loss_virt",     avg_virt,     cycle)

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

    final_R   = _effective_R(pose_correction)
    final_err = _rotation_error_deg(final_R, gt_l2c_R)
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
    parser = argparse.ArgumentParser(description="Reset-primitive LiDAR-camera calibration")
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
    # ── Mode ─────────────────────────────────────────────────
    parser.add_argument("--mode",                default="noise",
                        choices=["reset", "noise"],
                        help="'reset': full Gaussian restore each cycle. "
                             "'noise': continuous training with periodic noise injection (default).")
    # ─────────────────────────────────────────────────────────
    parser.add_argument("--freeze_xyz",          action="store_true",
                        help="Freeze Gaussian mean positions (_xyz) during calibration. "
                             "Prevents xyz from absorbing translation errors, giving the "
                             "translation optimizer an uncontested depth gradient. "
                             "Recommended when optimising translation (use_gt_translation=False).")
    parser.add_argument("--freeze_colors",       action="store_true",
                        help="Freeze Gaussian SH color features during calibration. "
                             "Prevents view-dependent colors from absorbing camera-shift gradients.")
    parser.add_argument("--freeze_covariance",   action="store_true",
                        help="Freeze Gaussian scale and rotation (covariance) during calibration.")
    parser.add_argument("--translation_start_cycle", type=int, default=0,
                        help="Two-stage calibration: optimise rotation-only for this many cycles, "
                             "then freeze xyz and add translation optimisation. "
                             "0 = simultaneous (default). Recommended: ~75 (half of total_cycles). "
                             "Ignored when use_gt_translation=True.")
    parser.add_argument("--warmup_cycles",         type=int, default=0,
                        help="Freeze pose optimizer for this many cycles at the start, "
                             "letting Gaussians train first before pose updates begin. "
                             "Useful when training from scratch. Default: 0 (no warmup).")
    parser.add_argument("--freeze_rotation",       action="store_true",
                        help="Freeze rotation, optimise translation only. "
                             "Use with --init_rot_deg to fix a pre-calibrated rotation.")
    parser.add_argument("--lr_patience",          type=int,   default=0,
                        help="ReduceLROnPlateau patience (cycles). If rot_err doesn't improve "
                             "for this many cycles, pose rotation_lr is multiplied by lr_factor. "
                             "0 = disabled (default). Recommended: 50-100.")
    parser.add_argument("--lr_factor",            type=float, default=0.5,
                        help="Multiplicative factor for ReduceLROnPlateau. Default: 0.5.")
    parser.add_argument("--lr_min",               type=float, default=1e-5,
                        help="Minimum rotation_lr floor for ReduceLROnPlateau. Default: 1e-5.")
    parser.add_argument("--lambda_reproj",        type=float, default=0.0,
                        help="Weight of LiDAR→camera reprojection loss. Provides a direct "
                             "geometric gradient for translation via the projection Jacobian. "
                             "Applied alongside the standard RGB loss. Recommended: 0.5–1.0.")
    parser.add_argument("--reset_gaussians_stage2", action="store_true",
                        help="Reset Gaussians to the original checkpoint state at stage 2 "
                             "and freeze ALL Gaussian parameters. This gives the RGB loss a "
                             "clean photometric signal for translation because the original "
                             "Gaussians were trained with the correct extrinsic, so a wrong T "
                             "produces a real image residual.  Requires --translation_start_cycle > 0.")
    parser.add_argument("--no_freeze_stage2",        action="store_true",
                        help="At stage 2 activation, do NOT freeze xyz or color parameters. "
                             "Gaussians continue training freely alongside translation. "
                             "Only the translation optimizer is added.")
    parser.add_argument("--dc_only",                 action="store_true",
                        help="Force DC-only SH (active_sh_degree=0) and freeze higher-order "
                             "SH coefficients throughout calibration.  Prevents Gaussians from "
                             "learning view-dependent colours that would compensate for wrong "
                             "camera poses and kill the translation gradient.  No data leakage.")
    parser.add_argument("--lambda_cross_view",    type=float, default=0.0,
                        help="Weight of pure geometric cross-view consistency loss: "
                             "GT_i(π_i(T,P)) vs GT_j(π_j(T,P)).  No Gaussian rendering — "
                             "gradient flows only through the projection Jacobian ∂(u,v)/∂T. "
                             "Gaussian colors cannot compensate.  Requires camera data.")
    parser.add_argument("--lambda_virtual_frame", type=float, default=0.0,
                        help="Weight of virtual-frame loss: render an interpolated camera pose "
                             "with 3DGS, supervise rendered colors with GT from a real frame "
                             "warped via LiDAR geometry.  Combines Gaussian color gradient "
                             "(render) with geometric T gradient (real-frame projection).")
    parser.add_argument("--num_virtual_frames",    type=int,   default=1,
                        help="Virtual frames to sample per iteration. Default: 1.")
    parser.add_argument("--vf_neighbor_dist",      type=int,   default=5,
                        help="Max frame-index distance for interpolation partner. Default: 5.")
    parser.add_argument("--vf_perturb_rot_deg",    type=float, default=5.0,
                        help="Max rotation perturbation added on top of interpolated pose (°). Default: 5.")
    parser.add_argument("--vf_perturb_trans_m",    type=float, default=0.3,
                        help="Max translation perturbation added on top of interpolated pose (m). Default: 0.3.")
    parser.add_argument("--vf_min_overlap",        type=int,   default=200,
                        help="Min overlapping LiDAR points to accept a virtual frame. Default: 200.")
    parser.add_argument("--use_gt_translation",  action="store_true",
                        help="Lock translation to GT (skip translation optimisation). "
                             "Default: False — both rotation and translation are calibrated.")
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
                             "Only the 2 most recent are kept. Set 0 to disable.")
    parser.add_argument("--output_dir",          default=None)
    parser.add_argument("--gpu",                 type=int, default=None)
    cli = parser.parse_args()

    if cli.gpu is not None:
        torch.cuda.set_device(cli.gpu)

    args = parse(cli.exp_config)
    args = parse(cli.data_config, args)
    _dtype = str(getattr(args, "data_type", "")).lower()

    scene_id = getattr(args, "scene_id", "reset_prim_scene")
    mode_tag = f"{cli.mode}_calib"
    out_dir  = cli.output_dir or os.path.join("output", mode_tag, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # ── Scene ─────────────────────────────────────────────────
    print(blue("[ResetPrim] Loading scene..."))
    scene = dataloader.load_scene(args.source_dir, args)
    gaussians = scene.gaussians_assets[0]

    # ── Camera data ───────────────────────────────────────────
    camera_scale = float(getattr(args, "camera_scale", 1))
    if "kitticalib" in _dtype.replace("-", "").replace("_", ""):
        scene_name = getattr(args, "kitti_calib_scene", None)
        if scene_name is None:
            print(red("[ResetPrim] kitti_calib_scene not set. Exiting."))
            sys.exit(1)
        frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
        cam_cameras, cam_images = load_kitti_calib_cameras(
            args.source_dir, args, scene_name=scene_name,
            frame_ids=frame_ids, scale=camera_scale,
        )
    else:
        print(red(f"[ResetPrim] Dataset '{_dtype}' not supported."))
        sys.exit(1)
    print(blue(f"[ResetPrim] Loaded {len(cam_cameras)} camera frames."))

    lidar_world_poses = {k: v.float() for k, v in scene.train_lidar.sensor2world.items()}

    # ── Load Gaussian checkpoint ──────────────────────────────
    if cli.checkpoint:
        print(blue(f"[ResetPrim] Loading checkpoint: {cli.checkpoint}"))
        model_params, _ = torch.load(cli.checkpoint, weights_only=False, map_location="cuda")
        scene.restore(model_params, args.opt)
        for attr in _GAUSSIAN_ATTRS:
            p = getattr(gaussians, attr, None)
            if p is not None and not p.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(p.data.cuda()))
        gaussians.training_setup(args.opt)
        print(blue(f"[ResetPrim] Checkpoint loaded: {gaussians.get_local_xyz.shape[0]} Gaussians."))
    else:
        print(blue("[ResetPrim] No checkpoint — using initial point-cloud state."))
        gaussians.training_setup(args.opt)

    # ── Optional Gaussian downsampling ───────────────────────
    if cli.voxel_size > 0.0:
        xyz = gaussians.get_local_xyz.detach()          # (N, 3)
        origin = xyz.min(dim=0).values
        ijk = ((xyz - origin) / cli.voxel_size).long()  # (N, 3)
        # Encode voxel key as single int64 (scene fits in ~1000^3 grid safely)
        grid_size = ijk.max(dim=0).values + 1
        voxel_key = ijk[:, 0] * (grid_size[1] * grid_size[2]) + \
                    ijk[:, 1] * grid_size[2] + ijk[:, 2]
        # Sort by voxel key, then keep first occurrence per voxel
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
        print(blue(f"[ResetPrim] Voxel-downsampled {n_total} → {n_after} Gaussians "
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
        print(blue(f"[ResetPrim] Random-downsampled {n_total} → {n_after} Gaussians "
                   f"(ratio={cli.downsample_ratio:.2f})."))

    # ── Save base state (restored at start of every cycle) ────
    base_state = save_gaussian_state(gaussians)
    print(blue(f"[ResetPrim] Base state saved: {gaussians.get_local_xyz.shape[0]} Gaussians."))

    # ── CameraPoseCorrection ──────────────────────────────────
    model_cfg = getattr(args, "model", None)
    pose_cfg  = getattr(model_cfg, "pose_correction", None)
    pose_correction = CameraPoseCorrection(
        cam_cameras, pose_cfg, lidar_poses=lidar_world_poses
    ).cuda()
    pose_correction.use_gt_translation = cli.use_gt_translation

    gt_l2c_R = pose_correction.gt_lidar_to_camera_rotation[0].float().cuda()
    gt_l2c_T = pose_correction.gt_lidar_to_camera_translation[0].float().cuda()
    print(blue(f"[ResetPrim] GT l2c translation: {gt_l2c_T.cpu().numpy()}"))

    # ── Apply initial rotation bias ───────────────────────────
    gt_l2c_q = matrix_to_quaternion(gt_l2c_R)
    if cli.resume_from is not None:
        npz = np.load(cli.resume_from)
        resume_R = torch.tensor(npz["final_R"], dtype=torch.float32, device="cuda")
        init_q = matrix_to_quaternion(resume_R)
        resume_err = _rotation_error_deg(resume_R, gt_l2c_R)
        print(blue(f"[ResetPrim] Resuming from: {cli.resume_from}  (err={resume_err:.4f}°)"))
    elif cli.init_rot_deg > 0.0:
        if cli.init_rot_axis is not None:
            axis = F.normalize(
                torch.tensor(cli.init_rot_axis, dtype=torch.float32, device="cuda"), dim=0
            )
        else:
            axis = F.normalize(torch.randn(3, dtype=torch.float32, device="cuda"), dim=0)
        init_delta_q = axis_angle_to_quaternion(axis, math.radians(cli.init_rot_deg))
        init_q = quaternion_multiply(init_delta_q, gt_l2c_q)
        print(blue(f"[ResetPrim] Init perturbation: {cli.init_rot_deg:.1f}° along {axis.cpu().numpy().round(3)}"))
    else:
        init_q = gt_l2c_q.clone()
        print(blue("[ResetPrim] Starting from GT rotation."))

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
        print(blue(f"[ResetPrim] Init translation perturbation: {cli.init_trans_xyz}  "
                   f"(err={init_t_err:.4f} m)"))

    # ── Run ───────────────────────────────────────────────────
    if cli.mode == "noise":
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
            freeze_xyz=cli.freeze_xyz,
            freeze_colors=cli.freeze_colors,
            freeze_covariance=cli.freeze_covariance,
            translation_start_cycle=cli.translation_start_cycle,
            warmup_cycles=cli.warmup_cycles,
            freeze_rotation=cli.freeze_rotation,
            lr_patience=cli.lr_patience,
            lr_factor=cli.lr_factor,
            lr_min=cli.lr_min,
            lambda_reproj=cli.lambda_reproj,
            lambda_cross_view=cli.lambda_cross_view,
            lambda_virtual_frame=cli.lambda_virtual_frame,
            reset_gaussians_stage2=cli.reset_gaussians_stage2,
            no_freeze_stage2=cli.no_freeze_stage2,
            initial_gaussian_state=base_state,
            dc_only=cli.dc_only,
            tb_writer=tb_writer,
            cycle_ckpt_dir=os.path.join(out_dir, "cycle_ckpts"),
            save_cycle_every=cli.save_cycle_every,
            resume_cycle_ckpt=cli.resume_cycle_ckpt,
            num_virtual_frames=cli.num_virtual_frames,
            vf_neighbor_dist=cli.vf_neighbor_dist,
            vf_perturb_rot_deg=cli.vf_perturb_rot_deg,
            vf_perturb_trans_m=cli.vf_perturb_trans_m,
            vf_min_overlap=cli.vf_min_overlap,
        )
    else:
        final_R = run_reset_prim_calib(
            gaussians=gaussians,
            pose_correction=pose_correction,
            cam_cameras=cam_cameras,
            cam_images=cam_images,
            scene=scene,
            base_state=base_state,
            gt_l2c_R=gt_l2c_R,
            args=args,
            total_cycles=cli.total_cycles,
            iters_per_cycle=cli.iters_per_cycle,
            rotation_lr=cli.rotation_lr,
            freeze_gaussians=cli.freeze_gaussians,
            tb_writer=tb_writer,
        )

    # ── Save result ───────────────────────────────────────────
    out_path = os.path.join(out_dir, "best_rotation.npz")
    np.savez(out_path,
             final_R=final_R.detach().cpu().numpy(),
             gt_R=gt_l2c_R.detach().cpu().numpy(),
             init_rot_deg=cli.init_rot_deg)
    print(green(f"[{cli.mode.capitalize()}] Saved to: {out_path}"))


if __name__ == "__main__":
    main()
