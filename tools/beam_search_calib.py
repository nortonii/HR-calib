#!/usr/bin/env python3
"""Beam search for LiDAR-camera rotation calibration — from-scratch PSNR convergence.

Follows the SensorTrajectories quaternion-delta-compose update paradigm used in
HiGS-Calib (IRMVLab/HiGS-Calib):

  rotation_cl       — base quaternion  [w,x,y,z] (current centre)
  rotation_cl_delta — perturbation quaternion (one per candidate, identity = no change)
  effective_rotation = quaternion_multiply(rotation_cl_delta, rotation_cl)

After each beam step the best candidate becomes the new base
(update_extrinsic_in_search: base ← best, delta ← identity).  This guarantees
that subsequent deltas are always small rotations around the current best
estimate, matching the update paradigm from the upstream repo.

For each candidate the script runs a short (--iters_per_eval) joint
LiDAR+Camera training from a fixed Gaussian base state (no gradient on pose).
The rotation with the highest camera PSNR after N iterations is selected —
correct rotations make the two supervisory signals geometrically consistent so
the scene converges faster.

Pipeline
--------
1. Load dataset + camera data (same config system as train.py).
2. Create the initial Gaussian model from the point cloud.
   Optionally load a LiDAR-only checkpoint to start from a warmed-up base.
3. Save the "base Gaussian state" — a CPU copy of all Gaussian tensors.
4. Extract the GT lidar-to-camera extrinsic via CameraPoseCorrection.
5. Apply optional translation bias (--trans_bias dx dy dz) to fix translation.
6. Beam search over rotation (quaternion-delta paradigm):
   a. Sample K delta quaternions within ±radius around identity.
   b. Compose each with the current base: candidate_q = delta_q ⊗ current_q.
   c. For each candidate: restore Gaussian base state, run N joint iterations
      with the candidate rotation FIXED (no pose gradient), record final PSNR.
   d. Sort by PSNR; best_q → new base (update_extrinsic_in_search). Radius decays.
7. Print/save the final best rotation.

Usage
-----
python tools/beam_search_calib.py \\
    -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --init_rot_deg 15.0 \\
    --beam_steps 5 --beam_candidates 5 \\
    --iters_per_eval 200 \\
    --radius_deg 10.0 --radius_decay 0.6

Use --checkpoint to start from a LiDAR-pretrained base (faster signal):
    --checkpoint output/.../model_it_3000.pth
"""

import argparse
import math
import os
import random
import sys
import time

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
from lib.utils.console_utils import blue, green, red, yellow
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss, ssim


# ─────────────────────────────────────────────────────────────
# Quaternion utilities  [w, x, y, z] convention
# (mirrors pytorch3d.transforms used in sensor_trajectories.py,
#  implemented without pytorch3d to avoid the dependency)
# ─────────────────────────────────────────────────────────────

def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert a (3, 3) rotation matrix → quaternion [w, x, y, z] (float32).

    Uses the Shepperd / eigenvector method for numerical stability.
    Equivalent to pytorch3d.transforms.matrix_to_quaternion.
    """
    R = R.float()
    # Build the 4×4 K matrix (Shepperd 1978 / Horn 1987)
    K = torch.stack([
        torch.stack([R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1], R[2,0]+R[0,2], R[2,1]-R[1,2]]),
        torch.stack([R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2], R[0,2]-R[2,0]]),
        torch.stack([R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], R[1,0]-R[0,1]]),
        torch.stack([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]]),
    ]) / 3.0
    _, v = torch.linalg.eigh(K)        # eigenvalues in ascending order
    q_xyzw = v[:, -1]                  # eigenvector for largest eigenvalue → [x,y,z,w]
    q = torch.stack([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # → [w,x,y,z]
    return q if q[0] >= 0 else -q      # canonical form: w ≥ 0


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] → (3, 3) rotation matrix (float32).

    Equivalent to pytorch3d.transforms.quaternion_to_matrix.
    """
    q = F.normalize(q.float(), dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)]),
        torch.stack([  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)]),
        torch.stack([  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]),
    ])


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2  ([w,x,y,z] convention).

    Applies q2 first, then q1 — i.e. the composed rotation is q1 after q2.
    Equivalent to pytorch3d.transforms.quaternion_multiply.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_identity(device="cuda") -> torch.Tensor:
    """Identity quaternion [1, 0, 0, 0]."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)


def axis_angle_to_quaternion(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Convert axis-angle to quaternion [w, x, y, z]."""
    axis = F.normalize(axis.float(), dim=0)
    half = angle_rad / 2.0
    return torch.cat([
        torch.tensor([math.cos(half)], dtype=torch.float32, device=axis.device),
        math.sin(half) * axis,
    ])


def _rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> float:
    R_rel = R_pred @ R_gt.T
    cos_a = ((R_rel.diagonal().sum().clamp(-1, 3) - 1) / 2).clamp(-1, 1)
    return math.degrees(math.acos(cos_a.item()))


def _rotation_to_euler_deg(R: torch.Tensor):
    R = R.cpu().float()
    sy = math.sqrt(R[0, 0].item() ** 2 + R[1, 0].item() ** 2)
    if sy > 1e-6:
        rx = math.degrees(math.atan2(R[2, 1].item(), R[2, 2].item()))
        ry = math.degrees(math.atan2(-R[2, 0].item(), sy))
        rz = math.degrees(math.atan2(R[1, 0].item(), R[0, 0].item()))
    else:
        rx = math.degrees(math.atan2(-R[1, 2].item(), R[1, 1].item()))
        ry = math.degrees(math.atan2(-R[2, 0].item(), sy))
        rz = 0.0
    return rx, ry, rz


def _sample_delta_quaternions(radius_deg: float, n: int, device="cuda") -> list:
    """Sample n small-rotation delta quaternions within ±radius_deg of identity.

    This is the beam-search exploration step in the quaternion-delta paradigm:
      rotation_cl_delta ~ Uniform(random axis, angle ∈ [-radius, +radius])
    Identity quaternion [1,0,0,0] corresponds to zero perturbation.
    """
    deltas = []
    max_angle = math.radians(radius_deg)
    for _ in range(n):
        axis = F.normalize(torch.randn(3, dtype=torch.float32, device=device), dim=0)
        angle = (2 * torch.rand(1, device=device).item() - 1) * max_angle
        deltas.append(axis_angle_to_quaternion(axis, angle))
    return deltas


# ─────────────────────────────────────────────────────────────
# Per-frame camera R,T from a given l2c extrinsic
# ─────────────────────────────────────────────────────────────

def _l2c_to_camera_rt(l2c_R, l2c_T, lidar_world):
    """Mirrors CameraPoseCorrection.corrected_rt (shared-extrinsic mode)."""
    lidar_R = lidar_world[:3, :3]
    lidar_t = lidar_world[:3, 3]
    c2l_R = l2c_R.T
    c2l_t = -(c2l_R @ l2c_T)
    cam_R = lidar_R @ c2l_R
    cam_center = lidar_R @ c2l_t + lidar_t
    cam_T = -(cam_R.T @ cam_center)
    return cam_R, cam_T


# ─────────────────────────────────────────────────────────────
# Gaussian state save / restore (no densification → fixed size)
# ─────────────────────────────────────────────────────────────

_GAUSSIAN_ATTRS = [
    "_xyz", "_features_dc", "_features_rest",
    "_features_rgb_dc", "_features_rgb_rest",
    "_scaling", "_rotation", "_opacity", "_opacity_cam",
]


def save_gaussian_state(gaussians) -> dict:
    """Deep-copy all Gaussian tensors to CPU."""
    state = {"active_sh_degree": gaussians.active_sh_degree}
    for attr in _GAUSSIAN_ATTRS:
        p = getattr(gaussians, attr, None)
        if p is not None:
            state[attr] = p.data.detach().clone().cpu()
    return state


def restore_gaussian_state(gaussians, state: dict, args):
    """
    Restore Gaussian tensors from CPU state and reinitialise the Adam
    optimizer. Densification is intentionally skipped during beam-search
    evaluations so the point count stays fixed and state restores cleanly.
    """
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
            # If shapes mismatch (densification happened previously), replace.
            new_p = torch.nn.Parameter(
                state[attr].to(p.device).requires_grad_(True)
            )
            setattr(gaussians, attr, new_p)

    n = gaussians.get_local_xyz.shape[0]
    gaussians.max_radii2D = torch.zeros(n, device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
    gaussians.denom = torch.zeros((n, 1), device="cuda")

    # Re-enable gradients for all params.
    for attr in _GAUSSIAN_ATTRS:
        p = getattr(gaussians, attr, None)
        if p is not None:
            p.requires_grad_(True)

    # Fresh Adam optimizer.
    gaussians.training_setup(args.opt)


# ─────────────────────────────────────────────────────────────
# Core evaluation: fixed rotation, Gaussian-only gradient
# ─────────────────────────────────────────────────────────────

def evaluate_rotation(
    candidate_q: torch.Tensor,
    pose_correction,
    gaussians,
    base_state: dict,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    args,
    n_iters: int = 200,
    rotation_lr: float = 2e-3,   # unused; kept for API compatibility
    lambda_rgb: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_dssim: float = 0.2,
) -> tuple:
    """Evaluate a candidate rotation: fix rotation, optimise Gaussians only.

    Setup per candidate:
      base_q  ← candidate_q  (fixed, frozen for the whole evaluation)
      delta_q ← identity     (frozen — no rotation gradient)

    Each iteration:
      cam_R, cam_T = corrected_rt(frame)   ← detached, constant
      Gaussian optimizer step (joint LiDAR + camera loss)

    The PSNR averaged over the last few frames serves as the quality
    signal: a rotation closer to GT lets Gaussians converge better in
    the limited n_iters budget → higher PSNR.

    Returns:
        (final_psnr: float, candidate_q: Tensor [4])
          candidate_q is returned unchanged (rotation is frozen).
    """
    # ── Reset Gaussian state ──────────────────────────────────
    restore_gaussian_state(gaussians, base_state, args)

    # ── Fix pose: base = candidate_q, delta = identity (frozen) ─
    with torch.no_grad():
        cq = candidate_q.detach().to(
            device="cuda", dtype=pose_correction.base_lidar_to_camera_quat.dtype
        )
        if cq[0] < 0:
            cq = -cq
        pose_correction.base_lidar_to_camera_quat.data[0].copy_(cq)
        pose_correction.base_lidar_to_camera_rotation.data[0].copy_(
            quaternion_to_matrix(cq)
        )
        pose_correction.delta_rotations_quat.data.fill_(0.0)
        pose_correction.delta_rotations_quat.data[:, 0] = 1.0   # identity

    background = torch.tensor([0, 0, 1], device="cuda").float()
    frame_ids_train = sorted(scene.train_lidar.train_frames)
    frame_stack = list(frame_ids_train)
    random.shuffle(frame_stack)

    psnr_accum = 0.0
    psnr_count = 0
    eval_window = max(1, n_iters // 5)   # average over last 20% of iters

    for it in range(1, n_iters + 1):
        gaussians.update_learning_rate(it)

        if not frame_stack:
            frame_stack = list(frame_ids_train)
            random.shuffle(frame_stack)
        frame = frame_stack.pop()

        # ── LiDAR render (OptiX) ──────────────────────────────
        render_pkg = raytracing(
            frame, [gaussians], scene.train_lidar, background, args, depth_only=True
        )
        depth    = render_pkg["depth"].squeeze(-1)
        gt_mask  = scene.train_lidar.get_mask(frame).cuda()
        dyn_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
        static_mask = gt_mask & ~dyn_mask
        gt_depth = scene.train_lidar.get_depth(frame).cuda()
        loss_depth = lambda_depth * l1_loss(depth[static_mask], gt_depth[static_mask])

        # ── Camera render with FIXED rotation (no grad on pose) ──
        loss_rgb = torch.tensor(0.0, device="cuda")
        if frame in cam_cameras:
            with torch.no_grad():
                cam_R, cam_T = pose_correction.corrected_rt(frame, device="cuda")
            camera = cam_cameras[frame].cuda()
            gt_rgb = cam_images[frame].cuda()

            cam_render = render_camera(
                camera, [gaussians], args,
                cam_rotation=cam_R.detach(), cam_translation=cam_T.detach(),
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
                if it > n_iters - eval_window:
                    psnr_accum += psnr(pred_chw, gt_chw).item()
                    psnr_count += 1

        # ── Gaussian step only (rotation frozen) ──────────────
        total_loss = loss_depth + loss_rgb
        total_loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    final_psnr = psnr_accum / psnr_count if psnr_count > 0 else 0.0
    return final_psnr, candidate_q


# ─────────────────────────────────────────────────────────────
# Beam search  (quaternion-delta-compose paradigm)
# ─────────────────────────────────────────────────────────────

def beam_search(
    gaussians,
    pose_correction,
    cam_cameras: dict,
    cam_images: dict,
    scene,
    base_state: dict,
    gt_l2c_R: torch.Tensor,
    init_R: torch.Tensor,
    args,
    beam_steps: int = 5,
    beam_candidates: int = 5,
    iters_per_eval: int = 200,
    rotation_lr: float = 2e-3,
    radius_deg: float = 10.0,
    radius_decay: float = 0.6,
    tb_writer=None,
) -> torch.Tensor:
    """Beam search with gradient on delta_q during each candidate evaluation.

    Outer loop (gradient-free discrete search):
      current_q — base quaternion, updated each beam step

    Inner loop (gradient-based, n_iters per candidate):
      base_q  ← candidate_q  (fixed for this evaluation)
      delta_q ← identity     (learnable, Adam, lr=rotation_lr)
      RGB loss → delta_q grad → pose_opt.step → update_extrinsics each iter
      final PSNR = signal of how well delta converged (closer to GT → higher)

    After all candidates are evaluated, best_final_q (the effective rotation
    after training, not just the initial candidate) becomes the new current_q.
    """
    current_q = matrix_to_quaternion(init_R.to("cuda"))

    init_R_mat = quaternion_to_matrix(current_q)
    print(blue(f"\n[BeamSearch] Init rotation error vs GT: "
               f"{_rotation_error_deg(init_R_mat, gt_l2c_R):.2f}°"))
    print(blue(f"[BeamSearch] Steps={beam_steps}, K={beam_candidates}, "
               f"iters/eval={iters_per_eval}, radius={radius_deg:.1f}°, decay={radius_decay}"))
    print(blue(f"[BeamSearch] rotation_lr={rotation_lr}  "
               f"(delta_q gets gradient from RGB loss each iter)"))
    print()

    global_step = 0
    for step in range(beam_steps):
        # Sample delta quaternions + exploit (identity delta = current centre)
        delta_qs    = _sample_delta_quaternions(radius_deg, beam_candidates, device="cuda")
        delta_qs.append(quaternion_identity(device="cuda"))
        candidate_qs = [quaternion_multiply(dq, current_q) for dq in delta_qs]

        results = []
        for k, (delta_q, cand_q) in enumerate(zip(delta_qs, candidate_qs)):
            is_exploit = (k == len(candidate_qs) - 1)
            tag = "(exploit)" if is_exploit else f"(explore {k+1}/{beam_candidates})"

            cand_R    = quaternion_to_matrix(cand_q)
            euler     = _rotation_to_euler_deg(cand_R)
            init_err  = _rotation_error_deg(cand_R, gt_l2c_R)
            delta_deg = _rotation_error_deg(quaternion_to_matrix(delta_q),
                                            torch.eye(3, device="cuda"))
            print(yellow(
                f"  Step {step+1}/{beam_steps} {tag}: "
                f"delta={delta_deg:.2f}°  "
                f"euler=({euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°)  "
                f"init_err_vs_gt={init_err:.2f}°"
            ))

            t0 = time.time()
            final_psnr, final_q = evaluate_rotation(
                cand_q, pose_correction, gaussians, base_state,
                cam_cameras, cam_images, scene, args,
                n_iters=iters_per_eval, rotation_lr=rotation_lr,
            )
            elapsed   = time.time() - t0
            final_R   = quaternion_to_matrix(final_q)
            final_err = _rotation_error_deg(final_R, gt_l2c_R)
            print(yellow(
                f"    → PSNR={final_psnr:.2f} dB  "
                f"final_err={final_err:.2f}°  ({elapsed:.1f}s)"
            ))

            results.append((final_q, final_psnr, init_err, final_err))
            global_step += 1
            if tb_writer:
                tb_writer.add_scalar("beam/candidate_psnr",     final_psnr,  global_step)
                tb_writer.add_scalar("beam/init_rot_err",       init_err,    global_step)
                tb_writer.add_scalar("beam/final_rot_err",      final_err,   global_step)
                tb_writer.add_scalar("beam/delta_magnitude_deg", delta_deg,  global_step)

        results.sort(key=lambda x: x[1], reverse=True)
        best_q, best_psnr, best_init_err, best_final_err = results[0]

        # update_extrinsic_in_search: current_q ← best final effective rotation
        current_q = best_q.clone()

        best_R = quaternion_to_matrix(current_q)
        print(green(
            f"\n[BeamSearch] Step {step+1} done: "
            f"best PSNR={best_psnr:.2f} dB, "
            f"init_err={best_init_err:.2f}° → final_err={best_final_err:.2f}°, "
            f"radius={radius_deg:.2f}°"
        ))
        print("  Ranking:")
        for rank, (_, p, ie, fe) in enumerate(results):
            print(f"    [{rank+1}] PSNR={p:.2f} dB  init_err={ie:.2f}°  final_err={fe:.2f}°")

        if tb_writer:
            tb_writer.add_scalar("beam/best_psnr",          best_psnr,       step)
            tb_writer.add_scalar("beam/best_init_err_deg",  best_init_err,   step)
            tb_writer.add_scalar("beam/best_final_err_deg", best_final_err,  step)
            tb_writer.add_scalar("beam/search_radius_deg",  radius_deg,      step)

        radius_deg *= radius_decay
        print()

    # Return best rotation as (3, 3) matrix
    return quaternion_to_matrix(current_q)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Beam search LiDAR-camera calibration via from-scratch PSNR convergence."
    )
    parser.add_argument("-dc", "--data_config", required=True)
    parser.add_argument("-ec", "--exp_config", required=True)
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional LiDAR-pretrained .pth (used as fixed Gaussian base). "
             "If omitted, starts from the initial point cloud.",
    )
    parser.add_argument(
        "--init_rot_deg", type=float, default=0.0,
        help="Initial rotation perturbation from GT in degrees.",
    )
    parser.add_argument(
        "--init_rot_axis", type=float, nargs=3, default=None,
        metavar=("AX", "AY", "AZ"),
        help="Fixed axis for initial perturbation (default: random).",
    )
    parser.add_argument("--beam_steps", type=int, default=5)
    parser.add_argument("--beam_candidates", type=int, default=5)
    parser.add_argument("--iters_per_eval", type=int, default=200)
    parser.add_argument("--rotation_lr", type=float, default=2e-3,
                        help="Adam lr for delta_q during each candidate evaluation.")
    parser.add_argument("--radius_deg", type=float, default=10.0)
    parser.add_argument("--radius_decay", type=float, default=0.6)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    cli = parser.parse_args()

    if cli.gpu is not None:
        torch.cuda.set_device(cli.gpu)

    # ── Configs ──────────────────────────────────────────────
    args = parse(cli.exp_config)
    args = parse(cli.data_config, args)
    _dtype = str(getattr(args, "data_type", "")).lower()

    scene_id = getattr(args, "scene_id", "beam_scene")
    out_dir = cli.output_dir or os.path.join("output", scene_id, "beam_search_grad")
    os.makedirs(out_dir, exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # ── Scene ────────────────────────────────────────────────
    print(blue("[BeamSearch] Loading scene..."))
    scene = dataloader.load_scene(args.source_dir, args)
    gaussians = scene.gaussians_assets[0]

    # ── Camera data ──────────────────────────────────────────
    camera_scale = float(getattr(args, "camera_scale", 1))
    if "kitticalib" in _dtype.replace("-", "").replace("_", ""):
        scene_name = getattr(args, "kitti_calib_scene", None)
        if scene_name is None:
            print(red("[BeamSearch] kitti_calib_scene not set. Exiting."))
            sys.exit(1)
        frame_ids = list(range(args.frame_length[0], args.frame_length[1] + 1))
        cam_cameras, cam_images = load_kitti_calib_cameras(
            args.source_dir, args, scene_name=scene_name,
            frame_ids=frame_ids, scale=camera_scale,
        )
    else:
        print(red(f"[BeamSearch] Dataset '{_dtype}' not supported yet."))
        sys.exit(1)
    print(blue(f"[BeamSearch] Loaded {len(cam_cameras)} camera frames."))

    lidar_world_poses = {k: v.float() for k, v in scene.train_lidar.sensor2world.items()}

    # ── Load optional checkpoint ─────────────────────────────
    if cli.checkpoint:
        print(blue(f"[BeamSearch] Loading checkpoint: {cli.checkpoint}"))
        model_params, _ = torch.load(cli.checkpoint, weights_only=False)
        scene.restore(model_params, args.opt)
        for attr in _GAUSSIAN_ATTRS:
            p = getattr(gaussians, attr, None)
            if p is not None and not p.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(p.data.cuda()))
        gaussians.training_setup(args.opt)
        print(blue(f"[BeamSearch] Checkpoint loaded: {gaussians.get_local_xyz.shape[0]} Gaussians."))
    else:
        print(blue("[BeamSearch] No checkpoint — using initial point-cloud state."))

    # ── Save base Gaussian state ──────────────────────────────
    base_state = save_gaussian_state(gaussians)
    print(blue(f"[BeamSearch] Base state saved: {gaussians.get_local_xyz.shape[0]} Gaussians."))

    # ── Build CameraPoseCorrection (shared extrinsic, GT translation) ──
    model_cfg = getattr(args, "model", None)
    pose_cfg  = getattr(model_cfg, "pose_correction", None)
    pose_correction = CameraPoseCorrection(
        cam_cameras, pose_cfg, lidar_poses=lidar_world_poses
    ).cuda()
    pose_correction.use_gt_translation = True
    gt_l2c_R = pose_correction.gt_lidar_to_camera_rotation[0].float().cuda()
    gt_l2c_T = pose_correction.gt_lidar_to_camera_translation[0].float().cuda()
    print(blue(f"[BeamSearch] GT l2c Euler XYZ: {_rotation_to_euler_deg(gt_l2c_R)}"))
    print(blue(f"[BeamSearch] GT l2c translation: {gt_l2c_T.cpu().numpy()}"))

    # ── Initial rotation ──────────────────────────────────────
    gt_l2c_q = matrix_to_quaternion(gt_l2c_R)
    if cli.init_rot_deg > 0.0:
        if cli.init_rot_axis is not None:
            axis = F.normalize(
                torch.tensor(cli.init_rot_axis, dtype=torch.float32, device="cuda"), dim=0
            )
        else:
            axis = F.normalize(torch.randn(3, dtype=torch.float32, device="cuda"), dim=0)
        init_delta_q = axis_angle_to_quaternion(axis, math.radians(cli.init_rot_deg))
        init_q = quaternion_multiply(init_delta_q, gt_l2c_q)
        init_R = quaternion_to_matrix(init_q)
        print(blue(f"[BeamSearch] Init perturbation: {cli.init_rot_deg:.1f}° along {axis.cpu().numpy().round(3)}"))
        print(blue(f"  Init error vs GT: {_rotation_error_deg(init_R, gt_l2c_R):.2f}°"))
    else:
        init_R = gt_l2c_R.clone()
        print(blue("[BeamSearch] Starting from GT rotation (init_rot_deg=0)."))

    # ── Beam search ───────────────────────────────────────────
    best_R = beam_search(
        gaussians=gaussians,
        pose_correction=pose_correction,
        cam_cameras=cam_cameras,
        cam_images=cam_images,
        scene=scene,
        base_state=base_state,
        gt_l2c_R=gt_l2c_R,
        init_R=init_R,
        args=args,
        beam_steps=cli.beam_steps,
        beam_candidates=cli.beam_candidates,
        iters_per_eval=cli.iters_per_eval,
        rotation_lr=cli.rotation_lr,
        radius_deg=cli.radius_deg,
        radius_decay=cli.radius_decay,
        tb_writer=tb_writer,
    )

    # ── Report ────────────────────────────────────────────────
    init_err  = _rotation_error_deg(init_R, gt_l2c_R)
    final_err = _rotation_error_deg(best_R, gt_l2c_R)
    euler     = _rotation_to_euler_deg(best_R)

    print(green("=" * 60))
    print(green(f"[BeamSearch] Init error : {init_err:.4f}°"))
    print(green(f"[BeamSearch] Final error: {final_err:.4f}°"))
    print(green(f"[BeamSearch] Improvement: {init_err - final_err:+.4f}°"))
    print(green(f"[BeamSearch] Best rotation Euler XYZ: "
                f"rx={euler[0]:.4f}°, ry={euler[1]:.4f}°, rz={euler[2]:.4f}°"))
    print(green("=" * 60))

    result_path = os.path.join(out_dir, "best_rotation.npz")
    np.savez(
        result_path,
        best_rotation=best_R.cpu().numpy(),
        gt_rotation=gt_l2c_R.cpu().numpy(),
        rotation_error_deg=final_err,
        init_error_deg=init_err,
        gt_translation=gt_l2c_T.cpu().numpy(),
    )
    print(green(f"[BeamSearch] Saved to: {result_path}"))

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
