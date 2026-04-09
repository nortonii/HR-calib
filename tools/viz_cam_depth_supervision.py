#!/usr/bin/env python3
"""Visualise camera-view depth supervision signal.

Loads a calibration cycle checkpoint (Gaussians + pose correction), then for
a set of frames produces a 3-panel figure:
  Left:   Camera GT RGB with LiDAR points projected at CURRENT pose (colour = depth)
  Centre: GT-projected reference depth image (fixed, computed at GT T_l2c)
  Right:  Supervision error: |d_lidar(current_pose) - d_ref| coloured by magnitude
          (green=match, red=large error)

The reference depth is built by projecting LiDAR points at GT T_l2c (same as the
training loss), so Panel 2 shows exactly what the loss is trying to match.

Usage
-----
python tools/viz_cam_depth_supervision.py \\
    -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \\
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \\
    --cycle_ckpt output/noise_inject_calib/test44_from36ckpt_trans_freezexyz_colors/cycle_ckpts/cycle_1200.pth \\
    --output_dir output/viz_cam_depth
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import dataloader
from lib.arguments import parse
from lib.dataloader.kitti_calib_loader import load_kitti_calib_cameras
from lib.scene.camera_pose_correction import CameraPoseCorrection
from lib.utils.console_utils import blue, green, red

# Import helpers from the calibration script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from reset_prim_calib import restore_gaussian_state, _build_cam_depth_refs


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q.float(), dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)]),
        torch.stack([  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)]),
        torch.stack([  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]),
    ])


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def get_l2c(ckpt: dict, pose_correction: CameraPoseCorrection,
            frame: int, device: str = "cuda"):
    """Return (R_l2c, T_l2c) from checkpoint deltas."""
    pidx  = pose_correction._pose_index(frame)
    dT    = ckpt["delta_translations"][pidx].to(device)
    bT    = pose_correction.base_lidar_to_camera_translation[0].to(device)
    T_l2c = bT + dT
    dq    = F.normalize(ckpt["delta_rotations_quat"][pidx].to(device), dim=0)
    bq    = pose_correction.base_lidar_to_camera_quat[0].to(device)
    R_l2c = _quat_to_matrix(_quat_mul(dq, bq))
    return R_l2c, T_l2c


def get_corrected_rt(ckpt: dict, pose_correction: CameraPoseCorrection,
                     frame: int, device: str = "cuda"):
    """Return (cam_R_c2w, cam_T_w2c) for the Gaussian transform during rendering."""
    fidx  = pose_correction._frame_index(frame)
    R_l2c, T_l2c = get_l2c(ckpt, pose_correction, frame, device)
    R_l2w = pose_correction.lidar_world_rotations[fidx].to(device)
    T_l2w = pose_correction.lidar_world_translations[fidx].to(device)
    R_c2l = R_l2c.T
    T_c2l = -(R_c2l @ T_l2c)
    R_c2w = R_l2w @ R_c2l
    cam_center = R_l2w @ T_c2l + T_l2w
    T_w2c = -(R_c2w.T @ cam_center)
    return R_c2w, T_w2c


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_lidar_to_cam(scene, frame, R_l2c, T_l2c, W, H, fx, fy):
    """Project static LiDAR hit-points into camera frame.
    Returns u, v (pixel coords), d_ray (Euclidean ray depth) all on CUDA.
    """
    gt_mask  = scene.train_lidar.get_mask(frame).cuda()
    dyn_mask = scene.train_lidar.get_dynamic_mask(frame).cuda()
    valid    = (gt_mask & ~dyn_mask).reshape(-1)

    rays_o, rays_d = scene.train_lidar.get_range_rays(frame)
    gt_depth = scene.train_lidar.get_depth(frame).cuda()
    P_world  = (rays_o + rays_d * gt_depth.unsqueeze(-1)).reshape(-1, 3)
    P_valid  = P_world[valid]

    s2w     = scene.train_lidar.sensor2world[frame].cuda()
    P_local = (P_valid - s2w[:3, 3]) @ s2w[:3, :3]
    P_cam   = P_local @ R_l2c.T + T_l2c             # camera frame

    front = P_cam[:, 2] > 0.1
    P_f   = P_cam[front]
    d_ray = P_f.norm(dim=-1)                         # Euclidean ray depth

    u = fx * P_f[:, 0] / P_f[:, 2] + W / 2.0
    v = fy * P_f[:, 1] / P_f[:, 2] + H / 2.0
    in_fov = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[in_fov], v[in_fov], d_ray[in_fov]


def sample_rendered_depth(depth_render: torch.Tensor,
                          u: torch.Tensor, v: torch.Tensor,
                          W: int, H: int) -> torch.Tensor:
    """Bilinear-sample depth_render (H, W) at pixel locations (u, v)."""
    u_n = (u / (W - 1)) * 2.0 - 1.0
    v_n = (v / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([u_n, v_n], dim=-1)[None, None]
    sampled = F.grid_sample(
        depth_render[None, None],
        grid, mode="bilinear", align_corners=True, padding_mode="zeros",
    ).squeeze()
    return sampled


# ---------------------------------------------------------------------------
# Per-frame visualisation
# ---------------------------------------------------------------------------

def visualise_frame(frame, scene, ckpt, pose_correction,
                    cam_cameras, cam_images, args, out_path,
                    use_gt_pose=False, gt_R=None, gt_T=None,
                    cam_depth_ref=None):
    """
    3-panel visualisation using the reference-based depth supervision.
    cam_depth_ref: Tensor(H, W) pre-computed GT-projected reference depth (or None).
    """
    print(blue(f"  Frame {frame}..."))

    # ── Extrinsics ──────────────────────────────────────────────────────────
    if use_gt_pose:
        device = "cuda"
        R_l2c = gt_R.to(device)
        T_l2c = gt_T.to(device)
        print(blue("    [use_gt_pose] Using GT extrinsics"))
    else:
        R_l2c, T_l2c = get_l2c(ckpt, pose_correction, frame)

    cam = cam_cameras[frame].cuda()
    W   = int(cam.image_width)
    H   = int(cam.image_height)
    fx  = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy  = H / (2.0 * math.tan(cam.FoVy * 0.5))

    # ── Project LiDAR at CURRENT pose ───────────────────────────────────────
    u, v, d_lidar = project_lidar_to_cam(scene, frame, R_l2c, T_l2c, W, H, fx, fy)
    u_np     = u.cpu().numpy()
    v_np     = v.cpu().numpy()
    d_np     = d_lidar.cpu().numpy()

    # ── Sample reference depth at projected pixels ───────────────────────────
    if cam_depth_ref is not None:
        d_ref_gpu    = cam_depth_ref.to("cuda")
        d_ref_sampled = sample_rendered_depth(d_ref_gpu, u, v, W, H)
        ds_np         = d_ref_sampled.cpu().numpy()
        hit           = ds_np > 0.1
    else:
        ds_np = np.zeros_like(d_np)
        hit   = np.zeros(len(d_np), dtype=bool)

    if hit.sum() > 0:
        err = np.abs(d_np[hit] - ds_np[hit])
        mae = float(err.mean())
        med = float(np.median(err))
        print(green(f"    hits={hit.sum()}/{len(hit)}  MAE={mae:.3f}m  median={med:.3f}m"))
    else:
        err = np.array([])
        mae = med = float("nan")
        print(red("    no reference hits — check GT projection or camera/LiDAR overlap"))

    # ── Build reference depth image for display ──────────────────────────────
    if cam_depth_ref is not None:
        ref_np = cam_depth_ref.cpu().numpy()
    else:
        ref_np = np.zeros((H, W), dtype=np.float32)

    # ── Plot ────────────────────────────────────────────────────────────────
    gt_rgb = cam_images[frame].cpu().numpy()
    vmin   = float(np.percentile(d_np, 2))  if len(d_np) > 0 else 0.0
    vmax   = float(np.percentile(d_np, 98)) if len(d_np) > 0 else 30.0

    pose_label = "GT pose" if use_gt_pose else "checkpoint pose"
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=100)
    fig.suptitle(
        f"Frame {frame}  |  Camera-view depth supervision ({pose_label})\n"
        f"hits={hit.sum()}/{len(hit)}  MAE={mae:.3f}m  median={med:.3f}m",
        fontsize=11)

    # Panel 1: GT RGB + LiDAR projections at current pose
    ax = axes[0]
    ax.imshow(gt_rgb)
    if len(u_np) > 0:
        sc1 = ax.scatter(u_np, v_np, c=d_np, cmap="jet",
                         vmin=vmin, vmax=vmax, s=0.8, linewidths=0, rasterized=True)
        plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label="depth (m)")
    ax.set_title(f"GT RGB + LiDAR projection ({pose_label})")
    ax.axis("off")

    # Panel 2: GT-projected reference depth image
    ax = axes[1]
    ref_vis = np.where(ref_np > 0.1, ref_np, np.nan)
    im2 = ax.imshow(ref_vis, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label="depth (m)")
    ref_pts = int((ref_np > 0.1).sum())
    ax.set_title(f"GT-projected reference depth\n({ref_pts} pixels with observations)")
    ax.axis("off")

    # Panel 3: Supervision error map |d_lidar(current) - d_ref|
    ax = axes[2]
    ax.imshow(gt_rgb, alpha=0.35)
    if (~hit).sum() > 0:
        ax.scatter(u_np[~hit], v_np[~hit],
                   c="grey", s=0.4, alpha=0.4, linewidths=0, rasterized=True,
                   label="no reference hit")
    if hit.sum() > 0:
        err_norm = plt.Normalize(vmin=0.0, vmax=1.0)
        err_frac  = np.clip(err / 1.0, 0, 1)
        sc3 = ax.scatter(u_np[hit], v_np[hit],
                         c=err_frac, cmap="RdYlGn_r", norm=err_norm,
                         s=1.0, linewidths=0, rasterized=True,
                         label="supervised")
        plt.colorbar(sc3, ax=ax, fraction=0.03, pad=0.02, label="|error| (m)")
    ax.set_title("|d_lidar(current) - d_ref(GT)| (green=0m, red≥1m, grey=no ref)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(green(f"    saved: {out_path}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise camera-view depth supervision signal")
    parser.add_argument("-dc", "--data_config",  required=True)
    parser.add_argument("-ec", "--exp_config",   required=True)
    parser.add_argument("--cycle_ckpt", required=True,
                        help="Path to cycle_NNNN.pth checkpoint")
    parser.add_argument("--output_dir", default="output/viz_cam_depth")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="Frame IDs to visualise (default: first 3 train frames)")
    parser.add_argument("--use_gt_pose", action="store_true",
                        help="Override pose with GT l2c (useful for verifying the "
                             "supervision signal when checkpoint rotation is unavailable)")
    parser.add_argument("--gpu", type=int, default=None)
    cli = parser.parse_args()

    if cli.gpu is not None:
        torch.cuda.set_device(cli.gpu)

    os.makedirs(cli.output_dir, exist_ok=True)

    args = parse(cli.exp_config)
    args = parse(cli.data_config, args)
    _dtype = str(getattr(args, "data_type", "")).lower()

    print(blue("[VizCamDepth] Loading scene..."))
    scene     = dataloader.load_scene(args.source_dir, args)
    gaussians = scene.gaussians_assets[0]

    camera_scale = float(getattr(args, "camera_scale", 1))
    if "kitticalib" in _dtype.replace("-", "").replace("_", ""):
        scene_name = getattr(args, "kitti_calib_scene", None)
        frame_ids  = list(range(args.frame_length[0], args.frame_length[1] + 1))
        cam_cameras, cam_images = load_kitti_calib_cameras(
            args.source_dir, args, scene_name=scene_name,
            frame_ids=frame_ids, scale=camera_scale,
        )
    else:
        print(red(f"Dataset type '{_dtype}' not supported."))
        sys.exit(1)
    print(blue(f"[VizCamDepth] {len(cam_cameras)} camera frames loaded."))

    print(blue(f"[VizCamDepth] Loading checkpoint: {cli.cycle_ckpt}"))
    ckpt = torch.load(cli.cycle_ckpt, map_location="cpu")
    print(blue(f"  cycle={ckpt['cycle']}  best_T_err={ckpt['best_T_err']:.4f}m"))

    restore_gaussian_state(gaussians, ckpt["gaussian_state"], args)

    lidar_world_poses = {k: v.float() for k, v in scene.train_lidar.sensor2world.items()}
    model_cfg = getattr(args, "model", None)
    pose_cfg  = getattr(model_cfg, "pose_correction", None)
    pose_correction = CameraPoseCorrection(
        cam_cameras, pose_cfg, lidar_poses=lidar_world_poses
    ).cuda()

    with torch.no_grad():
        if "pose_correction_state" in ckpt:
            pose_correction.load_state_dict(
                {k: v.to("cuda") for k, v in ckpt["pose_correction_state"].items()})
            print(blue("  Loaded full pose_correction state (includes accumulated base_q)."))
        else:
            pose_correction.delta_rotations_quat.copy_(
                ckpt["delta_rotations_quat"].cuda())
            pose_correction.delta_translations.copy_(
                ckpt["delta_translations"].cuda())
            print(red("  Legacy checkpoint: base_q not saved -- rotation may be incorrect."))
            print(red("  Run with --use_gt_pose to verify signal at GT pose."))

    # Report current extrinsic error
    gt_R  = pose_correction.gt_lidar_to_camera_rotation[0].float().cuda()
    gt_T  = pose_correction.gt_lidar_to_camera_translation[0].float().cuda()
    f0    = sorted(cam_cameras.keys())[0]
    with torch.no_grad():
        R_l2c0, T_l2c0 = get_l2c(ckpt, pose_correction, f0)
        R_rel   = R_l2c0 @ gt_R.T
        cos_a   = ((R_rel.diagonal().sum().clamp(-1, 3) - 1) / 2).clamp(-1, 1)
        rot_err = math.degrees(math.acos(cos_a.item()))
        T_err   = (T_l2c0 - gt_T).norm().item()
    print(blue(f"  Current rot_err={rot_err:.4f} deg  T_err={T_err:.4f} m"))

    # ── Pre-compute GT-projected reference depth images ──────────────────────
    print(blue("[VizCamDepth] Building GT-projected reference depth images..."))
    cam_depth_refs = _build_cam_depth_refs(scene, pose_correction, cam_cameras)

    train_frames  = sorted(scene.train_lidar.train_frames)
    cam_frame_ids = [f for f in train_frames if f in cam_cameras]
    if cli.frames is not None:
        frames_to_viz = [f for f in cli.frames if f in cam_cameras]
        if not frames_to_viz:
            print(red("None of the requested frames have camera data."))
            sys.exit(1)
    else:
        frames_to_viz = cam_frame_ids[:3]
    print(blue(f"[VizCamDepth] Visualising frames: {frames_to_viz}"))

    with torch.no_grad():
        for frame in frames_to_viz:
            out_path = os.path.join(cli.output_dir, f"frame_{frame:04d}.png")
            visualise_frame(frame, scene, ckpt, pose_correction,
                            cam_cameras, cam_images, args, out_path,
                            use_gt_pose=cli.use_gt_pose,
                            gt_R=gt_R, gt_T=gt_T,
                            cam_depth_ref=cam_depth_refs.get(frame))

    print(green(f"\n[VizCamDepth] Done. Saved to: {cli.output_dir}"))


if __name__ == "__main__":
    main()
