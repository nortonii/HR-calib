#!/usr/bin/env python3
"""Multi-frame RGB-D extrinsic calibration using MatchAnything RoMa via vismatch."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataloader.kitti_calib_loader import _parse_kitti_calib_file
from lib.utils.rgbd_calibration import (
    apply_temporal_support,
    build_temporal_residuals,
    build_temporal_track_residuals,
    build_frame_correspondence,
    build_matcher,
    depth_to_match_image,
    initialize_from_extrinsic,
    initialize_shared_extrinsic,
    load_camera_model,
    load_depth_map,
    load_rgb_image,
    make_depth_overlay,
    match_cross_modal,
    match_temporal_frames,
    optimize_shared_extrinsic,
    project_depth_to_rgb,
    score_temporal_support,
    save_extrinsic_yaml,
)


def _collect_paths(path_value: str, patterns: tuple[str, ...]) -> list[Path]:
    input_path = Path(path_value)
    if input_path.is_dir():
        files = []
        for pattern in patterns:
            files.extend(input_path.glob(pattern))
        return sorted(path for path in files if path.is_file())
    matched = sorted(Path(path) for path in glob.glob(path_value))
    return [path for path in matched if path.is_file()]


def _load_kitti_rgb_camera_poses(rgb_paths: list[Path]) -> list[np.ndarray]:
    if not rgb_paths:
        return []
    scene_dir = rgb_paths[0].parent
    if any(path.parent != scene_dir for path in rgb_paths):
        raise RuntimeError("Temporal residuals require RGB frames to come from one scene directory.")

    pose_file = scene_dir / "LiDAR_poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"LiDAR_poses.txt not found next to RGB frames: {pose_file}")

    scene_name = scene_dir.name
    seq_num = int(scene_name.split("-")[0])
    calib_path = scene_dir.parent / "calibs" / f"{seq_num:02d}.txt"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    _, Tr = _parse_kitti_calib_file(str(calib_path))
    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3] = Tr
    cam_to_lidar = np.linalg.inv(Tr4)
    lidar_poses = np.loadtxt(pose_file).reshape(-1, 4, 4)

    poses = []
    for rgb_path in rgb_paths:
        frame_id = int(rgb_path.stem)
        if frame_id >= len(lidar_poses):
            raise IndexError(f"Frame {frame_id} exceeds LiDAR pose table length {len(lidar_poses)}")
        poses.append((lidar_poses[frame_id] @ cam_to_lidar).astype(np.float64))
    return poses


def _load_initial_extrinsic(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    extrinsic_path = Path(path)
    with open(extrinsic_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if "T_rgb_d" not in payload:
        raise KeyError(f"T_rgb_d not found in {extrinsic_path}")
    pose_payload = payload["T_rgb_d"]
    if "rotation_vector" in pose_payload:
        rvec = np.asarray(pose_payload["rotation_vector"], dtype=np.float64).reshape(3)
    elif "rotation_matrix" in pose_payload:
        rotation_matrix = np.asarray(pose_payload["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        rvec = rvec.reshape(3)
    else:
        raise KeyError(f"Rotation not found in {extrinsic_path}")
    if "translation_xyz" not in pose_payload:
        raise KeyError(f"translation_xyz not found in {extrinsic_path}")
    tvec = np.asarray(pose_payload["translation_xyz"], dtype=np.float64).reshape(3)
    return rvec, tvec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-D multi-frame extrinsic calibration with MatchAnything RoMa")
    parser.add_argument("--rgb", required=True, help="RGB image directory or glob pattern.")
    parser.add_argument("--depth", required=True, help="Depth map directory or glob pattern.")
    parser.add_argument("--rgb_intrinsics", required=True, help="RGB camera intrinsics file (.yaml/.json/.npz/.npy).")
    parser.add_argument("--depth_intrinsics", required=True, help="Depth camera intrinsics file (.yaml/.json/.npz/.npy).")
    parser.add_argument("--output_dir", required=True, help="Output directory for extrinsic.yaml and visualizations.")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor applied after loading each depth map.")
    parser.add_argument("--depth_npz_key", default=None, help="Array key for .npz depth files.")
    parser.add_argument("--device", default="cuda", help="Matcher device: cuda or cpu.")
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com", help="Hugging Face endpoint used when local cache is missing.")
    parser.add_argument("--matcher_name", choices=("matchanything-roma", "minima-roma"), default="matchanything-roma",
                        help="Sparse cross-modal matcher backend.")
    parser.add_argument("--matcher_minima_root", type=str, default=None,
                        help="Optional MINIMA repo root for matcher_name=minima-roma.")
    parser.add_argument("--matcher_minima_ckpt", type=str, default=None,
                        help="Optional MINIMA RoMa checkpoint path for matcher_name=minima-roma.")
    parser.add_argument("--matcher_resize", type=int, default=832, help="Input resize for the matcher backend.")
    parser.add_argument("--matcher_max_keypoints", type=int, default=2048, help="Max keypoints passed to MatchAnything.")
    parser.add_argument("--matcher_ransac_reproj_thresh", type=float, default=3.0, help="RANSAC reprojection threshold inside MatchAnything.")
    parser.add_argument("--matcher_match_threshold", type=float, default=0.2, help="MatchAnything coarse match threshold.")
    parser.add_argument("--min_cross_matches", type=int, default=20, help="Minimum cross-modal matches per frame before depth filtering.")
    parser.add_argument("--min_depth_matches", type=int, default=12, help="Minimum valid RGB-depth 2D-3D correspondences per frame.")
    parser.add_argument("--min_pnp_inliers", type=int, default=8, help="Minimum shared-PnP inliers required for initialization.")
    parser.add_argument("--pnp_reproj_error", type=float, default=4.0, help="solvePnPRansac reprojection error for shared initialization.")
    parser.add_argument("--pnp_iterations", type=int, default=1000, help="solvePnPRansac iteration count for shared initialization.")
    parser.add_argument("--initial_extrinsic", default=None, help="Optional extrinsic.yaml used as the initialization for refinement.")
    parser.add_argument("--depth_min", type=float, default=0.1, help="Minimum valid depth value.")
    parser.add_argument("--depth_max", type=float, default=80.0, help="Maximum valid depth value.")
    parser.add_argument("--depth_search_radius", type=int, default=2, help="Pixel radius to search a nearby valid depth.")
    parser.add_argument("--depth_percentile_low", type=float, default=5.0, help="Low percentile for depth visualization.")
    parser.add_argument("--depth_percentile_high", type=float, default=95.0, help="High percentile for depth visualization.")
    parser.add_argument("--depth_use_inverse", action="store_true", help="Invert depth normalization before color mapping.")
    parser.add_argument("--temporal_neighbors", type=int, default=1, help="Number of forward RGB-RGB neighbor pairs per frame.")
    parser.add_argument("--temporal_support_mode", choices=("none", "weight", "filter", "filter_weight"), default="none", help="Use temporal matches to weight or filter cross-modal correspondences before second-stage optimization.")
    parser.add_argument("--temporal_support_min", type=int, default=1, help="Minimum temporal support count required when temporal support filtering is enabled.")
    parser.add_argument("--temporal_support_scale", type=float, default=1.0, help="Additional per-point weight scale applied for each temporal support hit.")
    parser.add_argument("--temporal_support_tolerance", type=float, default=6.0, help="Projection tolerance in pixels used to count a temporal support hit under the initialization pose.")
    parser.add_argument("--temporal_residual_mode", choices=("pairwise", "track"), default="pairwise", help="How to convert RGB-RGB matches into temporal residuals.")
    parser.add_argument("--temporal_residual_weight", type=float, default=0.0, help="Weight for lightweight cross-frame projection residuals during global refinement.")
    parser.add_argument("--temporal_residual_match_radius", type=float, default=4.0, help="Max RGB pixel distance to attach a temporal match to an existing source 3D point.")
    parser.add_argument("--temporal_residual_min_matches", type=int, default=8, help="Minimum attached temporal matches required to keep one directed temporal residual block.")
    parser.add_argument("--temporal_track_min_length", type=int, default=3, help="Minimum number of frame observations required for one temporal track when using track mode.")
    parser.add_argument("--temporal_regularization", type=float, default=0.0, help="Optional weak initialization prior weight on both rotation and translation during global refinement.")
    parser.add_argument("--staged_refinement", action="store_true", help="Run rotation-first then translation-focused staged refinement before the final joint optimization.")
    parser.add_argument("--staged_depth_split", type=float, default=20.0, help="Depth threshold in meters used to separate far points for rotation and near points for translation during staged refinement.")
    parser.add_argument("--solver_backend", choices=("auto", "opencv", "scipy"), default="auto", help="Shared extrinsic refinement backend. 'auto' prefers OpenCV solvePnPRefineLM when compatible, otherwise falls back to SciPy.")
    parser.add_argument("--save_visualizations", action="store_true", help="Save per-frame RGB-depth overlay images.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optionally truncate the sequence after this many frame pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = _collect_paths(args.rgb, ("*.png", "*.jpg", "*.jpeg", "*.bmp"))
    depth_paths = _collect_paths(args.depth, ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.exr", "*.npy", "*.npz"))
    if args.max_frames is not None:
        rgb_paths = rgb_paths[: args.max_frames]
        depth_paths = depth_paths[: args.max_frames]

    if not rgb_paths:
        raise RuntimeError("No RGB images found.")
    if len(rgb_paths) != len(depth_paths):
        raise RuntimeError(f"RGB/depth frame count mismatch: {len(rgb_paths)} vs {len(depth_paths)}")

    rgb_camera = load_camera_model(args.rgb_intrinsics)
    depth_camera = load_camera_model(args.depth_intrinsics)
    matcher = build_matcher(
        matcher_name=args.matcher_name,
        device=args.device,
        max_num_keypoints=args.matcher_max_keypoints,
        ransac_reproj_thresh=args.matcher_ransac_reproj_thresh,
        img_resize=args.matcher_resize,
        match_threshold=args.matcher_match_threshold,
        hf_endpoint=args.hf_endpoint,
        minima_root=args.matcher_minima_root,
        minima_ckpt=args.matcher_minima_ckpt,
    )

    rgb_images = [load_rgb_image(path) for path in rgb_paths]
    depth_maps = [load_depth_map(path, depth_scale=args.depth_scale, npz_key=args.depth_npz_key) for path in depth_paths]
    depth_match_images = [
        depth_to_match_image(
            depth_map,
            percentile_low=args.depth_percentile_low,
            percentile_high=args.depth_percentile_high,
            use_inverse=args.depth_use_inverse,
        )
        for depth_map in depth_maps
    ]

    temporal_weights, temporal_summaries, temporal_matches = match_temporal_frames(
        matcher=matcher,
        rgb_images=rgb_images,
        max_pairs=max(0, args.temporal_neighbors),
    )
    needs_camera_poses = args.temporal_residual_weight > 0.0 or args.temporal_support_mode != "none"
    rgb_camera_poses_c2w = _load_kitti_rgb_camera_poses(rgb_paths) if needs_camera_poses else []

    frame_data_list = []
    for index, (rgb_path, depth_path, rgb_image, depth_map, depth_match_image) in enumerate(
        zip(rgb_paths, depth_paths, rgb_images, depth_maps, depth_match_images)
    ):
        rgb_points, depth_points, _ = match_cross_modal(matcher, rgb_image, depth_match_image)
        if rgb_points.shape[0] < args.min_cross_matches:
            print(f"[skip] {rgb_path.name}: cross-modal matches {rgb_points.shape[0]} < {args.min_cross_matches}")
            continue

        frame_data = build_frame_correspondence(
            frame_name=rgb_path.stem,
            rgb_path=rgb_path,
            depth_path=depth_path,
            rgb_points=rgb_points,
            depth_points=depth_points,
            depth_map=depth_map,
            depth_camera=depth_camera,
            min_depth=args.depth_min,
            max_depth=args.depth_max,
            search_radius=args.depth_search_radius,
        )
        if frame_data is None or frame_data.points_3d.shape[0] < args.min_depth_matches:
            valid_count = 0 if frame_data is None else frame_data.points_3d.shape[0]
            print(f"[skip] {rgb_path.name}: valid 2D-3D matches {valid_count} < {args.min_depth_matches}")
            continue

        frame_data.temporal_weight = float(temporal_weights[index])
        frame_data.frame_index = index
        frame_data.frame_id = int(rgb_path.stem)
        frame_data_list.append(frame_data)

    if not frame_data_list:
        raise RuntimeError("No valid frames remained after matching and depth filtering.")

    initialization_mode = "shared_pnp"
    initial_extrinsic_path = None
    if args.initial_extrinsic is not None:
        initial_extrinsic_path = str(Path(args.initial_extrinsic).resolve())
        initial_rvec, initial_tvec = _load_initial_extrinsic(initial_extrinsic_path)
        initial_rvec, initial_tvec = initialize_from_extrinsic(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            rvec=initial_rvec,
            tvec=initial_tvec,
            reproj_error=args.pnp_reproj_error,
            min_inliers=args.min_pnp_inliers,
        )
        initialization_mode = "external_extrinsic"
        print(f"[info] seeded initialization from {initial_extrinsic_path}")
    else:
        initial_rvec, initial_tvec = initialize_shared_extrinsic(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            reproj_error=args.pnp_reproj_error,
            iterations=args.pnp_iterations,
            min_inliers=args.min_pnp_inliers,
            filter_frames=False,
        )
    for frame_data in frame_data_list:
        print(
            f"[keep] {Path(frame_data.rgb_path).name}: global_inliers={frame_data.pnp_inliers} "
            f"temporal_w={frame_data.temporal_weight:.2f} init_rmse={frame_data.pnp_reproj_error:.3f}px"
        )
    temporal_support_points = 0
    temporal_supported_points = 0
    if args.temporal_support_mode != "none":
        support_scores = score_temporal_support(
            frame_data_list=frame_data_list,
            temporal_matches=temporal_matches,
            rgb_camera_poses_c2w=rgb_camera_poses_c2w,
            rgb_camera=rgb_camera,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
            match_radius_px=args.temporal_residual_match_radius,
            projection_tolerance_px=args.temporal_support_tolerance,
        )
        temporal_support_points = int(sum(len(scores) for scores in support_scores.values()))
        temporal_supported_points = int(sum(np.count_nonzero(scores >= args.temporal_support_min) for scores in support_scores.values()))
        frame_data_list = apply_temporal_support(
            frame_data_list=frame_data_list,
            support_scores=support_scores,
            mode=args.temporal_support_mode,
            min_support=args.temporal_support_min,
            support_scale=args.temporal_support_scale,
        )
        print(
            f"[info] temporal support mode={args.temporal_support_mode} "
            f"supported_points={temporal_supported_points}/{temporal_support_points}"
        )
    temporal_track_count = 0
    if args.temporal_residual_weight > 0.0:
        if args.temporal_residual_mode == "track":
            temporal_residuals, temporal_track_count = build_temporal_track_residuals(
                frame_data_list=frame_data_list,
                temporal_matches=temporal_matches,
                rgb_camera_poses_c2w=rgb_camera_poses_c2w,
                match_radius_px=args.temporal_residual_match_radius,
                min_track_length=args.temporal_track_min_length,
                min_block_points=args.temporal_residual_min_matches,
            )
        else:
            temporal_residuals = build_temporal_residuals(
                frame_data_list=frame_data_list,
                temporal_matches=temporal_matches,
                rgb_camera_poses_c2w=rgb_camera_poses_c2w,
                match_radius_px=args.temporal_residual_match_radius,
                min_matches=args.temporal_residual_min_matches,
            )
    else:
        temporal_residuals = []
    if temporal_residuals:
        total_temporal_points = int(sum(res.source_points_3d.shape[0] for res in temporal_residuals))
        print(
            f"[info] temporal residual mode={args.temporal_residual_mode} blocks={len(temporal_residuals)} "
            f"points={total_temporal_points} tracks={temporal_track_count} weight={args.temporal_residual_weight:.3f}"
        )
    comparison = optimize_shared_extrinsic(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        initial_rvec=initial_rvec,
        initial_tvec=initial_tvec,
        temporal_residuals=temporal_residuals,
        temporal_residual_weight=args.temporal_residual_weight,
        temporal_regularization=args.temporal_regularization,
        staged_refinement=args.staged_refinement,
        staged_depth_split=args.staged_depth_split,
        solver_backend=args.solver_backend,
    )
    calibration = comparison.optimized

    extra_metrics = {
        "sequence_frames": len(rgb_paths),
        "valid_frames": len(frame_data_list),
        "initialization_mode": initialization_mode,
        "initial_extrinsic": initial_extrinsic_path,
        "temporal_edges": len(temporal_summaries),
        "temporal_mean_matches": float(np.mean([summary.num_matches for summary in temporal_summaries])) if temporal_summaries else 0.0,
        "temporal_support_mode": args.temporal_support_mode,
        "temporal_support_min": int(args.temporal_support_min),
        "temporal_support_scale": float(args.temporal_support_scale),
        "temporal_support_tolerance": float(args.temporal_support_tolerance),
        "temporal_support_points": temporal_support_points,
        "temporal_supported_points": temporal_supported_points,
        "temporal_residual_mode": args.temporal_residual_mode,
        "temporal_track_count": temporal_track_count,
        "temporal_residual_blocks": len(temporal_residuals),
        "temporal_residual_points": int(sum(res.source_points_3d.shape[0] for res in temporal_residuals)) if temporal_residuals else 0,
        "staged_refinement": bool(args.staged_refinement),
        "staged_depth_split": float(args.staged_depth_split),
    }
    save_extrinsic_yaml(
        output_path=output_dir / "extrinsic.yaml",
        comparison=comparison,
        rgb_camera=rgb_camera,
        depth_camera=depth_camera,
        extra_metrics=extra_metrics,
    )

    if args.save_visualizations:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for rgb_path, rgb_image, depth_map in zip(rgb_paths, rgb_images, depth_maps):
            warped_depth = project_depth_to_rgb(
                depth_map=depth_map,
                depth_camera=depth_camera,
                rgb_camera=rgb_camera,
                rotation_matrix=calibration.rotation_matrix,
                translation=calibration.translation,
                rgb_shape=rgb_image.shape[:2],
            )
            overlay = make_depth_overlay(rgb_image, warped_depth)
            cv2.imwrite(str(overlay_dir / f"{rgb_path.stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[done] saved {output_dir / 'extrinsic.yaml'}")
    print(
        "[done] initial -> optimized mean reprojection: "
        f"{comparison.initial.mean_reprojection_error:.4f}px -> "
        f"{comparison.optimized.mean_reprojection_error:.4f}px"
    )
    print(
        "[done] initial -> optimized median reprojection: "
        f"{comparison.initial.median_reprojection_error:.4f}px -> "
        f"{comparison.optimized.median_reprojection_error:.4f}px"
    )
    print(f"[done] mean reprojection error: {calibration.mean_reprojection_error:.4f}px")
    print(f"[done] median reprojection error: {calibration.median_reprojection_error:.4f}px")


if __name__ == "__main__":
    main()
