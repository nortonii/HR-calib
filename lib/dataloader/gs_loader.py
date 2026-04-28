import os
import random
from typing import Dict

import numpy as np
import open3d as o3d
import torch
from lib.scene import BoundingBox, GaussianModel, LiDARSensor, Scene
from lib.utils import general_utils
from lib.utils.camera_utils import cameraList_from_camInfos
from lib.utils.general_utils import build_rotation
from lib.utils.graphics_utils import BasicPointCloud
from PIL import Image


def _sample_inverse_distance_sphere_points(points, color_intensity, num_points):
    if num_points <= 0 or points.shape[0] == 0:
        empty_points = points.new_zeros((0, 3))
        empty_color_intensity = color_intensity.new_zeros((0, color_intensity.shape[1]))
        empty_normals = points.new_zeros((0, 3))
        return empty_points, empty_color_intensity, empty_normals

    scene_center = points.mean(dim=0)
    radii = torch.norm(points - scene_center, dim=1)
    valid_radii = radii[radii > 1.0e-3]
    if valid_radii.numel() == 0:
        empty_points = points.new_zeros((0, 3))
        empty_color_intensity = color_intensity.new_zeros((0, color_intensity.shape[1]))
        empty_normals = points.new_zeros((0, 3))
        return empty_points, empty_color_intensity, empty_normals

    radius_min = torch.quantile(valid_radii, 0.10)
    radius_max = torch.quantile(valid_radii, 0.90)
    radius_max = torch.maximum(radius_max, radius_min * 1.01)

    sample_u = torch.rand(num_points, dtype=points.dtype, device=points.device)
    sample_radii = torch.exp(
        torch.log(radius_min) + sample_u * (torch.log(radius_max) - torch.log(radius_min))
    )

    directions = torch.randn((num_points, 3), dtype=points.dtype, device=points.device)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    sampled_points = scene_center.unsqueeze(0) + directions * sample_radii.unsqueeze(-1)
    sampled_normals = -directions

    base_ci = color_intensity.mean(dim=0, keepdim=True)
    sampled_color_intensity = base_ci.repeat(num_points, 1)
    return sampled_points, sampled_color_intensity, sampled_normals


def _resolve_init_scale(configured_scale: float, fallback_scale: float) -> float:
    scale = float(configured_scale)
    if scale > 0.0:
        return scale
    return max(float(fallback_scale), 1.0e-3)


def _resolve_inverse_distance_init_num(args) -> int:
    model_cfg = getattr(args, "model", None)
    if model_cfg is None:
        return 0
    data_type = str(getattr(args, "data_type", "")).strip().lower()
    if data_type == "kitticalib":
        override = getattr(model_cfg, "kitticalib_inverse_distance_init_num", None)
        if override is not None:
            return max(int(override), 0)
    return max(int(getattr(model_cfg, "inverse_distance_init_num", 0)), 0)


def _filter_points_by_min_distance(
    points: torch.Tensor,
    intensities: torch.Tensor,
    sensor_center: torch.Tensor,
    min_distance: float,
    normals: torch.Tensor | None = None,
):
    threshold = float(min_distance)
    if threshold <= 0.0 or points.numel() == 0:
        return points, intensities, normals

    center = sensor_center.to(device=points.device, dtype=points.dtype).reshape(1, 3)
    keep_mask = torch.norm(points - center, dim=1) >= threshold
    filtered_normals = None if normals is None else normals[keep_mask]
    return points[keep_mask], intensities[keep_mask], filtered_normals


class SceneLidar(Scene):
    def __init__(self, args, waymo_raw_pkg, shuffle=True, resize_ratio=1, test=False):
        scene_id = (
            str(args.scene_id) if isinstance(args.scene_id, int) else args.scene_id
        )
        self.output_dir = os.path.join(
            args.model_dir, args.task_name, args.exp_name, "scene_" + scene_id
        )
        self.model_save_dir = os.path.join(self.output_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.loaded_iter = None
        self.camera_extent = 0
        self.gaussians_assets = [
            GaussianModel(
                args.model.dimension, args.model.sh_degree, extent=self.camera_extent
            )
        ]
        dc_only_sh = bool(getattr(args.model, "dc_only_sh", False))
        if dc_only_sh:
            self.gaussians_assets[0].set_dc_only_sh(True)

        lidar: Dict[int, LiDARSensor] = waymo_raw_pkg[0]
        bboxes: Dict[str, BoundingBox] = waymo_raw_pkg[1]
        frame_range = args.frame_length
        available_frame_ids = sorted(int(frame_id) for frame_id in lidar.raw_points_local.keys())
        if available_frame_ids:
            eval_frame_set = {int(frame_id) for frame_id in args.eval_frames}
            eval_frames = [frame_id for frame_id in available_frame_ids if frame_id in eval_frame_set]
            train_frames = [frame_id for frame_id in available_frame_ids if frame_id not in eval_frame_set]
        else:
            eval_frames = args.eval_frames
            train_frames = [
                frame_id
                for frame_id in range(frame_range[0], frame_range[1] + 1)
                if frame_id not in eval_frames
            ]

        self.train_lidar = lidar
        self.train_lidar.set_frames(train_frames, eval_frames)

        print("[Loaded] background guassians")

        # initialize objects with bounding boxes
        if args.dynamic:
            obj_ids = list(bboxes.keys())
            for obj_id in obj_ids:
                bbox = bboxes[obj_id]
                general_utils.fill_zeros_with_previous_nonzero(
                    available_frame_ids if available_frame_ids else range(frame_range[0], frame_range[1] + 1),
                    bbox.frame,
                )

                abs_velocities = []
                velocity_frame_ids = available_frame_ids if available_frame_ids else list(range(frame_range[0], frame_range[1] + 1))
                for frame, next_frame in zip(velocity_frame_ids[:-1], velocity_frame_ids[1:]):
                    velocity = bbox.frame[next_frame][0] - bbox.frame[frame][0]
                    abs_velocities.append(torch.norm(velocity).item())
                avg_velocity = torch.tensor(abs_velocities).mean().item()

                if avg_velocity > 0.01 and bbox.object_type == 1:
                    extent = (
                        torch.norm(bbox.size, keepdim=False).item()
                        * args.model.object_extent_factor
                    )
                    gaussian_model = GaussianModel(
                        args.model.dimension,
                        args.model.sh_degree,
                        extent=extent,
                        bounding_box=bbox,
                    )
                    if dc_only_sh:
                        gaussian_model.set_dc_only_sh(True)
                    gaussian_model.tmp_points_intensities_list = []
                    self.gaussians_assets.append(gaussian_model)

            if not bboxes:
                print("No dynamic objects in the scene")
                args.dynamic = False

        # initialize bkgd points — with disk cache to skip repeated normal
        # estimation and voxel downsampling (the dominant startup cost).
        source_dir  = getattr(args, "source_dir", "data")
        scene_id    = str(getattr(args, "scene_id", "scene"))
        data_scene  = str(getattr(args, "kitti_calib_scene", scene_id))
        voxel_size  = float(getattr(args.model, "voxel_size", 0.15))
        min_init_distance = float(getattr(args.model, "bkgd_init_min_distance", 0.0))
        use_voxel   = bool(getattr(args.opt, "use_voxel_init", True))
        cache_key   = (
            f"v4_fr{frame_range[0]}_{frame_range[1]}_"
            f"vs{voxel_size}_vx{int(use_voxel)}_md{min_init_distance:.3f}"
        )
        cache_path  = os.path.join(source_dir, data_scene, f".init_cache_{cache_key}.npz")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            print(f"[Init] Loading cached point cloud from {cache_path}")
            cached = np.load(cache_path)
            all_points = torch.from_numpy(cached["all_points"]).float()
            ip         = torch.from_numpy(cached["ip"]).float()
            normals    = torch.from_numpy(cached["normals"]).float()
        else:
            all_points = []
            all_intensity = []
            all_normals = []
            init_frame_ids = available_frame_ids if available_frame_ids else list(range(frame_range[0], frame_range[1] + 1))
            for frame in init_frame_ids:
                lidar_pts, lidar_intensity = lidar.get_raw_points(frame, world=True)
                raw_point_count = int(lidar_pts.shape[0])
                lidar_pts, lidar_intensity, _ = _filter_points_by_min_distance(
                    lidar_pts,
                    lidar_intensity,
                    lidar.sensor_center[frame],
                    min_init_distance,
                )
                removed_near_points = raw_point_count - int(lidar_pts.shape[0])
                if removed_near_points > 0:
                    print(
                        f"[Init] Frame {frame}: filtered {removed_near_points} near-self points "
                        f"(min_distance={min_init_distance:.2f}m)"
                    )

                points_lidar = o3d.geometry.PointCloud()
                points_lidar.points = o3d.utility.Vector3dVector(
                    lidar_pts.cpu().numpy().astype(np.float64)
                )
                points_lidar.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6)
                )
                normals = torch.from_numpy(np.asarray(points_lidar.normals)).float()
                for gaussian_model in self.gaussians_assets[1:]:
                    bbox = gaussian_model.bounding_box
                    T = bbox.frame[frame][0].cpu()
                    R = build_rotation(bbox.frame[frame][1])[0].cpu()
                    points_in_local = (lidar_pts - T) @ R.inverse().T
                    normals_in_local = normals @ R.inverse().T
                    mask = (torch.abs(points_in_local) < bbox.size.cpu() / 2).all(dim=1)
                    gaussian_model.tmp_points_intensities_list.append(
                        (
                            points_in_local[mask],
                            lidar_intensity[mask],
                            normals_in_local[mask],
                        )
                    )
                    lidar_pts, lidar_intensity = lidar_pts[~mask], lidar_intensity[~mask]
                    normals = normals[~mask]

                all_points.append(lidar_pts)
                all_intensity.append(lidar_intensity)
                all_normals.append(normals)

            all_points = torch.cat(all_points, dim=0)
            all_intensity = torch.cat(all_intensity, dim=0)
            all_normals = torch.cat(all_normals, dim=0)
            hit_probs = torch.ones(all_points.shape[0])
            drop_probs = torch.zeros(all_points.shape[0])
            ip = torch.stack([all_intensity, hit_probs, drop_probs], dim=1)

            if use_voxel:
                points_lidar = o3d.geometry.PointCloud()
                points_lidar.points = o3d.utility.Vector3dVector(
                    all_points.cpu().numpy().astype(np.float64)
                )
                points_lidar.colors = o3d.utility.Vector3dVector(
                    ip.cpu().numpy().astype(np.float64)
                )
                points_lidar.normals = o3d.utility.Vector3dVector(
                    all_normals.cpu().numpy().astype(np.float64)
                )
                downsample_points_lidar = points_lidar.voxel_down_sample(
                    voxel_size=voxel_size
                )
                all_points = torch.tensor(
                    np.asarray(downsample_points_lidar.points)
                ).float()
                ip = torch.tensor(np.asarray(downsample_points_lidar.colors)).float()
                normals = torch.tensor(np.asarray(downsample_points_lidar.normals)).float()
            else:
                normals = all_normals

            np.savez_compressed(
                cache_path,
                all_points=all_points.numpy(),
                ip=ip.numpy(),
                normals=normals.numpy(),
            )
            print(f"[Init] Saved point cloud cache to {cache_path}")

        inverse_distance_init_num = _resolve_inverse_distance_init_num(args)
        if inverse_distance_init_num > 0:
            sampled_points, sampled_ip, sampled_normals = _sample_inverse_distance_sphere_points(
                all_points, ip, inverse_distance_init_num
            )
            if sampled_points.shape[0] > 0:
                print(f"[Init] Adding {sampled_points.shape[0]} inverse-distance sphere samples")
                all_points = torch.cat([all_points, sampled_points], dim=0)
                ip = torch.cat([ip, sampled_ip], dim=0)
                normals = torch.cat([normals, sampled_normals], dim=0)

        # calcualte extent
        scene_center = all_points.mean(dim=0)
        point_extent = 2 * torch.norm(all_points - scene_center, dim=1)
        self.camera_extent = (
            args.model.bkgd_extent_factor
            * torch.quantile(point_extent, 0.90).int().item()
        )
        self.gaussians_assets[0].extent = self.camera_extent
        background_init_scale = _resolve_init_scale(
            getattr(args.model, "bkgd_init_scale", 0.0),
            voxel_size * 0.5,
        )

        pcd = BasicPointCloud(all_points, ip, normals=normals)
        self.gaussians_assets[0].create_from_pcd(
            pcd,
            args.opt.use_normal_init,
            init_scale=background_init_scale,
        )

        # initialize objects points
        points_num = args.model.obj_pt_num
        object_init_scale = _resolve_init_scale(
            getattr(args.model, "obj_init_scale", 0.0),
            background_init_scale,
        )
        for gaussian_model in self.gaussians_assets[1:]:
            points = torch.cat(
                [point for point, _, _ in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            intensities = torch.cat(
                [
                    intensitie
                    for _, intensitie, _ in gaussian_model.tmp_points_intensities_list
                ],
                dim=0,
            )
            normals = torch.cat(
                [normal for _, _, normal in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            if points.shape[0] < points_num:
                extra_num = points_num - points.shape[0]
                extra_points = torch.zeros((extra_num, 3))
                for i in range(3):
                    extra_points[:, i] = (
                        torch.rand(size=(extra_num,)) * bbox.size[i].cpu()
                        + bbox.min_xyz[i].cpu()
                    )
                points = torch.cat([points, extra_points], dim=0)

                extra_points_intensity = torch.rand(extra_num)
                intensities = torch.cat([intensities, extra_points_intensity], dim=0)

                theta = np.random.uniform(0, 2 * np.pi, extra_num)
                phi = np.random.uniform(0, np.pi, extra_num)
                extra_normals = np.zeros((extra_num, 3))
                extra_normals[:, 0] = np.sin(phi) * np.cos(theta)
                extra_normals[:, 1] = np.sin(phi) * np.sin(theta)
                extra_normals[:, 2] = np.cos(phi)

                normals = torch.cat(
                    [normals, torch.tensor(extra_normals).float()], dim=0
                )

            elif points.shape[0] > points_num:
                mask = torch.randperm(points.shape[0])[:points_num]
                points = points[mask]
                intensities = intensities[mask]
                normals = normals[mask]

            hit_probs = torch.ones(points.shape[0])
            drop_probs = torch.zeros(points.shape[0])
            ip = torch.stack([intensities, hit_probs, drop_probs], dim=1)
            pcd = BasicPointCloud(points, ip, normals=normals)
            gaussian_model.create_from_pcd(
                pcd,
                args.opt.use_normal_init,
                init_scale=object_init_scale,
            )
            del gaussian_model.tmp_points_intensities_list

        print("[Loaded] object guassians")

    def training_setup(self, args):
        for gs in self.gaussians_assets:
            gs.training_setup(args)

    def restore(self, model_params, args):
        for i, gs in enumerate(self.gaussians_assets):
            gs.restore(model_params[i], args)

    def update_learning_rate(self, iteration):
        for gs in self.gaussians_assets:
            gs.update_learning_rate(iteration)

    def oneupSHdegree(self):
        for gs in self.gaussians_assets:
            gs.oneupSHdegree()

    def save(self, iteration, model_name):
        model_pth = os.path.join(self.model_save_dir, model_name + ".pth")
        model_params = []
        for i, gs in enumerate(self.gaussians_assets):
            model_params.append(gs.capture())
        torch.save((model_params, iteration), model_pth)

    def optimize(
        self,
        args,
        iteration,
        mean_grads,
        accum_weights,
        visibility_filter_list,
        radii_list,
    ):

        clone_num, split_num, prune_scale_num, prune_opacity_num = 0, 0, 0, 0
        prune_nonfinite_num = 0

        begin_index = 0
        for gaussians in self.gaussians_assets:
            points_num = gaussians.get_local_xyz.shape[0]
            if mean_grads is None or accum_weights is None:
                instance_mean_grads = None
                instance_accum_weights = None
            else:
                instance_mean_grads = mean_grads[begin_index : begin_index + points_num]
                instance_accum_weights = (
                    accum_weights[begin_index : begin_index + points_num] > 0
                )
            begin_index += points_num

            # Densification
            if bool(getattr(args.opt, "enable_densification", True)) and iteration < args.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if instance_mean_grads is not None and instance_accum_weights is not None:
                    gaussians.add_densification_stats(
                        instance_mean_grads, instance_accum_weights
                    )

                if (
                    iteration > args.opt.densify_from_iter
                    and iteration % args.opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > args.opt.opacity_reset_interval else None
                    )
                    densify_info = gaussians.densify_and_prune(
                        args.opt, 0.005, size_threshold
                    )
                    clone_num += densify_info[0]
                    split_num += densify_info[1]
                    prune_scale_num += densify_info[2]
                    prune_opacity_num += densify_info[3]

                if iteration % args.opt.opacity_reset_interval == 0 or (
                    args.model.white_background
                    and iteration == args.opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # args.optimizer step
            if iteration < args.opt.iterations:
                if bool(getattr(args.opt, "prune_nonfinite_gaussians", False)):
                    prune_nonfinite_num += gaussians.prune_nonfinite_points()
                    gaussians.sanitize_gradients()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        return clone_num, split_num, prune_scale_num, prune_opacity_num, prune_nonfinite_num
