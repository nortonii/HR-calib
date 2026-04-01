#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
import os
from simple_knn._C import distCUDA2
from lib.scene.bounding_box import BoundingBox
from lib.utils.console_utils import *
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from lib.utils.general_utils import generate_random_quaternion_with_fixed_normal
from lib.utils.sh_utils import RGB2SH

class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, dimension, sh_degree : int, extent : int = 200, bounding_box : BoundingBox= None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.dc_only_sh = False
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_rgb_dc = torch.empty(0)
        self._features_rgb_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity_cam = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.densify_scale_threshold = 0
        self.spatial_lr_scale = 0

        self.bounding_box = bounding_box
        self.extent = extent

        self.dimension = dimension # 3dgs/2dgs

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_cam,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._features_rgb_dc,
            self._features_rgb_rest,
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 15:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._opacity_cam,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._features_rgb_dc,
                self._features_rgb_rest,
            ) = model_args
        elif len(model_args) == 13:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._opacity_cam,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._init_rgb_from_lidar_features()
        elif len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._opacity_cam = self._opacity.clone()
            self._init_rgb_from_lidar_features()
        else:
            raise ValueError(f"Unsupported GaussianModel checkpoint format with {len(model_args)} entries.")
        if self.dc_only_sh:
            self.active_sh_degree = 0
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self._load_optimizer_state_compat(opt_dict)
        if self.dc_only_sh:
            for group in self.optimizer.param_groups:
                if group["name"] in {"f_rest", "f_rgb_rest"}:
                    group["lr"] = 0.0
        print("Number of points of restored gaussian: ", self.get_local_xyz.shape[0])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)

    #@property
    def get_rotation(self, timestamp = 0.0):
        # if self.bounding_box is not None and self.bounding_box.frame:
        #     rot_in_local = torch.nn.functional.normalize(self._rotation, dim=1)
        #     quaternion = quaternion_raw_multiply(timer, rot_in_local, self.bounding_box.frame[timestamp][1])
        #     return quaternion
        # else:
        #     return self.rotation_activation(self._rotation)
        if self.bounding_box is not None and timestamp in self.bounding_box.frame:
            obj_rot = self.bounding_box.frame[timestamp][1]
        else:
            obj_rot = torch.zeros((1, 4), device="cuda")
        return obj_rot, self.rotation_activation(self._rotation)

    def get_world_xyz(self, timestamp = 0.0):
        if self.bounding_box is not None and timestamp in self.bounding_box.frame:
            R = build_rotation(self.bounding_box.frame[timestamp][1]).squeeze(0)
            return self._xyz @ R.T + self.bounding_box.frame[timestamp][0]
        else:
            return self._xyz

    @property
    def get_local_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.dc_only_sh:
            features_rest = torch.zeros_like(self._features_rest)
        else:
            features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_camera_features(self):
        features_dc = self._features_rgb_dc
        if self.dc_only_sh:
            features_rest = torch.zeros_like(self._features_rgb_rest)
        else:
            features_rest = self._features_rgb_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_cam(self):
        return self.opacity_activation(self._opacity_cam)


    def oneupSHdegree(self):
        if self.dc_only_sh:
            self.active_sh_degree = 0
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_dc_only_sh(self, enabled=True):
        self.dc_only_sh = bool(enabled)
        if self.dc_only_sh:
            self.active_sh_degree = 0
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group["name"] in {"f_rest", "f_rgb_rest"}:
                    group["lr"] = 0.0 if self.dc_only_sh else group["lr"]

    def freeze_positions(self, enabled=True):
        enabled = bool(enabled)
        self._xyz.requires_grad_(not enabled)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    group["lr"] = 0.0 if enabled else group["lr"]

    def scale_optimizer_gradients(self, scale, group_names=None):
        if self.optimizer is None:
            return
        scale = float(scale)
        if scale == 1.0:
            return
        if group_names is None:
            group_names = {"xyz", "scaling", "rotation", "opacity", "opacity_cam"}
        else:
            group_names = set(group_names)
        for group in self.optimizer.param_groups:
            if group["name"] not in group_names:
                continue
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.mul_(scale)

    def _pointwise_nonfinite_mask(self, tensor):
        if tensor is None:
            return None
        if tensor.dim() == 1:
            flat = tensor.unsqueeze(-1)
        else:
            flat = tensor.reshape(tensor.shape[0], -1)
        return ~torch.isfinite(flat).all(dim=1)

    def prune_nonfinite_points(self):
        num_points = int(self.get_local_xyz.shape[0])
        if num_points == 0:
            return 0

        prune_mask = torch.zeros((num_points,), dtype=torch.bool, device=self.get_local_xyz.device)
        tracked_tensors = (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._features_rgb_dc,
            self._features_rgb_rest,
            self._opacity,
            self._opacity_cam,
            self._scaling,
            self._rotation,
        )
        for tensor in tracked_tensors:
            mask = self._pointwise_nonfinite_mask(tensor)
            if mask is not None:
                prune_mask |= mask
            grad = None if tensor is None else tensor.grad
            if grad is not None:
                grad_mask = self._pointwise_nonfinite_mask(grad)
                if grad_mask is not None:
                    prune_mask |= grad_mask

        prune_count = int(prune_mask.sum().item())
        if prune_count > 0:
            self.prune_points(prune_mask)
        return prune_count

    def sanitize_gradients(self):
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

    def _make_rgb_feature_tensors(self, base_rgb):
        coeff = (self.max_sh_degree + 1) ** 2
        fused_rgb = RGB2SH(base_rgb)
        rgb_features = torch.zeros(
            (fused_rgb.shape[0], 3, coeff),
            dtype=fused_rgb.dtype,
            device=fused_rgb.device,
        )
        rgb_features[:, :3, 0] = fused_rgb
        rgb_dc = rgb_features[:, :, 0:1].transpose(1, 2).contiguous()
        rgb_rest = rgb_features[:, :, 1:].transpose(1, 2).contiguous()
        return rgb_dc, rgb_rest

    def _init_rgb_from_lidar_features(self):
        if self._features_dc.numel() == 0:
            self._features_rgb_dc = torch.empty_like(self._features_dc)
            self._features_rgb_rest = torch.empty_like(self._features_rest)
            return
        intensity_dc = self._features_dc[..., 0:1].repeat(1, 1, 3).contiguous()
        rgb_rest = torch.zeros(
            (self._features_rest.shape[0], self._features_rest.shape[1], 3),
            dtype=self._features_rest.dtype,
            device=self._features_rest.device,
        )
        self._features_rgb_dc = intensity_dc
        self._features_rgb_rest = rgb_rest

    def _load_optimizer_state_compat(self, opt_dict):
        if self.optimizer is None or opt_dict is None:
            return
        try:
            self.optimizer.load_state_dict(opt_dict)
            return
        except Exception:
            pass

        loaded_state = opt_dict.get("state", {})
        current_groups = {group["name"]: group for group in self.optimizer.param_groups}
        for loaded_group in opt_dict.get("param_groups", []):
            name = loaded_group.get("name")
            if name not in current_groups or len(loaded_group.get("params", [])) == 0:
                continue
            current_group = current_groups[name]
            current_group["lr"] = loaded_group.get("lr", current_group["lr"])
            for hyper_name in ("betas", "eps", "weight_decay", "amsgrad"):
                if hyper_name in loaded_group:
                    current_group[hyper_name] = loaded_group[hyper_name]
            loaded_param_id = loaded_group["params"][0]
            loaded_param_state = loaded_state.get(loaded_param_id)
            if loaded_param_state is None:
                continue
            current_param = current_group["params"][0]
            copied_state = {}
            state_is_compatible = True
            for key, value in loaded_param_state.items():
                if torch.is_tensor(value):
                    if value.shape != current_param.shape:
                        state_is_compatible = False
                        break
                    copied_state[key] = value.to(
                        device=current_param.device,
                        dtype=current_param.dtype,
                    )
                else:
                    copied_state[key] = value
            if state_is_compatible:
                self.optimizer.state[current_param] = copied_state

    def create_from_pcd(self, pcd, use_normals=False):
        self.spatial_lr_scale = self.extent
        fused_point_cloud = pcd.points.float().cuda()

        color_intensity = pcd.color_intensity.float().cuda()
        fused_color = RGB2SH(color_intensity)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        base_rgb = color_intensity[:, :1].repeat(1, 3)
        features_rgb_dc, features_rgb_rest = self._make_rgb_feature_tensors(base_rgb)

        print("Number of points at initialization: ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, self.dimension)
        
        if use_normals:
            normals = pcd.normals.float().cuda()
            rots = generate_random_quaternion_with_fixed_normal(normals)
        else:
            rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rgb_dc = nn.Parameter(features_rgb_dc.requires_grad_(True))
        self._features_rgb_rest = nn.Parameter(features_rgb_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._opacity_cam = nn.Parameter(opacities.clone().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_local_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.densify_scale_threshold = training_args.densify_scale_threshold
        self.densify_weight_threshold = training_args.densify_weight_threshold
        self.xyz_gradient_accum = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        rest_lr = 0.0 if self.dc_only_sh else training_args.feature_lr / 20.0
        if self._features_rgb_dc.numel() == 0 or self._features_rgb_rest.numel() == 0:
            self._init_rgb_from_lidar_features()
        if not isinstance(self._features_rgb_dc, nn.Parameter):
            self._features_rgb_dc = nn.Parameter(self._features_rgb_dc.requires_grad_(True))
        if not isinstance(self._features_rgb_rest, nn.Parameter):
            self._features_rgb_rest = nn.Parameter(self._features_rgb_rest.requires_grad_(True))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': rest_lr, "name": "f_rest"},
            {'params': [self._features_rgb_dc], 'lr': training_args.feature_lr, "name": "f_rgb_dc"},
            {'params': [self._features_rgb_rest], 'lr': rest_lr, "name": "f_rgb_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._opacity_cam], 'lr': training_args.opacity_lr, "name": "opacity_cam"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_rgb_dc = optimizable_tensors["f_rgb_dc"]
        self._features_rgb_rest = optimizable_tensors["f_rgb_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._opacity_cam = optimizable_tensors["opacity_cam"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_features_rgb_dc, new_features_rgb_rest, new_opacities, new_opacities_cam, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "f_rgb_dc": new_features_rgb_dc,
        "f_rgb_rest": new_features_rgb_rest,
        "opacity": new_opacities,
        "opacity_cam": new_opacities_cam,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_rgb_dc = optimizable_tensors["f_rgb_dc"]
        self._features_rgb_rest = optimizable_tensors["f_rgb_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._opacity_cam = optimizable_tensors["opacity_cam"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_local_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self.get_local_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded = torch.zeros((n_init_points), device="cuda")
        padded[:grads.shape[0]] = grads
        grad_mask = torch.where(padded >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grad_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.densify_scale_threshold*self.extent)
        num = selected_pts_mask.sum().item()

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        if self.dimension == 2:
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_local_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features_rgb_dc = self._features_rgb_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rgb_rest = self._features_rgb_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_opacity_cam = self._opacity_cam[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_rgb_dc, new_features_rgb_rest, new_opacity, new_opacity_cam, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return num

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        grad_mask = torch.where(grads >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grad_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.densify_scale_threshold*self.extent)
        num = selected_pts_mask.sum().item()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_rgb_dc = self._features_rgb_dc[selected_pts_mask]
        new_features_rgb_rest = self._features_rgb_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_opacities_cam = self._opacity_cam[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_rgb_dc, new_features_rgb_rest, new_opacities, new_opacities_cam, new_scaling, new_rotation)
        return num

    def densify_and_prune(self, opt, min_opacity, max_screen_size):
        mean_grads = (self.xyz_gradient_accum / self.denom).nan_to_num(0.0).squeeze(-1)

        clone_num = self.densify_and_clone(mean_grads, opt.densify_grad_threshold)
        split_num = self.densify_and_split(mean_grads, opt.densify_grad_threshold)
        print(f"clone_num: {clone_num}, split_num: {split_num}")

        low_opacity = (self.get_opacity < opt.thresh_opa_prune).squeeze()
        prune_mask = low_opacity
        prune_opacity_num = low_opacity.sum().item()
        prune_scale_num = 0
        if max_screen_size:

            #big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * self.extent * opt.prune_size_threshold
            prune_scale_num = big_points_ws.sum().item()
            #prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(low_opacity, big_points_ws)
            print(f'Prune big points in world: {prune_scale_num} lower than min opacity: {prune_opacity_num}')

            if self.bounding_box is not None:
                # Refer to Street Gaussian by Yan et al. in 2024.
                # Prune points outside the tracking box
                repeat_num = 2
                stds = self.get_scaling
                if self.dimension == 2:
                    stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
                stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1]
                means = torch.zeros_like(self.get_local_xyz)
                means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
                samples = torch.normal(mean=means, std=stds) # [N, M, 3]
                rots = build_rotation(self._rotation) # [N, 3, 3]
                rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
                origins = self.get_local_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]

                samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]
                num_gaussians = self.get_local_xyz.shape[0]
                if num_gaussians > 0:
                    points_inside_box = torch.logical_and(
                        torch.all((samples_xyz >= self.bounding_box.min_xyz).view(num_gaussians, -1), dim=-1),
                        torch.all((samples_xyz <= self.bounding_box.max_xyz).view(num_gaussians, -1), dim=-1),
                    )
                    points_outside_box = torch.logical_not(points_inside_box)
                    prune_mask = torch.logical_or(prune_mask, points_outside_box)

                    print(f'Prune points outside bbox: {points_outside_box.sum()}')
    
        if prune_mask.sum() < self.get_local_xyz.shape[0]:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return clone_num, split_num, prune_scale_num, prune_opacity_num

    def add_densification_stats(self, mean_grads, update_filter):
        self.xyz_gradient_accum += torch.norm(mean_grads, dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # Refer to Street Gaussian by Yan et al. in 2024.
    def box_reg_loss(self):
        reg_loss = 0
        if self.bounding_box is not None:
            box_loss_1 = torch.clamp_min(self.get_local_xyz - self.bounding_box.max_xyz, min=0.).mean()
            box_loss_2 = torch.clamp_min(self.bounding_box.min_xyz - self.get_local_xyz, min=0.).mean()
            box_loss = (box_loss_1 + box_loss_2) / self.extent
            scale_loss = (self.get_scaling.max(dim=1).values / self.extent).mean()
            reg_loss = box_loss * 100 + scale_loss

        return reg_loss
