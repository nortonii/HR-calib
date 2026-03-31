import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.scene.cameras import Camera


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1 = rotation_6d[:, 0:3]
    a2 = rotation_6d[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def _matrix_to_euler_xyz(rotation: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2)
    singular = sy < 1.0e-6

    x = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    y = torch.atan2(-rotation[..., 2, 0], sy)
    z = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    x_s = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    y_s = torch.atan2(-rotation[..., 2, 0], sy)
    z_s = torch.zeros_like(z)

    x = torch.where(singular, x_s, x)
    y = torch.where(singular, y_s, y)
    z = torch.where(singular, z_s, z)
    return torch.stack((x, y, z), dim=-1)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_minus_c = 1.0 - c

    rotation = torch.zeros((axis.shape[0], 3, 3), dtype=axis.dtype, device=axis.device)
    rotation[:, 0, 0] = c + x * x * one_minus_c
    rotation[:, 0, 1] = x * y * one_minus_c - z * s
    rotation[:, 0, 2] = x * z * one_minus_c + y * s
    rotation[:, 1, 0] = y * x * one_minus_c + z * s
    rotation[:, 1, 1] = c + y * y * one_minus_c
    rotation[:, 1, 2] = y * z * one_minus_c - x * s
    rotation[:, 2, 0] = z * x * one_minus_c - y * s
    rotation[:, 2, 1] = z * y * one_minus_c + x * s
    rotation[:, 2, 2] = c + z * z * one_minus_c
    return rotation


def _higs_se3_to_delta_transform(se3: torch.Tensor):
    rho = se3[:, :3]
    phi = se3[:, 3:]
    angle = torch.norm(phi, dim=-1, keepdim=True)

    rotation = torch.eye(3, dtype=se3.dtype, device=se3.device).unsqueeze(0).repeat(se3.shape[0], 1, 1)
    v_matrix = torch.eye(3, dtype=se3.dtype, device=se3.device).unsqueeze(0).repeat(se3.shape[0], 1, 1)

    valid = angle.squeeze(-1) >= 1.0e-8
    if torch.any(valid):
        axis = phi[valid] / angle[valid]
        ax, ay, az = axis.unbind(dim=-1)
        k = torch.zeros((axis.shape[0], 3, 3), dtype=se3.dtype, device=se3.device)
        k[:, 0, 1] = -az
        k[:, 0, 2] = ay
        k[:, 1, 0] = az
        k[:, 1, 2] = -ax
        k[:, 2, 0] = -ay
        k[:, 2, 1] = ax

        valid_angle = angle[valid].unsqueeze(-1)
        sin_angle = torch.sin(valid_angle)
        cos_angle = torch.cos(valid_angle)
        rotation[valid] = (
            torch.eye(3, dtype=se3.dtype, device=se3.device).unsqueeze(0)
            + sin_angle * k
            + (1.0 - cos_angle) * (k @ k)
        )
        v_matrix[valid] = (
            torch.eye(3, dtype=se3.dtype, device=se3.device).unsqueeze(0)
            + ((1.0 - cos_angle) / valid_angle) * k
            + ((valid_angle - sin_angle) / valid_angle) * (k @ k)
        )

    translation = torch.bmm(v_matrix, rho.unsqueeze(-1)).squeeze(-1)
    return rotation, translation


def _rotation_angle_deg(rotation: torch.Tensor) -> torch.Tensor:
    trace = torch.clamp((torch.trace(rotation) - 1.0) * 0.5, min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(trace))


class CameraPoseCorrection(nn.Module):
    def __init__(self, cameras: dict, config=None, lidar_poses: dict = None):
        super().__init__()
        frame_ids = sorted(cameras.keys())
        if not frame_ids:
            raise ValueError("CameraPoseCorrection requires at least one camera frame.")

        sample_camera = cameras[frame_ids[0]]
        dtype = sample_camera.R.dtype
        pose_mode = "frame" if config is None else str(getattr(config, "mode", "frame")).lower()
        if pose_mode not in {"frame", "all"}:
            raise ValueError(f"Unsupported pose_correction.mode '{pose_mode}'. Expected 'frame' or 'all'.")
        num_pose_params = 1 if pose_mode == "all" else len(frame_ids)
        use_shared_lidar_extrinsic = pose_mode == "all" and lidar_poses is not None

        base_rotations = []
        base_translations = []
        for frame_id in frame_ids:
            camera = cameras[frame_id]
            base_rotations.append(camera.R.detach().clone().to(dtype=dtype))
            base_translations.append(camera.T.detach().clone().to(dtype=dtype))

        base_rotations = torch.stack(base_rotations, dim=0)
        base_translations = torch.stack(base_translations, dim=0)

        pose_cfg = config
        init_translation_std = 0.0 if pose_cfg is None else getattr(pose_cfg, "init_translation_std", 0.0)
        init_rotation_deg = 0.0 if pose_cfg is None else getattr(pose_cfg, "init_rotation_deg", 0.0)

        init_rotation = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(num_pose_params, 1, 1)
        init_translation = torch.zeros((num_pose_params, 3), dtype=dtype)

        if init_translation_std > 0.0:
            init_translation.normal_(mean=0.0, std=float(init_translation_std))

        if init_rotation_deg > 0.0:
            max_angle_rad = math.radians(float(init_rotation_deg))
            axis = torch.randn((num_pose_params, 3), dtype=dtype)
            axis = F.normalize(axis, dim=-1)
            angle = (2.0 * torch.rand((num_pose_params,), dtype=dtype) - 1.0) * max_angle_rad
            init_rotation = _axis_angle_to_matrix(axis, angle)

        init_se3 = None if pose_cfg is None else getattr(pose_cfg, "init_se3", None)
        if init_se3 is not None:
            init_se3_tensor = torch.as_tensor(init_se3, dtype=dtype)
            if init_se3_tensor.numel() != 6:
                raise ValueError("pose_correction.init_se3 must contain 6 values: [tx, ty, tz, rx, ry, rz].")
            init_se3_tensor = init_se3_tensor.reshape(1, 6).repeat(num_pose_params, 1)
            init_rotation, init_translation = _higs_se3_to_delta_transform(init_se3_tensor)

        lidar_world_rotations = None
        lidar_world_translations = None
        gt_extrinsic_rotation = None
        gt_extrinsic_translation = None
        base_extrinsic_rotation = None
        base_extrinsic_translation = None

        if use_shared_lidar_extrinsic:
            lidar_world = []
            for frame_id in frame_ids:
                if frame_id not in lidar_poses:
                    raise KeyError(f"LiDAR pose for frame {frame_id} not found.")
                lidar_world.append(torch.as_tensor(lidar_poses[frame_id], dtype=dtype))
            lidar_world = torch.stack(lidar_world, dim=0)
            lidar_world_rotations = lidar_world[:, :3, :3]
            lidar_world_translations = lidar_world[:, :3, 3]

            gt_camera_centers = -(torch.bmm(base_rotations, base_translations.unsqueeze(-1)).squeeze(-1))
            gt_c2w = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(len(frame_ids), 1, 1)
            gt_c2w[:, :3, :3] = base_rotations
            gt_c2w[:, :3, 3] = gt_camera_centers

            lidar_world_inv = torch.linalg.inv(lidar_world)
            gt_c2l = torch.bmm(lidar_world_inv, gt_c2w)
            gt_l2c = torch.linalg.inv(gt_c2l)
            gt_extrinsic_rotation = gt_l2c[0, :3, :3]
            gt_extrinsic_translation = gt_l2c[0, :3, 3]

            init_rotation_single = init_rotation[0]
            init_translation_single = init_translation[0]
            base_extrinsic_rotation = init_rotation_single @ gt_extrinsic_rotation
            base_extrinsic_translation = (
                init_rotation_single @ gt_extrinsic_translation.unsqueeze(-1)
            ).squeeze(-1) + init_translation_single

            c2l_rotation = base_extrinsic_rotation.transpose(0, 1)
            c2l_translation = -(c2l_rotation @ base_extrinsic_translation)
            initialized_rotations = torch.bmm(lidar_world_rotations, c2l_rotation.unsqueeze(0).repeat(len(frame_ids), 1, 1))
            initialized_centers = (
                torch.bmm(
                    lidar_world_rotations,
                    c2l_translation.unsqueeze(0).unsqueeze(-1).repeat(len(frame_ids), 1, 1),
                ).squeeze(-1)
                + lidar_world_translations
            )
            initialized_translations = -torch.bmm(
                initialized_rotations.transpose(1, 2), initialized_centers.unsqueeze(-1)
            ).squeeze(-1)
        else:
            if pose_mode == "all":
                init_rotation_per_frame = init_rotation.repeat(len(frame_ids), 1, 1)
                init_translation_per_frame = init_translation.repeat(len(frame_ids), 1)
            else:
                init_rotation_per_frame = init_rotation
                init_translation_per_frame = init_translation

            initialized_rotations = torch.bmm(base_rotations, init_rotation_per_frame)
            initialized_centers = -(torch.bmm(base_rotations, base_translations.unsqueeze(-1)).squeeze(-1))
            initialized_centers = initialized_centers + torch.bmm(
                base_rotations, init_translation_per_frame.unsqueeze(-1)
            ).squeeze(-1)
            initialized_translations = -torch.bmm(
                initialized_rotations.transpose(1, 2), initialized_centers.unsqueeze(-1)
            ).squeeze(-1)

        self.frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
        self.pose_mode = pose_mode
        self.use_shared_lidar_extrinsic = use_shared_lidar_extrinsic
        self.frame_to_index = {frame_id: index for index, frame_id in enumerate(self.frame_ids)}
        self.frame_to_pose_index = {
            frame_id: (0 if pose_mode == "all" else index) for index, frame_id in enumerate(self.frame_ids)
        }
        self.register_buffer("gt_rotations", base_rotations)
        self.register_buffer("gt_translations", base_translations)
        self.register_buffer("base_rotations", initialized_rotations)
        self.register_buffer("base_translations", initialized_translations)
        self.register_buffer("pose_init_rotations", init_rotation)
        self.register_buffer("pose_init_translations", init_translation)
        if lidar_world_rotations is not None:
            self.register_buffer("lidar_world_rotations", lidar_world_rotations)
            self.register_buffer("lidar_world_translations", lidar_world_translations)
            self.register_buffer("gt_lidar_to_camera_rotation", gt_extrinsic_rotation.unsqueeze(0))
            self.register_buffer("gt_lidar_to_camera_translation", gt_extrinsic_translation.unsqueeze(0))
            self.register_buffer("base_lidar_to_camera_rotation", base_extrinsic_rotation.unsqueeze(0))
            self.register_buffer("base_lidar_to_camera_translation", base_extrinsic_translation.unsqueeze(0))

        self.delta_translations = nn.Parameter(torch.zeros((num_pose_params, 3), dtype=dtype))
        self.delta_rotations_6d = nn.Parameter(
            _matrix_to_rotation_6d(torch.eye(3, dtype=dtype).unsqueeze(0).repeat(num_pose_params, 1, 1))
        )
        self.use_gt_translation = False

    def _frame_index(self, frame_id: int) -> int:
        frame_id = int(frame_id)
        if frame_id not in self.frame_to_index:
            raise KeyError(f"Frame {frame_id} not found in camera pose correction table.")
        return self.frame_to_index[frame_id]

    def _pose_index(self, frame_id: int) -> int:
        frame_id = int(frame_id)
        if frame_id not in self.frame_to_pose_index:
            raise KeyError(f"Frame {frame_id} not found in camera pose correction table.")
        return self.frame_to_pose_index[frame_id]

    def corrected_rt(self, frame_id: int, device=None):
        frame_index = self._frame_index(frame_id)
        pose_index = self._pose_index(frame_id)
        if device is None:
            device = self.delta_translations.device
        use_gt_translation = bool(self.use_gt_translation)

        if self.use_shared_lidar_extrinsic:
            delta_translation = self.delta_translations[pose_index].to(device=device)
            delta_rotation = _rotation_6d_to_matrix(
                self.delta_rotations_6d[pose_index:pose_index + 1].to(device=device)
            )[0]
            base_extrinsic_rotation = self.base_lidar_to_camera_rotation[0].to(device=device)
            base_extrinsic_translation = self.base_lidar_to_camera_translation[0].to(device=device)
            lidar_world_rotation = self.lidar_world_rotations[frame_index].to(device=device)
            lidar_world_translation = self.lidar_world_translations[frame_index].to(device=device)

            extrinsic_rotation = delta_rotation @ base_extrinsic_rotation
            if use_gt_translation:
                extrinsic_translation = self.gt_lidar_to_camera_translation[0].to(device=device)
            else:
                extrinsic_translation = (
                    delta_rotation @ base_extrinsic_translation.unsqueeze(-1)
                ).squeeze(-1) + delta_translation

            camera_to_lidar_rotation = extrinsic_rotation.transpose(0, 1)
            camera_to_lidar_translation = -(camera_to_lidar_rotation @ extrinsic_translation)
            corrected_rotation = lidar_world_rotation @ camera_to_lidar_rotation
            corrected_center = (
                lidar_world_rotation @ camera_to_lidar_translation + lidar_world_translation
            )
            corrected_translation = -(corrected_rotation.T @ corrected_center)
            return corrected_rotation, corrected_translation

        base_rotation = self.base_rotations[frame_index].to(device=device)
        base_translation = self.base_translations[frame_index].to(device=device)
        delta_translation = self.delta_translations[pose_index].to(device=device)
        delta_rotation = _rotation_6d_to_matrix(
            self.delta_rotations_6d[pose_index:pose_index + 1].to(device=device)
        )[0]

        camera_center = -(base_rotation @ base_translation)
        corrected_center = camera_center + base_rotation @ delta_translation
        corrected_rotation = base_rotation @ delta_rotation
        corrected_translation = -(corrected_rotation.T @ corrected_center)
        return corrected_rotation, corrected_translation

    def corrected_lidar_to_camera(self, frame_id: int, device=None):
        if not self.use_shared_lidar_extrinsic:
            raise RuntimeError(
                "corrected_lidar_to_camera is only available in shared-extrinsic mode."
            )
        pose_index = self._pose_index(frame_id)
        if device is None:
            device = self.delta_translations.device
        delta_translation = self.delta_translations[pose_index].to(device=device)
        delta_rotation = _rotation_6d_to_matrix(
            self.delta_rotations_6d[pose_index:pose_index + 1].to(device=device)
        )[0]
        base_extrinsic_rotation = self.base_lidar_to_camera_rotation[0].to(device=device)
        base_extrinsic_translation = self.base_lidar_to_camera_translation[0].to(device=device)

        extrinsic_rotation = delta_rotation @ base_extrinsic_rotation
        if bool(self.use_gt_translation):
            extrinsic_translation = self.gt_lidar_to_camera_translation[0].to(device=device)
        else:
            extrinsic_translation = (
                delta_rotation @ base_extrinsic_translation.unsqueeze(-1)
                ).squeeze(-1) + delta_translation
        return extrinsic_rotation, extrinsic_translation

    @torch.no_grad()
    def set_lidar_to_camera(self, frame_id: int, extrinsic_rotation: torch.Tensor, extrinsic_translation: torch.Tensor):
        if not self.use_shared_lidar_extrinsic:
            raise RuntimeError("set_lidar_to_camera is only available in shared-extrinsic mode.")
        pose_index = self._pose_index(frame_id)
        device = self.delta_translations.device
        dtype = self.delta_translations.dtype
        extrinsic_rotation = extrinsic_rotation.to(device=device, dtype=dtype)
        extrinsic_translation = extrinsic_translation.to(device=device, dtype=dtype)
        base_rotation = self.base_lidar_to_camera_rotation[0].to(device=device, dtype=dtype)
        base_translation = self.base_lidar_to_camera_translation[0].to(device=device, dtype=dtype)
        delta_rotation = extrinsic_rotation @ base_rotation.transpose(0, 1)
        delta_translation = extrinsic_translation - (
            delta_rotation @ base_translation.unsqueeze(-1)
        ).squeeze(-1)
        self.delta_rotations_6d[pose_index].copy_(
            _matrix_to_rotation_6d(delta_rotation.unsqueeze(0))[0]
        )
        self.delta_translations[pose_index].copy_(delta_translation)

    @torch.no_grad()
    def apply_relative_camera_transform(
        self,
        frame_id: int,
        relative_rotation: torch.Tensor,
        relative_translation: torch.Tensor,
    ):
        if not self.use_shared_lidar_extrinsic:
            raise RuntimeError(
                "apply_relative_camera_transform is only available in shared-extrinsic mode."
            )
        device = self.delta_translations.device
        dtype = self.delta_translations.dtype
        relative_rotation = relative_rotation.to(device=device, dtype=dtype)
        relative_translation = relative_translation.to(device=device, dtype=dtype)
        current_rotation, current_translation = self.corrected_lidar_to_camera(frame_id, device=device)
        target_rotation = relative_rotation @ current_rotation
        target_translation = (
            relative_rotation @ current_translation.unsqueeze(-1)
        ).squeeze(-1) + relative_translation
        self.set_lidar_to_camera(frame_id, target_rotation, target_translation)

    def corrected_camera(self, camera: Camera, device="cuda"):
        corrected_rotation, corrected_translation = self.corrected_rt(camera.timestamp, device=device)
        depth_map = camera.depth_map
        intensity_map = camera.intensity_map
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.to(device=device)
        if isinstance(intensity_map, torch.Tensor):
            intensity_map = intensity_map.to(device=device)
        trans = camera.trans
        if isinstance(trans, torch.Tensor):
            trans = trans.to(device=device)
        return Camera(
            timestamp=camera.timestamp,
            R=corrected_rotation,
            T=corrected_translation,
            w=camera.image_width,
            h=camera.image_height,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            depth=depth_map,
            intensity=intensity_map,
            trans=trans,
            scale=camera.scale,
            data_device=device,
        )

    def regularization_loss(self, frame_id: int, config=None):
        index = self._pose_index(frame_id)
        trans_weight = 0.0 if config is None else getattr(config, "lambda_translation", 0.0)
        rot_weight = 0.0 if config is None else getattr(config, "lambda_rotation", 0.0)
        loss = torch.zeros((), device=self.delta_translations.device, dtype=self.delta_translations.dtype)
        if trans_weight > 0.0:
            loss = loss + float(trans_weight) * self.delta_translations[index].pow(2).sum()
        if rot_weight > 0.0:
            identity_6d = _matrix_to_rotation_6d(
                torch.eye(3, dtype=self.delta_rotations_6d.dtype, device=self.delta_rotations_6d.device).unsqueeze(0)
            )[0]
            loss = loss + float(rot_weight) * (self.delta_rotations_6d[index] - identity_6d).pow(2).sum()
        return loss

    def pose_magnitude(self, frame_id: int):
        index = self._pose_index(frame_id)
        translation_norm = self.delta_translations[index].norm()
        rotation = _rotation_6d_to_matrix(self.delta_rotations_6d[index:index + 1])[0]
        rotation_deg = _rotation_angle_deg(rotation)
        return translation_norm, rotation_deg

    def global_pose_statistics(self):
        translation_norms = self.delta_translations.norm(dim=-1)
        rotations = _rotation_6d_to_matrix(self.delta_rotations_6d)
        traces = torch.clamp((rotations[:, 0, 0] + rotations[:, 1, 1] + rotations[:, 2, 2] - 1.0) * 0.5, min=-1.0, max=1.0)
        rotation_degs = torch.rad2deg(torch.acos(traces))
        return {
            "translation_mean": translation_norms.mean(),
            "translation_max": translation_norms.max(),
            "rotation_mean_deg": rotation_degs.mean(),
            "rotation_max_deg": rotation_degs.max(),
        }

    def delta_pose_error(self, frame_id: int, device=None):
        frame_index = self._frame_index(frame_id)
        pose_index = self._pose_index(frame_id)
        if device is None:
            device = self.delta_translations.device
        use_gt_translation = bool(self.use_gt_translation)

        current_delta_rotation = _rotation_6d_to_matrix(
            self.delta_rotations_6d[pose_index:pose_index + 1].to(device=device)
        )[0]
        current_delta_translation = self.delta_translations[pose_index].to(device=device)

        if self.use_shared_lidar_extrinsic:
            gt_rotation = self.gt_lidar_to_camera_rotation[0].to(device=device)
            base_rotation = self.base_lidar_to_camera_rotation[0].to(device=device)
            gt_delta_rotation = gt_rotation @ base_rotation.transpose(0, 1)
            if use_gt_translation:
                gt_delta_translation = torch.zeros_like(current_delta_translation)
                current_delta_translation = torch.zeros_like(current_delta_translation)
            else:
                gt_translation = self.gt_lidar_to_camera_translation[0].to(device=device)
                base_translation = self.base_lidar_to_camera_translation[0].to(device=device)
                gt_delta_translation = gt_translation - (
                    gt_delta_rotation @ base_translation.unsqueeze(-1)
                ).squeeze(-1)
        elif self.pose_mode == "all":
            init_rotation = self.pose_init_rotations[pose_index].to(device=device)
            init_translation = self.pose_init_translations[pose_index].to(device=device)
            gt_delta_rotation = init_rotation.transpose(0, 1)
            gt_delta_translation = -(
                init_rotation.transpose(0, 1) @ init_translation.unsqueeze(-1)
            ).squeeze(-1)
        else:
            gt_rotation = self.gt_rotations[frame_index].to(device=device)
            gt_translation = self.gt_translations[frame_index].to(device=device)
            base_rotation = self.base_rotations[frame_index].to(device=device)
            base_translation = self.base_translations[frame_index].to(device=device)
            gt_center = -(gt_rotation @ gt_translation)
            base_center = -(base_rotation @ base_translation)
            gt_delta_rotation = base_rotation.transpose(0, 1) @ gt_rotation
            gt_delta_translation = (
                base_rotation.transpose(0, 1) @ (gt_center - base_center).unsqueeze(-1)
            ).squeeze(-1)

        relative_rotation = current_delta_rotation.transpose(0, 1) @ gt_delta_rotation
        translation_error = gt_delta_translation - current_delta_translation
        return {
            "translation_error_norm": translation_error.norm(),
            "translation_error": translation_error,
            "translation_error_abs": torch.abs(translation_error),
            "rotation_error_deg": _rotation_angle_deg(relative_rotation),
            "rotation_euler_error_deg": torch.rad2deg(torch.abs(_matrix_to_euler_xyz(relative_rotation))),
        }

    def extrinsic_error(self, frame_id: int, device=None):
        index = self._frame_index(frame_id)
        rotation_cur, translation_cur = self.corrected_rt(frame_id, device=device)
        gt_rotation = self.gt_rotations[index].to(device=rotation_cur.device)
        gt_translation = self.gt_translations[index].to(device=translation_cur.device)

        gt_center = -(gt_rotation @ gt_translation)
        cur_center = -(rotation_cur @ translation_cur)
        relative_rotation = gt_rotation.transpose(0, 1) @ rotation_cur
        relative_translation = gt_rotation.transpose(0, 1) @ (cur_center - gt_center)
        trace = torch.clamp((torch.trace(relative_rotation) - 1.0) * 0.5, min=-1.0, max=1.0)
        rotation_error_deg = torch.rad2deg(torch.acos(trace))
        euler_error_deg = torch.rad2deg(torch.abs(_matrix_to_euler_xyz(relative_rotation)))
        translation_error_norm = relative_translation.norm()
        return {
            "translation_error_norm": translation_error_norm,
            "translation_error": relative_translation,
            "translation_error_abs": torch.abs(relative_translation),
            "rotation_error_deg": rotation_error_deg,
            "rotation_euler_error_deg": euler_error_deg,
        }

    def global_extrinsic_error(self):
        rotations_cur = []
        translations_cur = []
        for frame_id in self.frame_ids:
            rotation_cur, translation_cur = self.corrected_rt(frame_id, device=self.delta_translations.device)
            rotations_cur.append(rotation_cur)
            translations_cur.append(translation_cur)
        rotations_cur = torch.stack(rotations_cur, dim=0)
        translations_cur = torch.stack(translations_cur, dim=0)

        gt_rotations = self.gt_rotations.to(rotations_cur.device)
        gt_translations = self.gt_translations.to(translations_cur.device)
        gt_centers = -torch.bmm(gt_rotations, gt_translations.unsqueeze(-1)).squeeze(-1)
        cur_centers = -torch.bmm(rotations_cur, translations_cur.unsqueeze(-1)).squeeze(-1)
        relative_rotations = torch.matmul(gt_rotations.transpose(1, 2), rotations_cur)
        relative_translations = torch.bmm(
            gt_rotations.transpose(1, 2), (cur_centers - gt_centers).unsqueeze(-1)
        ).squeeze(-1)
        traces = torch.clamp(
            (relative_rotations[:, 0, 0] + relative_rotations[:, 1, 1] + relative_rotations[:, 2, 2] - 1.0) * 0.5,
            min=-1.0,
            max=1.0,
        )
        rotation_error_deg = torch.rad2deg(torch.acos(traces))
        euler_error_deg = torch.rad2deg(torch.abs(_matrix_to_euler_xyz(relative_rotations)))
        translation_error_norm = relative_translations.norm(dim=-1)
        return {
            "translation_mean": translation_error_norm.mean(),
            "translation_max": translation_error_norm.max(),
            "rotation_mean_deg": rotation_error_deg.mean(),
            "rotation_max_deg": rotation_error_deg.max(),
            "rotation_euler_mean_deg": euler_error_deg.mean(),
            "rotation_euler_max_deg": euler_error_deg.max(),
        }

    def shared_extrinsic_error(self, device=None):
        if device is None:
            device = self.delta_translations.device
        if self.pose_mode != "all":
            return self.extrinsic_error(self.frame_ids[0], device=device)
        if self.use_shared_lidar_extrinsic:
            delta_rotation = _rotation_6d_to_matrix(self.delta_rotations_6d[0:1].to(device=device))[0]
            delta_translation = self.delta_translations[0].to(device=device)
            base_rotation = self.base_lidar_to_camera_rotation[0].to(device=device)
            base_translation = self.base_lidar_to_camera_translation[0].to(device=device)
            gt_rotation = self.gt_lidar_to_camera_rotation[0].to(device=device)
            gt_translation = self.gt_lidar_to_camera_translation[0].to(device=device)

            pred_rotation = delta_rotation @ base_rotation
            pred_translation = (delta_rotation @ base_translation.unsqueeze(-1)).squeeze(-1) + delta_translation
            relative_rotation = pred_rotation.transpose(0, 1) @ gt_rotation
            translation_error = gt_translation - pred_translation
            return {
                "translation_error_norm": translation_error.norm(),
                "translation_error": translation_error,
                "translation_error_abs": torch.abs(translation_error),
                "rotation_error_deg": _rotation_angle_deg(relative_rotation),
                "rotation_euler_error_deg": torch.rad2deg(torch.abs(_matrix_to_euler_xyz(relative_rotation))),
            }

        init_rotation = self.pose_init_rotations[0].to(device=device)
        init_translation = self.pose_init_translations[0].to(device=device)
        delta_rotation = _rotation_6d_to_matrix(self.delta_rotations_6d[0:1].to(device=device))[0]
        delta_translation = self.delta_translations[0].to(device=device)

        total_rotation = init_rotation @ delta_rotation
        total_translation = init_translation + init_rotation @ delta_translation
        return {
            "translation_error_norm": total_translation.norm(),
            "translation_error": total_translation,
            "translation_error_abs": torch.abs(total_translation),
            "rotation_error_deg": _rotation_angle_deg(total_rotation),
            "rotation_euler_error_deg": torch.rad2deg(torch.abs(_matrix_to_euler_xyz(total_rotation))),
        }
