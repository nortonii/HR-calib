import numpy as np
import torch
import torch.nn.functional as F
from diff_lidar_tracer import Tracer, TracingSettings
from lib.scene import Camera, GaussianModel, LiDARSensor
from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from lib.utils.graphics_utils import camera_to_rays
from lib.utils.primitive_utils import primitiveTypeCallbacks
from lib.utils.sh_utils import eval_sh

tracer_2dgs = Tracer()
vertices, faces = None, None


def raytracing(
    frame: int,
    gaussian_assets: list[GaussianModel],
    sensor: LiDARSensor | Camera,
    background: torch.Tensor,
    args,
    scaling_modifier=1.0,
    override_color=None,
    decomp=False,
    depth_only=False,
    gaussian_transform_rotation=None,
    gaussian_transform_translation=None,
    feature_mode="lidar",
):

    if decomp == "background":
        gaussian_assets = gaussian_assets[:1]
    elif decomp == "object":
        gaussian_assets = gaussian_assets[1:]

    if isinstance(sensor, Camera):
        sensor_center = sensor.camera_center
        rays_o, rays_d = camera_to_rays(sensor)
    elif isinstance(sensor, LiDARSensor):
        rays_o, rays_d = sensor.get_range_rays(frame)
        sensor_center = sensor.sensor_center[frame]
    elif isinstance(sensor, tuple):
        rays_o, rays_d = sensor[0], sensor[1]
        sensor_center = sensor[2]
    else:
        raise ValueError("sensor type not supported")

    tracer = tracer_2dgs
    primitiveCallback = primitiveTypeCallbacks["2DRectangle"]

    tracer_settings = TracingSettings(
        image_height=None,
        image_width=None,
        tanfovx=None,
        tanfovy=None,
        bg=background.cuda(),
        scale_modifier=1.0,
        viewmatrix=torch.Tensor([]).cuda(),
        projmatrix=torch.Tensor([]).cuda(),
        sh_degree=gaussian_assets[0].active_sh_degree,
        campos=sensor_center.cuda(),
        prefiltered=False,
        debug=False,
    )

    all_means3D = []
    all_opacities = []
    all_scales = []
    obj_rot, rot_in_local = [], []
    all_shs = []
    all_colors_precomp = []
    all_features = []
    for pc in gaussian_assets[:]:
        means3D = pc.get_world_xyz(frame)
        if isinstance(sensor, Camera) and bool(getattr(args, "camera_dual_opacity", True)):
            opacity = pc.get_opacity_cam
        else:
            opacity = pc.get_opacity
        all_means3D.append(means3D)
        all_opacities.append(opacity)

        all_scales.append(pc.get_scaling)
        r1, r2 = pc.get_rotation(frame)
        obj_rot.append(r1.expand(r2.shape[0], -1))
        rot_in_local.append(r2)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        if override_color is None:
            features = pc.get_camera_features if feature_mode == "camera" else pc.get_features
            all_features.append(features)
            if not args.pipe.convert_SHs_python:
                all_shs.append(features)
        else:
            all_colors_precomp.append(override_color)

    # Concatenate all tensors
    means3D = torch.cat(all_means3D, dim=0)
    opacity = torch.cat(all_opacities, dim=0)
    scales = torch.cat(all_scales, dim=0) if all_scales else None

    if decomp == "background" or not args.dynamic:
        rotations = rot_in_local[0]
    elif decomp == "object":
        obj_rot = torch.cat(obj_rot, dim=0)  # exclude the background
        rot_in_local = torch.cat(rot_in_local, dim=0)
        rot_in_local = torch.nn.functional.normalize(rot_in_local, dim=1)
        rotations = quaternion_raw_multiply(None, obj_rot, rot_in_local)
    else:
        obj_rot = torch.cat(obj_rot[1:], dim=0)  # exclude the background
        rots_bkgd = rot_in_local[0]
        rot_in_local = torch.cat(rot_in_local[1:], dim=0)
        rot_in_local = torch.nn.functional.normalize(rot_in_local, dim=1)
        rotations = quaternion_raw_multiply(None, obj_rot, rot_in_local)
        rotations = torch.cat([rots_bkgd, rotations], dim=0)

    if gaussian_transform_rotation is not None and gaussian_transform_translation is not None:
        means3D = means3D @ gaussian_transform_rotation + gaussian_transform_translation
        transform_quaternion = matrix_to_quaternion(
            gaussian_transform_rotation.T.unsqueeze(0)
        ).expand(rotations.shape[0], -1)
        rotations = quaternion_raw_multiply(None, transform_quaternion, rotations)

    shs = torch.cat(all_shs, dim=0) if all_shs else None
    if override_color is None and args.pipe.convert_SHs_python:
        features = torch.cat(all_features, dim=0)
        sh_channels = features.shape[-1]
        shs_view = features.transpose(1, 2).view(
            -1, sh_channels, (gaussian_assets[0].max_sh_degree + 1) ** 2
        )
        dir_pp = means3D - sensor_center.repeat(means3D.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
        sh2rgb = eval_sh(gaussian_assets[0].active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = (
            torch.cat(all_colors_precomp, dim=0) if all_colors_precomp else None
        )
    if depth_only and shs is not None:
        # Detach SH features so no gradient flows through them
        shs = shs.detach()

    grads3D = torch.zeros_like(means3D, requires_grad=True)
    try:
        means3D.retain_grad()
    except:
        pass

    vertices, faces, mesh_normals = primitiveCallback(
        means3D, scales, rotations, opacity
    )  # (V, 3), (F, 3)
    tracer.build_acceleration_structure(vertices, faces, rebuild=True)

    rendered_tensor, accum_gaussian_weights = tracer(
        ray_o=rays_o,  # (H, W, 3)
        ray_d=rays_d,  # (H, W, 3)
        mesh_normals=mesh_normals,  # (V, 3)
        means3D=means3D,  # (P, 3)
        grads3D=grads3D,  # (P, 3)
        shs=shs,  # (P, 3, M)
        colors_precomp=colors_precomp,
        opacities=opacity,  # (P, 1)
        scales=scales,  # (P, 3)
        rotations=rotations,  # (P, 4)
        cov3Ds_precomp=None,
        tracer_settings=tracer_settings,
    )

    # mean2D
    rendered_attrs = rendered_tensor[:, :, 0:3]
    intensities = rendered_attrs[:, :, 0:1]
    rayhit_logits = rendered_attrs[:, :, 1:2]
    raydrop_logits = rendered_attrs[:, :, 2:3]
    depth = rendered_tensor[:, :, 3:4]
    accum = rendered_tensor[:, :, 4:5]
    normals = rendered_tensor[:, :, 5:8]
    final_transmittance = rendered_tensor[:, :, 8:9]

    if args.opt.use_rayhit:
        logits = torch.cat([rayhit_logits, raydrop_logits], dim=-1)
        prob = F.softmax(logits, dim=-1)
        raydrop_prob = prob[..., 1:2]
    else:
        raydrop_prob = torch.sigmoid(raydrop_logits)

    return {
        "rendered_attrs": rendered_attrs,
        "rgb": rendered_attrs,
        "depth": depth,
        "intensity": intensities,
        "raydrop": raydrop_prob,
        "accumulation": accum,
        "normal": normals,
        "final_transmittance": final_transmittance,
        "means3D": means3D,
        "accum_gaussian_weight": accum_gaussian_weights.unsqueeze(-1),
    }
