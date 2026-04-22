import torch
import torch.nn.functional as F

try:
    from diff_lidar_tracer import Tracer, TracingSettings
    _LEGACY_LIDAR_RT_AVAILABLE = True
except ImportError:
    Tracer = None
    TracingSettings = None
    _LEGACY_LIDAR_RT_AVAILABLE = False

from lib.scene import Camera, GaussianModel, LiDARSensor
from lib.gaussian_renderer.threedgrut_backend import render_lidar_with_3dgrt
from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from lib.utils.graphics_utils import camera_to_rays
from lib.utils.primitive_utils import primitiveTypeCallbacks
from lib.utils.sh_utils import eval_sh

_legacy_tracer_by_device = {}
vertices, faces = None, None


def _resolve_expected_depth(integrated_depth: torch.Tensor, accumulation: torch.Tensor) -> torch.Tensor:
    expected_depth = torch.zeros_like(integrated_depth)
    valid_accumulation = accumulation > 1.0e-8
    expected_depth[valid_accumulation] = (
        integrated_depth[valid_accumulation] / accumulation[valid_accumulation]
    )
    return expected_depth


def _get_legacy_tracer():
    if not _LEGACY_LIDAR_RT_AVAILABLE:
        raise RuntimeError("Legacy diff_lidar_tracer backend is not installed.")
    device_index = torch.cuda.current_device()
    tracer = _legacy_tracer_by_device.get(device_index)
    if tracer is None:
        tracer = Tracer()
        _legacy_tracer_by_device[device_index] = tracer
    return tracer


def _get_training_render_mode(args) -> str | None:
    model = getattr(args, "model", None)
    mode = getattr(model, "training_render_mode", None)
    if mode is None:
        return None
    return str(mode).lower()


def get_lidar_raytrace_backend(args) -> str:
    training_mode = _get_training_render_mode(args)
    if training_mode in {"hybrid_3dgrut", "hybrid-3dgrut", "3dgrut_hybrid"}:
        return "3dgrt"
    model = getattr(args, "model", None)
    backend = str(getattr(model, "raytrace_backend", "3dgrt")).lower()
    aliases = {
        "3dgrut": "3dgrt",
        "threedgrut": "3dgrt",
        "threedgrt": "3dgrt",
        "3dgut": "3dgrt",
        "lidar_rt": "legacy",
        "diff_lidar_tracer": "legacy",
        "legacy_diff_lidar_tracer": "legacy",
    }
    return aliases.get(backend, backend)


def _raytracing_legacy(
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
        height, width = int(sensor.image_height), int(sensor.image_width)
        rays_o = rays_o.view(height, width, 3).contiguous()
        rays_d = rays_d.view(height, width, 3).contiguous()
    elif isinstance(sensor, LiDARSensor):
        rays_o, rays_d = sensor.get_range_rays(frame)
        sensor_center = sensor.sensor_center[frame]
    elif isinstance(sensor, tuple):
        rays_o, rays_d = sensor[0], sensor[1]
        sensor_center = sensor[2]
    else:
        raise ValueError("sensor type not supported")

    tracer = _get_legacy_tracer()
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
    depth_integrated = rendered_tensor[:, :, 3:4]
    accum = rendered_tensor[:, :, 4:5]
    depth = _resolve_expected_depth(depth_integrated, accum)
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
        "depth_expected": depth,
        "depth_integrated": depth_integrated,
        "intensity": intensities,
        "raydrop": raydrop_prob,
        "accumulation": accum,
        "normal": normals,
        "final_transmittance": final_transmittance,
        "means3D": means3D,
        "accum_gaussian_weight": accum_gaussian_weights.unsqueeze(-1),
    }


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
    backend = get_lidar_raytrace_backend(args)
    if backend == "3dgrt":
        return render_lidar_with_3dgrt(
            frame=frame,
            gaussian_assets=gaussian_assets,
            sensor=sensor,
            background=background,
            args=args,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            decomp=decomp,
            depth_only=depth_only,
            gaussian_transform_rotation=gaussian_transform_rotation,
            gaussian_transform_translation=gaussian_transform_translation,
            feature_mode=feature_mode,
        )
    if backend == "legacy":
        return _raytracing_legacy(
            frame=frame,
            gaussian_assets=gaussian_assets,
            sensor=sensor,
            background=background,
            args=args,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            decomp=decomp,
            depth_only=depth_only,
            gaussian_transform_rotation=gaussian_transform_rotation,
            gaussian_transform_translation=gaussian_transform_translation,
            feature_mode=feature_mode,
        )
    raise ValueError(f"Unsupported model.raytrace_backend '{backend}'.")
