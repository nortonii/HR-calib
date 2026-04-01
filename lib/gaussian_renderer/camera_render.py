"""
Camera rendering helpers.

Both rasterization and OptiX-style ray tracing can render camera RGB/depth.
"""
import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization import GaussianRasterizationSettings as SurfelRasterizationSettings
from diff_surfel_rasterization import GaussianRasterizer as SurfelRasterizer

from lib.gaussian_renderer import raytracing
from lib.scene.cameras import Camera
from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from lib.utils.graphics_utils import geom_transform_points
from lib.utils.sh_utils import eval_sh


def _make_camera_like(source_camera, rotation=None, translation=None, device="cuda"):
    target_device = torch.device(device)

    def _move_tensor(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=target_device)
        return value

    return Camera(
        timestamp=source_camera.timestamp,
        R=_move_tensor(source_camera.R if rotation is None else rotation),
        T=_move_tensor(source_camera.T if translation is None else translation),
        w=source_camera.image_width,
        h=source_camera.image_height,
        FoVx=source_camera.FoVx,
        FoVy=source_camera.FoVy,
        depth=_move_tensor(source_camera.depth_map),
        intensity=_move_tensor(source_camera.intensity_map),
        trans=_move_tensor(source_camera.trans),
        scale=source_camera.scale,
        data_device=str(target_device),
    )


def get_camera_render_backend(args, require_rgb=False):
    backend = str(getattr(getattr(args, "model", None), "camera_render_backend", "rasterization")).lower()
    backend_aliases = {
        "3dgs": "rasterization",
        "raster": "rasterization",
        "rasterizer": "rasterization",
        "2dgs": "surfel_rasterization",
        "surfel": "surfel_rasterization",
        "surfel_raster": "surfel_rasterization",
        "2d_rasterization": "surfel_rasterization",
        "optix": "raytracing",
        "raytrace": "raytracing",
    }
    backend = backend_aliases.get(backend, backend)
    if backend not in {"rasterization", "surfel_rasterization", "raytracing"}:
        raise ValueError(
            f"Unsupported model.camera_render_backend '{backend}'. "
            "Expected 'rasterization', 'surfel_rasterization', or 'raytracing'."
        )
    return backend


def render_camera_3dgs(camera, gaussian_assets, args, scaling_modifier=1.0,
                       cam_rotation=None, cam_translation=None):
    """Render a camera view with standard 3DGS tile rasterization.

    Args:
        camera:           Camera object (world_view_transform, full_proj_transform, etc.)
        gaussian_assets:  list[GaussianModel] – [background, ...objects]
        args:             training args (needs args.dynamic, args.pipe)
        scaling_modifier: Gaussian scale multiplier

    Returns:
        dict:
            'rgb'              – (H, W, 3) float32, values in [0, 1]
            'screenspace_points' – means2D tensor (for gradient)
    """
    bg = torch.zeros(3, device="cuda")  # black RGB background

    if cam_rotation is not None and cam_translation is not None:
        fixed_camera = _make_camera_like(
            camera,
            rotation=torch.eye(3, dtype=torch.float32, device="cuda"),
            translation=torch.zeros(3, dtype=torch.float32, device="cuda"),
            device="cuda",
        )
        viewmatrix = fixed_camera.world_view_transform
        projmatrix = fixed_camera.full_proj_transform
        campos = fixed_camera.camera_center
    else:
        viewmatrix = camera.world_view_transform.cuda()
        projmatrix = camera.full_proj_transform.cuda()
        campos = camera.camera_center.cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=gaussian_assets[0].active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings)

    frame = camera.timestamp
    all_means3D, all_dc, all_sh, all_opacities, all_scales = [], [], [], [], []
    obj_rot, rot_in_local = [], []

    use_dual_opacity = getattr(args, "camera_dual_opacity", True)

    for pc in gaussian_assets:
        means3D = pc.get_world_xyz(frame)
        features = pc.get_camera_features  # (N, D, 3): camera-only RGB SH branch
        all_means3D.append(means3D)
        all_dc.append(features[:, :1, :])      # (N, 1, 3) – DC component
        all_sh.append(features[:, 1:, :])      # (N, D-1, 3) – higher orders
        all_opacities.append(pc.get_opacity_cam if use_dual_opacity else pc.get_opacity)
        all_scales.append(pc.get_scaling)
        r1, r2 = pc.get_rotation(frame)
        obj_rot.append(r1.expand(r2.shape[0], -1))
        rot_in_local.append(r2)

    means3D = torch.cat(all_means3D, 0)
    dc = torch.cat(all_dc, 0)
    sh = torch.cat(all_sh, 0)
    opacities = torch.cat(all_opacities, 0)
    scales = torch.cat(all_scales, 0)

    # 2D Gaussian models store scale as (N, 2); 3DGS rasterizer needs (N, 3).
    # LiDAR Gaussian disks are edge-on to cameras → visible=0 → CUDA backward
    # crashes with grid_dim=0.  Fix: make them isotropic by setting z = max(x,y),
    # clamped to avoid tile-buffer overflow (large scales → billions of sort keys).
    _MAX_CAM_SCALE = 3.0   # metres; keeps tile count manageable
    if scales.shape[1] == 2:
        sc2 = scales.clamp(max=_MAX_CAM_SCALE)
        # z = max(x,y) with detach so no scatter-grad through argmax
        z_pad = sc2.detach().max(dim=1, keepdim=True).values
        scales = torch.cat([sc2, z_pad], dim=1)
    else:
        scales = scales.clamp(max=_MAX_CAM_SCALE)

    # Rotation: mirror the same static/dynamic logic as raytracing()
    if not args.dynamic or len(gaussian_assets) == 1:
        rotations = rot_in_local[0]
    else:
        obj_rot_objs = torch.cat(obj_rot[1:], 0)
        rots_bkgd = rot_in_local[0]
        rot_local_objs = torch.cat(rot_in_local[1:], 0)
        rot_local_objs = F.normalize(rot_local_objs, dim=1)
        rotations_objs = quaternion_raw_multiply(None, obj_rot_objs, rot_local_objs)
        rotations = torch.cat([rots_bkgd, rotations_objs], 0)

    if cam_rotation is not None and cam_translation is not None:
        means3D = means3D @ cam_rotation + cam_translation
        cam_rotation_cw = cam_rotation.T
        cam_quaternion = matrix_to_quaternion(cam_rotation_cw.unsqueeze(0)).expand(rotations.shape[0], -1)
        rotations = quaternion_raw_multiply(None, cam_quaternion, rotations)

    screenspace_points = torch.zeros_like(means3D, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    rendered_image, _radii, _invdepths = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        dc=dc,
        shs=sh,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    if _invdepths.dim() == 3 and _invdepths.shape[0] == 1:
        invdepth = _invdepths[0]
    else:
        invdepth = _invdepths.squeeze()
    valid_depth = invdepth > 1.0e-8
    depth = torch.where(valid_depth, invdepth.reciprocal(), torch.zeros_like(invdepth))
    # rendered_image: (3, H, W) → (H, W, 3)
    return {
        "rgb": rendered_image.permute(1, 2, 0),
        "depth": depth,
        "invdepth": invdepth,
        "screenspace_points": screenspace_points,
        "radii": _radii,
        "visibility_filter": _radii > 0,
        "num_visible": int((_radii > 0).sum().item()),
    }


def render_camera_2dgs(camera, gaussian_assets, args, scaling_modifier=1.0,
                       cam_rotation=None, cam_translation=None):
    """Render a camera view with native 2DGS surfel rasterization."""
    bg = torch.zeros(3, device="cuda")

    if cam_rotation is not None and cam_translation is not None:
        fixed_camera = _make_camera_like(
            camera,
            rotation=torch.eye(3, dtype=torch.float32, device="cuda"),
            translation=torch.zeros(3, dtype=torch.float32, device="cuda"),
            device="cuda",
        )
        viewmatrix = fixed_camera.world_view_transform
        projmatrix = fixed_camera.full_proj_transform
        campos = fixed_camera.camera_center
    else:
        viewmatrix = camera.world_view_transform.cuda()
        projmatrix = camera.full_proj_transform.cuda()
        campos = camera.camera_center.cuda()

    raster_settings = SurfelRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=gaussian_assets[0].active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = SurfelRasterizer(raster_settings)

    frame = camera.timestamp
    all_means3D, all_features, all_opacities, all_scales = [], [], [], []
    obj_rot, rot_in_local = [], []

    use_dual_opacity = getattr(args, "camera_dual_opacity", True)

    for pc in gaussian_assets:
        means3D = pc.get_world_xyz(frame)
        all_means3D.append(means3D)
        all_features.append(pc.get_camera_features)
        all_opacities.append(pc.get_opacity_cam if use_dual_opacity else pc.get_opacity)
        all_scales.append(pc.get_scaling)
        r1, r2 = pc.get_rotation(frame)
        obj_rot.append(r1.expand(r2.shape[0], -1))
        rot_in_local.append(r2)

    means3D = torch.cat(all_means3D, 0)
    features = torch.cat(all_features, 0)
    opacities = torch.cat(all_opacities, 0)
    scales = torch.cat(all_scales, 0).clamp(max=1.0)

    if not args.dynamic or len(gaussian_assets) == 1:
        rotations = rot_in_local[0]
    else:
        obj_rot_objs = torch.cat(obj_rot[1:], 0)
        rots_bkgd = rot_in_local[0]
        rot_local_objs = torch.cat(rot_in_local[1:], 0)
        rot_local_objs = F.normalize(rot_local_objs, dim=1)
        rotations_objs = quaternion_raw_multiply(None, obj_rot_objs, rot_local_objs)
        rotations = torch.cat([rots_bkgd, rotations_objs], 0)

    if cam_rotation is not None and cam_translation is not None:
        means3D = means3D @ cam_rotation + cam_translation
        cam_rotation_cw = cam_rotation.T
        cam_quaternion = matrix_to_quaternion(cam_rotation_cw.unsqueeze(0)).expand(rotations.shape[0], -1)
        rotations = quaternion_raw_multiply(None, cam_quaternion, rotations)
    rotations = F.normalize(rotations, dim=1)

    cam_space_means = geom_transform_points(means3D, viewmatrix)
    visible_mask = rasterizer.markVisible(means3D)
    visible_mask = (
        visible_mask
        & torch.isfinite(means3D).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(rotations).all(dim=1)
        & (cam_space_means[:, 2] > float(camera.znear))
        & (cam_space_means[:, 2] < float(camera.zfar))
    )
    means3D = means3D[visible_mask]
    features = features[visible_mask]
    opacities = opacities[visible_mask]
    scales = scales[visible_mask]
    rotations = rotations[visible_mask]

    shs_view = features.transpose(1, 2).view(
        -1, 3, (gaussian_assets[0].max_sh_degree + 1) ** 2
    )
    dir_pp = means3D - campos.repeat(means3D.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
    colors_rgb = torch.clamp_min(
        eval_sh(gaussian_assets[0].active_sh_degree, shs_view, dir_pp_normalized) + 0.5,
        0.0,
    )
    colors_precomp = torch.cat(
        [colors_rgb, torch.zeros((colors_rgb.shape[0], 1), device=colors_rgb.device, dtype=colors_rgb.dtype)],
        dim=1,
    )

    screenspace_points = torch.zeros_like(means3D, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    rendered_image, _radii, allmap = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    rgb = rendered_image[:3]
    alpha = allmap[1:2]
    expected_depth = allmap[0:1] / alpha.clamp_min(1.0e-8)
    depth = expected_depth.squeeze(0)
    return {
        "rgb": rgb.permute(1, 2, 0),
        "depth": depth,
        "alpha": alpha.squeeze(0),
        "screenspace_points": screenspace_points,
        "radii": _radii,
        "visibility_filter": _radii > 0,
        "num_visible": int((_radii > 0).sum().item()),
        "render_backend": "surfel_rasterization",
    }


def render_camera_raytracing(camera, gaussian_assets, args, scaling_modifier=1.0,
                             cam_rotation=None, cam_translation=None, require_rgb=False):
    if cam_rotation is not None and cam_translation is not None:
        traced_camera = _make_camera_like(
            camera,
            rotation=torch.eye(3, dtype=torch.float32, device="cuda"),
            translation=torch.zeros(3, dtype=torch.float32, device="cuda"),
            device="cuda",
        )
    else:
        traced_camera = _make_camera_like(camera, device="cuda")
    background = torch.zeros(3, device="cuda").float() if require_rgb else torch.tensor([0, 0, 1], device="cuda").float()
    traced = raytracing(
        traced_camera.timestamp,
        gaussian_assets,
        traced_camera,
        background,
        args,
        scaling_modifier=scaling_modifier,
        depth_only=not require_rgb,
        gaussian_transform_rotation=cam_rotation,
        gaussian_transform_translation=cam_translation,
        feature_mode="camera" if require_rgb else "lidar",
    )
    depth = traced["depth"]
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    result = {
        "depth": depth,
        "intensity": traced["intensity"],
        "raydrop": traced["raydrop"],
        "rgb": traced["rgb"],
        "accumulation": traced["accumulation"],
        "normal": traced["normal"],
        "final_transmittance": traced["final_transmittance"],
        "means3D": traced["means3D"],
        "accum_gaussian_weight": traced["accum_gaussian_weight"],
        "render_backend": "raytracing",
        "num_visible": int((depth > 0.0).sum().item()),
    }
    return result


def render_camera(camera, gaussian_assets, args, scaling_modifier=1.0,
                  cam_rotation=None, cam_translation=None, require_rgb=False):
    backend = get_camera_render_backend(args, require_rgb=require_rgb)
    if backend == "raytracing":
        return render_camera_raytracing(
            camera,
            gaussian_assets,
            args,
            scaling_modifier=scaling_modifier,
            cam_rotation=cam_rotation,
            cam_translation=cam_translation,
            require_rgb=require_rgb,
        )
    if backend == "surfel_rasterization":
        return render_camera_2dgs(
            camera,
            gaussian_assets,
            args,
            scaling_modifier=scaling_modifier,
            cam_rotation=cam_rotation,
            cam_translation=cam_translation,
        )
    return render_camera_3dgs(
        camera,
        gaussian_assets,
        args,
        scaling_modifier=scaling_modifier,
        cam_rotation=cam_rotation,
        cam_translation=cam_translation,
    )
