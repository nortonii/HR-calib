"""
Standard 3DGS rasterization-based camera rendering (not OptiX ray tracing).
Used for camera RGB supervision during training.
"""
import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from lib.scene.cameras import Camera
from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply


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
        fixed_camera = Camera(
            timestamp=camera.timestamp,
            R=torch.eye(3, dtype=torch.float32, device="cuda"),
            T=torch.zeros(3, dtype=torch.float32, device="cuda"),
            w=camera.image_width,
            h=camera.image_height,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            data_device="cuda",
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
        features = pc.get_features  # (N, D, 3): D=16 for sh_degree=3
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
