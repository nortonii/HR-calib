import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from lib.utils.velodyne_utils import assign_inclinations_to_rows

DEFAULT_KITTI_LIDAR_WIDTH = 2048


def resolve_kitti_lidar_width(args, default: int = DEFAULT_KITTI_LIDAR_WIDTH) -> int:
    candidate = None
    if args is not None:
        candidate = getattr(args, "kitti_lidar_width", None)
        if candidate is None:
            data_cfg = getattr(args, "data", None)
            if data_cfg is not None:
                candidate = getattr(data_cfg, "kitti_lidar_width", None)
    width = default if candidate is None else int(candidate)
    if width <= 0:
        raise ValueError(f"kitti_lidar_width must be positive, got {width}")
    return width


def _compute_kitti_column_indices(
    azimuth: np.ndarray,
    width: int,
    azimuth_left: float,
    azimuth_right: float,
) -> np.ndarray:
    h_res = (azimuth_right - azimuth_left) / float(width)
    col = np.rint((azimuth - azimuth_left) / h_res).astype(np.int64)
    return np.mod(col, width).astype(np.int64)


def _expand_ray_origin_map(ray_origin, height, width, default_origin):
    if ray_origin is None:
        ray_origin = default_origin

    ray_origin = np.asarray(ray_origin, dtype=np.float32)
    if ray_origin.shape == (3,):
        return np.broadcast_to(ray_origin.reshape(1, 1, 3), (height, width, 3)).copy()
    if ray_origin.shape == (height, 3):
        return np.broadcast_to(ray_origin[:, None, :], (height, width, 3)).copy()
    if ray_origin.shape == (width, 3):
        return np.broadcast_to(ray_origin[None, :, :], (height, width, 3)).copy()
    if ray_origin.shape == (height, width, 3):
        return ray_origin.copy()
    raise ValueError(
        f"ray_origin must be (3,), ({height}, 3), ({width}, 3) or ({height}, {width}, 3), got {ray_origin.shape}"
    )


def _expand_sensor2world_map(sensor2world, width):
    sensor2world = np.asarray(sensor2world, dtype=np.float32)
    if sensor2world.shape == (4, 4):
        return np.broadcast_to(sensor2world[None, :, :], (width, 4, 4)).copy()
    if sensor2world.shape == (width, 4, 4):
        return sensor2world.copy()
    raise ValueError(
        f"sensor2world must be (4, 4) or ({width}, 4, 4), got {sensor2world.shape}"
    )


def interpolate_sensor2world_columns(prev_sensor2world, current_sensor2world, width):
    current_sensor2world = np.asarray(current_sensor2world, dtype=np.float32).reshape(4, 4)
    if prev_sensor2world is None:
        return _expand_sensor2world_map(current_sensor2world, width)

    prev_sensor2world = np.asarray(prev_sensor2world, dtype=np.float32).reshape(4, 4)
    alpha = np.linspace(0.0, 1.0, width, dtype=np.float32)

    rotations = SciRotation.from_matrix(
        np.stack(
            [
                prev_sensor2world[:3, :3].astype(np.float64),
                current_sensor2world[:3, :3].astype(np.float64),
            ],
            axis=0,
        )
    )
    slerp = Slerp([0.0, 1.0], rotations)
    interp_rot = slerp(alpha.astype(np.float64)).as_matrix().astype(np.float32)
    interp_trans = (
        (1.0 - alpha)[:, None] * prev_sensor2world[:3, 3][None, :]
        + alpha[:, None] * current_sensor2world[:3, 3][None, :]
    ).astype(np.float32)

    pose_map = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], width, axis=0)
    pose_map[:, :3, :3] = interp_rot
    pose_map[:, :3, 3] = interp_trans
    return pose_map


def build_kitti_range_image_from_points(
    xyzs,
    intensities,
    beam_inclinations_top_to_bottom,
    width,
    sensor2world,
    ray_origin=None,
    max_depth=80.0,
    min_depth=0.1,
    azimuth_left=np.pi,
    azimuth_right=-np.pi,
):
    beam_inclinations_top_to_bottom = np.asarray(
        beam_inclinations_top_to_bottom, dtype=np.float32
    ).reshape(-1)
    xyzs = np.asarray(xyzs, dtype=np.float32).reshape(-1, 3)
    intensities = np.asarray(intensities, dtype=np.float32).reshape(-1)

    height = beam_inclinations_top_to_bottom.size
    sensor2world_map = _expand_sensor2world_map(sensor2world, width)
    default_origin = sensor2world_map[:, :3, 3]
    ray_origin_map = _expand_ray_origin_map(ray_origin, height, width, default_origin)
    empty_ray_direction_map = np.zeros((height, width, 3), dtype=np.float32)

    if xyzs.shape[0] == 0:
        return (
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            empty_ray_direction_map,
        )

    dists = np.linalg.norm(xyzs, axis=1)
    valid = (dists >= min_depth) & (dists <= max_depth)
    xyzs = xyzs[valid]
    intensities = intensities[valid]

    if xyzs.shape[0] == 0:
        return (
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            empty_ray_direction_map,
        )

    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

    w_idx = _compute_kitti_column_indices(
        azimuth=azimuth,
        width=width,
        azimuth_left=azimuth_left,
        azimuth_right=azimuth_right,
    )
    h_idx, h_valid = assign_inclinations_to_rows(
        inclination, beam_inclinations_top_to_bottom
    )
    in_bounds = (
        h_valid
        & (h_idx >= 0)
        & (h_idx < height)
    )
    xyzs = xyzs[in_bounds]
    intensities = intensities[in_bounds]
    h_idx = h_idx[in_bounds]
    w_idx = w_idx[in_bounds]

    if xyzs.shape[0] == 0:
        return (
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            empty_ray_direction_map,
        )

    point_rotations = sensor2world_map[w_idx, :3, :3]
    point_translations = sensor2world_map[w_idx, :3, 3]
    points_world = np.einsum("nc,nkc->nk", xyzs, point_rotations)
    points_world = points_world + point_translations
    origins = ray_origin_map[h_idx, w_idx]
    ray_vectors = points_world - origins
    ray_depths = np.linalg.norm(ray_vectors, axis=1)
    valid_rays = ray_depths > 1.0e-8

    points_world = points_world[valid_rays]
    intensities = intensities[valid_rays]
    h_idx = h_idx[valid_rays]
    w_idx = w_idx[valid_rays]
    ray_vectors = ray_vectors[valid_rays]
    ray_depths = ray_depths[valid_rays]

    depth_map = np.zeros((height, width), dtype=np.float32)
    intensity_map = np.zeros((height, width), dtype=np.float32)
    ray_direction_map = empty_ray_direction_map.copy()

    if ray_depths.shape[0] == 0:
        return depth_map, intensity_map, ray_direction_map

    order = np.argsort(ray_depths)[::-1]
    h_ord = h_idx[order]
    w_ord = w_idx[order]
    depth_map[h_ord, w_ord] = ray_depths[order]
    intensity_map[h_ord, w_ord] = intensities[order]
    ray_direction_map[h_ord, w_ord] = (
        ray_vectors[order] / ray_depths[order, None]
    ).astype(np.float32)

    return depth_map, intensity_map, ray_direction_map

def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view

def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    lidar_K = np.asarray(lidar_K, dtype=np.float32).reshape(-1)
    use_beam_angles = lidar_K.size > 2
    if use_beam_angles:
        row_angles = lidar_K
        if row_angles.size != lidar_H:
            raise ValueError(
                f"Expected {lidar_H} beam angles for lidar_H={lidar_H}, got {row_angles.size}"
            )
    else:
        fov_up, fov = lidar_K
        fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        c = int(round(beta / (2 * np.pi / lidar_W)))
        if use_beam_angles:
            inclination = np.arctan2(z, np.sqrt(x**2 + y**2))
            row_idx, valid_row = assign_inclinations_to_rows(
                np.asarray([inclination], dtype=np.float32), row_angles
            )
            if not bool(valid_row[0]):
                continue
            r = int(row_idx[0])
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities


def lidar_to_pano(
    local_points: np.ndarray, lidar_H: int, lidar_W: int, lidar_K: int, max_dpeth=80
):
    """
    Convert lidar frame to pano frame. Lidar points are in local coordinates.

    Args:
        local_points: (N, 3), float32, in lidar frame.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
    """

    # (N, 3) -> (N, 4), filled with zeros.
    local_points_with_intensities = np.concatenate(
        [local_points, np.zeros((local_points.shape[0], 1))], axis=1
    )
    pano, _ = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=lidar_K,
        max_dpeth=max_dpeth,
    )
    return pano


def pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """
    H, W = pano.shape
    lidar_K = np.asarray(lidar_K, dtype=np.float32).reshape(-1)
    use_beam_angles = lidar_K.size > 2
    if use_beam_angles and lidar_K.size != H:
        raise ValueError(f"Expected {H} beam angles for pano height {H}, got {lidar_K.size}")
    if not use_beam_angles:
        fov_up, fov = lidar_K
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    beta = -(i - W / 2) / W * 2 * np.pi
    if use_beam_angles:
        alpha = np.broadcast_to(lidar_K.reshape(H, 1), (H, W))
    else:
        alpha = (fov_up - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]

    return local_points_with_intensities


def pano_to_lidar(pano, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
    )
    return local_points_with_intensities[:, :3]
