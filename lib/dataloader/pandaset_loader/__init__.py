from __future__ import annotations

import gzip
import math
import os
import pickle
from typing import Dict

import numpy as np
import torch
try:
    from pandaset import DataSet
except ImportError:
    DataSet = None
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from lib.scene import LiDARSensor


# PandaSet / Pandar64 distribution from the NeurAD dataparser.
PANDAR64_VERT_DEG = [
    14.882, 11.032, 8.059, 5.057, 3.040, 2.028, 1.860, 1.688,
    1.522, 1.351, 1.184, 1.013, 0.846, 0.675, 0.508, 0.337,
    0.169, 0.000, -0.169, -0.337, -0.508, -0.675, -0.846, -1.013,
    -1.184, -1.351, -1.522, -1.688, -1.860, -2.028, -2.198, -2.365,
    -2.536, -2.700, -2.873, -3.040, -3.210, -3.375, -3.548, -3.712,
    -3.884, -4.050, -4.221, -4.385, -4.558, -4.720, -4.892, -5.057,
    -5.229, -5.391, -5.565, -5.726, -5.898, -6.061, -7.063, -8.059,
    -9.060, -9.885, -11.032, -12.006, -12.974, -13.930, -18.889, -24.897,
]
PANDAR64_ROT_DEG = [
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, 1.042, 3.125,
    5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208,
    -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
    1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125,
    5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208,
    -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
    1.042, 3.125, 5.208, -5.208, -3.125, -1.042, -1.042, -1.042,
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,
]


def _debug_enabled() -> bool:
    return str(os.getenv("PANDASET_DEBUG_RAYS", "0")).lower() not in {"", "0", "false", "no"}


def _fmt_vec(vec: np.ndarray | torch.Tensor, precision: int = 4) -> str:
    if torch.is_tensor(vec):
        vec = vec.detach().cpu().numpy()
    return np.array2string(np.asarray(vec), precision=precision, suppress_small=True)


def _maybe_print_ray_debug(
    frame: int,
    origin_mode: str,
    current_center: np.ndarray,
    ray_origin: np.ndarray,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> None:
    if not _debug_enabled():
        return
    sample_idx = [
        (0, 0),
        (0, rays_d.shape[1] // 2),
        (rays_d.shape[0] // 2, rays_d.shape[1] // 2),
        (rays_d.shape[0] - 1, rays_d.shape[1] - 1),
    ]
    samples = []
    for r, c in sample_idx:
        samples.append(
            f"({r},{c}) o={_fmt_vec(rays_o[r, c])} d={_fmt_vec(rays_d[r, c])}"
        )
    print(
        "[PandaSet ray debug] "
        f"frame={frame} mode={origin_mode} "
        f"center={_fmt_vec(current_center)} "
        f"ray_origin_shape={tuple(ray_origin.shape)} "
        f"rays_o_shape={tuple(rays_o.shape)} "
        f"rays_d_shape={tuple(rays_d.shape)} "
        + " | ".join(samples)
    )


def _resolve_source_root(base_dir: str) -> str:
    base_dir = os.path.abspath(base_dir)
    if os.path.isdir(base_dir):
        direct_children = set(os.listdir(base_dir))
        if {"lidar", "meta"}.issubset(direct_children):
            return base_dir
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs and any(name.isdigit() for name in subdirs):
            return base_dir
        nested = os.path.join(base_dir, "pandaset")
        if os.path.isdir(nested):
            return nested
    raise FileNotFoundError(f"Cannot locate PandaSet root under: {base_dir}")


def _pose_dict_to_matrix(pose: dict) -> np.ndarray:
    position = pose["position"]
    heading = pose["heading"]
    w = float(heading["w"])
    x = float(heading["x"])
    y = float(heading["y"])
    z = float(heading["z"])
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array([position["x"], position["y"], position["z"]], dtype=np.float32)
    return mat


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    rot = mat[:3, :3]
    tr = float(np.trace(rot))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1.0e-8)) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1.0e-8)) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1.0e-8)) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / max(np.linalg.norm(q0), 1.0e-8)
    q1 = q1 / max(np.linalg.norm(q1), 1.0e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / max(np.linalg.norm(q), 1.0e-8)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1.0e-8:
        return q0
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / max(np.linalg.norm(q), 1.0e-8)


def _pose_at_timestamp(timestamps: np.ndarray, pose_mats: np.ndarray, query_t: float) -> np.ndarray:
    if len(timestamps) == 0:
        return np.eye(4, dtype=np.float32)
    if len(timestamps) == 1:
        return pose_mats[0]
    idx1 = int(np.searchsorted(timestamps, query_t, side="right"))
    idx1 = int(np.clip(idx1, 1, len(timestamps) - 1))
    idx0 = idx1 - 1
    t0 = float(timestamps[idx0])
    t1 = float(timestamps[idx1])
    alpha = 0.0 if abs(t1 - t0) < 1.0e-8 else float((query_t - t0) / (t1 - t0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    pose0 = pose_mats[idx0]
    pose1 = pose_mats[idx1]
    trans = (1.0 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]
    q0 = _mat_to_quat_wxyz(pose0)
    q1 = _mat_to_quat_wxyz(pose1)
    quat = _quat_slerp(q0, q1, alpha)
    w, x, y, z = quat.tolist()
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = trans.astype(np.float32)
    return pose


def _poses_at_timestamps(
    query_timestamps: np.ndarray,
    timestamps: np.ndarray,
    pose_mats: np.ndarray,
) -> np.ndarray:
    if len(query_timestamps) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    if len(timestamps) == 0:
        return np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(query_timestamps), axis=0)
    if len(timestamps) == 1:
        return np.repeat(pose_mats[:1], len(query_timestamps), axis=0)

    timestamps = np.asarray(timestamps, dtype=np.float64)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    query_timestamps = np.clip(query_timestamps, timestamps[0], timestamps[-1])

    rotations = SciRotation.from_matrix(pose_mats[:, :3, :3].astype(np.float64))
    slerp = Slerp(timestamps, rotations)
    query_rots = slerp(query_timestamps).as_matrix().astype(np.float32)

    idx1 = np.searchsorted(timestamps, query_timestamps, side="right")
    idx1 = np.clip(idx1, 1, len(timestamps) - 1)
    idx0 = idx1 - 1
    t0 = timestamps[idx0]
    t1 = timestamps[idx1]
    alpha = np.zeros_like(query_timestamps, dtype=np.float64)
    denom = np.clip(t1 - t0, 1.0e-8, None)
    alpha = (query_timestamps - t0) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    trans0 = pose_mats[idx0, :3, 3].astype(np.float64)
    trans1 = pose_mats[idx1, :3, 3].astype(np.float64)
    query_trans = ((1.0 - alpha)[:, None] * trans0 + alpha[:, None] * trans1).astype(np.float32)

    query_pose = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(query_timestamps), axis=0)
    query_pose[:, :3, :3] = query_rots
    query_pose[:, :3, 3] = query_trans
    return query_pose


def _interpolated_centers(
    query_timestamps: np.ndarray,
    pose_timestamps: np.ndarray,
    pose_mats: np.ndarray,
) -> np.ndarray:
    centers = np.zeros((len(query_timestamps), 3), dtype=np.float32)
    for i, ts in enumerate(query_timestamps):
        centers[i] = _pose_at_timestamp(pose_timestamps, pose_mats, float(ts))[:3, 3]
    return centers


def _load_cuboids(source_root: str, seq: str, frame: int):
    """Load cuboid annotations for a given frame. Returns DataFrame or None."""
    path = os.path.join(source_root, seq, "annotations", "cuboids", f"{frame:02d}.pkl.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _dynamic_flags(points_world: np.ndarray, cuboids_df) -> np.ndarray:
    """Return boolean array: True if point falls inside a non-stationary cuboid."""
    dynamic = np.zeros(len(points_world), dtype=bool)
    if cuboids_df is None or len(cuboids_df) == 0:
        return dynamic
    moving = cuboids_df[cuboids_df["stationary"] == False]
    for _, row in moving.iterrows():
        cx, cy, cz = row["position.x"], row["position.y"], row["position.z"]
        dx, dy, dz = row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]
        yaw = float(row["yaw"])
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        tx = points_world[:, 0] - cx
        ty = points_world[:, 1] - cy
        tz = points_world[:, 2] - cz
        lx = cos_y * tx - sin_y * ty
        ly = sin_y * tx + cos_y * ty
        lz = tz
        inside = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(lz) <= dz / 2)
        dynamic |= inside
    return dynamic


def _project_range_image(
    points_world: np.ndarray,
    intensities: np.ndarray,
    pose_inv: np.ndarray,
    ray_origin: np.ndarray,
    width: int,
    height: int,
    max_depth: float,
    is_dynamic: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty_dyn = np.zeros((height, width), dtype=np.float32)
    if points_world.size == 0:
        return np.zeros((height, width), dtype=np.float32), -np.ones((height, width), dtype=np.float32), empty_dyn

    points_sensor = (pose_inv[:3, :3] @ points_world.T).T + pose_inv[:3, 3]
    x = points_sensor[:, 0]
    y = points_sensor[:, 1]
    z = points_sensor[:, 2]
    xy = np.sqrt(x * x + y * y)
    ranges = np.sqrt(x * x + y * y + z * z)
    valid = (ranges > 0.1) & (ranges < max_depth) & np.isfinite(ranges)
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.float32), -np.ones((height, width), dtype=np.float32), empty_dyn

    points_world = points_world[valid]
    intensities = intensities[valid]
    if is_dynamic is not None:
        is_dynamic = is_dynamic[valid]
    else:
        is_dynamic = np.zeros(len(points_world), dtype=bool)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    xy = xy[valid]
    ranges = ranges[valid]
    ring_angles = np.radians(np.asarray(PANDAR64_VERT_DEG, dtype=np.float32))
    row_offsets = np.radians(np.asarray(PANDAR64_ROT_DEG, dtype=np.float32))

    elevation = np.arctan2(z, np.clip(xy, 1.0e-6, None))
    diff = np.abs(elevation[:, None] - ring_angles[None, :])
    ring_id = np.argmin(diff, axis=1).astype(np.int64)

    azimuth = np.arctan2(y, x) - row_offsets[ring_id]
    azimuth = (azimuth + math.pi) % (2.0 * math.pi) - math.pi
    # Use the same column convention as get_range_rays: azimuth decreases with column index
    # (column 0 = azimuth π, column W-1 ≈ azimuth -π), matching KITTI's beam_id formula.
    beam_id = np.floor(float(width) * (math.pi - azimuth) / (2.0 * math.pi)).astype(np.int64)
    beam_id = np.mod(beam_id, width)

    if ray_origin.ndim == 1:
        ray_origin = np.repeat(ray_origin[None, None, :], height, axis=0)
        ray_origin = np.repeat(ray_origin, width, axis=1)
    elif ray_origin.ndim == 2 and ray_origin.shape == (width, 3):
        ray_origin = np.repeat(ray_origin[None, :, :], height, axis=0)
    elif ray_origin.ndim == 3 and ray_origin.shape == (height, width, 3):
        pass
    else:
        raise ValueError(
            "ray_origin must be (3,), (W, 3) or (H, W, 3) for PandaSet, "
            f"got {ray_origin.shape}"
        )

    depths = np.linalg.norm(points_world - ray_origin[ring_id, beam_id], axis=1).astype(np.float32)
    order = np.argsort(depths)
    ring_id = ring_id[order]
    beam_id = beam_id[order]
    depths = depths[order]
    intensities = intensities[order]
    is_dynamic = is_dynamic[order]

    depth_map = np.zeros((height, width), dtype=np.float32)
    intensity_map = -np.ones((height, width), dtype=np.float32)
    dynamic_map = np.zeros((height, width), dtype=np.float32)
    seen = np.zeros((height, width), dtype=bool)
    for r, b, d, inten, dyn in zip(ring_id, beam_id, depths, intensities, is_dynamic, strict=False):
        if not seen[r, b]:
            depth_map[r, b] = d
            intensity_map[r, b] = float(inten)
            dynamic_map[r, b] = float(dyn)
            seen[r, b] = True
    return depth_map, intensity_map, dynamic_map


def load_pandaset_raw(base_dir, args):
    source_root = _resolve_source_root(base_dir)
    dataset = DataSet(source_root)

    if hasattr(args, "seq"):
        seq = str(args.seq)
    elif hasattr(args, "pandaset_seq"):
        seq = str(args.pandaset_seq)
    elif hasattr(args, "scene_id"):
        seq = str(args.scene_id)
    else:
        seq = "001"
    if seq not in dataset.sequences():
        available = ", ".join(sorted(dataset.sequences()))
        raise ValueError(f"PandaSet sequence {seq} not found under {source_root}. Available: {available}")

    sequence = dataset[seq]
    sequence.load_lidar()

    lidar_sensor_id = int(getattr(args, "pandaset_lidar_sensor_id", 0))
    sequence.lidar.set_sensor(lidar_sensor_id)
    lidar_data = sequence.lidar.data
    lidar_pose_dicts = sequence.lidar.poses or []
    lidar_timestamps = np.asarray(sequence.lidar.timestamps or [], dtype=np.float64)
    if lidar_pose_dicts:
        lidar_pose_mats = np.stack([_pose_dict_to_matrix(p) for p in lidar_pose_dicts], axis=0)
    else:
        lidar_pose_mats = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(lidar_data), axis=0)
    if lidar_timestamps.size == 0:
        lidar_timestamps = np.arange(len(lidar_pose_mats), dtype=np.float64)

    frame_start, frame_end = map(int, args.frame_length)
    frame_start = max(0, frame_start)
    frame_end = min(frame_end, len(lidar_data) - 1)
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame range: {args.frame_length}, sequence has {len(lidar_data)} frames")

    origin_mode = str(getattr(args, "lidar_origin_mode", "center")).lower()
    max_depth = float(getattr(args, "pandaset_max_depth", 80.0))

    lidar = LiDARSensor(
        sensor2ego=np.eye(4, dtype=np.float32),
        name=f"pandaset_{seq}",
        inclination_bounds=np.radians(np.asarray(PANDAR64_VERT_DEG, dtype=np.float32)).tolist(),
        data_type="PandaSet",
    )
    lidar.azimuth_offsets = np.radians(
        np.asarray(PANDAR64_ROT_DEG, dtype=np.float32)
    ).tolist()

    for frame in range(frame_start, frame_end + 1):
        cuboids_df = _load_cuboids(source_root, seq, frame)
        df = lidar_data[frame]
        if len(df) == 0:
            depth = np.zeros((64, 2048), dtype=np.float32)
            intensity = np.zeros((64, 2048), dtype=np.float32)
            current_pose = lidar_pose_mats[frame]
            ray_origin = current_pose[:3, 3].astype(np.float32)
            lidar.add_frame(
                frame,
                current_pose,
                np.stack([depth, intensity, np.zeros_like(depth), np.zeros_like(depth)], axis=-1),
                -np.ones((64, 2048, 4), dtype=np.float32),
                ray_origin=ray_origin,
            )
            if origin_mode == "center":
                rays_o, rays_d = lidar.get_range_rays(frame)
                _maybe_print_ray_debug(frame, origin_mode, ray_origin, ray_origin, rays_o, rays_d)
            continue

        if "d" in df.columns:
            df = df.loc[df["d"] == lidar_sensor_id]
        points_world = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        intensities = df["i"].to_numpy(dtype=np.float32) / 255.0 if "i" in df.columns else np.zeros((len(df),), dtype=np.float32)
        point_timestamps = df["t"].to_numpy(dtype=np.float64) if "t" in df.columns else None
        is_dynamic = _dynamic_flags(points_world, cuboids_df)

        current_pose = lidar_pose_mats[frame]
        pose_inv = np.linalg.inv(current_pose)
        current_center = current_pose[:3, 3].astype(np.float32)

        ray_origin = current_center[None, :].repeat(2048, axis=0)
        if origin_mode == "column_interp":
            if frame > 0:
                prev_center = lidar_pose_mats[frame - 1][:3, 3].astype(np.float32)
                sweep = current_center - prev_center
                alpha = np.linspace(0.0, 1.0, 2048, dtype=np.float32)
                ray_origin = current_center[None, :] + alpha[:, None] * sweep[None, :]
        elif (
            origin_mode == "timestamp_column"
            and point_timestamps is not None
            and len(point_timestamps) > 1
            and len(lidar_timestamps) > 1
            and len(lidar_pose_mats) > 1
        ):
            col_ts = np.linspace(point_timestamps.min(), point_timestamps.max(), 2048, dtype=np.float64)
            ray_origin = _interpolated_centers(col_ts, lidar_timestamps, lidar_pose_mats)
        elif (
            origin_mode == "timestamp_point"
            and point_timestamps is not None
            and len(point_timestamps) > 0
            and len(lidar_timestamps) > 1
            and len(lidar_pose_mats) > 1
        ):
            point_poses = _poses_at_timestamps(point_timestamps, lidar_timestamps, lidar_pose_mats)
            point_pose_inv = np.linalg.inv(point_poses)
            points_sensor = (point_pose_inv[:, :3, :3] @ points_world[..., None]).squeeze(-1) + point_pose_inv[:, :3, 3]
            x = points_sensor[:, 0]
            y = points_sensor[:, 1]
            z = points_sensor[:, 2]
            xy = np.sqrt(x * x + y * y)
            ranges = np.sqrt(x * x + y * y + z * z)
            valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < max_depth)
            if not np.any(valid):
                depth_map = np.zeros((64, 2048), dtype=np.float32)
                intensity_map = -np.ones((64, 2048), dtype=np.float32)
                ray_origin = np.repeat(current_center[None, :], 2048, axis=0)
            else:
                points_world = points_world[valid]
                intensities = intensities[valid]
                is_dynamic_valid = is_dynamic[valid]
                point_poses = point_poses[valid]
                x = x[valid]
                y = y[valid]
                z = z[valid]
                xy = xy[valid]
                ring_angles = np.radians(np.asarray(PANDAR64_VERT_DEG, dtype=np.float32))
                row_offsets = np.radians(np.asarray(PANDAR64_ROT_DEG, dtype=np.float32))
                elevation = np.arctan2(z, np.clip(xy, 1.0e-6, None))
                diff = np.abs(elevation[:, None] - ring_angles[None, :])
                ring_id = np.argmin(diff, axis=1).astype(np.int64)
                azimuth = np.arctan2(y, x) - row_offsets[ring_id]
                azimuth = (azimuth + math.pi) % (2.0 * math.pi) - math.pi
                beam_id = np.floor(2048.0 * (math.pi - azimuth) / (2.0 * math.pi)).astype(np.int64)
                beam_id = np.mod(beam_id, 2048)
                ray_origin_map = np.repeat(current_center[None, None, :], 64, axis=0)
                ray_origin_map = np.repeat(ray_origin_map, 2048, axis=1)
                origin_world = point_poses[:, :3, 3].astype(np.float32)
                depths = np.linalg.norm(points_world - origin_world, axis=1).astype(np.float32)
                order = np.argsort(depths)
                ring_id = ring_id[order]
                beam_id = beam_id[order]
                depths = depths[order]
                intensities = intensities[order]
                is_dynamic_valid = is_dynamic_valid[order]
                origin_world = origin_world[order]
                depth_map = np.zeros((64, 2048), dtype=np.float32)
                intensity_map = -np.ones((64, 2048), dtype=np.float32)
                dynamic_map = np.zeros((64, 2048), dtype=np.float32)
                seen = np.zeros((64, 2048), dtype=bool)
                for r, b, d, inten, dyn, ori in zip(ring_id, beam_id, depths, intensities, is_dynamic_valid, origin_world, strict=False):
                    if not seen[r, b]:
                        depth_map[r, b] = d
                        intensity_map[r, b] = float(inten)
                        dynamic_map[r, b] = float(dyn)
                        ray_origin_map[r, b] = ori
                        seen[r, b] = True
                range_image_r1 = np.stack(
                    [depth_map, intensity_map, dynamic_map, np.zeros_like(depth_map)],
                    axis=-1,
                )
                range_image_r2 = -np.ones_like(range_image_r1, dtype=np.float32)
                lidar.add_frame(frame, current_pose, range_image_r1, range_image_r2, ray_origin=ray_origin_map)
                continue

        depth_map, intensity_map, dynamic_map = _project_range_image(
            points_world=points_world,
            intensities=intensities,
            pose_inv=pose_inv,
            ray_origin=ray_origin,
            width=2048,
            height=64,
            max_depth=max_depth,
            is_dynamic=is_dynamic,
        )
        range_image_r1 = np.stack(
            [depth_map, intensity_map, dynamic_map, np.zeros_like(depth_map)],
            axis=-1,
        )
        range_image_r2 = -np.ones_like(range_image_r1, dtype=np.float32)
        lidar.add_frame(frame, current_pose, range_image_r1, range_image_r2, ray_origin=ray_origin)
        if origin_mode == "center":
            rays_o, rays_d = lidar.get_range_rays(frame)
            _maybe_print_ray_debug(frame, origin_mode, current_center, ray_origin, rays_o, rays_d)

    return lidar, {}
