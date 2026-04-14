from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray


@dataclass
class FrameCalibrationData:
    frame_name: str
    rgb_path: str
    depth_path: str
    rgb_points: np.ndarray
    depth_points: np.ndarray
    points_3d: np.ndarray
    point_weights: np.ndarray | None = None
    frame_index: int = -1
    frame_id: int = -1
    temporal_weight: float = 1.0
    pnp_inliers: int = 0
    pnp_reproj_error: float = np.nan


@dataclass
class TemporalMatchSummary:
    frame_i: int
    frame_j: int
    num_matches: int


@dataclass
class TemporalMatchData:
    frame_i: int
    frame_j: int
    points_i: np.ndarray
    points_j: np.ndarray


@dataclass
class TemporalResidualData:
    source_frame_index: int
    target_frame_index: int
    source_points_3d: np.ndarray
    target_rgb_points: np.ndarray
    source_c2w: np.ndarray
    target_w2c: np.ndarray
    weight: float = 1.0


@dataclass
class CalibrationResult:
    rotation_matrix: np.ndarray
    translation: np.ndarray
    rotation_vector: np.ndarray
    mean_reprojection_error: float
    median_reprojection_error: float
    frames_used: int
    matches_used: int


@dataclass
class CalibrationComparison:
    initial: CalibrationResult
    optimized: CalibrationResult
    rotation_delta_deg: float
    translation_delta_m: float
    mean_reprojection_improvement_px: float
    median_reprojection_improvement_px: float


def _as_array(value: Any, dtype=np.float64) -> np.ndarray:
    if isinstance(value, dict):
        if {"fx", "fy", "cx", "cy"} <= set(value.keys()):
            return np.array(
                [
                    [value["fx"], 0.0, value["cx"]],
                    [0.0, value["fy"], value["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=dtype,
            )
        if "data" in value:
            data = np.asarray(value["data"], dtype=dtype)
            rows = value.get("rows")
            cols = value.get("cols")
            if rows is not None and cols is not None:
                return data.reshape(int(rows), int(cols))
            return data
        if "matrix" in value:
            return _as_array(value["matrix"], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def load_camera_model(path: str | Path) -> CameraModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera intrinsics file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif path.suffix.lower() == ".npz":
        arrays = np.load(path)
        data = {key: arrays[key] for key in arrays.files}
    elif path.suffix.lower() == ".npy":
        matrix = np.load(path)
        data = {"K": matrix}
    else:
        raise ValueError(f"Unsupported camera model format: {path.suffix}")

    if isinstance(data, dict):
        k_value = None
        for key in ("K", "k", "camera_matrix", "intrinsics", "matrix"):
            if key in data:
                k_value = data[key]
                break
        if k_value is None:
            raise KeyError(f"Could not find camera matrix in {path}")

        dist_value = None
        for key in ("dist", "distortion", "dist_coeffs", "distortion_coefficients"):
            if key in data:
                dist_value = data[key]
                break
    else:
        k_value = data
        dist_value = None

    K = _as_array(k_value).reshape(3, 3)
    dist = np.zeros((0,), dtype=np.float64) if dist_value is None else _as_array(dist_value).reshape(-1)
    return CameraModel(K=K.astype(np.float64), dist=dist.astype(np.float64))


def load_depth_map(path: str | Path, depth_scale: float = 1.0, npz_key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if npz_key is not None:
            depth = data[npz_key]
        elif len(data.files) == 1:
            depth = data[data.files[0]]
        else:
            raise KeyError(f"{path} contains multiple arrays; specify --depth_npz_key")
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth map: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]

    depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
    return depth


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def depth_to_match_image(
    depth: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    use_inverse: bool = True,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)

    values = depth[valid]
    lo = np.percentile(values, percentile_low)
    hi = np.percentile(values, percentile_high)
    if hi <= lo:
        hi = lo + 1.0e-6

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    if use_inverse:
        normalized = 1.0 - normalized
    gray = (normalized * 255.0).astype(np.uint8)
    gray[~valid] = 0
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def build_matcher(
    device: str = "cuda",
    max_num_keypoints: int = 2048,
    ransac_reproj_thresh: float = 3.0,
    img_resize: int | None = 832,
    match_threshold: float = 0.2,
    hf_endpoint: str = "https://hf-mirror.com",
):
    _configure_matchanything_hub(
        repo_id="vismatch/matchanything-roma",
        hf_endpoint=hf_endpoint,
    )

    try:
        from vismatch import get_matcher
    except ImportError as exc:
        raise ImportError(
            "vismatch is required for RGB-D calibration. Install it with `pip install vismatch`."
        ) from exc

    return get_matcher(
        "matchanything-roma",
        device=device,
        max_num_keypoints=max_num_keypoints,
        ransac_reproj_thresh=ransac_reproj_thresh,
        img_resize=img_resize,
        match_threshold=match_threshold,
    )


def _configure_matchanything_hub(
    repo_id: str,
    hf_endpoint: str,
) -> str:
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name
    from huggingface_hub.utils import LocalEntryNotFoundError
    from huggingface_hub import snapshot_download

    cache_dir = Path(constants.HF_HUB_CACHE)
    storage_folder = cache_dir / repo_folder_name(repo_id=repo_id, repo_type="model")

    try:
        snapshot_path = snapshot_download(
            repo_id,
            local_files_only=True,
            allow_patterns=["model.safetensors"],
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        constants.HF_HUB_OFFLINE = True
        return snapshot_path
    except LocalEntryNotFoundError:
        os.environ.pop("HF_HUB_OFFLINE", None)
        constants.HF_HUB_OFFLINE = False

    endpoint = hf_endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    constants.ENDPOINT = endpoint
    constants.HUGGINGFACE_CO_URL_TEMPLATE = endpoint + "/{repo_id}/resolve/{revision}/{filename}"

    storage_folder.mkdir(parents=True, exist_ok=True)
    return str(storage_folder)


def select_match_points(match_result: dict, prefer_inliers: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if prefer_inliers and int(match_result.get("num_inliers", 0)) > 0:
        points0 = match_result["inlier_kpts0"]
        points1 = match_result["inlier_kpts1"]
    else:
        points0 = match_result["matched_kpts0"]
        points1 = match_result["matched_kpts1"]
    return np.asarray(points0, dtype=np.float64), np.asarray(points1, dtype=np.float64)


def _format_matcher_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Matcher input must be an RGB image, got shape {image.shape}")
    if image.shape[0] == 3:
        chw = image
    if image.shape[-1] == 3:
        chw = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Matcher input must have 3 channels, got shape {image.shape}")
    chw = np.ascontiguousarray(chw.astype(np.float32))
    if chw.max(initial=0.0) > 1.2 or chw.min(initial=0.0) < -0.2:
        chw = chw / 255.0
    return chw


def match_cross_modal(matcher, rgb_image: np.ndarray, depth_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    result = matcher(_format_matcher_image(rgb_image), _format_matcher_image(depth_image))
    rgb_points, depth_points = select_match_points(result)
    return rgb_points, depth_points, result


def match_cross_modal_dense(
    matcher,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    query_stride: int = 4,
    cert_threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense cross-modal matching using RoMa's dense warp field.

    Bypasses the sparse keypoint sampling limit (max_num_keypoints) and instead
    queries the full dense warp field at a strided grid of pixels from the RGB
    image.  This yields ~10-100x more supervision points than sparse matching.

    Args:
        matcher: MatchAnythingMatcher with variant="roma" (built by build_matcher).
        rgb_image: RGB image, HWC uint8 or CHW/HWC float [0, 1].
        depth_image: Depth colormap, same format.
        query_stride: Sample every N pixels along both axes of the RGB image.
                      stride=1 is fully dense; stride=4 gives 16x less points.
        cert_threshold: Minimum RoMa certainty score to keep a correspondence.

    Returns:
        rgb_points:   [N, 2] float64, (x, y) pixel coords in the RGB image.
        depth_points: [N, 2] float64, (x, y) pixel coords in the depth colormap.
        certainties:  [N]   float32,  per-point certainty from the warp field.
    """
    import torch

    net = matcher.net          # MatchAnything_Model
    roma_model = net.model     # RegressionMatcher

    img0_chw = _format_matcher_image(rgb_image)   # [3, H, W] float32 [0,1]
    img1_chw = _format_matcher_image(depth_image)  # [3, H, W] float32 [0,1]

    H_A, W_A = int(img0_chw.shape[1]), int(img0_chw.shape[2])
    H_B, W_B = int(img1_chw.shape[1]), int(img1_chw.shape[2])

    img0_t = torch.from_numpy(img0_chw).to(matcher.device)
    img1_t = torch.from_numpy(img1_chw).to(matcher.device)

    with torch.no_grad():
        warp, dense_certainty = roma_model.self_inference_time_match(
            img0_t,
            img1_t,
            resize_by_stretch=net.resize_by_stretch,
            norm_img=net.norm_image,
        )

        # Build strided query grid in RGB-image pixel space (x, y)
        ys = np.arange(0, H_A, query_stride, dtype=np.float32)
        xs = np.arange(0, W_A, query_stride, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        query_pts = torch.from_numpy(
            np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # [N, 2] (x, y)
        ).float().to(matcher.device)

        warped_pts, certs = roma_model.warp_keypoints(
            query_pts, warp, dense_certainty, H_A, W_A, H_B, W_B
        )

    q_np = query_pts.cpu().numpy().astype(np.float64)       # [N, 2]
    w_np = warped_pts.cpu().numpy().astype(np.float64)      # [N, 2]
    c_np = certs.cpu().numpy().astype(np.float32)           # [N]

    valid = (
        (c_np >= cert_threshold)
        & (w_np[:, 0] >= 0) & (w_np[:, 0] <= W_B - 1)
        & (w_np[:, 1] >= 0) & (w_np[:, 1] <= H_B - 1)
    )

    return q_np[valid], w_np[valid], c_np[valid]


def match_temporal_frames(
    matcher,
    rgb_images: list[np.ndarray],
    max_pairs: int = 1,
) -> tuple[np.ndarray, list[TemporalMatchSummary], list[TemporalMatchData]]:
    num_frames = len(rgb_images)
    supports = np.ones(num_frames, dtype=np.float64)
    summaries: list[TemporalMatchSummary] = []
    matches: list[TemporalMatchData] = []

    for frame_i in range(num_frames):
        for offset in range(1, max_pairs + 1):
            frame_j = frame_i + offset
            if frame_j >= num_frames:
                break
            result = matcher(
                _format_matcher_image(rgb_images[frame_i]),
                _format_matcher_image(rgb_images[frame_j]),
            )
            points0, points1 = select_match_points(result)
            num_matches = int(points0.shape[0])
            summaries.append(
                TemporalMatchSummary(
                    frame_i=frame_i,
                    frame_j=frame_j,
                    num_matches=num_matches,
                )
            )
            matches.append(
                TemporalMatchData(
                    frame_i=frame_i,
                    frame_j=frame_j,
                    points_i=points0,
                    points_j=points1,
                )
            )
            supports[frame_i] += num_matches
            supports[frame_j] += num_matches

    median_support = np.median(supports)
    if median_support > 0:
        supports = supports / median_support
    supports = np.clip(supports, 0.25, 4.0)
    return supports, summaries, matches


def _associate_nearest_points(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if query_points.size == 0 or reference_points.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=bool)
    distances = np.linalg.norm(
        query_points[:, None, :] - reference_points[None, :, :],
        axis=2,
    )
    nearest_indices = np.argmin(distances, axis=1).astype(np.int32)
    keep_mask = distances[np.arange(len(query_points)), nearest_indices] <= float(max_distance_px)
    return nearest_indices, keep_mask


def build_temporal_residuals(
    frame_data_list: list[FrameCalibrationData],
    temporal_matches: list[TemporalMatchData],
    rgb_camera_poses_c2w: list[np.ndarray],
    match_radius_px: float = 4.0,
    min_matches: int = 8,
) -> list[TemporalResidualData]:
    frame_data_by_index = {frame_data.frame_index: frame_data for frame_data in frame_data_list}
    temporal_residuals: list[TemporalResidualData] = []

    def append_direction(
        source_index: int,
        target_index: int,
        source_match_points: np.ndarray,
        target_match_points: np.ndarray,
    ) -> None:
        source_frame = frame_data_by_index.get(source_index)
        target_frame = frame_data_by_index.get(target_index)
        if source_frame is None or target_frame is None:
            return
        if source_index >= len(rgb_camera_poses_c2w) or target_index >= len(rgb_camera_poses_c2w):
            return

        nearest_indices, keep_mask = _associate_nearest_points(
            query_points=np.asarray(source_match_points, dtype=np.float64),
            reference_points=source_frame.rgb_points,
            max_distance_px=match_radius_px,
        )
        if not np.any(keep_mask):
            return

        source_points_3d = source_frame.points_3d[nearest_indices[keep_mask]]
        target_rgb_points = np.asarray(target_match_points[keep_mask], dtype=np.float64)
        if source_points_3d.shape[0] < int(min_matches):
            return

        temporal_residuals.append(
            TemporalResidualData(
                source_frame_index=source_index,
                target_frame_index=target_index,
                source_points_3d=source_points_3d.astype(np.float64),
                target_rgb_points=target_rgb_points.astype(np.float64),
                source_c2w=np.asarray(rgb_camera_poses_c2w[source_index], dtype=np.float64),
                target_w2c=np.linalg.inv(np.asarray(rgb_camera_poses_c2w[target_index], dtype=np.float64)),
                weight=float(np.sqrt(source_frame.temporal_weight * target_frame.temporal_weight)),
            )
        )

    for match in temporal_matches:
        append_direction(match.frame_i, match.frame_j, match.points_i, match.points_j)
        append_direction(match.frame_j, match.frame_i, match.points_j, match.points_i)

    return temporal_residuals


def build_temporal_track_residuals(
    frame_data_list: list[FrameCalibrationData],
    temporal_matches: list[TemporalMatchData],
    rgb_camera_poses_c2w: list[np.ndarray],
    match_radius_px: float = 4.0,
    min_track_length: int = 3,
    min_block_points: int = 8,
) -> tuple[list[TemporalResidualData], int]:
    frame_data_by_index = {frame_data.frame_index: frame_data for frame_data in frame_data_list}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    rank: dict[tuple[int, int], int] = {}
    node_votes: dict[tuple[int, int], int] = defaultdict(int)

    def make_node(node: tuple[int, int]) -> None:
        if node not in parent:
            parent[node] = node
            rank[node] = 0

    def find(node: tuple[int, int]) -> tuple[int, int]:
        make_node(node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(node_a: tuple[int, int], node_b: tuple[int, int]) -> None:
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for match in temporal_matches:
        frame_i = frame_data_by_index.get(match.frame_i)
        frame_j = frame_data_by_index.get(match.frame_j)
        if frame_i is None or frame_j is None:
            continue

        nearest_i, keep_i = _associate_nearest_points(match.points_i, frame_i.rgb_points, match_radius_px)
        nearest_j, keep_j = _associate_nearest_points(match.points_j, frame_j.rgb_points, match_radius_px)
        keep_mask = keep_i & keep_j
        if not np.any(keep_mask):
            continue

        for point_i, point_j in zip(nearest_i[keep_mask], nearest_j[keep_mask]):
            node_i = (match.frame_i, int(point_i))
            node_j = (match.frame_j, int(point_j))
            make_node(node_i)
            make_node(node_j)
            union(node_i, node_j)
            node_votes[node_i] += 1
            node_votes[node_j] += 1

    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)

    block_points: dict[tuple[int, int], dict[str, Any]] = {}
    num_tracks = 0
    for members in components.values():
        if len(members) < int(min_track_length):
            continue

        best_per_frame: dict[int, tuple[int, int]] = {}
        for frame_index, point_index in members:
            current = best_per_frame.get(frame_index)
            node = (frame_index, point_index)
            if current is None or node_votes[node] > node_votes[current]:
                best_per_frame[frame_index] = node
        observations = list(best_per_frame.values())
        if len(observations) < int(min_track_length):
            continue

        observations.sort(key=lambda item: item[0])
        anchor = max(
            observations,
            key=lambda node: (
                node_votes[node],
                frame_data_by_index[node[0]].temporal_weight,
            ),
        )
        anchor_frame = frame_data_by_index[anchor[0]]
        anchor_point_index = anchor[1]

        contributed = False
        for target_frame_index, target_point_index in observations:
            if target_frame_index == anchor[0]:
                continue
            target_frame = frame_data_by_index[target_frame_index]
            key = (anchor[0], target_frame_index)
            block = block_points.setdefault(
                key,
                {
                    "source_points_3d": [],
                    "target_rgb_points": [],
                    "weight": [],
                    "source_c2w": np.asarray(rgb_camera_poses_c2w[anchor[0]], dtype=np.float64),
                    "target_w2c": np.linalg.inv(np.asarray(rgb_camera_poses_c2w[target_frame_index], dtype=np.float64)),
                },
            )
            block["source_points_3d"].append(anchor_frame.points_3d[anchor_point_index])
            block["target_rgb_points"].append(target_frame.rgb_points[target_point_index])
            block["weight"].append(np.sqrt(float(anchor_frame.temporal_weight * target_frame.temporal_weight)))
            contributed = True
        if contributed:
            num_tracks += 1

    temporal_residuals: list[TemporalResidualData] = []
    for (source_index, target_index), block in block_points.items():
        if len(block["source_points_3d"]) < int(min_block_points):
            continue
        temporal_residuals.append(
            TemporalResidualData(
                source_frame_index=source_index,
                target_frame_index=target_index,
                source_points_3d=np.asarray(block["source_points_3d"], dtype=np.float64),
                target_rgb_points=np.asarray(block["target_rgb_points"], dtype=np.float64),
                source_c2w=block["source_c2w"],
                target_w2c=block["target_w2c"],
                weight=float(np.mean(block["weight"])) if block["weight"] else 1.0,
            )
        )

    return temporal_residuals, num_tracks


def sample_depth_values_vectorized(
    depth_map: np.ndarray,
    points: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast O(N) vectorized depth sampling at rounded pixel coords (no search radius).

    Suitable when depth_map is dense (e.g. rendered 2DGS depth), where the
    point coords are expected to land on valid depth pixels.
    """
    h, w = depth_map.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)
    depths = depth_map[ys, xs].astype(np.float64)
    _min = float(min_depth) if min_depth is not None else -np.inf
    _max = float(max_depth) if max_depth is not None else np.inf
    valid = np.isfinite(depths) & (depths > 0) & (depths >= _min) & (depths <= _max)
    indices = np.where(valid)[0].astype(np.int32)
    return indices, depths[valid]


def sample_depth_values(
    depth_map: np.ndarray,
    points: np.ndarray,
    min_depth: float,
    max_depth: float,
    search_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_map.shape[:2]
    sampled_depths = []
    sampled_indices = []

    for index, point in enumerate(points):
        center_x = int(round(float(point[0])))
        center_y = int(round(float(point[1])))
        best_depth = None

        for radius in range(search_radius + 1):
            x0 = max(center_x - radius, 0)
            x1 = min(center_x + radius + 1, w)
            y0 = max(center_y - radius, 0)
            y1 = min(center_y + radius + 1, h)
            patch = depth_map[y0:y1, x0:x1]
            valid = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
            if not np.any(valid):
                continue
            patch_values = patch[valid]
            best_depth = float(np.median(patch_values))
            break

        if best_depth is None:
            continue

        sampled_indices.append(index)
        sampled_depths.append(best_depth)

    return np.asarray(sampled_indices, dtype=np.int32), np.asarray(sampled_depths, dtype=np.float64)


def undistort_points(points: np.ndarray, camera: CameraModel) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 2)
    return cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64),
        camera.K,
        camera.dist if camera.dist.size > 0 else None,
        P=camera.K,
    ).reshape(-1, 2)


def backproject_depth_points(points: np.ndarray, depth_values: np.ndarray, camera: CameraModel) -> np.ndarray:
    fx = float(camera.K[0, 0])
    fy = float(camera.K[1, 1])
    cx = float(camera.K[0, 2])
    cy = float(camera.K[1, 2])
    x = (points[:, 0] - cx) * depth_values / fx
    y = (points[:, 1] - cy) * depth_values / fy
    return np.stack((x, y, depth_values), axis=1).astype(np.float64)


def build_frame_correspondence(
    frame_name: str,
    rgb_path: str | Path,
    depth_path: str | Path,
    rgb_points: np.ndarray,
    depth_points: np.ndarray,
    depth_map: np.ndarray,
    depth_camera: CameraModel,
    min_depth: float,
    max_depth: float,
    search_radius: int,
) -> FrameCalibrationData | None:
    keep_indices, sampled_depths = sample_depth_values(
        depth_map=depth_map,
        points=depth_points,
        min_depth=min_depth,
        max_depth=max_depth,
        search_radius=search_radius,
    )
    if keep_indices.size == 0:
        return None

    rgb_points = np.asarray(rgb_points[keep_indices], dtype=np.float64)
    depth_points = np.asarray(depth_points[keep_indices], dtype=np.float64)
    depth_points_undistorted = undistort_points(depth_points, depth_camera)
    points_3d = backproject_depth_points(depth_points_undistorted, sampled_depths, depth_camera)
    return FrameCalibrationData(
        frame_name=frame_name,
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        rgb_points=rgb_points,
        depth_points=depth_points,
        points_3d=points_3d,
        point_weights=np.ones((points_3d.shape[0],), dtype=np.float64),
    )


def score_temporal_support(
    frame_data_list: list[FrameCalibrationData],
    temporal_matches: list[TemporalMatchData],
    rgb_camera_poses_c2w: list[np.ndarray],
    rgb_camera: CameraModel,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    match_radius_px: float = 4.0,
    projection_tolerance_px: float = 6.0,
) -> dict[int, np.ndarray]:
    frame_data_by_index = {frame_data.frame_index: frame_data for frame_data in frame_data_list}
    support_scores = {
        frame_data.frame_index: np.zeros((frame_data.points_3d.shape[0],), dtype=np.float64)
        for frame_data in frame_data_list
    }

    rotation_matrix, _ = cv2.Rodrigues(np.asarray(initial_rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(initial_tvec, dtype=np.float64).reshape(1, 3)

    def score_direction(
        source_index: int,
        target_index: int,
        source_match_points: np.ndarray,
        target_match_points: np.ndarray,
    ) -> None:
        source_frame = frame_data_by_index.get(source_index)
        if source_frame is None:
            return
        if source_index >= len(rgb_camera_poses_c2w) or target_index >= len(rgb_camera_poses_c2w):
            return
        nearest_indices, keep_mask = _associate_nearest_points(
            query_points=np.asarray(source_match_points, dtype=np.float64),
            reference_points=source_frame.rgb_points,
            max_distance_px=match_radius_px,
        )
        if not np.any(keep_mask):
            return

        point_indices = nearest_indices[keep_mask]
        source_points_3d = source_frame.points_3d[point_indices]
        source_rgb = (rotation_matrix @ source_points_3d.T).T + translation
        source_c2w = np.asarray(rgb_camera_poses_c2w[source_index], dtype=np.float64)
        target_w2c = np.linalg.inv(np.asarray(rgb_camera_poses_c2w[target_index], dtype=np.float64))
        world_points = (source_c2w[:3, :3] @ source_rgb.T).T + source_c2w[:3, 3]
        target_rgb = (target_w2c[:3, :3] @ world_points.T).T + target_w2c[:3, 3]
        valid_mask = target_rgb[:, 2] > 1.0e-6
        if not np.any(valid_mask):
            return

        projected, _ = cv2.projectPoints(
            target_rgb[valid_mask].astype(np.float64),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = np.linalg.norm(
            projected.reshape(-1, 2) - np.asarray(target_match_points[keep_mask], dtype=np.float64)[valid_mask],
            axis=1,
        )
        supported = residual <= float(projection_tolerance_px)
        if not np.any(supported):
            return
        valid_point_indices = point_indices[valid_mask][supported]
        support_scores[source_index][valid_point_indices] += 1.0

    for match in temporal_matches:
        score_direction(match.frame_i, match.frame_j, match.points_i, match.points_j)
        score_direction(match.frame_j, match.frame_i, match.points_j, match.points_i)

    return support_scores


def apply_temporal_support(
    frame_data_list: list[FrameCalibrationData],
    support_scores: dict[int, np.ndarray],
    mode: str = "weight",
    min_support: int = 1,
    support_scale: float = 1.0,
) -> list[FrameCalibrationData]:
    if mode == "none":
        return frame_data_list

    updated_frames: list[FrameCalibrationData] = []
    for frame_data in frame_data_list:
        scores = np.asarray(
            support_scores.get(frame_data.frame_index, np.zeros((frame_data.points_3d.shape[0],), dtype=np.float64)),
            dtype=np.float64,
        )
        if scores.shape[0] != frame_data.points_3d.shape[0]:
            scores = np.zeros((frame_data.points_3d.shape[0],), dtype=np.float64)

        if mode in {"filter", "filter_weight"}:
            keep_mask = scores >= float(min_support)
            if not np.any(keep_mask):
                keep_mask = np.ones((scores.shape[0],), dtype=bool)
            frame_data.points_3d = frame_data.points_3d[keep_mask]
            frame_data.rgb_points = frame_data.rgb_points[keep_mask]
            frame_data.depth_points = frame_data.depth_points[keep_mask]
            scores = scores[keep_mask]

        if mode in {"weight", "filter_weight"}:
            frame_data.point_weights = 1.0 + float(support_scale) * scores
        else:
            frame_data.point_weights = np.ones((frame_data.points_3d.shape[0],), dtype=np.float64)

        updated_frames.append(frame_data)

    return [frame for frame in updated_frames if frame.points_3d.shape[0] > 0]


def solve_frame_pnp(
    frame_data: FrameCalibrationData,
    rgb_camera: CameraModel,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if frame_data.points_3d.shape[0] < max(4, min_inliers):
        return None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=frame_data.points_3d.astype(np.float64),
        imagePoints=frame_data.rgb_points.astype(np.float64),
        cameraMatrix=rgb_camera.K,
        distCoeffs=rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(reproj_error),
        iterationsCount=int(iterations),
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None

    inlier_mask = inliers.reshape(-1)
    projected, _ = cv2.projectPoints(
        frame_data.points_3d[inlier_mask],
        rvec,
        tvec,
        rgb_camera.K,
        rgb_camera.dist if rgb_camera.dist.size > 0 else None,
    )
    residual = projected.reshape(-1, 2) - frame_data.rgb_points[inlier_mask]
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    frame_data.points_3d = frame_data.points_3d[inlier_mask]
    frame_data.rgb_points = frame_data.rgb_points[inlier_mask]
    frame_data.depth_points = frame_data.depth_points[inlier_mask]
    frame_data.pnp_inliers = int(len(inliers))
    frame_data.pnp_reproj_error = rmse
    return rvec.reshape(3), tvec.reshape(3), rmse


def initialize_shared_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if not frame_data_list:
        raise ValueError("At least one frame with valid correspondences is required for initialization.")

    counts = [int(frame_data.points_3d.shape[0]) for frame_data in frame_data_list]
    stacked_points_3d = np.concatenate([frame_data.points_3d for frame_data in frame_data_list], axis=0)
    stacked_rgb_points = np.concatenate([frame_data.rgb_points for frame_data in frame_data_list], axis=0)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=stacked_points_3d.astype(np.float64),
        imagePoints=stacked_rgb_points.astype(np.float64),
        cameraMatrix=rgb_camera.K,
        distCoeffs=rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(reproj_error),
        iterationsCount=int(iterations),
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        raise ValueError("Shared initialization PnP failed to find a valid global solution.")

    global_inlier_mask = np.zeros(stacked_points_3d.shape[0], dtype=bool)
    global_inlier_mask[inliers.reshape(-1)] = True

    filtered_frames: list[FrameCalibrationData] = []
    start = 0
    for frame_data, count in zip(frame_data_list, counts):
        frame_mask = global_inlier_mask[start : start + count]
        start += count
        inlier_count = int(np.count_nonzero(frame_mask))
        if inlier_count == 0:
            continue

        frame_points_3d = frame_data.points_3d[frame_mask]
        frame_rgb_points = frame_data.rgb_points[frame_mask]

        projected, _ = cv2.projectPoints(
            frame_points_3d.astype(np.float64),
            rvec,
            tvec,
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = projected.reshape(-1, 2) - frame_rgb_points
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

        frame_data.pnp_inliers = inlier_count
        frame_data.pnp_reproj_error = rmse
        filtered_frames.append(frame_data)

    if not filtered_frames:
        raise ValueError("Shared initialization rejected all frame correspondences.")

    frame_data_list[:] = filtered_frames
    return rvec.reshape(3), tvec.reshape(3)


def initialize_from_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reproj_error: float = 4.0,
    min_inliers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if not frame_data_list:
        raise ValueError("At least one frame with valid correspondences is required for initialization.")

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    filtered_frames: list[FrameCalibrationData] = []
    for frame_data in frame_data_list:
        projected, _ = cv2.projectPoints(
            frame_data.points_3d.astype(np.float64),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = projected.reshape(-1, 2) - frame_data.rgb_points
        reproj = np.sqrt(np.sum(residual**2, axis=1))
        inlier_mask = reproj <= float(reproj_error)
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < min_inliers:
            continue

        frame_data.points_3d = frame_data.points_3d[inlier_mask]
        frame_data.rgb_points = frame_data.rgb_points[inlier_mask]
        frame_data.depth_points = frame_data.depth_points[inlier_mask]
        if frame_data.point_weights is not None:
            frame_data.point_weights = np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)[inlier_mask]
        frame_data.pnp_inliers = inlier_count
        frame_data.pnp_reproj_error = float(np.sqrt(np.mean(np.sum(residual[inlier_mask] ** 2, axis=1))))
        filtered_frames.append(frame_data)

    if not filtered_frames:
        raise ValueError("Provided initialization pose rejected all frame correspondences.")

    frame_data_list[:] = filtered_frames
    return rvec, tvec


def evaluate_shared_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> CalibrationResult:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)

    reprojection_errors = []
    num_matches = 0
    for frame_data in frame_data_list:
        point_weights = (
            np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)
            if frame_data.point_weights is not None
            else np.ones((frame_data.points_3d.shape[0],), dtype=np.float64)
        )
        projected, _ = cv2.projectPoints(
            frame_data.points_3d.astype(np.float64),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = projected.reshape(-1, 2) - frame_data.rgb_points
        reprojection_errors.extend((np.sqrt(np.sum(residual**2, axis=1)) * np.sqrt(point_weights)).tolist())
        num_matches += int(frame_data.rgb_points.shape[0])

    reprojection_errors = np.asarray(reprojection_errors, dtype=np.float64)
    return CalibrationResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation=translation.astype(np.float64),
        rotation_vector=np.asarray(rvec, dtype=np.float64).reshape(3),
        mean_reprojection_error=float(np.mean(reprojection_errors)) if reprojection_errors.size else float("nan"),
        median_reprojection_error=float(np.median(reprojection_errors)) if reprojection_errors.size else float("nan"),
        frames_used=len(frame_data_list),
        matches_used=num_matches,
    )


def compare_calibrations(initial: CalibrationResult, optimized: CalibrationResult) -> CalibrationComparison:
    relative_rotation = optimized.rotation_matrix @ initial.rotation_matrix.T
    trace = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_delta_deg = float(np.degrees(np.arccos(trace)))
    translation_delta_m = float(np.linalg.norm(optimized.translation - initial.translation))
    return CalibrationComparison(
        initial=initial,
        optimized=optimized,
        rotation_delta_deg=rotation_delta_deg,
        translation_delta_m=translation_delta_m,
        mean_reprojection_improvement_px=float(initial.mean_reprojection_error - optimized.mean_reprojection_error),
        median_reprojection_improvement_px=float(initial.median_reprojection_error - optimized.median_reprojection_error),
    )


def _clone_frame_subset(
    frame_data: FrameCalibrationData,
    mask: np.ndarray,
) -> FrameCalibrationData | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != frame_data.points_3d.shape[0] or not np.any(mask):
        return None
    point_weights = None
    if frame_data.point_weights is not None:
        point_weights = np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)[mask]
    return FrameCalibrationData(
        frame_name=frame_data.frame_name,
        rgb_path=frame_data.rgb_path,
        depth_path=frame_data.depth_path,
        rgb_points=frame_data.rgb_points[mask].copy(),
        depth_points=frame_data.depth_points[mask].copy(),
        points_3d=frame_data.points_3d[mask].copy(),
        point_weights=None if point_weights is None else point_weights.copy(),
        frame_index=frame_data.frame_index,
        frame_id=frame_data.frame_id,
        temporal_weight=frame_data.temporal_weight,
        pnp_inliers=frame_data.pnp_inliers,
        pnp_reproj_error=frame_data.pnp_reproj_error,
    )


def subset_frame_data_by_depth(
    frame_data_list: list[FrameCalibrationData],
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> list[FrameCalibrationData]:
    subsets: list[FrameCalibrationData] = []
    for frame_data in frame_data_list:
        depths = frame_data.points_3d[:, 2]
        mask = np.ones((depths.shape[0],), dtype=bool)
        if min_depth is not None:
            mask &= depths >= float(min_depth)
        if max_depth is not None:
            mask &= depths <= float(max_depth)
        subset = _clone_frame_subset(frame_data, mask)
        if subset is not None:
            subsets.append(subset)
    return subsets


def optimize_shared_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    temporal_residuals: list[TemporalResidualData] | None = None,
    temporal_residual_weight: float = 0.0,
    temporal_regularization: float = 0.0,
    staged_refinement: bool = False,
    staged_depth_split: float = 20.0,
) -> CalibrationComparison:
    if not frame_data_list:
        raise ValueError("No frame correspondences available for optimization.")
    temporal_residuals = temporal_residuals or []

    def residual_function(
        params: np.ndarray,
        active_frame_data_list: list[FrameCalibrationData],
        optimize_rotation: bool = True,
        optimize_translation: bool = True,
        fixed_rvec: np.ndarray | None = None,
        fixed_tvec: np.ndarray | None = None,
        include_temporal_residuals: bool = True,
    ) -> np.ndarray:
        current_rvec = params[:3] if optimize_rotation else np.asarray(fixed_rvec, dtype=np.float64).reshape(3)
        current_tvec = params[3:6] if optimize_translation else np.asarray(fixed_tvec, dtype=np.float64).reshape(3)
        rvec = current_rvec.reshape(3, 1)
        tvec = current_tvec.reshape(3, 1)
        residuals: list[np.ndarray] = []

        for frame_data in active_frame_data_list:
            point_weights = (
                np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)
                if frame_data.point_weights is not None
                else np.ones((frame_data.points_3d.shape[0],), dtype=np.float64)
            )
            projected, _ = cv2.projectPoints(
                frame_data.points_3d.astype(np.float64),
                rvec,
                tvec,
                rgb_camera.K,
                rgb_camera.dist if rgb_camera.dist.size > 0 else None,
            )
            projected = projected.reshape(-1, 2)
            weight = np.sqrt(float(frame_data.temporal_weight)) * np.sqrt(point_weights)[:, None]
            residuals.append((weight * (projected - frame_data.rgb_points)).reshape(-1))

        if include_temporal_residuals and temporal_residual_weight > 0.0 and temporal_residuals:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            translation = current_tvec.reshape(1, 3)
            for temporal_residual in temporal_residuals:
                source_rgb = (rotation_matrix @ temporal_residual.source_points_3d.T).T + translation
                world_points = (
                    temporal_residual.source_c2w[:3, :3] @ source_rgb.T
                ).T + temporal_residual.source_c2w[:3, 3]
                target_rgb = (
                    temporal_residual.target_w2c[:3, :3] @ world_points.T
                ).T + temporal_residual.target_w2c[:3, 3]
                valid_mask = target_rgb[:, 2] > 1.0e-6
                if not np.any(valid_mask):
                    continue

                projected, _ = cv2.projectPoints(
                    target_rgb[valid_mask].astype(np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    rgb_camera.K,
                    rgb_camera.dist if rgb_camera.dist.size > 0 else None,
                )
                weight = np.sqrt(float(temporal_residual_weight) * float(temporal_residual.weight))
                residuals.append(
                    weight
                    * (projected.reshape(-1, 2) - temporal_residual.target_rgb_points[valid_mask]).reshape(-1)
                )

        if temporal_regularization > 0.0:
            prior_weights = [
                float(frame_data.temporal_weight)
                for frame_data in active_frame_data_list
                if np.isfinite(frame_data.pnp_reproj_error)
            ]
            prior_scale = float(temporal_regularization) * np.sqrt(np.sum(prior_weights)) if prior_weights else float(temporal_regularization)
            residuals.append(prior_scale * (current_rvec - initial_rvec.reshape(3)).reshape(-1))
            residuals.append(prior_scale * (current_tvec - initial_tvec.reshape(3)).reshape(-1))

        return np.concatenate(residuals, axis=0)

    initial_params = np.concatenate([initial_rvec.reshape(3), initial_tvec.reshape(3)], axis=0)
    if staged_refinement:
        far_frame_data_list = subset_frame_data_by_depth(frame_data_list, min_depth=staged_depth_split)
        near_frame_data_list = subset_frame_data_by_depth(frame_data_list, max_depth=staged_depth_split)

        stage1_rvec = initial_rvec.reshape(3)
        stage1_tvec = initial_tvec.reshape(3)
        if far_frame_data_list:
            stage1 = least_squares(
                lambda rot_params: residual_function(
                    np.concatenate([rot_params.reshape(3), stage1_tvec], axis=0),
                    active_frame_data_list=far_frame_data_list,
                    optimize_rotation=True,
                    optimize_translation=False,
                    fixed_tvec=stage1_tvec,
                    include_temporal_residuals=True,
                ),
                stage1_rvec,
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=100,
            )
            stage1_rvec = stage1.x.reshape(3)

        stage2_tvec = stage1_tvec.copy()
        stage2_frame_data_list = near_frame_data_list or frame_data_list
        stage2 = least_squares(
            lambda trans_params: residual_function(
                np.concatenate([stage1_rvec, trans_params.reshape(3)], axis=0),
                active_frame_data_list=stage2_frame_data_list,
                optimize_rotation=False,
                optimize_translation=True,
                fixed_rvec=stage1_rvec,
                include_temporal_residuals=False,
            ),
            stage2_tvec,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=100,
        )
        stage2_tvec = stage2.x.reshape(3)
        staged_initial_params = np.concatenate([stage1_rvec, stage2_tvec], axis=0)
        result = least_squares(
            lambda params: residual_function(
                params,
                active_frame_data_list=frame_data_list,
                optimize_rotation=True,
                optimize_translation=True,
                include_temporal_residuals=True,
            ),
            staged_initial_params,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
        )
    else:
        result = least_squares(
            lambda params: residual_function(
                params,
                active_frame_data_list=frame_data_list,
                optimize_rotation=True,
                optimize_translation=True,
                include_temporal_residuals=True,
            ),
            initial_params,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
        )
    initial_calibration = evaluate_shared_extrinsic(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=initial_rvec,
        tvec=initial_tvec,
    )
    optimized_calibration = evaluate_shared_extrinsic(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=result.x[:3],
        tvec=result.x[3:6],
    )
    return compare_calibrations(initial_calibration, optimized_calibration)


def rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def project_depth_to_rgb(
    depth_map: np.ndarray,
    depth_camera: CameraModel,
    rgb_camera: CameraModel,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
    rgb_shape: tuple[int, int],
) -> np.ndarray:
    height, width = depth_map.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    depth_pixels = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1).astype(np.float64)
    depth_values = depth_map.reshape(-1)
    valid = np.isfinite(depth_values) & (depth_values > 0.0)
    if not np.any(valid):
        return np.zeros(rgb_shape, dtype=np.float32)

    depth_pixels = depth_pixels[valid]
    depth_values = depth_values[valid].astype(np.float64)
    undistorted = undistort_points(depth_pixels, depth_camera)
    points_3d = backproject_depth_points(undistorted, depth_values, depth_camera)
    transformed = (rotation_matrix @ points_3d.T).T + translation.reshape(1, 3)
    positive = transformed[:, 2] > 0.0
    transformed = transformed[positive]
    depth_values = transformed[:, 2]

    projected, _ = cv2.projectPoints(
        transformed.astype(np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        rgb_camera.K,
        rgb_camera.dist if rgb_camera.dist.size > 0 else None,
    )
    projected = projected.reshape(-1, 2)
    rgb_h, rgb_w = rgb_shape
    projected_u = np.round(projected[:, 0]).astype(np.int32)
    projected_v = np.round(projected[:, 1]).astype(np.int32)
    inside = (
        (projected_u >= 0)
        & (projected_u < rgb_w)
        & (projected_v >= 0)
        & (projected_v < rgb_h)
    )
    projected_u = projected_u[inside]
    projected_v = projected_v[inside]
    depth_values = depth_values[inside].astype(np.float32)

    warped_depth = np.zeros((rgb_h, rgb_w), dtype=np.float32)
    for u, v, depth_value in zip(projected_u, projected_v, depth_values):
        current = warped_depth[v, u]
        if current == 0.0 or depth_value < current:
            warped_depth[v, u] = depth_value
    return warped_depth


def make_depth_overlay(rgb_image: np.ndarray, warped_depth: np.ndarray) -> np.ndarray:
    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    depth_viz = depth_to_match_image(warped_depth, use_inverse=True)
    valid = warped_depth > 0.0
    overlay = rgb_image.copy()
    overlay[valid] = (
        0.55 * overlay[valid].astype(np.float32) + 0.45 * depth_viz[valid].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def save_extrinsic_yaml(
    output_path: str | Path,
    comparison: CalibrationComparison,
    rgb_camera: CameraModel,
    depth_camera: CameraModel,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    extra_metrics = extra_metrics or {}
    calibration = comparison.optimized
    initial = comparison.initial
    payload = {
        "T_rgb_d": {
            "rotation_matrix": calibration.rotation_matrix.tolist(),
            "rotation_vector": calibration.rotation_vector.tolist(),
            "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(calibration.rotation_matrix).tolist(),
            "translation_xyz": calibration.translation.tolist(),
        },
        "optimization_comparison": {
            "initial_T_rgb_d": {
                "rotation_matrix": initial.rotation_matrix.tolist(),
                "rotation_vector": initial.rotation_vector.tolist(),
                "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(initial.rotation_matrix).tolist(),
                "translation_xyz": initial.translation.tolist(),
            },
            "optimized_T_rgb_d": {
                "rotation_matrix": calibration.rotation_matrix.tolist(),
                "rotation_vector": calibration.rotation_vector.tolist(),
                "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(calibration.rotation_matrix).tolist(),
                "translation_xyz": calibration.translation.tolist(),
            },
            "rotation_change_deg": comparison.rotation_delta_deg,
            "translation_change_m": comparison.translation_delta_m,
            "mean_reprojection_improvement_px": comparison.mean_reprojection_improvement_px,
            "median_reprojection_improvement_px": comparison.median_reprojection_improvement_px,
            "initial_mean_reprojection_error_px": initial.mean_reprojection_error,
            "initial_median_reprojection_error_px": initial.median_reprojection_error,
            "optimized_mean_reprojection_error_px": calibration.mean_reprojection_error,
            "optimized_median_reprojection_error_px": calibration.median_reprojection_error,
        },
        "rgb_camera": {
            "K": rgb_camera.K.tolist(),
            "dist": rgb_camera.dist.tolist(),
        },
        "depth_camera": {
            "K": depth_camera.K.tolist(),
            "dist": depth_camera.dist.tolist(),
        },
        "metrics": {
            "mean_reprojection_error_px": calibration.mean_reprojection_error,
            "median_reprojection_error_px": calibration.median_reprojection_error,
            "frames_used": calibration.frames_used,
            "matches_used": calibration.matches_used,
            **extra_metrics,
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
