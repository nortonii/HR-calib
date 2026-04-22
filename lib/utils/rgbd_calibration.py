from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
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
    point_depths: np.ndarray | None = None
    point_weights: np.ndarray | None = None
    support_scores: np.ndarray | None = None
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
    point_weights: np.ndarray | None = None


@dataclass
class TemporalPhotometricResidualData:
    source_frame_id: int
    target_frame_id: int
    source_points_3d: np.ndarray
    source_colors: np.ndarray
    source_c2w: np.ndarray
    target_w2c: np.ndarray
    weight: float = 1.0
    point_weights: np.ndarray | None = None


@dataclass
class TemporalDepthResidualData:
    source_frame_id: int
    target_frame_id: int
    source_points_3d: np.ndarray
    source_c2w: np.ndarray
    target_w2c: np.ndarray
    weight: float = 1.0
    point_weights: np.ndarray | None = None


def _stack_frame_correspondences(
    frame_data_list: list[FrameCalibrationData],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([frame_data.points_3d for frame_data in frame_data_list], axis=0),
        np.concatenate([frame_data.rgb_points for frame_data in frame_data_list], axis=0),
    )


def _has_nontrivial_point_weights(frame_data_list: list[FrameCalibrationData]) -> bool:
    for frame_data in frame_data_list:
        if frame_data.point_weights is None:
            continue
        weights = np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)
        if weights.size > 0 and not np.allclose(weights, 1.0):
            return True
    return False


def _opencv_shared_pnp_unsupported_reason(
    frame_data_list: list[FrameCalibrationData],
    *,
    temporal_residuals: list[TemporalResidualData],
    temporal_residual_weight: float,
    photometric_residuals: list[TemporalPhotometricResidualData],
    photometric_residual_weight: float,
    depth_residuals: list[TemporalDepthResidualData],
    depth_residual_weight: float,
    temporal_regularization: float,
    staged_refinement: bool,
    optimize_rotation: bool,
    optimize_translation: bool,
    gt_pose_residual_weight: float,
) -> str | None:
    if not optimize_rotation or not optimize_translation:
        return "OpenCV shared PnP refine does not support freezing only rotation or translation."
    if staged_refinement:
        return "OpenCV shared PnP refine does not support staged refinement."
    if temporal_regularization > 0.0:
        return "OpenCV shared PnP refine does not support temporal regularization terms."
    if temporal_residual_weight > 0.0 and temporal_residuals:
        return "OpenCV shared PnP refine does not support temporal geometric residuals."
    if photometric_residual_weight > 0.0 and photometric_residuals:
        return "OpenCV shared PnP refine does not support photometric residuals."
    if depth_residual_weight > 0.0 and depth_residuals:
        return "OpenCV shared PnP refine does not support temporal depth residuals."
    if gt_pose_residual_weight > 0.0:
        return "OpenCV shared PnP refine does not support GT pose residual terms."
    if _has_nontrivial_point_weights(frame_data_list):
        return "OpenCV shared PnP refine does not support non-uniform per-point weights."
    return None


def _refine_shared_extrinsic_opencv(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stacked_points_3d, stacked_rgb_points = _stack_frame_correspondences(frame_data_list)
    refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
        objectPoints=stacked_points_3d.astype(np.float64),
        imagePoints=stacked_rgb_points.astype(np.float64),
        cameraMatrix=rgb_camera.K,
        distCoeffs=rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        rvec=np.asarray(initial_rvec, dtype=np.float64).reshape(3, 1).copy(),
        tvec=np.asarray(initial_tvec, dtype=np.float64).reshape(3, 1).copy(),
    )
    return (
        np.asarray(refined_rvec, dtype=np.float64).reshape(3),
        np.asarray(refined_tvec, dtype=np.float64).reshape(3),
    )


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
    percentile_low: float = 2.0,
    percentile_high: float = 85.0,
    use_inverse: bool = True,
) -> np.ndarray:
    """Convert a metric depth map to a 3-channel grayscale image for MatchAnything-RoMa.

    Follows the official MatchAnything training convention (read_megadepth_depth_gray,
    read_gray=False): normalize by percentile, optionally invert (closer=brighter,
    following ControlNet convention), then stack the same grayscale channel 3 times.
    No colormap is applied — the model was trained on grayscale×3 depth images.
    """
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
        normalized = 1.0 - normalized   # closer = brighter (ControlNet convention)
    gray = (normalized * 255.0).astype(np.uint8)
    gray[~valid] = 0
    # Stack grayscale 3× → HWC RGB (matches MatchAnything training distribution)
    return np.stack([gray, gray, gray], axis=-1)


def build_matcher(
    matcher_name: str = "matchanything-roma",
    device: str = "cuda",
    max_num_keypoints: int = 2048,
    ransac_reproj_thresh: float = 3.0,
    img_resize: int | None = 832,
    match_threshold: float = 0.2,
    hf_endpoint: str = "https://hf-mirror.com",
    minima_root: str | None = None,
    minima_ckpt: str | None = None,
):
    matcher_name = str(matcher_name).lower()
    if matcher_name == "matchanything-roma":
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
    if matcher_name == "minima-roma":
        return _build_minima_roma_matcher(
            device=device,
            img_resize=img_resize,
            minima_root=minima_root,
            minima_ckpt=minima_ckpt,
        )
    raise ValueError(f"Unsupported matcher_name={matcher_name!r}")


def _resolve_minima_root(minima_root: str | None = None) -> Path:
    candidates = []
    if minima_root:
        candidates.append(Path(minima_root).expanduser())
    candidates.append(Path(__file__).resolve().parents[2] / "submodules" / "MINIMA")
    env_root = os.environ.get("MINIMA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "MINIMA repo not found. Set MINIMA_ROOT or place it at submodules/MINIMA."
    )


def _resolve_minima_ckpt(minima_root: Path, minima_ckpt: str | None = None) -> Path:
    candidates = []
    if minima_ckpt:
        candidates.append(Path(minima_ckpt).expanduser())
    env_ckpt = os.environ.get("MINIMA_ROMA_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())
    candidates.append(minima_root / "weights" / "minima_roma.pth")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "MINIMA RoMa checkpoint not found. Set MINIMA_ROMA_CKPT or place minima_roma.pth in submodules/MINIMA/weights."
    )


class MinimaRoMaMatcher:
    def __init__(
        self,
        matcher,
        wrapped_matcher,
        roma_model,
        device: str,
    ) -> None:
        self.matcher = matcher
        self.device = device
        self.wrapped_matcher = wrapped_matcher
        self.roma_model = roma_model
        self.torch_device = torch.device(device)

    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"MINIMA matcher expects 3-channel image, got shape {arr.shape}")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:
            raise ValueError(f"MINIMA matcher expects HWC/CHW RGB image, got shape {arr.shape}")
        arr = np.ascontiguousarray(arr.astype(np.float32))
        if arr.max(initial=0.0) <= 1.2 and arr.min(initial=0.0) >= -0.2:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> dict:
        result = self.matcher(
            self._to_bgr_uint8(image0),
            self._to_bgr_uint8(image1),
        )
        mkpts0 = np.asarray(result.get("mkpts0", np.zeros((0, 2), dtype=np.float32)), dtype=np.float64).reshape(-1, 2)
        mkpts1 = np.asarray(result.get("mkpts1", np.zeros((0, 2), dtype=np.float32)), dtype=np.float64).reshape(-1, 2)
        mconf = np.asarray(result.get("mconf", np.zeros((mkpts0.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        return {
            "matched_kpts0": mkpts0,
            "matched_kpts1": mkpts1,
            "inlier_kpts0": mkpts0,
            "inlier_kpts1": mkpts1,
            "matched_confidence": mconf,
            "num_inliers": int(mkpts0.shape[0]),
            "backend": "minima-roma",
            "raw_result": result,
        }

    def dense_match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        query_stride: int = 4,
        cert_threshold: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch.nn.functional as F
        from torchvision.transforms import ToPILImage

        img0_bgr = self._to_bgr_uint8(image0)
        img1_bgr = self._to_bgr_uint8(image1)
        orig_h0, orig_w0 = img0_bgr.shape[:2]
        orig_h1, orig_w1 = img1_bgr.shape[:2]

        img0_tensor, scale0, _, _, _ = self.wrapped_matcher.preprocess_image(
            img0_bgr,
            self.wrapped_matcher.device,
            resize=self.wrapped_matcher.img0_size,
            df=self.wrapped_matcher.df,
            padding=self.wrapped_matcher.padding,
        )
        img1_tensor, scale1, _, _, _ = self.wrapped_matcher.preprocess_image(
            img1_bgr,
            self.wrapped_matcher.device,
            resize=self.wrapped_matcher.img1_size,
            df=self.wrapped_matcher.df,
            padding=self.wrapped_matcher.padding,
        )

        img0_tensor = img0_tensor.squeeze(0).to(self.torch_device)
        img1_tensor = img1_tensor.squeeze(0).to(self.torch_device)
        H_A, W_A = int(img0_tensor.size(1)), int(img0_tensor.size(2))
        H_B, W_B = int(img1_tensor.size(1)), int(img1_tensor.size(2))

        to_pil = ToPILImage()
        pil0 = to_pil(img0_tensor.cpu())
        pil1 = to_pil(img1_tensor.cpu())

        with torch.no_grad():
            warp, certainty = self.roma_model.match(
                pil0,
                pil1,
                batched=False,
                device=self.torch_device,
            )

            ys = torch.arange(0, H_A, query_stride, device=self.torch_device, dtype=torch.float32)
            xs = torch.arange(0, W_A, query_stride, device=self.torch_device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            query_pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            query_norm = torch.stack(
                [
                    2.0 / float(W_A) * query_pts[:, 0] - 1.0,
                    2.0 / float(H_A) * query_pts[:, 1] - 1.0,
                ],
                dim=-1,
            )

            warped_norm = F.grid_sample(
                warp[..., -2:].permute(2, 0, 1)[None],
                query_norm[None, None],
                align_corners=False,
                mode="bilinear",
            )[0, :, 0].mT
            certs = F.grid_sample(
                certainty[None, None, ...],
                query_norm[None, None],
                align_corners=False,
                mode="bilinear",
            )[0, 0, 0]
            warped_pts = self.roma_model.to_pixel_coordinates(warped_norm, H_B, W_B)

        query_np = query_pts.cpu().numpy().astype(np.float64) * np.asarray(scale0, dtype=np.float64)
        warped_np = warped_pts.cpu().numpy().astype(np.float64) * np.asarray(scale1, dtype=np.float64)
        cert_np = certs.cpu().numpy().astype(np.float32)

        valid = (
            (cert_np >= cert_threshold)
            & (query_np[:, 0] >= 0.0) & (query_np[:, 0] <= orig_w0 - 1)
            & (query_np[:, 1] >= 0.0) & (query_np[:, 1] <= orig_h0 - 1)
            & (warped_np[:, 0] >= 0.0) & (warped_np[:, 0] <= orig_w1 - 1)
            & (warped_np[:, 1] >= 0.0) & (warped_np[:, 1] <= orig_h1 - 1)
        )
        return query_np[valid], warped_np[valid], cert_np[valid]


def _build_minima_roma_matcher(
    device: str = "cuda",
    img_resize: int | None = 640,
    minima_root: str | None = None,
    minima_ckpt: str | None = None,
):
    minima_root_path = _resolve_minima_root(minima_root)
    minima_ckpt_path = _resolve_minima_ckpt(minima_root_path, minima_ckpt)
    roma_root = minima_root_path / "third_party" / "RoMa_minima"
    if not roma_root.exists():
        raise FileNotFoundError(f"MINIMA RoMa submodule not found: {roma_root}")

    sys.path.insert(0, str(minima_root_path))
    sys.path.insert(0, str(roma_root))
    try:
        from src.config.default import get_cfg_defaults
        from src.utils.data_io_roma import DataIOWrapper, lower_config
        from romatch import roma_outdoor
    except Exception as exc:
        raise ImportError(
            f"Failed to import MINIMA RoMa components from {minima_root_path}: {exc}"
        ) from exc

    config = lower_config(get_cfg_defaults(inference=True))["test"]
    resize = 640 if img_resize is None else int(img_resize)
    config["img0_resize"] = resize
    config["img1_resize"] = resize
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    state_dict = torch.load(str(minima_ckpt_path), map_location=torch_device)
    matcher = roma_outdoor(device=torch_device, weights=state_dict)
    wrapped = DataIOWrapper(matcher, config=config)
    wrapped.device = torch_device
    wrapped.model = wrapped.model.eval().to(torch_device)
    return MinimaRoMaMatcher(
        wrapped.from_cv_imgs,
        wrapped,
        matcher,
        str(torch_device),
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
        depth_image: Depth grayscale×3 image (same format), produced by depth_to_match_image.
        query_stride: Sample every N pixels along both axes of the RGB image.
                      stride=1 is fully dense; stride=4 gives 16x less points.
        cert_threshold: Minimum RoMa certainty score to keep a correspondence.

    Returns:
        rgb_points:   [N, 2] float64, (x, y) pixel coords in the RGB image.
        depth_points: [N, 2] float64, (x, y) pixel coords in the depth image.
        certainties:  [N]   float32,  per-point certainty from the warp field.
    """
    if hasattr(matcher, "dense_match"):
        return matcher.dense_match(
            rgb_image,
            depth_image,
            query_stride=query_stride,
            cert_threshold=cert_threshold,
        )
    if not hasattr(matcher, "net"):
        raise ValueError("Dense cross-modal matching is only available for MatchAnything/RoMa-style matchers.")
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


def _build_far_depth_boost_weights(
    depth_values: np.ndarray,
    boost_scale: float,
    start_percentile: float = 60.0,
    end_percentile: float = 95.0,
) -> np.ndarray:
    depth_values = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    weights = np.ones((depth_values.shape[0],), dtype=np.float64)
    if depth_values.size == 0 or float(boost_scale) <= 0.0:
        return weights

    valid = np.isfinite(depth_values) & (depth_values > 1.0e-6)
    if not np.any(valid):
        return weights

    valid_depths = depth_values[valid]
    lo = float(np.percentile(valid_depths, start_percentile))
    hi = float(np.percentile(valid_depths, max(float(end_percentile), float(start_percentile) + 1.0)))
    if hi <= lo + 1.0e-6:
        return weights

    alpha = np.clip((depth_values - lo) / (hi - lo), 0.0, 1.0)
    weights[valid] = 1.0 + float(boost_scale) * alpha[valid]
    return weights


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
    far_depth_boost: float = 0.0,
    far_depth_start_percentile: float = 60.0,
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
    point_weights = _build_far_depth_boost_weights(
        sampled_depths,
        boost_scale=far_depth_boost,
        start_percentile=far_depth_start_percentile,
    )
    return FrameCalibrationData(
        frame_name=frame_name,
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        rgb_points=rgb_points,
        depth_points=depth_points,
        points_3d=points_3d,
        point_depths=np.asarray(sampled_depths, dtype=np.float64),
        point_weights=point_weights,
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
            if frame_data.point_depths is not None:
                frame_data.point_depths = np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)[keep_mask]
            scores = scores[keep_mask]

        frame_data.support_scores = scores.copy()
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
    if frame_data.point_depths is not None:
        frame_data.point_depths = np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)[inlier_mask]
    if frame_data.point_weights is not None:
        frame_data.point_weights = np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)[inlier_mask]
    if frame_data.support_scores is not None:
        frame_data.support_scores = np.asarray(frame_data.support_scores, dtype=np.float64).reshape(-1)[inlier_mask]
    frame_data.pnp_inliers = int(len(inliers))
    frame_data.pnp_reproj_error = rmse
    return rvec.reshape(3), tvec.reshape(3), rmse


def _rotation_delta_deg_from_rvecs(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rmat_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    rmat_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    relative = rmat_a @ rmat_b.T
    trace = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _relative_rotation_residual_rvec(current_rvec: np.ndarray, target_rvec: np.ndarray) -> np.ndarray:
    current_rot = Rotation.from_rotvec(np.asarray(current_rvec, dtype=np.float64).reshape(3))
    target_rot = Rotation.from_rotvec(np.asarray(target_rvec, dtype=np.float64).reshape(3))
    relative_rot = current_rot * target_rot.inv()
    return relative_rot.as_rotvec().astype(np.float64).reshape(3)


def filter_frame_data_by_pose_disagreement(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    shared_rvec: np.ndarray,
    shared_tvec: np.ndarray,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
    mad_scale: float = 2.5,
    min_keep_ratio: float = 0.7,
    min_keep_frames: int = 12,
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
) -> tuple[list[FrameCalibrationData], dict[str, Any]]:
    frame_infos: list[dict[str, Any]] = []
    valid_rot_deltas: list[float] = []
    valid_trans_deltas: list[float] = []

    for frame_data in frame_data_list:
        candidate = _clone_frame_subset(
            frame_data,
            np.ones((frame_data.points_3d.shape[0],), dtype=bool),
        )
        if candidate is None:
            frame_infos.append(
                {
                    "frame_index": int(frame_data.frame_index),
                    "frame_id": int(frame_data.frame_id),
                    "frame_name": str(frame_data.frame_name),
                    "total_matches": int(frame_data.points_3d.shape[0]),
                    "status": "empty",
                }
            )
            continue
        solved = solve_frame_pnp(
            candidate,
            rgb_camera=rgb_camera,
            reproj_error=reproj_error,
            iterations=iterations,
            min_inliers=min_inliers,
        )
        if solved is None:
            frame_infos.append(
                {
                    "frame_index": int(frame_data.frame_index),
                    "frame_id": int(frame_data.frame_id),
                    "frame_name": str(frame_data.frame_name),
                    "total_matches": int(frame_data.points_3d.shape[0]),
                    "status": "failed",
                }
            )
            continue
        frame_rvec, frame_tvec, frame_rmse = solved
        rot_delta = _rotation_delta_deg_from_rvecs(frame_rvec, shared_rvec)
        trans_delta = float(
            np.linalg.norm(
                np.asarray(frame_tvec, dtype=np.float64).reshape(3)
                - np.asarray(shared_tvec, dtype=np.float64).reshape(3)
            )
        )
        info = {
            "frame_index": int(frame_data.frame_index),
            "frame_id": int(frame_data.frame_id),
            "frame_name": str(frame_data.frame_name),
            "total_matches": int(frame_data.points_3d.shape[0]),
            "single_frame_inliers": int(candidate.pnp_inliers),
            "single_frame_reproj_rmse": float(frame_rmse),
            "rotation_delta_deg": float(rot_delta),
            "translation_delta_m": float(trans_delta),
            "status": "ok",
        }
        if gt_rvec is not None and gt_tvec is not None:
            info["gt_rotation_delta_deg"] = _rotation_delta_deg_from_rvecs(frame_rvec, gt_rvec)
            info["gt_translation_delta_m"] = float(
                np.linalg.norm(
                    np.asarray(frame_tvec, dtype=np.float64).reshape(3)
                    - np.asarray(gt_tvec, dtype=np.float64).reshape(3)
                )
            )
        frame_infos.append(info)
        valid_rot_deltas.append(rot_delta)
        valid_trans_deltas.append(trans_delta)

    if valid_rot_deltas:
        rot_values = np.asarray(valid_rot_deltas, dtype=np.float64)
        rot_median = float(np.median(rot_values))
        rot_sigma = float(1.4826 * np.median(np.abs(rot_values - rot_median)))
        rot_threshold = rot_median + float(mad_scale) * max(rot_sigma, 0.05)
    else:
        rot_threshold = float("inf")
    if valid_trans_deltas:
        trans_values = np.asarray(valid_trans_deltas, dtype=np.float64)
        trans_median = float(np.median(trans_values))
        trans_sigma = float(1.4826 * np.median(np.abs(trans_values - trans_median)))
        trans_threshold = trans_median + float(mad_scale) * max(trans_sigma, 0.005)
    else:
        trans_threshold = float("inf")

    keep_indices: list[int] = []
    score_rows: list[tuple[float, int]] = []
    for idx, info in enumerate(frame_infos):
        if info.get("status") != "ok":
            continue
        rot_delta = float(info["rotation_delta_deg"])
        trans_delta = float(info["translation_delta_m"])
        rot_norm = rot_delta / max(rot_threshold, 1.0e-6)
        trans_norm = trans_delta / max(trans_threshold, 1.0e-6)
        score = max(rot_norm, trans_norm)
        info["rotation_threshold_deg"] = float(rot_threshold)
        info["translation_threshold_m"] = float(trans_threshold)
        info["consensus_score"] = float(score)
        info["kept"] = bool(rot_delta <= rot_threshold and trans_delta <= trans_threshold)
        score_rows.append((score, idx))
        if info["kept"]:
            keep_indices.append(idx)

    valid_ok = [idx for idx, info in enumerate(frame_infos) if info.get("status") == "ok"]
    min_keep = min(
        len(valid_ok),
        max(
            int(np.ceil(float(min_keep_ratio) * len(valid_ok))),
            int(min_keep_frames),
        ),
    )
    if len(keep_indices) < min_keep:
        score_rows.sort(key=lambda item: item[0])
        keep_indices = [idx for _, idx in score_rows[:min_keep]]
        keep_set = set(keep_indices)
        for idx in valid_ok:
            frame_infos[idx]["kept"] = idx in keep_set

    kept_frames = [frame_data_list[idx] for idx in keep_indices]
    dropped_infos = [
        info for info in frame_infos
        if info.get("status") == "ok" and not bool(info.get("kept", False))
    ]
    kept_infos = [
        info for info in frame_infos
        if info.get("status") == "ok" and bool(info.get("kept", False))
    ]

    diagnostics = {
        "mad_scale": float(mad_scale),
        "rotation_threshold_deg": None if not np.isfinite(rot_threshold) else float(rot_threshold),
        "translation_threshold_m": None if not np.isfinite(trans_threshold) else float(trans_threshold),
        "valid_frames": int(len(valid_ok)),
        "kept_frames": int(len(kept_frames)),
        "dropped_frames": int(len(dropped_infos)),
        "kept_frame_ids": [int(info["frame_id"]) for info in kept_infos],
        "dropped_frame_ids": [int(info["frame_id"]) for info in dropped_infos],
        "kept_rotation_delta_deg": _summarize_array([info["rotation_delta_deg"] for info in kept_infos]),
        "dropped_rotation_delta_deg": _summarize_array([info["rotation_delta_deg"] for info in dropped_infos]),
        "kept_translation_delta_m": _summarize_array([info["translation_delta_m"] for info in kept_infos]),
        "dropped_translation_delta_m": _summarize_array([info["translation_delta_m"] for info in dropped_infos]),
        "kept_gt_rotation_delta_deg": _summarize_array([info.get("gt_rotation_delta_deg", np.nan) for info in kept_infos]),
        "dropped_gt_rotation_delta_deg": _summarize_array([info.get("gt_rotation_delta_deg", np.nan) for info in dropped_infos]),
        "kept_gt_translation_delta_m": _summarize_array([info.get("gt_translation_delta_m", np.nan) for info in kept_infos]),
        "dropped_gt_translation_delta_m": _summarize_array([info.get("gt_translation_delta_m", np.nan) for info in dropped_infos]),
        "frames": frame_infos,
    }
    return kept_frames, diagnostics


def filter_frame_data_by_single_frame_pnp_stability(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    pnp_min_inliers: int = 8,
    keep_min_inliers: int = 0,
    keep_min_inlier_ratio: float = 0.0,
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
) -> tuple[list[FrameCalibrationData], dict[str, Any]]:
    filtered_frames: list[FrameCalibrationData] = []
    frame_infos: list[dict[str, Any]] = []
    keep_min_inliers = int(max(0, keep_min_inliers))
    keep_min_inlier_ratio = float(max(0.0, keep_min_inlier_ratio))

    for frame_data in frame_data_list:
        total_matches = int(frame_data.points_3d.shape[0])
        candidate = _clone_frame_subset(
            frame_data,
            np.ones((total_matches,), dtype=bool),
        )
        if candidate is None:
            frame_infos.append(
                {
                    "frame_index": int(frame_data.frame_index),
                    "frame_id": int(frame_data.frame_id),
                    "frame_name": str(frame_data.frame_name),
                    "total_matches": total_matches,
                    "status": "empty",
                    "kept": False,
                }
            )
            continue
        solved = solve_frame_pnp(
            candidate,
            rgb_camera=rgb_camera,
            reproj_error=reproj_error,
            iterations=iterations,
            min_inliers=pnp_min_inliers,
        )
        if solved is None:
            frame_infos.append(
                {
                    "frame_index": int(frame_data.frame_index),
                    "frame_id": int(frame_data.frame_id),
                    "frame_name": str(frame_data.frame_name),
                    "total_matches": total_matches,
                    "single_frame_inliers": 0,
                    "single_frame_inlier_ratio": 0.0,
                    "single_frame_reproj_rmse": float("nan"),
                    "status": "failed",
                    "kept": False,
                }
            )
            continue
        frame_rvec, frame_tvec, frame_rmse = solved
        inliers = int(candidate.pnp_inliers)
        inlier_ratio = float(inliers / max(total_matches, 1))
        kept = (
            inliers >= max(keep_min_inliers, pnp_min_inliers)
            and inlier_ratio >= keep_min_inlier_ratio
        )
        info = {
            "frame_index": int(frame_data.frame_index),
            "frame_id": int(frame_data.frame_id),
            "frame_name": str(frame_data.frame_name),
            "total_matches": total_matches,
            "single_frame_inliers": inliers,
            "single_frame_inlier_ratio": inlier_ratio,
            "single_frame_reproj_rmse": float(frame_rmse),
            "status": "ok",
            "kept": bool(kept),
        }
        if gt_rvec is not None and gt_tvec is not None:
            info["gt_rotation_delta_deg"] = _rotation_delta_deg_from_rvecs(frame_rvec, gt_rvec)
            info["gt_translation_delta_m"] = float(
                np.linalg.norm(
                    np.asarray(frame_tvec, dtype=np.float64).reshape(3)
                    - np.asarray(gt_tvec, dtype=np.float64).reshape(3)
                )
            )
        frame_infos.append(info)
        if kept:
            filtered_frames.append(candidate)

    diagnostics = {
        "pnp_min_inliers": int(pnp_min_inliers),
        "keep_min_inliers": int(max(keep_min_inliers, pnp_min_inliers)),
        "keep_min_inlier_ratio": float(keep_min_inlier_ratio),
        "valid_frames": int(len(frame_data_list)),
        "kept_frames": int(len(filtered_frames)),
        "dropped_frames": int(len(frame_data_list) - len(filtered_frames)),
        "kept_frame_ids": [int(frame.frame_id) for frame in filtered_frames],
        "dropped_frame_ids": [
            int(info["frame_id"])
            for info in frame_infos
            if not bool(info.get("kept", False))
        ],
        "frames": frame_infos,
    }
    return filtered_frames, diagnostics


def initialize_shared_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
    filter_frames: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if not frame_data_list:
        raise ValueError("At least one frame with valid correspondences is required for initialization.")

    counts = [int(frame_data.points_3d.shape[0]) for frame_data in frame_data_list]
    stacked_points_3d, stacked_rgb_points = _stack_frame_correspondences(frame_data_list)

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
        frame_data.pnp_inliers = inlier_count
        if inlier_count == 0:
            frame_data.pnp_reproj_error = float("nan")
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

        frame_data.pnp_reproj_error = rmse
        filtered_frames.append(frame_data)

    if filter_frames:
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
    point_depths = None
    if frame_data.point_depths is not None:
        point_depths = np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)[mask]
    support_scores = None
    if frame_data.support_scores is not None:
        support_scores = np.asarray(frame_data.support_scores, dtype=np.float64).reshape(-1)[mask]
    return FrameCalibrationData(
        frame_name=frame_data.frame_name,
        rgb_path=frame_data.rgb_path,
        depth_path=frame_data.depth_path,
        rgb_points=frame_data.rgb_points[mask].copy(),
        depth_points=frame_data.depth_points[mask].copy(),
        points_3d=frame_data.points_3d[mask].copy(),
        point_depths=None if point_depths is None else point_depths.copy(),
        point_weights=None if point_weights is None else point_weights.copy(),
        support_scores=None if support_scores is None else support_scores.copy(),
        frame_index=frame_data.frame_index,
        frame_id=frame_data.frame_id,
        temporal_weight=frame_data.temporal_weight,
        pnp_inliers=frame_data.pnp_inliers,
        pnp_reproj_error=frame_data.pnp_reproj_error,
    )


def filter_frame_correspondence_by_reference_cloud(
    frame_data: FrameCalibrationData,
    reference_points: np.ndarray,
    max_distance: float,
) -> tuple[FrameCalibrationData | None, dict[str, Any]]:
    max_distance = float(max_distance)
    points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    total = int(frame_data.points_3d.shape[0])
    if max_distance <= 0.0:
        return frame_data, {
            "applied": False,
            "total_points": total,
            "kept_points": total,
            "dropped_points": 0,
            "kept_ratio": 1.0,
            "max_distance": max_distance,
        }
    if points.shape[0] == 0:
        return None, {
            "applied": True,
            "total_points": total,
            "kept_points": 0,
            "dropped_points": total,
            "kept_ratio": 0.0,
            "max_distance": max_distance,
            "reason": "empty_reference_cloud",
        }

    tree = cKDTree(points)
    distances, _ = tree.query(
        np.asarray(frame_data.points_3d, dtype=np.float64),
        k=1,
        workers=-1,
    )
    keep_mask = np.isfinite(distances) & (distances <= max_distance)
    kept = int(np.count_nonzero(keep_mask))
    filtered = _clone_frame_subset(frame_data, keep_mask)
    diagnostics: dict[str, Any] = {
        "applied": True,
        "total_points": total,
        "kept_points": kept,
        "dropped_points": int(total - kept),
        "kept_ratio": float(kept / max(total, 1)),
        "max_distance": max_distance,
    }
    if distances.size > 0:
        diagnostics["nearest_distance_mean_m"] = float(np.mean(distances))
        diagnostics["nearest_distance_median_m"] = float(np.median(distances))
        diagnostics["nearest_distance_p90_m"] = float(np.percentile(distances, 90.0))
    return filtered, diagnostics


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


def stratify_frame_data_by_depth(
    frame_data_list: list[FrameCalibrationData],
    num_bins: int = 3,
    max_points_per_bin: int = 128,
) -> list[FrameCalibrationData]:
    stratified: list[FrameCalibrationData] = []
    bins = max(int(num_bins), 1)
    limit = max(int(max_points_per_bin), 0)
    if bins <= 1 or limit <= 0:
        return [
            _clone_frame_subset(frame_data, np.ones((frame_data.points_3d.shape[0],), dtype=bool))
            for frame_data in frame_data_list
            if frame_data.points_3d.shape[0] > 0
        ]

    for frame_data in frame_data_list:
        depths = np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        if depths.size == 0:
            continue
        valid = np.isfinite(depths) & (depths > 1.0e-6)
        if not np.any(valid):
            continue
        valid_depths = depths[valid]
        quantiles = np.linspace(0.0, 100.0, bins + 1)
        edges = np.percentile(valid_depths, quantiles)
        selected_indices: list[np.ndarray] = []
        point_weights = (
            np.asarray(frame_data.point_weights, dtype=np.float64).reshape(-1)
            if frame_data.point_weights is not None
            else np.ones((depths.shape[0],), dtype=np.float64)
        )

        for bin_index in range(bins):
            lo = float(edges[bin_index])
            hi = float(edges[bin_index + 1])
            if bin_index == bins - 1:
                mask = valid & (depths >= lo) & (depths <= hi)
            else:
                mask = valid & (depths >= lo) & (depths < hi)
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            if indices.size > limit:
                priority = point_weights[indices]
                order = np.argsort(-priority, kind="stable")
                indices = indices[order[:limit]]
            selected_indices.append(indices.astype(np.int32, copy=False))

        if not selected_indices:
            continue

        keep_indices = np.unique(np.concatenate(selected_indices, axis=0))
        keep_mask = np.zeros((depths.shape[0],), dtype=bool)
        keep_mask[keep_indices] = True
        subset = _clone_frame_subset(frame_data, keep_mask)
        if subset is not None:
            stratified.append(subset)
    return stratified


def optimize_shared_extrinsic(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    temporal_residuals: list[TemporalResidualData] | None = None,
    temporal_residual_weight: float = 0.0,
    photometric_residuals: list[TemporalPhotometricResidualData] | None = None,
    photometric_target_images: dict[int, np.ndarray] | None = None,
    photometric_residual_weight: float = 0.0,
    depth_residuals: list[TemporalDepthResidualData] | None = None,
    depth_target_maps: dict[int, np.ndarray] | None = None,
    depth_target_cameras: dict[int, CameraModel] | None = None,
    depth_residual_weight: float = 0.0,
    temporal_regularization: float = 0.0,
    staged_refinement: bool = False,
    staged_depth_split: float = 20.0,
    optimize_rotation: bool = True,
    optimize_translation: bool = True,
    solver_backend: str = "auto",
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
    gt_pose_residual_weight: float = 0.0,
) -> CalibrationComparison:
    if not frame_data_list:
        raise ValueError("No frame correspondences available for optimization.")
    if not optimize_rotation and not optimize_translation:
        raise ValueError("At least one of optimize_rotation or optimize_translation must be enabled.")
    if solver_backend not in {"auto", "opencv", "scipy"}:
        raise ValueError(f"Unsupported solver_backend={solver_backend!r}. Expected 'auto', 'opencv', or 'scipy'.")
    temporal_residuals = temporal_residuals or []
    photometric_residuals = photometric_residuals or []
    photometric_target_images = photometric_target_images or {}
    depth_residuals = depth_residuals or []
    depth_target_maps = depth_target_maps or {}
    depth_target_cameras = depth_target_cameras or {}
    expected_residual_length: int | None = None

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
                point_weights = (
                    np.asarray(temporal_residual.point_weights, dtype=np.float64).reshape(-1)
                    if temporal_residual.point_weights is not None
                    else np.ones((temporal_residual.source_points_3d.shape[0],), dtype=np.float64)
                )
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
                point_weight = np.sqrt(point_weights[valid_mask])[:, None]
                residuals.append(
                    weight
                    * (point_weight * (projected.reshape(-1, 2) - temporal_residual.target_rgb_points[valid_mask])).reshape(-1)
                )

        if photometric_residual_weight > 0.0 and photometric_residuals:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            translation = current_tvec.reshape(1, 3)
            for photometric_residual in photometric_residuals:
                target_image = photometric_target_images.get(int(photometric_residual.target_frame_id))
                if target_image is None:
                    continue
                point_weights = (
                    np.asarray(photometric_residual.point_weights, dtype=np.float64).reshape(-1)
                    if photometric_residual.point_weights is not None
                    else np.ones((photometric_residual.source_points_3d.shape[0],), dtype=np.float64)
                )
                photometric_block = np.zeros(
                    (
                        photometric_residual.source_points_3d.shape[0],
                        int(target_image.shape[2]),
                    ),
                    dtype=np.float64,
                )
                source_rgb = (rotation_matrix @ photometric_residual.source_points_3d.T).T + translation
                world_points = (
                    photometric_residual.source_c2w[:3, :3] @ source_rgb.T
                ).T + photometric_residual.source_c2w[:3, 3]
                target_rgb = (
                    photometric_residual.target_w2c[:3, :3] @ world_points.T
                ).T + photometric_residual.target_w2c[:3, 3]
                valid_mask = target_rgb[:, 2] > 1.0e-6
                if np.any(valid_mask):
                    valid_indices = np.flatnonzero(valid_mask)
                    projected, _ = cv2.projectPoints(
                        target_rgb[valid_indices].astype(np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        rgb_camera.K,
                        rgb_camera.dist if rgb_camera.dist.size > 0 else None,
                    )
                    projected = projected.reshape(-1, 2)
                    inside = (
                        (projected[:, 0] >= 0.0)
                        & (projected[:, 0] <= float(target_image.shape[1] - 1))
                        & (projected[:, 1] >= 0.0)
                        & (projected[:, 1] <= float(target_image.shape[0] - 1))
                    )
                    if np.any(inside):
                        inside_indices = np.flatnonzero(inside)
                        coords = np.vstack([
                            projected[inside_indices, 1].astype(np.float64, copy=False),
                            projected[inside_indices, 0].astype(np.float64, copy=False),
                        ])
                        sampled_target = np.stack(
                            [
                                map_coordinates(
                                    target_image[:, :, channel],
                                    coords,
                                    order=1,
                                    mode="nearest",
                                ).astype(np.float64)
                                for channel in range(target_image.shape[2])
                            ],
                            axis=1,
                        )
                        global_indices = valid_indices[inside_indices]
                        source_colors = photometric_residual.source_colors[global_indices].astype(np.float64)
                        photometric_block[global_indices] = (
                            sampled_target - source_colors
                        ) * np.sqrt(point_weights[global_indices])[:, None]
                weight = np.sqrt(float(photometric_residual_weight) * float(photometric_residual.weight))
                residuals.append(weight * photometric_block.reshape(-1))

        if depth_residual_weight > 0.0 and depth_residuals:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            translation = current_tvec.reshape(1, 3)
            for depth_residual in depth_residuals:
                target_depth = depth_target_maps.get(int(depth_residual.target_frame_id))
                if target_depth is None:
                    continue
                target_depth_camera = depth_target_cameras.get(int(depth_residual.target_frame_id), rgb_camera)
                point_weights = (
                    np.asarray(depth_residual.point_weights, dtype=np.float64).reshape(-1)
                    if depth_residual.point_weights is not None
                    else np.ones((depth_residual.source_points_3d.shape[0],), dtype=np.float64)
                )
                depth_block = np.zeros((depth_residual.source_points_3d.shape[0],), dtype=np.float64)
                source_rgb = (rotation_matrix @ depth_residual.source_points_3d.T).T + translation
                world_points = (
                    depth_residual.source_c2w[:3, :3] @ source_rgb.T
                ).T + depth_residual.source_c2w[:3, 3]
                target_rgb = (
                    depth_residual.target_w2c[:3, :3] @ world_points.T
                ).T + depth_residual.target_w2c[:3, 3]
                valid_mask = target_rgb[:, 2] > 1.0e-6
                if np.any(valid_mask):
                    valid_indices = np.flatnonzero(valid_mask)
                    projected, _ = cv2.projectPoints(
                        target_rgb[valid_indices].astype(np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        target_depth_camera.K,
                        target_depth_camera.dist if target_depth_camera.dist.size > 0 else None,
                    )
                    projected = projected.reshape(-1, 2)
                    inside = (
                        (projected[:, 0] >= 0.0)
                        & (projected[:, 0] <= float(target_depth.shape[1] - 1))
                        & (projected[:, 1] >= 0.0)
                        & (projected[:, 1] <= float(target_depth.shape[0] - 1))
                    )
                    if np.any(inside):
                        inside_indices = np.flatnonzero(inside)
                        coords = np.vstack([
                            projected[inside_indices, 1].astype(np.float64, copy=False),
                            projected[inside_indices, 0].astype(np.float64, copy=False),
                        ])
                        sampled_target_depth = map_coordinates(
                            target_depth,
                            coords,
                            order=1,
                            mode="nearest",
                        ).astype(np.float64)
                        global_indices = valid_indices[inside_indices]
                        predicted_depth = target_rgb[global_indices, 2].astype(np.float64)
                        valid_depth = (sampled_target_depth > 1.0e-3) & (predicted_depth > 1.0e-3)
                        if np.any(valid_depth):
                            sel_indices = global_indices[valid_depth]
                            depth_block[sel_indices] = (
                                (
                                    (1.0 / predicted_depth[valid_depth])
                                    - (1.0 / sampled_target_depth[valid_depth])
                                )
                                * np.sqrt(point_weights[sel_indices])
                            )
                weight = np.sqrt(float(depth_residual_weight) * float(depth_residual.weight))
                residuals.append(weight * depth_block)

        if temporal_regularization > 0.0:
            prior_weights = [
                float(frame_data.temporal_weight)
                for frame_data in active_frame_data_list
                if np.isfinite(frame_data.pnp_reproj_error)
            ]
            prior_scale = float(temporal_regularization) * np.sqrt(np.sum(prior_weights)) if prior_weights else float(temporal_regularization)
            residuals.append(prior_scale * (current_rvec - initial_rvec.reshape(3)).reshape(-1))
            residuals.append(prior_scale * (current_tvec - initial_tvec.reshape(3)).reshape(-1))

        if gt_pose_residual_weight > 0.0 and gt_rvec is not None and gt_tvec is not None:
            gt_scale = np.sqrt(float(gt_pose_residual_weight))
            residuals.append(
                gt_scale * _relative_rotation_residual_rvec(current_rvec, np.asarray(gt_rvec, dtype=np.float64).reshape(3))
            )
            residuals.append(
                gt_scale * (current_tvec - np.asarray(gt_tvec, dtype=np.float64).reshape(3))
            )

        nonlocal expected_residual_length
        residual_vector = np.concatenate(residuals, axis=0)
        if expected_residual_length is None:
            expected_residual_length = int(residual_vector.shape[0])
        elif residual_vector.shape[0] != expected_residual_length:
            if residual_vector.shape[0] < expected_residual_length:
                residual_vector = np.pad(
                    residual_vector,
                    (0, expected_residual_length - residual_vector.shape[0]),
                    mode="constant",
                )
            else:
                residual_vector = residual_vector[:expected_residual_length]
        return residual_vector

    initial_rvec = np.asarray(initial_rvec, dtype=np.float64).reshape(3)
    initial_tvec = np.asarray(initial_tvec, dtype=np.float64).reshape(3)
    opencv_unsupported_reason = _opencv_shared_pnp_unsupported_reason(
        frame_data_list,
        temporal_residuals=temporal_residuals,
        temporal_residual_weight=temporal_residual_weight,
        photometric_residuals=photometric_residuals,
        photometric_residual_weight=photometric_residual_weight,
        depth_residuals=depth_residuals,
        depth_residual_weight=depth_residual_weight,
        temporal_regularization=temporal_regularization,
        staged_refinement=staged_refinement,
        optimize_rotation=optimize_rotation,
        optimize_translation=optimize_translation,
        gt_pose_residual_weight=gt_pose_residual_weight,
    )
    if solver_backend == "opencv" and opencv_unsupported_reason is not None:
        raise ValueError(opencv_unsupported_reason)
    use_opencv_backend = (
        solver_backend == "opencv"
        or (solver_backend == "auto" and opencv_unsupported_reason is None)
    )
    if optimize_rotation and optimize_translation:
        initial_params = np.concatenate([initial_rvec, initial_tvec], axis=0)
    elif optimize_rotation:
        initial_params = initial_rvec.copy()
    else:
        initial_params = initial_tvec.copy()

    if use_opencv_backend:
        final_rvec, final_tvec = _refine_shared_extrinsic_opencv(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
        )
    elif staged_refinement and optimize_rotation and optimize_translation:
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
        final_rvec = np.asarray(result.x[:3], dtype=np.float64).reshape(3)
        final_tvec = np.asarray(result.x[3:6], dtype=np.float64).reshape(3)
    elif optimize_rotation and not optimize_translation:
        result = least_squares(
            lambda rot_params: residual_function(
                np.concatenate([np.asarray(rot_params, dtype=np.float64).reshape(3), initial_tvec], axis=0),
                active_frame_data_list=frame_data_list,
                optimize_rotation=True,
                optimize_translation=False,
                fixed_tvec=initial_tvec,
                include_temporal_residuals=True,
            ),
            initial_params,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
        )
        final_rvec = np.asarray(result.x, dtype=np.float64).reshape(3)
        final_tvec = initial_tvec.copy()
    elif optimize_translation and not optimize_rotation:
        result = least_squares(
            lambda trans_params: residual_function(
                np.concatenate([initial_rvec, np.asarray(trans_params, dtype=np.float64).reshape(3)], axis=0),
                active_frame_data_list=frame_data_list,
                optimize_rotation=False,
                optimize_translation=True,
                fixed_rvec=initial_rvec,
                include_temporal_residuals=True,
            ),
            initial_params,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
        )
        final_rvec = initial_rvec.copy()
        final_tvec = np.asarray(result.x, dtype=np.float64).reshape(3)
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
        final_rvec = np.asarray(result.x[:3], dtype=np.float64).reshape(3)
        final_tvec = np.asarray(result.x[3:6], dtype=np.float64).reshape(3)
    initial_calibration = evaluate_shared_extrinsic(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=initial_rvec,
        tvec=initial_tvec,
    )
    optimized_calibration = evaluate_shared_extrinsic(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=final_rvec,
        tvec=final_tvec,
    )
    return compare_calibrations(initial_calibration, optimized_calibration)


def compute_frame_reprojection_errors(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict[int, np.ndarray]:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    residuals_by_frame: dict[int, np.ndarray] = {}
    for frame_data in frame_data_list:
        if frame_data.points_3d.shape[0] == 0:
            residuals_by_frame[frame_data.frame_index] = np.zeros((0,), dtype=np.float64)
            continue
        projected, _ = cv2.projectPoints(
            frame_data.points_3d.astype(np.float64),
            rvec,
            tvec,
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = projected.reshape(-1, 2) - frame_data.rgb_points
        residuals_by_frame[frame_data.frame_index] = np.sqrt(np.sum(residual**2, axis=1)).astype(np.float64)
    return residuals_by_frame


def compute_frame_translation_sensitivity(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict[int, np.ndarray]:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    fx = float(rgb_camera.K[0, 0])
    fy = float(rgb_camera.K[1, 1])
    sensitivities_by_frame: dict[int, np.ndarray] = {}
    for frame_data in frame_data_list:
        if frame_data.points_3d.shape[0] == 0:
            sensitivities_by_frame[frame_data.frame_index] = np.zeros((0,), dtype=np.float64)
            continue
        rgb_points_3d = (rotation_matrix @ frame_data.points_3d.astype(np.float64).T).T + translation
        x = rgb_points_3d[:, 0]
        y = rgb_points_3d[:, 1]
        z = rgb_points_3d[:, 2]
        sensitivity = np.zeros((rgb_points_3d.shape[0],), dtype=np.float64)
        valid = z > 1.0e-6
        if np.any(valid):
            z_valid = z[valid]
            x_valid = x[valid]
            y_valid = y[valid]
            jacobian_norm_sq = (
                (fx / z_valid) ** 2
                + (fy / z_valid) ** 2
                + (fx * x_valid / (z_valid**2)) ** 2
                + (fy * y_valid / (z_valid**2)) ** 2
            )
            sensitivity[valid] = np.sqrt(jacobian_norm_sq)
        sensitivities_by_frame[frame_data.frame_index] = sensitivity
    return sensitivities_by_frame


def filter_frame_data_by_shared_ransac(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[FrameCalibrationData], dict[str, Any]]:
    if not frame_data_list:
        raise ValueError("At least one frame with valid correspondences is required for shared RANSAC filtering.")

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
        raise ValueError("Shared RANSAC filtering failed to find a valid global solution.")

    global_inlier_mask = np.zeros(stacked_points_3d.shape[0], dtype=bool)
    global_inlier_mask[inliers.reshape(-1)] = True
    init_residuals_by_frame = compute_frame_reprojection_errors(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
    )
    gt_residuals_by_frame = (
        compute_frame_reprojection_errors(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            rvec=np.asarray(gt_rvec, dtype=np.float64).reshape(3),
            tvec=np.asarray(gt_tvec, dtype=np.float64).reshape(3),
        )
        if gt_rvec is not None and gt_tvec is not None
        else {}
    )

    filtered_frames: list[FrameCalibrationData] = []
    frame_infos: list[dict[str, Any]] = []
    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_depth: list[np.ndarray] = []
    all_init_reproj: list[np.ndarray] = []
    all_gt_reproj: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    all_kept: list[np.ndarray] = []

    start = 0
    for frame_data, count in zip(frame_data_list, counts):
        frame_mask = global_inlier_mask[start : start + count]
        start += count
        total_count = int(count)
        inlier_count = int(np.count_nonzero(frame_mask))
        dropped_count = int(total_count - inlier_count)
        init_reproj = np.asarray(
            init_residuals_by_frame.get(frame_data.frame_index, np.zeros((count,), dtype=np.float64)),
            dtype=np.float64,
        )
        gt_reproj = np.asarray(
            gt_residuals_by_frame.get(frame_data.frame_index, np.full((count,), np.nan, dtype=np.float64)),
            dtype=np.float64,
        )
        depth_vals = (
            np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)
            if frame_data.point_depths is not None
            else np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        )

        all_u.append(np.asarray(frame_data.rgb_points[:, 0], dtype=np.float64))
        all_v.append(np.asarray(frame_data.rgb_points[:, 1], dtype=np.float64))
        all_depth.append(depth_vals)
        all_init_reproj.append(init_reproj)
        all_gt_reproj.append(gt_reproj)
        all_frame_ids.append(np.full((count,), int(frame_data.frame_id), dtype=np.int32))
        all_kept.append(frame_mask.astype(bool))

        info = {
            "frame_index": int(frame_data.frame_index),
            "frame_id": int(frame_data.frame_id),
            "frame_name": str(frame_data.frame_name),
            "total_matches": total_count,
            "inlier_matches": inlier_count,
            "dropped_matches": dropped_count,
            "keep_ratio": float(inlier_count / max(total_count, 1)),
            "kept_init_reproj_px": _summarize_array(init_reproj[frame_mask]),
            "dropped_init_reproj_px": _summarize_array(init_reproj[~frame_mask]),
            "kept_depth_m": _summarize_array(depth_vals[frame_mask]),
            "dropped_depth_m": _summarize_array(depth_vals[~frame_mask]),
        }
        if gt_rvec is not None and gt_tvec is not None:
            info["kept_gt_reproj_px"] = _summarize_array(gt_reproj[frame_mask])
            info["dropped_gt_reproj_px"] = _summarize_array(gt_reproj[~frame_mask])
        frame_infos.append(info)

        if inlier_count <= 0:
            continue
        filtered = _clone_frame_subset(frame_data, frame_mask)
        if filtered is None:
            continue
        projected, _ = cv2.projectPoints(
            filtered.points_3d.astype(np.float64),
            rvec,
            tvec,
            rgb_camera.K,
            rgb_camera.dist if rgb_camera.dist.size > 0 else None,
        )
        residual = projected.reshape(-1, 2) - filtered.rgb_points
        filtered.pnp_inliers = inlier_count
        filtered.pnp_reproj_error = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        filtered_frames.append(filtered)

    all_u_arr = np.concatenate(all_u, axis=0) if all_u else np.zeros((0,), dtype=np.float64)
    all_v_arr = np.concatenate(all_v, axis=0) if all_v else np.zeros((0,), dtype=np.float64)
    all_depth_arr = np.concatenate(all_depth, axis=0) if all_depth else np.zeros((0,), dtype=np.float64)
    all_init_arr = np.concatenate(all_init_reproj, axis=0) if all_init_reproj else np.zeros((0,), dtype=np.float64)
    all_gt_arr = np.concatenate(all_gt_reproj, axis=0) if all_gt_reproj else np.zeros((0,), dtype=np.float64)
    all_frame_ids_arr = np.concatenate(all_frame_ids, axis=0) if all_frame_ids else np.zeros((0,), dtype=np.int32)
    all_kept_arr = np.concatenate(all_kept, axis=0) if all_kept else np.zeros((0,), dtype=bool)
    width = float(np.max(all_u_arr) + 1.0) if all_u_arr.size else 1.0
    height = float(np.max(all_v_arr) + 1.0) if all_v_arr.size else 1.0

    def _partition_summary(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not np.any(mask):
            return {
                "count": 0,
                "init_reproj_px": _summarize_array(np.zeros((0,), dtype=np.float64)),
                "gt_reproj_px": _summarize_array(np.zeros((0,), dtype=np.float64)),
                "depth_m": _summarize_array(np.zeros((0,), dtype=np.float64)),
                "u_band_rate": {"left": None, "mid": None, "right": None},
                "v_band_rate": {"top": None, "center": None, "bottom": None},
                "top_frames": [],
            }
        left = all_u_arr[mask] < (width / 3.0)
        mid = (all_u_arr[mask] >= (width / 3.0)) & (all_u_arr[mask] < (2.0 * width / 3.0))
        right = all_u_arr[mask] >= (2.0 * width / 3.0)
        top = all_v_arr[mask] < (height / 3.0)
        center = (all_v_arr[mask] >= (height / 3.0)) & (all_v_arr[mask] < (2.0 * height / 3.0))
        bottom = all_v_arr[mask] >= (2.0 * height / 3.0)
        unique_frames, counts = np.unique(all_frame_ids_arr[mask], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        top_frames = [
            {"frame_id": int(unique_frames[idx]), "count": int(counts[idx])}
            for idx in order[:10]
        ]
        return {
            "count": int(np.count_nonzero(mask)),
            "init_reproj_px": _summarize_array(all_init_arr[mask]),
            "gt_reproj_px": _summarize_array(all_gt_arr[mask]),
            "depth_m": _summarize_array(all_depth_arr[mask]),
            "u_band_rate": {
                "left": float(np.mean(left)),
                "mid": float(np.mean(mid)),
                "right": float(np.mean(right)),
            },
            "v_band_rate": {
                "top": float(np.mean(top)),
                "center": float(np.mean(center)),
                "bottom": float(np.mean(bottom)),
            },
            "top_frames": top_frames,
        }

    diagnostics = {
        "reproj_error": float(reproj_error),
        "iterations": int(iterations),
        "total_matches": int(all_kept_arr.size),
        "kept_matches": int(np.count_nonzero(all_kept_arr)),
        "dropped_matches": int(np.count_nonzero(~all_kept_arr)),
        "kept_summary": _partition_summary(all_kept_arr),
        "dropped_summary": _partition_summary(~all_kept_arr),
        "frames": frame_infos,
    }
    return rvec.reshape(3), tvec.reshape(3), filtered_frames, diagnostics


def filter_frame_data_by_gt_reprojection(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    gt_rvec: np.ndarray,
    gt_tvec: np.ndarray,
    reference_rvec: np.ndarray | None = None,
    reference_tvec: np.ndarray | None = None,
    keep_quantile: float = 0.35,
    min_keep_per_frame: int = 24,
) -> tuple[list[FrameCalibrationData], dict[str, Any]]:
    gt_residuals_by_frame = compute_frame_reprojection_errors(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=np.asarray(gt_rvec, dtype=np.float64).reshape(3),
        tvec=np.asarray(gt_tvec, dtype=np.float64).reshape(3),
    )
    init_residuals_by_frame = (
        compute_frame_reprojection_errors(
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            rvec=np.asarray(reference_rvec, dtype=np.float64).reshape(3),
            tvec=np.asarray(reference_tvec, dtype=np.float64).reshape(3),
        )
        if reference_rvec is not None and reference_tvec is not None
        else gt_residuals_by_frame
    )

    all_gt = np.concatenate(
        [
            np.asarray(gt_residuals_by_frame.get(frame.frame_index, np.zeros((frame.points_3d.shape[0],), dtype=np.float64)), dtype=np.float64)
            for frame in frame_data_list
            if frame.points_3d.shape[0] > 0
        ],
        axis=0,
    )
    if all_gt.size == 0:
        raise ValueError("GT reprojection filtering found no correspondences.")
    threshold = float(np.quantile(all_gt, float(np.clip(keep_quantile, 0.0, 1.0))))

    filtered_frames: list[FrameCalibrationData] = []
    frame_infos: list[dict[str, Any]] = []
    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_depth: list[np.ndarray] = []
    all_gt_reproj: list[np.ndarray] = []
    all_init_reproj: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    all_kept: list[np.ndarray] = []

    for frame_data in frame_data_list:
        count = int(frame_data.points_3d.shape[0])
        if count <= 0:
            continue
        gt_reproj = np.asarray(gt_residuals_by_frame.get(frame_data.frame_index, np.zeros((count,), dtype=np.float64)), dtype=np.float64)
        init_reproj = np.asarray(
            init_residuals_by_frame.get(frame_data.frame_index, np.zeros((count,), dtype=np.float64)),
            dtype=np.float64,
        )
        keep_mask = gt_reproj <= threshold
        if int(np.count_nonzero(keep_mask)) < min(int(min_keep_per_frame), count):
            order = np.argsort(gt_reproj, kind="stable")
            keep_mask = np.zeros((count,), dtype=bool)
            keep_mask[order[: min(int(min_keep_per_frame), count)]] = True
        kept = _clone_frame_subset(frame_data, keep_mask)
        if kept is not None:
            filtered_frames.append(kept)

        depth_vals = (
            np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)
            if frame_data.point_depths is not None
            else np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        )
        all_u.append(np.asarray(frame_data.rgb_points[:, 0], dtype=np.float64))
        all_v.append(np.asarray(frame_data.rgb_points[:, 1], dtype=np.float64))
        all_depth.append(depth_vals)
        all_gt_reproj.append(gt_reproj)
        all_init_reproj.append(init_reproj)
        all_frame_ids.append(np.full((count,), int(frame_data.frame_id), dtype=np.int32))
        all_kept.append(keep_mask)
        frame_infos.append(
            {
                "frame_index": int(frame_data.frame_index),
                "frame_id": int(frame_data.frame_id),
                "frame_name": str(frame_data.frame_name),
                "total_matches": count,
                "kept_matches": int(np.count_nonzero(keep_mask)),
                "dropped_matches": int(np.count_nonzero(~keep_mask)),
                "keep_ratio": float(np.mean(keep_mask)),
                "kept_gt_reproj_px": _summarize_array(gt_reproj[keep_mask]),
                "dropped_gt_reproj_px": _summarize_array(gt_reproj[~keep_mask]),
                "kept_depth_m": _summarize_array(depth_vals[keep_mask]),
                "dropped_depth_m": _summarize_array(depth_vals[~keep_mask]),
            }
        )

    all_u_arr = np.concatenate(all_u, axis=0) if all_u else np.zeros((0,), dtype=np.float64)
    all_v_arr = np.concatenate(all_v, axis=0) if all_v else np.zeros((0,), dtype=np.float64)
    all_depth_arr = np.concatenate(all_depth, axis=0) if all_depth else np.zeros((0,), dtype=np.float64)
    all_gt_arr = np.concatenate(all_gt_reproj, axis=0) if all_gt_reproj else np.zeros((0,), dtype=np.float64)
    all_init_arr = np.concatenate(all_init_reproj, axis=0) if all_init_reproj else np.zeros((0,), dtype=np.float64)
    all_frame_ids_arr = np.concatenate(all_frame_ids, axis=0) if all_frame_ids else np.zeros((0,), dtype=np.int32)
    all_kept_arr = np.concatenate(all_kept, axis=0) if all_kept else np.zeros((0,), dtype=bool)
    width = float(np.max(all_u_arr) + 1.0) if all_u_arr.size else 1.0
    height = float(np.max(all_v_arr) + 1.0) if all_v_arr.size else 1.0

    def _partition(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not np.any(mask):
            return {"count": 0}
        unique_frames, counts = np.unique(all_frame_ids_arr[mask], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        return {
            "count": int(np.count_nonzero(mask)),
            "gt_reproj_px": _summarize_array(all_gt_arr[mask]),
            "init_reproj_px": _summarize_array(all_init_arr[mask]),
            "depth_m": _summarize_array(all_depth_arr[mask]),
            "u_band_rate": {
                "left": float(np.mean(all_u_arr[mask] < (width / 3.0))),
                "mid": float(np.mean((all_u_arr[mask] >= (width / 3.0)) & (all_u_arr[mask] < (2.0 * width / 3.0)))),
                "right": float(np.mean(all_u_arr[mask] >= (2.0 * width / 3.0))),
            },
            "v_band_rate": {
                "top": float(np.mean(all_v_arr[mask] < (height / 3.0))),
                "center": float(np.mean((all_v_arr[mask] >= (height / 3.0)) & (all_v_arr[mask] < (2.0 * height / 3.0)))),
                "bottom": float(np.mean(all_v_arr[mask] >= (2.0 * height / 3.0))),
            },
            "top_frames": [
                {"frame_id": int(unique_frames[idx]), "count": int(counts[idx])}
                for idx in order[:10]
            ],
        }

    diagnostics = {
        "keep_quantile": float(keep_quantile),
        "gt_reproj_threshold_px": threshold,
        "total_matches": int(all_kept_arr.size),
        "kept_matches": int(np.count_nonzero(all_kept_arr)),
        "dropped_matches": int(np.count_nonzero(~all_kept_arr)),
        "kept_summary": _partition(all_kept_arr),
        "dropped_summary": _partition(~all_kept_arr),
        "frames": frame_infos,
    }
    return filtered_frames, diagnostics


def _summarize_array(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def filter_frame_data_by_reprojection_consensus(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    rvec: np.ndarray,
    tvec: np.ndarray,
    mad_scale: float = 2.5,
    min_keep_ratio: float = 0.5,
    min_keep_per_frame: int = 24,
    max_reproj_error: float | None = None,
    gt_rvec: np.ndarray | None = None,
    gt_tvec: np.ndarray | None = None,
) -> tuple[list[FrameCalibrationData], dict[str, Any]]:
    residuals_by_frame = compute_frame_reprojection_errors(frame_data_list, rgb_camera, rvec, tvec)
    gt_residuals_by_frame = (
        compute_frame_reprojection_errors(frame_data_list, rgb_camera, gt_rvec, gt_tvec)
        if gt_rvec is not None and gt_tvec is not None
        else {}
    )
    filtered_frames: list[FrameCalibrationData] = []
    frame_summaries: list[dict[str, Any]] = []
    all_init_errors: list[np.ndarray] = []
    all_gt_errors: list[np.ndarray] = []
    kept_init_errors: list[np.ndarray] = []
    kept_gt_errors: list[np.ndarray] = []
    dropped_init_errors: list[np.ndarray] = []
    dropped_gt_errors: list[np.ndarray] = []
    kept_depths: list[np.ndarray] = []
    dropped_depths: list[np.ndarray] = []

    for frame_data in frame_data_list:
        reproj = np.asarray(residuals_by_frame.get(frame_data.frame_index, np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        if reproj.size == 0:
            continue
        median = float(np.median(reproj))
        mad = float(np.median(np.abs(reproj - median)))
        robust_sigma = 1.4826 * mad
        threshold = median + float(mad_scale) * max(robust_sigma, 0.5)
        if max_reproj_error is not None and float(max_reproj_error) > 0.0:
            threshold = min(threshold, float(max_reproj_error))
        keep_mask = reproj <= threshold

        min_keep = max(
            min(int(np.ceil(float(min_keep_ratio) * reproj.shape[0])), reproj.shape[0]),
            min(int(min_keep_per_frame), reproj.shape[0]),
        )
        if int(np.count_nonzero(keep_mask)) < min_keep:
            order = np.argsort(reproj, kind="stable")
            keep_mask = np.zeros((reproj.shape[0],), dtype=bool)
            keep_mask[order[:min_keep]] = True

        subset = _clone_frame_subset(frame_data, keep_mask)
        if subset is None:
            continue
        filtered_frames.append(subset)

        dropped_mask = ~keep_mask
        frame_depths = (
            np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)
            if frame_data.point_depths is not None
            else np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        )
        gt_reproj = np.asarray(
            gt_residuals_by_frame.get(frame_data.frame_index, np.full_like(reproj, np.nan)),
            dtype=np.float64,
        )
        all_init_errors.append(reproj)
        all_gt_errors.append(gt_reproj)
        kept_init_errors.append(reproj[keep_mask])
        dropped_init_errors.append(reproj[dropped_mask])
        kept_gt_errors.append(gt_reproj[keep_mask])
        dropped_gt_errors.append(gt_reproj[dropped_mask])
        kept_depths.append(frame_depths[keep_mask])
        dropped_depths.append(frame_depths[dropped_mask])
        frame_summaries.append(
            {
                "frame_index": int(frame_data.frame_index),
                "frame_id": int(frame_data.frame_id),
                "frame_name": str(frame_data.frame_name),
                "total_matches": int(reproj.shape[0]),
                "kept_matches": int(np.count_nonzero(keep_mask)),
                "dropped_matches": int(np.count_nonzero(dropped_mask)),
                "keep_ratio": float(np.count_nonzero(keep_mask) / max(reproj.shape[0], 1)),
                "threshold_px": float(threshold),
                "init_reproj": _summarize_array(reproj),
                "kept_init_reproj": _summarize_array(reproj[keep_mask]),
                "dropped_init_reproj": _summarize_array(reproj[dropped_mask]),
                "kept_depth_m": _summarize_array(frame_depths[keep_mask]),
                "dropped_depth_m": _summarize_array(frame_depths[dropped_mask]),
                "kept_gt_reproj": _summarize_array(gt_reproj[keep_mask]),
                "dropped_gt_reproj": _summarize_array(gt_reproj[dropped_mask]),
            }
        )

    def _concat(chunks: list[np.ndarray]) -> np.ndarray:
        non_empty = [np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in chunks if np.asarray(chunk).size > 0]
        if not non_empty:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(non_empty, axis=0)

    diagnostics = {
        "mad_scale": float(mad_scale),
        "min_keep_ratio": float(min_keep_ratio),
        "min_keep_per_frame": int(min_keep_per_frame),
        "max_reproj_error": None if max_reproj_error is None else float(max_reproj_error),
        "total_frames": int(len(frame_data_list)),
        "kept_frames": int(len(filtered_frames)),
        "total_matches": int(sum(frame.points_3d.shape[0] for frame in frame_data_list)),
        "kept_matches": int(sum(frame.points_3d.shape[0] for frame in filtered_frames)),
        "global_init_reproj": _summarize_array(_concat(all_init_errors)),
        "global_kept_init_reproj": _summarize_array(_concat(kept_init_errors)),
        "global_dropped_init_reproj": _summarize_array(_concat(dropped_init_errors)),
        "global_gt_reproj": _summarize_array(_concat(all_gt_errors)),
        "global_kept_gt_reproj": _summarize_array(_concat(kept_gt_errors)),
        "global_dropped_gt_reproj": _summarize_array(_concat(dropped_gt_errors)),
        "global_kept_depth_m": _summarize_array(_concat(kept_depths)),
        "global_dropped_depth_m": _summarize_array(_concat(dropped_depths)),
        "frames": frame_summaries,
    }
    return filtered_frames, diagnostics


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1.0e-12 or float(np.std(y)) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _normalize_positive_weights(weights: np.ndarray, min_weight: float = 1.0e-3, max_weight: float = 10.0) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return np.ones_like(weights, dtype=np.float64)
    normalized = np.ones_like(weights, dtype=np.float64)
    normalized[finite] = np.clip(weights[finite], float(min_weight), float(max_weight))
    mean_value = float(np.mean(normalized[finite]))
    if mean_value > 1.0e-12:
        normalized[finite] /= mean_value
    normalized[~finite] = float(min_weight)
    return normalized


def _gt_residual_to_oracle_weights(
    gt_reproj_px: np.ndarray,
    positive_threshold: float,
    negative_threshold: float,
) -> np.ndarray:
    gt_values = np.asarray(gt_reproj_px, dtype=np.float64).reshape(-1)
    log_gt = np.log1p(np.clip(gt_values, 0.0, None))
    log_pos = np.log1p(max(float(positive_threshold), 0.0))
    log_neg = np.log1p(max(float(negative_threshold), 0.0))
    center = 0.5 * (log_pos + log_neg)
    spread = max(log_neg - log_pos, 1.0e-3)
    oracle_weights = 1.0 / (1.0 + np.exp((log_gt - center) / spread))
    return _normalize_positive_weights(oracle_weights)


def _translation_sensitivity_to_balance_weights(
    translation_sensitivity: np.ndarray,
    alpha: float = 0.5,
    min_scale: float = 0.35,
    max_scale: float = 3.0,
) -> np.ndarray:
    sensitivity = np.asarray(translation_sensitivity, dtype=np.float64).reshape(-1)
    valid = np.isfinite(sensitivity) & (sensitivity > 0.0)
    if not np.any(valid):
        return np.ones_like(sensitivity, dtype=np.float64)
    median = float(np.median(sensitivity[valid]))
    if median <= 1.0e-12:
        return np.ones_like(sensitivity, dtype=np.float64)
    normalized = np.ones_like(sensitivity, dtype=np.float64)
    normalized[valid] = np.clip(
        sensitivity[valid] / median,
        float(min_scale),
        float(max_scale),
    )
    normalized[valid] = np.power(normalized[valid], float(max(alpha, 0.0)))
    normalized[~valid] = 1.0
    return normalized


def _build_weighted_frame_data_list(
    frame_data_list: list[FrameCalibrationData],
    point_weights: np.ndarray,
) -> list[FrameCalibrationData]:
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    weighted_frames: list[FrameCalibrationData] = []
    start = 0
    for frame_data in frame_data_list:
        count = int(frame_data.points_3d.shape[0])
        end = start + count
        if count <= 0:
            start = end
            continue
        subset = _clone_frame_subset(frame_data, np.ones((count,), dtype=bool))
        if subset is not None:
            subset.point_weights = weights[start:end].copy()
            weighted_frames.append(subset)
        start = end
    return weighted_frames


def build_weighted_frame_data_list(
    frame_data_list: list[FrameCalibrationData],
    point_weights: np.ndarray,
) -> list[FrameCalibrationData]:
    return _build_weighted_frame_data_list(frame_data_list, point_weights)


def _summarize_weight_family(
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    gt_values: np.ndarray,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    top_cut = float(np.quantile(weights, 0.9))
    bottom_cut = float(np.quantile(weights, 0.1))
    top_mask = weights >= top_cut
    bottom_mask = weights <= bottom_cut
    return {
        "top_weight_summary": {
            "weight": _summarize_array(weights[top_mask]),
            "gt_reproj_px": _summarize_array(gt_values[top_mask]),
            "init_reproj_px": _summarize_array(arrays["init_reproj_px"][top_mask]),
            "depth_m": _summarize_array(arrays["depth_m"][top_mask]),
            "translation_sensitivity": _summarize_array(arrays["translation_sensitivity"][top_mask]),
            "support_score": _summarize_array(arrays["support_score"][top_mask]),
            "frame_rotation_delta_deg": _summarize_array(arrays["frame_rotation_delta_deg"][top_mask]),
            "frame_translation_delta_m": _summarize_array(arrays["frame_translation_delta_m"][top_mask]),
        },
        "bottom_weight_summary": {
            "weight": _summarize_array(weights[bottom_mask]),
            "gt_reproj_px": _summarize_array(gt_values[bottom_mask]),
            "init_reproj_px": _summarize_array(arrays["init_reproj_px"][bottom_mask]),
            "depth_m": _summarize_array(arrays["depth_m"][bottom_mask]),
            "translation_sensitivity": _summarize_array(arrays["translation_sensitivity"][bottom_mask]),
            "support_score": _summarize_array(arrays["support_score"][bottom_mask]),
            "frame_rotation_delta_deg": _summarize_array(arrays["frame_rotation_delta_deg"][bottom_mask]),
            "frame_translation_delta_m": _summarize_array(arrays["frame_translation_delta_m"][bottom_mask]),
        },
        "weight_correlations": {
            "gt_reproj_px": _corrcoef_safe(weights, arrays["gt_reproj_px"]),
            "init_reproj_px": _corrcoef_safe(weights, arrays["init_reproj_px"]),
            "depth_m": _corrcoef_safe(weights, arrays["depth_m"]),
            "translation_sensitivity": _corrcoef_safe(weights, arrays["translation_sensitivity"]),
            "support_score": _corrcoef_safe(weights, arrays["support_score"]),
            "frame_rotation_delta_deg": _corrcoef_safe(weights, arrays["frame_rotation_delta_deg"]),
            "frame_translation_delta_m": _corrcoef_safe(weights, arrays["frame_translation_delta_m"]),
        },
    }


def _evaluate_weighted_solve_variant(
    name: str,
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    gt_rvec: np.ndarray,
    gt_tvec: np.ndarray,
    weights: np.ndarray,
    gt_pose_residual_weight: float = 0.0,
) -> dict[str, Any]:
    weighted_frames = _build_weighted_frame_data_list(frame_data_list, weights)
    comparison = optimize_shared_extrinsic(
        frame_data_list=weighted_frames,
        rgb_camera=rgb_camera,
        initial_rvec=initial_rvec,
        initial_tvec=initial_tvec,
        gt_rvec=gt_rvec,
        gt_tvec=gt_tvec,
        gt_pose_residual_weight=gt_pose_residual_weight,
    )
    return {
        "name": str(name),
        "mean_reproj_px": float(comparison.optimized.mean_reprojection_error),
        "median_reproj_px": float(comparison.optimized.median_reprojection_error),
        "matches_used": int(comparison.optimized.matches_used),
        "frames_used": int(comparison.optimized.frames_used),
        "rotation_delta_from_init_deg": float(comparison.rotation_delta_deg),
        "translation_delta_from_init_m": float(comparison.translation_delta_m),
        "rotation_error_to_gt_deg": float(
            _rotation_delta_deg_from_rvecs(comparison.optimized.rotation_vector, gt_rvec)
        ),
        "translation_error_to_gt_m": float(
            np.linalg.norm(
                np.asarray(comparison.optimized.translation, dtype=np.float64).reshape(3)
                - np.asarray(gt_tvec, dtype=np.float64).reshape(3)
            )
        ),
    }


def learn_gt_point_weight_predictor(
    frame_data_list: list[FrameCalibrationData],
    rgb_camera: CameraModel,
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
    gt_rvec: np.ndarray,
    gt_tvec: np.ndarray,
    gt_pose_residual_weight: float = 0.0,
    reproj_error: float = 4.0,
    iterations: int = 1000,
    min_inliers: int = 8,
    supervision_target: str = "gt_reproj",
    translation_sensitivity_alpha: float = 0.5,
    positive_quantile: float = 0.35,
    negative_quantile: float = 0.65,
    l2_reg: float = 1.0e-2,
    learning_rate: float = 0.2,
    num_iters: int = 400,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    init_residuals_by_frame = compute_frame_reprojection_errors(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=init_rvec,
        tvec=init_tvec,
    )
    gt_residuals_by_frame = compute_frame_reprojection_errors(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=gt_rvec,
        tvec=gt_tvec,
    )
    translation_sensitivity_by_frame = compute_frame_translation_sensitivity(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        rvec=init_rvec,
        tvec=init_tvec,
    )
    _, frame_diag = filter_frame_data_by_pose_disagreement(
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        shared_rvec=init_rvec,
        shared_tvec=init_tvec,
        reproj_error=reproj_error,
        iterations=iterations,
        min_inliers=min_inliers,
        mad_scale=2.5,
        min_keep_ratio=0.7,
        min_keep_frames=12,
        gt_rvec=gt_rvec,
        gt_tvec=gt_tvec,
    )
    frame_info_by_index = {
        int(info["frame_index"]): info
        for info in frame_diag.get("frames", [])
        if info.get("status") == "ok"
    }

    frame_ids: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    rgb_u: list[np.ndarray] = []
    rgb_v: list[np.ndarray] = []
    init_reproj: list[np.ndarray] = []
    gt_reproj: list[np.ndarray] = []
    depth_m: list[np.ndarray] = []
    translation_sensitivity: list[np.ndarray] = []
    support_score: list[np.ndarray] = []
    frame_rot_delta: list[np.ndarray] = []
    frame_trans_delta: list[np.ndarray] = []
    frame_single_reproj: list[np.ndarray] = []

    for frame_data in frame_data_list:
        count = int(frame_data.rgb_points.shape[0])
        if count <= 0:
            continue
        info = frame_info_by_index.get(int(frame_data.frame_index), {})
        frame_ids.append(np.full((count,), int(frame_data.frame_id), dtype=np.int32))
        frame_indices.append(np.full((count,), int(frame_data.frame_index), dtype=np.int32))
        rgb_u.append(np.asarray(frame_data.rgb_points[:, 0], dtype=np.float64))
        rgb_v.append(np.asarray(frame_data.rgb_points[:, 1], dtype=np.float64))
        init_reproj.append(np.asarray(init_residuals_by_frame.get(frame_data.frame_index, np.zeros((count,), dtype=np.float64)), dtype=np.float64))
        gt_reproj.append(np.asarray(gt_residuals_by_frame.get(frame_data.frame_index, np.zeros((count,), dtype=np.float64)), dtype=np.float64))
        if frame_data.point_depths is not None:
            depth_vals = np.asarray(frame_data.point_depths, dtype=np.float64).reshape(-1)
        else:
            depth_vals = np.asarray(frame_data.points_3d[:, 2], dtype=np.float64).reshape(-1)
        depth_m.append(depth_vals)
        translation_sensitivity.append(
            np.asarray(
                translation_sensitivity_by_frame.get(frame_data.frame_index, np.ones((count,), dtype=np.float64)),
                dtype=np.float64,
            )
        )
        support_vals = (
            np.asarray(frame_data.support_scores, dtype=np.float64).reshape(-1)
            if frame_data.support_scores is not None
            else np.zeros((count,), dtype=np.float64)
        )
        support_score.append(support_vals)
        frame_rot_delta.append(
            np.full((count,), float(info.get("rotation_delta_deg", 0.0)), dtype=np.float64)
        )
        frame_trans_delta.append(
            np.full((count,), float(info.get("translation_delta_m", 0.0)), dtype=np.float64)
        )
        frame_single_reproj.append(
            np.full((count,), float(info.get("single_frame_reproj_rmse", np.nan)), dtype=np.float64)
        )

    arrays = {
        "frame_id": np.concatenate(frame_ids, axis=0) if frame_ids else np.zeros((0,), dtype=np.int32),
        "frame_index": np.concatenate(frame_indices, axis=0) if frame_indices else np.zeros((0,), dtype=np.int32),
        "rgb_u": np.concatenate(rgb_u, axis=0) if rgb_u else np.zeros((0,), dtype=np.float64),
        "rgb_v": np.concatenate(rgb_v, axis=0) if rgb_v else np.zeros((0,), dtype=np.float64),
        "init_reproj_px": np.concatenate(init_reproj, axis=0) if init_reproj else np.zeros((0,), dtype=np.float64),
        "gt_reproj_px": np.concatenate(gt_reproj, axis=0) if gt_reproj else np.zeros((0,), dtype=np.float64),
        "depth_m": np.concatenate(depth_m, axis=0) if depth_m else np.zeros((0,), dtype=np.float64),
        "translation_sensitivity": (
            np.concatenate(translation_sensitivity, axis=0)
            if translation_sensitivity
            else np.zeros((0,), dtype=np.float64)
        ),
        "support_score": np.concatenate(support_score, axis=0) if support_score else np.zeros((0,), dtype=np.float64),
        "frame_rotation_delta_deg": np.concatenate(frame_rot_delta, axis=0) if frame_rot_delta else np.zeros((0,), dtype=np.float64),
        "frame_translation_delta_m": np.concatenate(frame_trans_delta, axis=0) if frame_trans_delta else np.zeros((0,), dtype=np.float64),
        "frame_single_reproj_rmse": np.concatenate(frame_single_reproj, axis=0) if frame_single_reproj else np.zeros((0,), dtype=np.float64),
    }
    if arrays["gt_reproj_px"].size == 0:
        raise ValueError("No correspondences available for GT weight learning.")

    feature_names = [
        "bias",
        "log_init_reproj",
        "log_depth",
        "log_translation_sensitivity",
        "support_score",
        "log_frame_rot_delta",
        "log_frame_trans_delta_cm",
        "log_frame_single_reproj",
    ]
    feature_matrix = np.stack(
        [
            np.ones_like(arrays["gt_reproj_px"], dtype=np.float64),
            np.log1p(np.clip(arrays["init_reproj_px"], 0.0, None)),
            np.log1p(np.clip(arrays["depth_m"], 0.0, None)),
            np.log1p(np.clip(arrays["translation_sensitivity"], 0.0, None)),
            np.asarray(arrays["support_score"], dtype=np.float64),
            np.log1p(np.clip(arrays["frame_rotation_delta_deg"], 0.0, None)),
            np.log1p(np.clip(arrays["frame_translation_delta_m"], 0.0, None) * 100.0),
            np.log1p(np.clip(arrays["frame_single_reproj_rmse"], 0.0, None)),
        ],
        axis=1,
    )

    gt_values = np.asarray(arrays["gt_reproj_px"], dtype=np.float64)
    pos_threshold = float(np.quantile(gt_values, float(positive_quantile)))
    neg_threshold = float(np.quantile(gt_values, float(negative_quantile)))
    oracle_gt_weights = _gt_residual_to_oracle_weights(
        gt_reproj_px=gt_values,
        positive_threshold=pos_threshold,
        negative_threshold=neg_threshold,
    )
    balance_weights = _translation_sensitivity_to_balance_weights(
        arrays["translation_sensitivity"],
        alpha=translation_sensitivity_alpha,
    )
    supervision_mode = str(supervision_target).strip().lower()
    if supervision_mode not in {"gt_reproj", "gt_pose_balance"}:
        raise ValueError(f"Unsupported supervision_target={supervision_target!r}")
    if supervision_mode == "gt_pose_balance":
        oracle_primary_weights = _normalize_positive_weights(
            oracle_gt_weights * balance_weights,
            min_weight=0.05,
            max_weight=10.0,
        )
        pos_weight_threshold = float(np.quantile(oracle_primary_weights, 1.0 - float(positive_quantile)))
        neg_weight_threshold = float(np.quantile(oracle_primary_weights, 1.0 - float(negative_quantile)))
        train_mask = (oracle_primary_weights >= pos_weight_threshold) | (oracle_primary_weights <= neg_weight_threshold)
        labels = (oracle_primary_weights >= pos_weight_threshold).astype(np.float64)
        positive_threshold_value = pos_weight_threshold
        negative_threshold_value = neg_weight_threshold
    else:
        oracle_primary_weights = oracle_gt_weights.copy()
        train_mask = (gt_values <= pos_threshold) | (gt_values >= neg_threshold)
        labels = (gt_values <= pos_threshold).astype(np.float64)
        positive_threshold_value = pos_threshold
        negative_threshold_value = neg_threshold
    oracle_capped_weights = _normalize_positive_weights(oracle_primary_weights, min_weight=0.2, max_weight=3.0)
    oracle_keep_pos_weights = np.where(labels > 0.5, 1.0, 0.05)
    oracle_keep_pos_weights = _normalize_positive_weights(oracle_keep_pos_weights, min_weight=0.05, max_weight=1.0)
    if supervision_mode == "gt_pose_balance":
        oracle_trim_tail_weights = np.where(
            oracle_primary_weights >= neg_weight_threshold,
            1.0,
            0.05,
        )
    else:
        oracle_trim_tail_weights = np.where(gt_values <= neg_threshold, 1.0, 0.05)
    oracle_trim_tail_weights = _normalize_positive_weights(oracle_trim_tail_weights, min_weight=0.05, max_weight=1.0)
    gt_good_labels = (gt_values <= pos_threshold).astype(np.float64)
    train_labels = labels[train_mask]
    train_features = feature_matrix[train_mask].copy()
    feature_mean = np.zeros((train_features.shape[1],), dtype=np.float64)
    feature_scale = np.ones((train_features.shape[1],), dtype=np.float64)
    if train_features.shape[0] == 0:
        raise ValueError("GT weight learning did not find training samples.")
    if train_features.shape[1] > 1:
        feature_mean[1:] = np.mean(train_features[:, 1:], axis=0)
        feature_scale[1:] = np.std(train_features[:, 1:], axis=0)
        feature_scale[feature_scale < 1.0e-6] = 1.0
        train_features[:, 1:] = (train_features[:, 1:] - feature_mean[1:]) / feature_scale[1:]
    normalized_all = feature_matrix.copy()
    if normalized_all.shape[1] > 1:
        normalized_all[:, 1:] = (normalized_all[:, 1:] - feature_mean[1:]) / feature_scale[1:]

    beta = np.zeros((train_features.shape[1],), dtype=np.float64)
    reg_mask = np.ones_like(beta)
    reg_mask[0] = 0.0
    for _ in range(max(int(num_iters), 1)):
        probs = _sigmoid(train_features @ beta)
        grad = (train_features.T @ (probs - train_labels)) / max(train_features.shape[0], 1)
        grad += float(l2_reg) * reg_mask * beta
        beta -= float(learning_rate) * grad

    learned_weights = _normalize_positive_weights(_sigmoid(normalized_all @ beta))
    arrays["learned_weight"] = learned_weights.astype(np.float64)
    arrays["oracle_gt_weight"] = oracle_gt_weights.astype(np.float64)
    arrays["oracle_pose_weight"] = oracle_primary_weights.astype(np.float64)
    arrays["translation_balance_weight"] = balance_weights.astype(np.float64)
    arrays["train_mask"] = train_mask.astype(np.uint8)
    arrays["gt_good_label"] = gt_good_labels.astype(np.uint8)
    arrays["supervision_good_label"] = labels.astype(np.uint8)
    unique_frame_ids = np.unique(arrays["frame_id"])
    frame_mean_weights = []
    for frame_id in unique_frame_ids:
        mask = arrays["frame_id"] == frame_id
        frame_mean_weights.append(
            {
                "frame_id": int(frame_id),
                "mean_weight": float(np.mean(learned_weights[mask])),
                "num_points": int(np.count_nonzero(mask)),
                "mean_gt_reproj_px": float(np.mean(gt_values[mask])),
            }
        )
    frame_mean_weights.sort(key=lambda item: item["mean_weight"], reverse=True)

    top_indices = np.argsort(-learned_weights, kind="stable")[: min(20, learned_weights.shape[0])]
    top_points = [
        {
            "rank": int(rank + 1),
            "frame_id": int(arrays["frame_id"][idx]),
            "u": float(arrays["rgb_u"][idx]),
            "v": float(arrays["rgb_v"][idx]),
            "weight": float(learned_weights[idx]),
            "gt_reproj_px": float(arrays["gt_reproj_px"][idx]),
            "init_reproj_px": float(arrays["init_reproj_px"][idx]),
            "depth_m": float(arrays["depth_m"][idx]),
            "support_score": float(arrays["support_score"][idx]),
            "frame_rotation_delta_deg": float(arrays["frame_rotation_delta_deg"][idx]),
            "frame_translation_delta_m": float(arrays["frame_translation_delta_m"][idx]),
        }
        for rank, idx in enumerate(top_indices)
    ]

    learned_family_summary = _summarize_weight_family(
        weights=learned_weights,
        arrays=arrays,
        gt_values=gt_values,
    )
    oracle_family_summary = _summarize_weight_family(
        weights=oracle_gt_weights,
        arrays=arrays,
        gt_values=gt_values,
    )
    oracle_capped_summary = _summarize_weight_family(
        weights=oracle_capped_weights,
        arrays=arrays,
        gt_values=gt_values,
    )
    oracle_keep_pos_summary = _summarize_weight_family(
        weights=oracle_keep_pos_weights,
        arrays=arrays,
        gt_values=gt_values,
    )
    oracle_trim_tail_summary = _summarize_weight_family(
        weights=oracle_trim_tail_weights,
        arrays=arrays,
        gt_values=gt_values,
    )
    learned_variant = _evaluate_weighted_solve_variant(
        name="learned_soft",
        frame_data_list=frame_data_list,
        rgb_camera=rgb_camera,
        initial_rvec=init_rvec,
        initial_tvec=init_tvec,
        gt_rvec=gt_rvec,
        gt_tvec=gt_tvec,
        weights=learned_weights,
        gt_pose_residual_weight=gt_pose_residual_weight,
    )
    oracle_variant_results = [
        _evaluate_weighted_solve_variant(
            name="oracle_soft",
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            initial_rvec=init_rvec,
            initial_tvec=init_tvec,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            weights=oracle_primary_weights,
            gt_pose_residual_weight=gt_pose_residual_weight,
        ),
        _evaluate_weighted_solve_variant(
            name="oracle_capped",
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            initial_rvec=init_rvec,
            initial_tvec=init_tvec,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            weights=oracle_capped_weights,
            gt_pose_residual_weight=gt_pose_residual_weight,
        ),
        _evaluate_weighted_solve_variant(
            name="oracle_keep_pos",
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            initial_rvec=init_rvec,
            initial_tvec=init_tvec,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            weights=oracle_keep_pos_weights,
            gt_pose_residual_weight=gt_pose_residual_weight,
        ),
        _evaluate_weighted_solve_variant(
            name="oracle_trim_tail",
            frame_data_list=frame_data_list,
            rgb_camera=rgb_camera,
            initial_rvec=init_rvec,
            initial_tvec=init_tvec,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            weights=oracle_trim_tail_weights,
            gt_pose_residual_weight=gt_pose_residual_weight,
        ),
    ]

    summary = {
        "feature_names": feature_names,
        "supervision_target": supervision_mode,
        "translation_sensitivity_alpha": float(translation_sensitivity_alpha),
        "feature_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, beta)
        },
        "positive_quantile": float(positive_quantile),
        "negative_quantile": float(negative_quantile),
        "positive_threshold_gt_reproj_px": float(pos_threshold),
        "negative_threshold_gt_reproj_px": float(neg_threshold),
        "positive_threshold_supervision_value": float(positive_threshold_value),
        "negative_threshold_supervision_value": float(negative_threshold_value),
        "train_count": int(np.count_nonzero(train_mask)),
        "all_point_count": int(learned_weights.shape[0]),
        "translation_sensitivity_summary": _summarize_array(arrays["translation_sensitivity"]),
        "top_weight_summary": learned_family_summary["top_weight_summary"],
        "bottom_weight_summary": learned_family_summary["bottom_weight_summary"],
        "weight_correlations": learned_family_summary["weight_correlations"],
        "oracle_gt_weight_summary": oracle_family_summary,
        "oracle_pose_weight_summary": _summarize_weight_family(
            weights=oracle_primary_weights,
            arrays=arrays,
            gt_values=gt_values,
        ),
        "oracle_capped_weight_summary": oracle_capped_summary,
        "oracle_keep_pos_weight_summary": oracle_keep_pos_summary,
        "oracle_trim_tail_weight_summary": oracle_trim_tail_summary,
        "oracle_weight_correlations": oracle_family_summary["weight_correlations"],
        "oracle_vs_learned_weight_corr": _corrcoef_safe(oracle_primary_weights, learned_weights),
        "gt_pose_residual_weight": float(gt_pose_residual_weight),
        "learned_weighted_solve": learned_variant,
        "oracle_weighted_solve": oracle_variant_results[0],
        "oracle_weighted_solve_variants": [learned_variant, *oracle_variant_results],
        "top_frames_by_mean_weight": frame_mean_weights[:10],
        "bottom_frames_by_mean_weight": list(reversed(frame_mean_weights[-10:])),
        "top_points": top_points,
    }
    return summary, arrays


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
