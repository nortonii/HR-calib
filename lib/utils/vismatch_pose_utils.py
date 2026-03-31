import cv2
import numpy as np
import torch

from lib.utils.console_utils import blue


def _torch_image_to_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3 and image.shape[0] in {1, 3}:
        return image.float().contiguous()
    if image.dim() == 3 and image.shape[-1] in {1, 3}:
        return image.permute(2, 0, 1).float().contiguous()
    raise ValueError(f"Unsupported image shape for matcher input: {tuple(image.shape)}")


def _depth_percentile_bounds(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1.0e-6:
        low = float(values.min())
        high = float(values.max())
    if abs(high - low) < 1.0e-6:
        high = low + 1.0
    return low, high


class VismatchPoseEstimator:
    def __init__(self, config=None, device: str = "cuda"):
        self.device = device
        self.matcher_name = str(getattr(config, "matcher_name", "xoftr"))
        self.max_num_keypoints = int(getattr(config, "matcher_max_num_keypoints", 2048))
        self.min_matches = int(getattr(config, "matcher_min_matches", 20))
        self.min_depth_matches = int(getattr(config, "matcher_min_depth_matches", 12))
        self.min_pnp_inliers = int(getattr(config, "matcher_min_pnp_inliers", 8))
        self.ransac_reproj_thresh = float(getattr(config, "matcher_ransac_reproj_thresh", 3.0))
        self.pnp_reproj_error = float(getattr(config, "matcher_pnp_reproj_error", 4.0))
        self.pnp_iterations = int(getattr(config, "matcher_pnp_iterations", 1000))
        self.depth_min = float(getattr(config, "matcher_depth_min", 0.1))
        self.depth_max = float(getattr(config, "matcher_depth_max", 80.0))
        self.depth_use_inverse = bool(getattr(config, "matcher_depth_use_inverse", True))
        self.depth_percentile_low = float(getattr(config, "matcher_depth_percentile_low", 5.0))
        self.depth_percentile_high = float(getattr(config, "matcher_depth_percentile_high", 95.0))
        self.update_interval = max(1, int(getattr(config, "matcher_update_interval", 1)))
        self.update_blend = float(getattr(config, "matcher_update_blend", 1.0))
        self._matcher = None

    def _get_matcher(self):
        if self._matcher is None:
            print(blue(f"[Camera] Loading vismatch matcher '{self.matcher_name}' on {self.device} ..."))
            if self.matcher_name == "orb-nn":
                from vismatch.im_models.handcrafted import OrbNNMatcher

                self._matcher = OrbNNMatcher(
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "sift-nn":
                from vismatch.im_models.handcrafted import SiftNNMatcher

                self._matcher = SiftNNMatcher(
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "xoftr":
                from vismatch.im_models.xoftr import XoFTRMatcher

                self._matcher = XoFTRMatcher(
                    device=self.device,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "matchanything-eloftr":
                from vismatch.im_models.matchanything import MatchAnythingMatcher

                self._matcher = MatchAnythingMatcher(
                    device=self.device,
                    variant="eloftr",
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "matchanything-roma":
                from vismatch.im_models.matchanything import MatchAnythingMatcher

                self._matcher = MatchAnythingMatcher(
                    device=self.device,
                    variant="roma",
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "roma":
                from vismatch.im_models.roma import RomaMatcher

                self._matcher = RomaMatcher(
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "tiny-roma":
                from vismatch.im_models.roma import TinyRomaMatcher

                self._matcher = TinyRomaMatcher(
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            elif self.matcher_name == "romav2":
                from vismatch.im_models.romav2 import RoMaV2Matcher

                self._matcher = RoMaV2Matcher(
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
            else:
                from vismatch import get_matcher

                self._matcher = get_matcher(
                    self.matcher_name,
                    device=self.device,
                    max_num_keypoints=self.max_num_keypoints,
                    ransac_reproj_thresh=self.ransac_reproj_thresh,
                )
        return self._matcher

    def _camera_intrinsics(self, camera) -> np.ndarray:
        fx = camera.image_width / (2.0 * np.tan(float(camera.FoVx) * 0.5))
        fy = camera.image_height / (2.0 * np.tan(float(camera.FoVy) * 0.5))
        cx = (float(camera.image_width) - 1.0) * 0.5
        cy = (float(camera.image_height) - 1.0) * 0.5
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def depth_to_match_image(self, depth_map: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        depth_np = depth_map.detach().float().cpu().numpy()
        valid = np.isfinite(depth_np) & (depth_np >= self.depth_min) & (depth_np <= self.depth_max)
        if not np.any(valid):
            empty = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.float32)
            return torch.from_numpy(empty).permute(2, 0, 1), empty

        values = depth_np[valid]
        if self.depth_use_inverse:
            values = 1.0 / np.clip(values, self.depth_min, None)
            display_base = np.zeros_like(depth_np, dtype=np.float32)
            display_base[valid] = 1.0 / np.clip(depth_np[valid], self.depth_min, None)
        else:
            display_base = depth_np.astype(np.float32, copy=True)
        low, high = _depth_percentile_bounds(values, self.depth_percentile_low, self.depth_percentile_high)
        normalized = np.zeros_like(display_base, dtype=np.float32)
        normalized[valid] = np.clip((display_base[valid] - low) / max(high - low, 1.0e-6), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(colored).permute(2, 0, 1), colored

    @staticmethod
    def _sample_depths(depth_map: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.rint(pixels).astype(np.int32)
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < depth_map.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < depth_map.shape[0])
        )
        sampled = np.zeros((pixels.shape[0],), dtype=np.float32)
        if np.any(valid):
            sampled[valid] = depth_map[xy[valid, 1], xy[valid, 0]]
        return sampled, valid

    def _backproject(self, pixels: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        x = (pixels[:, 0] - cx) * depths / fx
        y = (pixels[:, 1] - cy) * depths / fy
        return np.stack((x, y, depths), axis=-1).astype(np.float32)

    @staticmethod
    def _scale_relative_transform(rotation: np.ndarray, translation: np.ndarray, blend: float) -> tuple[np.ndarray, np.ndarray]:
        blend = float(np.clip(blend, 0.0, 1.0))
        if blend >= 1.0:
            return rotation, translation
        rvec, _ = cv2.Rodrigues(rotation)
        scaled_rotation, _ = cv2.Rodrigues(rvec * blend)
        return scaled_rotation.astype(np.float32), (translation * blend).astype(np.float32)

    def estimate_relative_pose(
        self,
        render_depth: torch.Tensor,
        gt_rgb: torch.Tensor,
        camera,
    ) -> dict:
        depth_input, depth_vis = self.depth_to_match_image(render_depth)
        rgb_input = _torch_image_to_chw(gt_rgb.detach().cpu()).clamp(0.0, 1.0)
        matcher = self._get_matcher()
        match_result = matcher(depth_input, rgb_input)

        matched_kpts0 = match_result["matched_kpts0"]
        matched_kpts1 = match_result["matched_kpts1"]
        inlier_kpts0 = match_result["inlier_kpts0"]
        inlier_kpts1 = match_result["inlier_kpts1"]

        use_h_inliers = len(inlier_kpts0) >= self.min_matches
        source_pixels = inlier_kpts0 if use_h_inliers else matched_kpts0
        target_pixels = inlier_kpts1 if use_h_inliers else matched_kpts1

        if source_pixels.shape[0] < self.min_matches:
            return {
                "success": False,
                "num_matches": int(source_pixels.shape[0]),
                "num_raw_matches": int(matched_kpts0.shape[0]),
                "num_h_inliers": int(inlier_kpts0.shape[0]),
                "num_depth_matches": 0,
                "num_pnp_inliers": 0,
                "mean_reproj_error": 0.0,
                "depth_vis": depth_vis,
            }

        depth_np = render_depth.detach().float().cpu().numpy()
        sampled_depths, depth_valid = self._sample_depths(depth_np, source_pixels)
        depth_valid &= np.isfinite(sampled_depths)
        depth_valid &= sampled_depths >= self.depth_min
        depth_valid &= sampled_depths <= self.depth_max

        source_pixels = source_pixels[depth_valid]
        target_pixels = target_pixels[depth_valid]
        sampled_depths = sampled_depths[depth_valid]

        if source_pixels.shape[0] < self.min_depth_matches:
            return {
                "success": False,
                "num_matches": int(source_pixels.shape[0]),
                "num_raw_matches": int(matched_kpts0.shape[0]),
                "num_h_inliers": int(inlier_kpts0.shape[0]),
                "num_depth_matches": int(source_pixels.shape[0]),
                "num_pnp_inliers": 0,
                "mean_reproj_error": 0.0,
                "depth_vis": depth_vis,
            }

        intrinsics = self._camera_intrinsics(camera)
        object_points = self._backproject(source_pixels, sampled_depths, intrinsics)
        image_points = target_pixels.astype(np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            intrinsics,
            None,
            iterationsCount=self.pnp_iterations,
            reprojectionError=self.pnp_reproj_error,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < self.min_pnp_inliers:
            return {
                "success": False,
                "num_matches": int(source_pixels.shape[0]),
                "num_raw_matches": int(matched_kpts0.shape[0]),
                "num_h_inliers": int(inlier_kpts0.shape[0]),
                "num_depth_matches": int(source_pixels.shape[0]),
                "num_pnp_inliers": 0 if inliers is None else int(len(inliers)),
                "mean_reproj_error": 0.0,
                "depth_vis": depth_vis,
            }

        inlier_indices = inliers.reshape(-1)
        object_points_inliers = object_points[inlier_indices]
        image_points_inliers = image_points[inlier_indices]
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points_inliers,
                image_points_inliers,
                intrinsics,
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        rotation_cv, _ = cv2.Rodrigues(rvec)
        translation_cv = tvec.reshape(3).astype(np.float32)
        projected, _ = cv2.projectPoints(object_points_inliers, rvec, tvec, intrinsics, None)
        projected = projected.reshape(-1, 2)
        mean_reproj_error = float(np.linalg.norm(projected - image_points_inliers, axis=-1).mean())
        rotation_cv, translation_cv = self._scale_relative_transform(
            rotation_cv.astype(np.float32),
            translation_cv,
            self.update_blend,
        )
        return {
            "success": True,
            "num_matches": int(source_pixels.shape[0]),
            "num_raw_matches": int(matched_kpts0.shape[0]),
            "num_h_inliers": int(inlier_kpts0.shape[0]),
            "num_depth_matches": int(source_pixels.shape[0]),
            "num_pnp_inliers": int(len(inlier_indices)),
            "mean_reproj_error": mean_reproj_error,
            "relative_rotation": torch.from_numpy(rotation_cv),
            "relative_translation": torch.from_numpy(translation_cv),
            "depth_vis": depth_vis,
        }

    def aggregate_relative_pose_estimates(self, match_results: list[dict]) -> dict:
        attempted_frames = len(match_results)
        if attempted_frames == 0:
            return {
                "success": False,
                "attempted_frames": 0,
                "successful_frames": 0,
                "success_ratio": 0.0,
                "mean_num_raw_matches": 0.0,
                "mean_num_matches": 0.0,
                "mean_num_h_inliers": 0.0,
                "mean_num_depth_matches": 0.0,
                "mean_num_pnp_inliers": 0.0,
                "mean_reproj_error": 0.0,
                "best_depth_vis": None,
            }

        successful = [result for result in match_results if result.get("success", False)]
        best_result = max(
            match_results,
            key=lambda result: (result.get("num_pnp_inliers", 0), result.get("num_depth_matches", 0)),
        )

        aggregate = {
            "success": len(successful) > 0,
            "attempted_frames": attempted_frames,
            "successful_frames": len(successful),
            "success_ratio": float(len(successful)) / float(max(attempted_frames, 1)),
            "mean_num_raw_matches": float(np.mean([result.get("num_raw_matches", 0) for result in match_results])),
            "mean_num_matches": float(np.mean([result.get("num_matches", 0) for result in match_results])),
            "mean_num_h_inliers": float(np.mean([result.get("num_h_inliers", 0) for result in match_results])),
            "mean_num_depth_matches": float(np.mean([result.get("num_depth_matches", 0) for result in match_results])),
            "mean_num_pnp_inliers": float(np.mean([result.get("num_pnp_inliers", 0) for result in match_results])),
            "mean_reproj_error": float(
                np.mean(
                    [
                        result.get("mean_reproj_error", 0.0)
                        for result in successful
                    ]
                )
            ) if successful else 0.0,
            "best_depth_vis": best_result.get("depth_vis"),
            "best_frame_id": best_result.get("frame_id", -1),
        }
        if not successful:
            return aggregate

        weights = np.asarray(
            [max(float(result.get("num_pnp_inliers", 0)), 1.0) for result in successful],
            dtype=np.float32,
        )
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0:
            weights = np.ones_like(weights, dtype=np.float32)
            weights_sum = float(weights.sum())
        weights /= weights_sum

        weighted_translation = np.zeros((3,), dtype=np.float32)
        weighted_rvec = np.zeros((3,), dtype=np.float32)
        for weight, result in zip(weights, successful, strict=False):
            rotation_np = result["relative_rotation"].detach().cpu().numpy().astype(np.float32)
            translation_np = result["relative_translation"].detach().cpu().numpy().astype(np.float32)
            rvec, _ = cv2.Rodrigues(rotation_np)
            weighted_rvec += float(weight) * rvec.reshape(3)
            weighted_translation += float(weight) * translation_np.reshape(3)

        aggregated_rotation, _ = cv2.Rodrigues(weighted_rvec.reshape(3, 1))
        aggregate["relative_rotation"] = torch.from_numpy(aggregated_rotation.astype(np.float32))
        aggregate["relative_translation"] = torch.from_numpy(weighted_translation.astype(np.float32))
        return aggregate
