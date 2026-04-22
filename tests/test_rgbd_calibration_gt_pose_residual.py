import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tools.calib import (
    _camera_rt_from_lidar_to_camera,
    _compute_camera_depth_normal_consistency_loss,
    _compute_lidar_depth_loss,
    _compute_lidar_normal_loss,
    _depth_to_camera_normal_map,
    _effective_T,
    _should_run_initial_pure_pnp,
)
from lib.scene.camera_pose_correction import CameraPoseCorrection
from lib.utils.rgbd_calibration import (
    CameraModel,
    FrameCalibrationData,
    _rotation_delta_deg_from_rvecs,
    _translation_sensitivity_to_balance_weights,
    compute_frame_translation_sensitivity,
    initialize_shared_extrinsic,
    optimize_shared_extrinsic,
)


class GTPoseResidualTests(unittest.TestCase):
    def test_lidar_normal_loss_is_zero_for_identical_normals(self):
        normal = torch.zeros((5, 5, 3), dtype=torch.float32)
        normal[..., 2] = -1.0
        valid = torch.ones((5, 5), dtype=torch.bool)
        loss = _compute_lidar_normal_loss(
            pred_normal_map=normal,
            gt_normal_map=normal,
            valid_mask=valid,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_depth_to_camera_normal_map_recovers_fronto_parallel_plane(self):
        camera = SimpleNamespace(
            K=np.array(
                [
                    [100.0, 0.0, 2.0],
                    [0.0, 100.0, 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        depth = torch.full((5, 5), 4.0, dtype=torch.float32)

        normal_map, valid = _depth_to_camera_normal_map(depth, camera)

        self.assertTrue(bool(valid[2, 2]))
        np.testing.assert_allclose(
            normal_map[2, 2].numpy(),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            atol=1.0e-4,
        )

    def test_camera_depth_normal_consistency_is_zero_for_matching_plane_normals(self):
        camera = SimpleNamespace(
            K=np.array(
                [
                    [100.0, 0.0, 2.0],
                    [0.0, 100.0, 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        depth = torch.full((5, 5), 4.0, dtype=torch.float32)
        pred_normal = torch.zeros((5, 5, 3), dtype=torch.float32)
        pred_normal[..., 2] = -1.0

        loss = _compute_camera_depth_normal_consistency_loss(
            pred_depth_map=depth,
            pred_normal_map=pred_normal,
            camera=camera,
        )

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_initial_pure_pnp_runs_on_last_cycle_when_warmup_consumes_all_cycles(self):
        self.assertTrue(
            _should_run_initial_pure_pnp(
                run_pending=True,
                cycle=50,
                warmup_cycles=50,
                total_cycles=50,
            )
        )
        self.assertFalse(
            _should_run_initial_pure_pnp(
                run_pending=True,
                cycle=49,
                warmup_cycles=50,
                total_cycles=50,
            )
        )

    def test_initial_pure_pnp_runs_one_cycle_after_warmup_when_extra_cycle_exists(self):
        self.assertTrue(
            _should_run_initial_pure_pnp(
                run_pending=True,
                cycle=21,
                warmup_cycles=20,
                total_cycles=21,
            )
        )
        self.assertFalse(
            _should_run_initial_pure_pnp(
                run_pending=True,
                cycle=20,
                warmup_cycles=20,
                total_cycles=21,
            )
        )

    def test_visibility_weighted_lidar_depth_loss_downweights_occluded_hits(self):
        pred = torch.tensor([5.1, 12.0, 0.0], dtype=torch.float32)
        gt = torch.tensor([5.0, 8.0, 4.0], dtype=torch.float32)
        accum = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

        weighted = _compute_lidar_depth_loss(
            pred,
            gt,
            accum,
            use_visibility_weights=True,
            visible_weight=2.0,
            occluded_weight=0.5,
            outside_weight=1.0,
            visibility_tolerance=0.25,
        )
        unweighted = _compute_lidar_depth_loss(pred, gt, accum, use_visibility_weights=False)

        self.assertLess(float(weighted), float(unweighted))

    def test_inverse_depth_loss_emphasizes_near_errors(self):
        pred = torch.tensor([2.0, 20.0], dtype=torch.float32)
        gt = torch.tensor([1.0, 19.0], dtype=torch.float32)

        loss = _compute_lidar_depth_loss(
            pred,
            gt,
            loss_mode="inverse_depth",
            inverse_min_depth=0.5,
            use_visibility_weights=False,
        )

        expected = 0.5 * (
            abs(1.0 / 2.0 - 1.0 / 1.0)
            + abs(1.0 / 20.0 - 1.0 / 19.0)
        )
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_translation_sensitivity_prefers_near_points(self):
        camera = CameraModel(
            K=np.array(
                [
                    [700.0, 0.0, 320.0],
                    [0.0, 700.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist=np.zeros((0,), dtype=np.float64),
        )
        frame = FrameCalibrationData(
            frame_name="synthetic",
            rgb_path="",
            depth_path="",
            rgb_points=np.array([[320.0, 240.0], [330.0, 240.0]], dtype=np.float64),
            depth_points=np.zeros((2, 2), dtype=np.float64),
            points_3d=np.array(
                [
                    [0.5, 0.0, 5.0],
                    [0.5, 0.0, 20.0],
                ],
                dtype=np.float64,
            ),
            frame_index=0,
            frame_id=0,
        )
        sensitivity = compute_frame_translation_sensitivity(
            frame_data_list=[frame],
            rgb_camera=camera,
            rvec=np.zeros((3,), dtype=np.float64),
            tvec=np.zeros((3,), dtype=np.float64),
        )[0]
        self.assertGreater(float(sensitivity[0]), float(sensitivity[1]))

        balance = _translation_sensitivity_to_balance_weights(sensitivity, alpha=0.5)
        self.assertGreater(float(balance[0]), float(balance[1]))

    def test_gt_pose_residual_pulls_solution_toward_gt(self):
        camera = CameraModel(
            K=np.array(
                [
                    [700.0, 0.0, 320.0],
                    [0.0, 700.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist=np.zeros((0,), dtype=np.float64),
        )
        points_3d = np.array(
            [
                [-1.0, -0.5, 12.0],
                [-0.5, 0.3, 11.5],
                [0.2, -0.4, 10.8],
                [0.8, 0.7, 13.2],
                [1.1, -0.2, 14.5],
                [-0.9, 0.9, 15.0],
                [0.4, 0.1, 9.8],
                [1.3, -0.6, 12.8],
            ],
            dtype=np.float64,
        )

        measurement_rvec = np.array([0.025, -0.018, 0.012], dtype=np.float64)
        measurement_tvec = np.array([0.18, -0.04, 0.09], dtype=np.float64)
        gt_rvec = np.zeros((3,), dtype=np.float64)
        gt_tvec = np.zeros((3,), dtype=np.float64)

        rgb_points, _ = cv2.projectPoints(
            points_3d,
            measurement_rvec.reshape(3, 1),
            measurement_tvec.reshape(3, 1),
            camera.K,
            None,
        )
        frame = FrameCalibrationData(
            frame_name="synthetic",
            rgb_path="",
            depth_path="",
            rgb_points=rgb_points.reshape(-1, 2).astype(np.float64),
            depth_points=np.zeros((points_3d.shape[0], 2), dtype=np.float64),
            points_3d=points_3d,
            frame_index=0,
            frame_id=0,
        )

        baseline = optimize_shared_extrinsic(
            frame_data_list=[frame],
            rgb_camera=camera,
            initial_rvec=measurement_rvec,
            initial_tvec=measurement_tvec,
        )
        gt_supervised = optimize_shared_extrinsic(
            frame_data_list=[frame],
            rgb_camera=camera,
            initial_rvec=measurement_rvec,
            initial_tvec=measurement_tvec,
            gt_rvec=gt_rvec,
            gt_tvec=gt_tvec,
            gt_pose_residual_weight=1.0e5,
        )

        baseline_rot_to_gt = _rotation_delta_deg_from_rvecs(baseline.optimized.rotation_vector, gt_rvec)
        guided_rot_to_gt = _rotation_delta_deg_from_rvecs(gt_supervised.optimized.rotation_vector, gt_rvec)
        baseline_trans_to_gt = np.linalg.norm(baseline.optimized.translation - gt_tvec)
        guided_trans_to_gt = np.linalg.norm(gt_supervised.optimized.translation - gt_tvec)

        self.assertLess(guided_rot_to_gt, baseline_rot_to_gt)
        self.assertLess(guided_trans_to_gt, baseline_trans_to_gt)
        self.assertGreater(
            gt_supervised.optimized.mean_reprojection_error,
            baseline.optimized.mean_reprojection_error,
        )

    def test_opencv_backend_refines_shared_pose(self):
        camera = CameraModel(
            K=np.array(
                [
                    [700.0, 0.0, 320.0],
                    [0.0, 700.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist=np.zeros((0,), dtype=np.float64),
        )
        points_3d = np.array(
            [
                [-1.2, -0.6, 10.0],
                [-0.8, 0.4, 11.0],
                [-0.2, -0.5, 9.5],
                [0.5, 0.7, 12.0],
                [0.9, -0.3, 13.5],
                [-1.0, 0.8, 14.0],
                [0.3, 0.1, 10.5],
                [1.1, -0.7, 12.5],
            ],
            dtype=np.float64,
        )
        gt_rvec = np.array([0.02, -0.015, 0.01], dtype=np.float64)
        gt_tvec = np.array([0.1, -0.02, 0.08], dtype=np.float64)
        initial_rvec = gt_rvec + np.array([0.04, -0.03, 0.02], dtype=np.float64)
        initial_tvec = gt_tvec + np.array([0.06, -0.01, 0.05], dtype=np.float64)

        rgb_points, _ = cv2.projectPoints(
            points_3d,
            gt_rvec.reshape(3, 1),
            gt_tvec.reshape(3, 1),
            camera.K,
            None,
        )
        frame = FrameCalibrationData(
            frame_name="synthetic-opencv",
            rgb_path="",
            depth_path="",
            rgb_points=rgb_points.reshape(-1, 2).astype(np.float64),
            depth_points=np.zeros((points_3d.shape[0], 2), dtype=np.float64),
            points_3d=points_3d,
            frame_index=0,
            frame_id=0,
        )

        comparison = optimize_shared_extrinsic(
            frame_data_list=[frame],
            rgb_camera=camera,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
            solver_backend="opencv",
        )

        self.assertLess(
            np.linalg.norm(comparison.optimized.translation - gt_tvec),
            np.linalg.norm(initial_tvec - gt_tvec),
        )
        self.assertGreater(comparison.mean_reprojection_improvement_px, 0.0)

    def test_translation_only_shared_optimization_keeps_rotation_fixed(self):
        camera = CameraModel(
            K=np.array(
                [
                    [700.0, 0.0, 320.0],
                    [0.0, 700.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist=np.zeros((0,), dtype=np.float64),
        )
        points_3d = np.array(
            [
                [-1.0, -0.5, 12.0],
                [-0.5, 0.3, 11.5],
                [0.2, -0.4, 10.8],
                [0.8, 0.7, 13.2],
                [1.1, -0.2, 14.5],
                [-0.9, 0.9, 15.0],
                [0.4, 0.1, 9.8],
                [1.3, -0.6, 12.8],
            ],
            dtype=np.float64,
        )
        gt_rvec = np.array([0.02, -0.015, 0.01], dtype=np.float64)
        gt_tvec = np.array([0.1, -0.02, 0.08], dtype=np.float64)
        initial_tvec = gt_tvec + np.array([0.08, -0.03, 0.04], dtype=np.float64)

        rgb_points, _ = cv2.projectPoints(
            points_3d,
            gt_rvec.reshape(3, 1),
            gt_tvec.reshape(3, 1),
            camera.K,
            None,
        )
        frame = FrameCalibrationData(
            frame_name="synthetic-translation-only",
            rgb_path="",
            depth_path="",
            rgb_points=rgb_points.reshape(-1, 2).astype(np.float64),
            depth_points=np.zeros((points_3d.shape[0], 2), dtype=np.float64),
            points_3d=points_3d,
            frame_index=0,
            frame_id=0,
        )

        comparison = optimize_shared_extrinsic(
            frame_data_list=[frame],
            rgb_camera=camera,
            initial_rvec=gt_rvec,
            initial_tvec=initial_tvec,
            optimize_rotation=False,
            optimize_translation=True,
        )

        self.assertAlmostEqual(
            _rotation_delta_deg_from_rvecs(comparison.optimized.rotation_vector, gt_rvec),
            0.0,
            places=6,
        )
        self.assertLess(
            np.linalg.norm(comparison.optimized.translation - gt_tvec),
            np.linalg.norm(initial_tvec - gt_tvec),
        )

    def test_shared_initialization_can_preserve_zero_inlier_frames(self):
        camera = CameraModel(
            K=np.array(
                [
                    [700.0, 0.0, 320.0],
                    [0.0, 700.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist=np.zeros((0,), dtype=np.float64),
        )
        points_3d = np.array(
            [
                [-1.2, -0.6, 10.0],
                [-0.8, 0.4, 11.0],
                [-0.2, -0.5, 9.5],
                [0.5, 0.7, 12.0],
                [0.9, -0.3, 13.5],
                [-1.0, 0.8, 14.0],
                [0.3, 0.1, 10.5],
                [1.1, -0.7, 12.5],
                [-0.4, 1.0, 15.0],
                [0.7, -1.1, 16.0],
            ],
            dtype=np.float64,
        )
        gt_rvec = np.array([0.015, -0.01, 0.02], dtype=np.float64)
        gt_tvec = np.array([0.12, -0.03, 0.08], dtype=np.float64)
        rgb_points_good, _ = cv2.projectPoints(
            points_3d,
            gt_rvec.reshape(3, 1),
            gt_tvec.reshape(3, 1),
            camera.K,
            None,
        )
        frame_good = FrameCalibrationData(
            frame_name="good",
            rgb_path="",
            depth_path="",
            rgb_points=rgb_points_good.reshape(-1, 2).astype(np.float64),
            depth_points=np.zeros((points_3d.shape[0], 2), dtype=np.float64),
            points_3d=points_3d.copy(),
            frame_index=0,
            frame_id=0,
        )
        frame_bad = FrameCalibrationData(
            frame_name="bad",
            rgb_path="",
            depth_path="",
            rgb_points=(rgb_points_good.reshape(-1, 2) + np.array([180.0, -140.0], dtype=np.float64)).astype(np.float64),
            depth_points=np.zeros((points_3d.shape[0], 2), dtype=np.float64),
            points_3d=points_3d.copy(),
            frame_index=1,
            frame_id=1,
        )

        frames = [frame_good, frame_bad]
        rvec, tvec = initialize_shared_extrinsic(
            frame_data_list=frames,
            rgb_camera=camera,
            reproj_error=2.0,
            iterations=500,
            min_inliers=8,
            filter_frames=False,
        )

        self.assertEqual(len(frames), 2)
        self.assertGreater(frame_good.pnp_inliers, 0)
        self.assertEqual(frame_bad.pnp_inliers, 0)
        self.assertTrue(np.isfinite(frame_good.pnp_reproj_error))
        self.assertTrue(np.isnan(frame_bad.pnp_reproj_error))
        self.assertLess(_rotation_delta_deg_from_rvecs(rvec, gt_rvec), 0.5)
        self.assertLess(np.linalg.norm(tvec - gt_tvec), 0.1)


class SharedExtrinsicPoseCorrectionTests(unittest.TestCase):
    def _make_pose_correction(self, init_se3):
        camera = SimpleNamespace(
            R=torch.eye(3, dtype=torch.float32),
            T=torch.zeros(3, dtype=torch.float32),
        )
        config = SimpleNamespace(
            mode="all",
            init_translation_std=0.0,
            init_rotation_deg=0.0,
            init_se3=init_se3,
        )
        lidar_pose = torch.eye(4, dtype=torch.float32)
        return CameraPoseCorrection(
            cameras={0: camera},
            config=config,
            lidar_poses={0: lidar_pose},
        )

    def test_corrected_lidar_to_camera_uses_decoupled_translation(self):
        pose_correction = self._make_pose_correction([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2.0])
        base_translation = (
            pose_correction.base_lidar_to_camera_translation[0].detach().cpu().numpy().copy()
        )

        _, extrinsic_translation = pose_correction.corrected_lidar_to_camera(0, device="cpu")

        np.testing.assert_allclose(
            extrinsic_translation.detach().cpu().numpy(),
            base_translation,
            atol=1.0e-5,
        )

    def test_relative_camera_transform_preserves_gt_translation(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, 0.0])
        pose_correction.use_gt_translation = True
        gt_translation = (
            pose_correction.gt_lidar_to_camera_translation[0].detach().cpu().numpy().copy()
        )

        pose_correction.apply_relative_camera_transform(
            0,
            torch.eye(3, dtype=torch.float32),
            torch.tensor([5.0, -4.0, 3.0], dtype=torch.float32),
        )

        _, extrinsic_translation = pose_correction.corrected_lidar_to_camera(0, device="cpu")
        np.testing.assert_allclose(
            extrinsic_translation.detach().cpu().numpy(),
            gt_translation,
            atol=1.0e-6,
        )

    def test_set_lidar_to_camera_roundtrip_preserves_translation(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, np.pi / 2.0])
        target_rotation = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        target_translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        pose_correction.set_lidar_to_camera(0, target_rotation, target_translation)

        read_rotation, read_translation = pose_correction.corrected_lidar_to_camera(0, device="cpu")
        np.testing.assert_allclose(
            read_rotation.detach().cpu().numpy(),
            target_rotation.detach().cpu().numpy(),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            read_translation.detach().cpu().numpy(),
            target_translation.detach().cpu().numpy(),
            atol=1.0e-6,
        )

    def test_delta_pose_error_zero_for_gt_shared_extrinsic(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, np.pi / 2.0])
        gt_rotation = pose_correction.gt_lidar_to_camera_rotation[0].detach().cpu()
        gt_translation = pose_correction.gt_lidar_to_camera_translation[0].detach().cpu()

        pose_correction.set_lidar_to_camera(0, gt_rotation, gt_translation)

        error = pose_correction.delta_pose_error(0, device="cpu")
        self.assertLess(error["rotation_error_deg"].detach().cpu().item(), 1.0e-5)
        self.assertLess(error["translation_error_norm"].detach().cpu().item(), 1.0e-6)

    def test_shared_extrinsic_error_zero_for_gt_shared_extrinsic(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, np.pi / 2.0])
        gt_rotation = pose_correction.gt_lidar_to_camera_rotation[0].detach().cpu()
        gt_translation = pose_correction.gt_lidar_to_camera_translation[0].detach().cpu()

        pose_correction.set_lidar_to_camera(0, gt_rotation, gt_translation)

        error = pose_correction.shared_extrinsic_error(device="cpu")
        self.assertLess(error["rotation_error_deg"].detach().cpu().item(), 1.0e-5)
        self.assertLess(error["translation_error_norm"].detach().cpu().item(), 1.0e-6)

    def test_effective_translation_respects_gt_translation_mode(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, 0.0])
        pose_correction.use_gt_translation = True
        pose_correction.delta_translations.data[0] = torch.tensor(
            [5.0, -4.0, 3.0],
            dtype=torch.float32,
        )

        eff_translation = _effective_T(pose_correction)

        np.testing.assert_allclose(
            eff_translation.detach().cpu().numpy(),
            pose_correction.gt_lidar_to_camera_translation[0].detach().cpu().numpy(),
            atol=1.0e-6,
        )

    def test_camera_rt_from_lidar_to_camera_matches_corrected_rt(self):
        pose_correction = self._make_pose_correction([0.2, -0.1, 0.05, 0.0, 0.0, 0.1])
        extrinsic_rotation, extrinsic_translation = pose_correction.corrected_lidar_to_camera(
            0,
            device="cpu",
        )

        helper_rotation, helper_translation = _camera_rt_from_lidar_to_camera(
            pose_correction,
            0,
            extrinsic_rotation,
            extrinsic_translation,
            device="cpu",
        )
        corrected_rotation, corrected_translation = pose_correction.corrected_rt(
            0,
            device="cpu",
        )

        np.testing.assert_allclose(
            helper_rotation.detach().cpu().numpy(),
            corrected_rotation.detach().cpu().numpy(),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            helper_translation.detach().cpu().numpy(),
            corrected_translation.detach().cpu().numpy(),
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
