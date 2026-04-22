import unittest

import numpy as np
import torch

from lib.utils.kitti_utils import (
    _compute_kitti_column_indices,
    build_kitti_range_image_from_points,
    interpolate_sensor2world_columns,
    pano_to_lidar_with_intensities,
    resolve_kitti_lidar_width,
)
from lib.utils.velodyne_utils import (
    KITTI_HDL64E_VERT_DEG_TOP_TO_BOTTOM,
    assign_inclinations_to_rows,
    get_kitti_hdl64e_beam_inclinations_rad,
)
from lib.scene.lidar_sensor import LiDARSensor
from lib.dataloader.gs_loader import (
    _filter_points_by_min_distance,
    _resolve_init_scale,
    _resolve_inverse_distance_init_num,
)


class KittiLidarBeamTests(unittest.TestCase):
    def test_kitti_beam_layout_is_nonuniform(self):
        angles_deg = KITTI_HDL64E_VERT_DEG_TOP_TO_BOTTOM
        self.assertEqual(len(angles_deg), 64)
        self.assertGreater(float(angles_deg[0]), float(angles_deg[-1]))
        diffs = np.diff(angles_deg)
        self.assertFalse(np.allclose(diffs, diffs[0]))

    def test_row_assignment_matches_exact_beam_angles(self):
        top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(order="top_to_bottom")
        sample_rows = np.array([0, 7, 31, 32, 48, 63], dtype=np.int32)
        sample_angles = top_to_bottom[sample_rows]
        row_idx, valid = assign_inclinations_to_rows(sample_angles, top_to_bottom)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.array_equal(row_idx, sample_rows))

    def test_pano_to_lidar_accepts_per_beam_intrinsics(self):
        top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(order="top_to_bottom")
        pano = np.zeros((64, 8), dtype=np.float32)
        intensities = np.zeros_like(pano)
        pano[0, 0] = 10.0
        pano[31, 4] = 15.0
        pano[63, 7] = 20.0
        intensities[pano > 0] = 1.0
        points = pano_to_lidar_with_intensities(pano, intensities, top_to_bottom)
        self.assertEqual(points.shape, (3, 4))
        self.assertTrue(np.all(points[:, 3] == 1.0))

    def test_lidar_sensor_raw_points_world_applies_sensor_pose(self):
        sensor = LiDARSensor(
            sensor2ego=torch.eye(4),
            name="velo",
            inclination_bounds=[-0.1, 0.1],
            data_type="KITTICalib",
        )
        ego2world = torch.eye(4)
        ego2world[:3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        range_image = torch.zeros((2, 2, 2), dtype=torch.float32)
        raw_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]], dtype=torch.float32)
        raw_intensity = torch.tensor([0.1, 0.9], dtype=torch.float32)
        sensor.add_frame(
            frame=0,
            ego2world=ego2world,
            r1=range_image,
            r2=range_image,
            raw_points=raw_points,
            raw_intensity=raw_intensity,
        )

        points_world, intensity = sensor.get_raw_points(0, world=True)

        np.testing.assert_allclose(
            points_world.numpy(),
            np.array([[1.0, 2.0, 3.0], [2.0, 0.0, 6.0]], dtype=np.float32),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            intensity.numpy(),
            np.array([0.1, 0.9], dtype=np.float32),
            atol=1.0e-6,
        )

    def test_column_pose_interpolation_uses_prev_to_current_endpoints(self):
        prev_pose = np.eye(4, dtype=np.float32)
        current_pose = np.eye(4, dtype=np.float32)
        prev_pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        current_pose[:3, 3] = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        pose_map = interpolate_sensor2world_columns(prev_pose, current_pose, width=5)
        self.assertTrue(np.allclose(pose_map[0], prev_pose, atol=1.0e-6))
        self.assertTrue(np.allclose(pose_map[-1], current_pose, atol=1.0e-6))

    def test_kitti_range_builder_keeps_empty_pixels_directionless(self):
        top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(order="top_to_bottom")
        sensor2world = np.eye(4, dtype=np.float32)
        depth_map, intensity_map, ray_direction = build_kitti_range_image_from_points(
            xyzs=np.zeros((0, 3), dtype=np.float32),
            intensities=np.zeros((0,), dtype=np.float32),
            beam_inclinations_top_to_bottom=top_to_bottom,
            width=8,
            sensor2world=sensor2world,
        )
        self.assertEqual(depth_map.shape, (64, 8))
        self.assertEqual(intensity_map.shape, (64, 8))
        self.assertEqual(ray_direction.shape, (64, 8, 3))
        self.assertTrue(np.allclose(ray_direction, 0.0, atol=1.0e-6))

    def test_kitti_range_builder_applies_sensor_rotation_consistently(self):
        top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(order="top_to_bottom")
        sensor2world = np.eye(4, dtype=np.float32)
        sensor2world[:3, :3] = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        sensor2world[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        xyzs = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        intensities = np.array([0.5], dtype=np.float32)

        depth_map, intensity_map, ray_direction = build_kitti_range_image_from_points(
            xyzs=xyzs,
            intensities=intensities,
            beam_inclinations_top_to_bottom=top_to_bottom,
            width=2048,
            sensor2world=sensor2world,
        )

        hit_rows, hit_cols = np.nonzero(depth_map > 0.0)
        self.assertEqual(hit_rows.size, 1)
        self.assertEqual(hit_cols.size, 1)
        hit_row = int(hit_rows[0])
        hit_col = int(hit_cols[0])
        self.assertAlmostEqual(float(depth_map[hit_row, hit_col]), 10.0, places=5)
        self.assertAlmostEqual(float(intensity_map[hit_row, hit_col]), 0.5, places=6)
        np.testing.assert_allclose(
            ray_direction[hit_row, hit_col],
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            atol=1.0e-6,
        )

    def test_kitti_column_indices_wrap_seam(self):
        col = _compute_kitti_column_indices(
            azimuth=np.array([np.pi, -np.pi, -np.pi + 1.0e-6], dtype=np.float32),
            width=16,
            azimuth_left=np.pi,
            azimuth_right=-np.pi,
        )
        np.testing.assert_array_equal(col, np.array([0, 0, 0], dtype=np.int64))

    def test_resolve_kitti_lidar_width_prefers_explicit_override(self):
        class Nested:
            pass

        class Args:
            pass

        args = Args()
        args.data = Nested()
        args.data.kitti_lidar_width = 1536
        self.assertEqual(resolve_kitti_lidar_width(args), 1536)
        args.kitti_lidar_width = 2048
        self.assertEqual(resolve_kitti_lidar_width(args), 2048)

    def test_resolve_init_scale_prefers_configured_value(self):
        self.assertAlmostEqual(_resolve_init_scale(0.12, 0.075), 0.12, places=6)
        self.assertAlmostEqual(_resolve_init_scale(0.0, 0.075), 0.075, places=6)
        self.assertAlmostEqual(_resolve_init_scale(-1.0, 0.0), 1.0e-3, places=9)

    def test_filter_points_by_min_distance_removes_near_sensor_points(self):
        points = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 0.0, 2.1],
            ],
            dtype=torch.float32,
        )
        intensity = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        filtered_points, filtered_intensity, filtered_normals = _filter_points_by_min_distance(
            points,
            intensity,
            sensor_center=torch.zeros(3, dtype=torch.float32),
            min_distance=2.0,
            normals=normals,
        )

        np.testing.assert_allclose(
            filtered_points.numpy(),
            np.array([[0.0, 0.0, 2.1]], dtype=np.float32),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            filtered_intensity.numpy(),
            np.array([0.3], dtype=np.float32),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            filtered_normals.numpy(),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            atol=1.0e-6,
        )

    def test_resolve_inverse_distance_init_num_allows_kitticalib_override(self):
        class Nested:
            pass

        class Args:
            pass

        args = Args()
        args.data_type = "KITTICalib"
        args.model = Nested()
        args.model.inverse_distance_init_num = 10000
        args.model.kitticalib_inverse_distance_init_num = 0
        self.assertEqual(_resolve_inverse_distance_init_num(args), 0)

        args.data_type = "KITTI"
        self.assertEqual(_resolve_inverse_distance_init_num(args), 10000)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for LiDARSensor ray tests")
    def test_lidar_sensor_prefers_explicit_ray_directions(self):
        sensor = LiDARSensor(
            sensor2ego=torch.eye(4),
            name="velo",
            inclination_bounds=[-0.1, 0.1],
            data_type="KITTICalib",
        )
        range_image = torch.zeros((2, 2, 2), dtype=torch.float32)
        ray_direction = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        sensor.add_frame(
            frame=0,
            ego2world=torch.eye(4),
            r1=range_image,
            r2=range_image,
            ray_direction=ray_direction,
        )
        _, rays_d = sensor.get_range_rays(0)
        rays_d = rays_d.cpu()
        expected = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(rays_d, expected, atol=1.0e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for LiDARSensor inverse projection tests")
    def test_inverse_projection_ignores_zero_depth_pixels(self):
        sensor = LiDARSensor(
            sensor2ego=torch.eye(4),
            name="velo",
            inclination_bounds=[-0.1, 0.1],
            data_type="KITTICalib",
        )
        range_image = torch.zeros((2, 2, 2), dtype=torch.float32)
        range_image[0, 0, 0] = 5.0
        range_image[0, 0, 1] = 0.7
        range_image[1, 1, 1] = 0.9
        ray_direction = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        sensor.add_frame(
            frame=0,
            ego2world=torch.eye(4),
            r1=range_image,
            r2=torch.zeros_like(range_image),
            ray_direction=ray_direction,
        )

        points, intensity = sensor.inverse_projection(0)

        self.assertEqual(points.shape[0], 1)
        np.testing.assert_allclose(points.numpy()[0], np.array([5.0, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(intensity.numpy(), np.array([0.7], dtype=np.float32), atol=1.0e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for LiDARSensor sparse ray tests")
    def test_valid_depth_rays_use_explicit_hit_directions(self):
        sensor = LiDARSensor(
            sensor2ego=torch.eye(4),
            name="velo",
            inclination_bounds=[-0.1, 0.1],
            data_type="KITTICalib",
        )
        range_image = torch.zeros((2, 2, 2), dtype=torch.float32)
        range_image[0, 0, 0] = 5.0
        range_image[0, 0, 1] = 0.7
        range_image[1, 0, 0] = 2.0
        range_image[1, 0, 1] = 0.2
        ray_direction = torch.tensor(
            [
                [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 4.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        sensor.add_frame(
            frame=0,
            ego2world=torch.eye(4),
            r1=range_image,
            r2=torch.zeros_like(range_image),
            ray_direction=ray_direction,
        )

        rays_o, rays_d, depth = sensor.get_valid_depth_rays(0)

        self.assertEqual(rays_o.shape, (2, 3))
        self.assertEqual(rays_d.shape, (2, 3))
        np.testing.assert_allclose(depth.cpu().numpy(), np.array([5.0, 2.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(
            rays_d.cpu().numpy(),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
