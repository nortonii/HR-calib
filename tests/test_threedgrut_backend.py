import sys
import types
import unittest

import torch

from lib.gaussian_renderer import get_lidar_raytrace_backend
from lib.gaussian_renderer.camera_render import get_camera_render_backend
from lib.gaussian_renderer.threedgrut_backend import (
    _camera_intrinsics,
    _expand_scales_for_threedgrut,
    _flatten_sh_features,
    _install_ncore_shim_if_missing,
)
from lib.scene.cameras import Camera
from lib.utils.graphics_utils import focal2fov, getProjectionMatrix


class _Args:
    def __init__(self, camera_backend="3dgut", raytrace_backend="3dgrt", training_render_mode=None):
        self.model = types.SimpleNamespace(
            camera_render_backend=camera_backend,
            raytrace_backend=raytrace_backend,
            training_render_mode=training_render_mode,
        )


class ThreeDGrutBackendTests(unittest.TestCase):
    def test_flatten_sh_features(self):
        features = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0], [41.0, 51.0, 61.0]],
            ]
        )
        flattened = _flatten_sh_features(features)
        self.assertEqual(tuple(flattened.shape), (2, 9))
        self.assertTrue(torch.equal(flattened[0], torch.tensor([1.0, 2.0, 3.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0])))

    def test_expand_2d_scales_to_3d(self):
        scales = torch.tensor([[1.0, 2.0], [3.5, 0.5]])
        expanded = _expand_scales_for_threedgrut(scales)
        self.assertTrue(torch.equal(expanded, torch.tensor([[1.0, 2.0, 2.0], [3.5, 0.5, 3.5]])))

    def test_camera_backend_aliases(self):
        self.assertEqual(get_camera_render_backend(_Args(camera_backend="3dgut")), "3dgut_rasterization")
        self.assertEqual(get_camera_render_backend(_Args(camera_backend="gut")), "3dgut_rasterization")
        self.assertEqual(get_camera_render_backend(_Args(camera_backend="raytrace")), "raytracing")

    def test_lidar_backend_aliases(self):
        self.assertEqual(get_lidar_raytrace_backend(_Args(raytrace_backend="3dgrut")), "3dgrt")
        self.assertEqual(get_lidar_raytrace_backend(_Args(raytrace_backend="diff_lidar_tracer")), "legacy")

    def test_hybrid_training_mode_overrides_backends(self):
        args = _Args(camera_backend="rasterization", raytrace_backend="legacy", training_render_mode="hybrid_3dgrut")
        self.assertEqual(get_camera_render_backend(args), "3dgut_rasterization")
        self.assertEqual(get_lidar_raytrace_backend(args), "3dgrt")

    def test_ncore_shim(self):
        sys.modules.pop("ncore", None)
        sys.modules.pop("ncore.data", None)
        _install_ncore_shim_if_missing()
        import ncore.data  # noqa: F401

        self.assertIn("ncore", sys.modules)
        self.assertIn("ncore.data", sys.modules)
        self.assertEqual(ncore.data.ShutterType.GLOBAL.name, "GLOBAL")
        self.assertEqual(
            ncore.data.FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE.name,
            "PIXELDIST_TO_ANGLE",
        )

    def test_camera_intrinsics_preserve_principal_point(self):
        width, height = 1226, 370
        fx, fy = 707.0912, 707.0912
        cx, cy = 601.8873, 183.1104
        camera = Camera(
            timestamp=0,
            R=torch.eye(3),
            T=torch.zeros(3),
            w=width,
            h=height,
            FoVx=focal2fov(fx, width),
            FoVy=focal2fov(fy, height),
            K=torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            data_device="cpu",
        )
        intrinsics = _camera_intrinsics(camera)
        self.assertEqual(len(intrinsics), 4)
        self.assertAlmostEqual(intrinsics[0], fx, places=4)
        self.assertAlmostEqual(intrinsics[1], fy, places=4)
        self.assertAlmostEqual(intrinsics[2], cx, places=4)
        self.assertAlmostEqual(intrinsics[3], cy, places=4)

    def test_projection_matrix_supports_offcenter_principal_point(self):
        width, height = 1226, 370
        fx, fy = 707.0912, 707.0912
        cx, cy = 601.8873, 183.1104
        projection = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=focal2fov(fx, width),
            fovY=focal2fov(fy, height),
            image_width=width,
            image_height=height,
            cx=cx,
            cy=cy,
        )
        self.assertAlmostEqual(float(projection[0, 2]), (width - 2.0 * cx) / width, places=6)
        self.assertAlmostEqual(float(projection[1, 2]), (2.0 * cy - height) / height, places=6)


if __name__ == "__main__":
    unittest.main()
