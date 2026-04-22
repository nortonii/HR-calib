import types
import unittest

import torch

from lib.dataloader.waymo_loader import (
    _build_waymo_camera_supervision_mask,
    _normalize_waymo_label_id,
)
from lib.scene.lidar_sensor import LiDARSensor


def _box(cx, cy, length, width):
    return types.SimpleNamespace(
        center_x=float(cx),
        center_y=float(cy),
        length=float(length),
        width=float(width),
    )


class WaymoDynamicMaskTests(unittest.TestCase):
    def test_normalize_waymo_label_id_strips_camera_suffix(self):
        self.assertEqual(
            _normalize_waymo_label_id("obj123_FRONT"),
            "obj123",
        )
        self.assertEqual(
            _normalize_waymo_label_id("obj123_SIDE_RIGHT"),
            "obj123",
        )
        self.assertEqual(_normalize_waymo_label_id("obj123"), "obj123")

    def test_camera_supervision_mask_marks_dynamic_projected_box(self):
        frame = types.SimpleNamespace(
            projected_lidar_labels=[
                types.SimpleNamespace(
                    name=1,
                    labels=[
                        types.SimpleNamespace(
                            id="moving-car_FRONT",
                            box=_box(40, 20, 20, 10),
                        )
                    ],
                )
            ],
            camera_labels=[],
        )
        mask = _build_waymo_camera_supervision_mask(
            frame_data=frame,
            camera_id=1,
            width=50,
            height=25,
            scale=2,
            dynamic_ids={"moving-car"},
        )
        self.assertEqual(tuple(mask.shape), (25, 50))
        self.assertTrue(bool(mask[10, 20]))
        self.assertFalse(bool(mask[0, 0]))

    def test_lidar_sensor_uses_explicit_dynamic_mask(self):
        sensor = LiDARSensor(
            sensor2ego=torch.eye(4),
            name=1,
            inclination_bounds=[-0.1, 0.1],
            data_type="Waymo",
        )
        sensor.set_dynamic_mask(
            frame=7,
            return1=torch.tensor([[True, False], [False, True]]),
        )
        mask = sensor.get_dynamic_mask(7)
        self.assertTrue(torch.equal(mask, torch.tensor([[True, False], [False, True]])))


if __name__ == "__main__":
    unittest.main()
