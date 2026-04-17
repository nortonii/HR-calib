import os
import math
import struct
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

tf.config.set_visible_devices([], "GPU")

from lib.dataloader.waymo_loader.waymo_protobuf import dataset_pb2
from lib.dataloader.waymo_loader.waymo_protobuf import label_pb2
from lib.scene.bounding_box import BoundingBox
from lib.scene.cameras import Camera
from lib.scene.lidar_sensor import LiDARSensor
from lib.utils.graphics_utils import getWorld2View2, getProjectionMatrix


def decompress_range_image(compressed):
    """
    Add two numbers and return the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    decompress_str = tf.io.decode_compressed(compressed, "ZLIB")
    decompress_data = dataset_pb2.MatrixFloat()
    decompress_data.ParseFromString(bytearray(decompress_str.numpy()))
    range_image_tensor = torch.tensor(
        decompress_data.data, dtype=torch.float32
    ).reshape(tuple(decompress_data.shape.dims))
    return range_image_tensor


def decompress_matrix_int32(compressed):
    decompress_str = tf.io.decode_compressed(compressed, "ZLIB")
    decompress_data = dataset_pb2.MatrixInt32()
    decompress_data.ParseFromString(bytearray(decompress_str.numpy()))
    matrix_tensor = torch.tensor(decompress_data.data, dtype=torch.int32).reshape(
        tuple(decompress_data.shape.dims)
    )
    return matrix_tensor


_WAYMO_DYNAMIC_LABEL_TYPES = {
    label_pb2.Label.TYPE_VEHICLE,
    label_pb2.Label.TYPE_PEDESTRIAN,
    label_pb2.Label.TYPE_CYCLIST,
}
_WAYMO_CAMERA_SUFFIXES = (
    "_FRONT",
    "_FRONT_LEFT",
    "_FRONT_RIGHT",
    "_SIDE_LEFT",
    "_SIDE_RIGHT",
    "_REAR_LEFT",
    "_REAR",
    "_REAR_RIGHT",
)


def _normalize_waymo_label_id(label_id: str) -> str:
    if not label_id:
        return ""
    for suffix in _WAYMO_CAMERA_SUFFIXES:
        if label_id.endswith(suffix):
            return label_id[: -len(suffix)]
    return label_id


def _collect_waymo_dynamic_label_ids(frame_data, speed_threshold_mps: float = 0.5):
    dynamic_ids = set()
    dynamic_boxes = []
    for label in frame_data.laser_labels:
        if int(label.type) not in _WAYMO_DYNAMIC_LABEL_TYPES:
            continue
        speed = math.sqrt(
            float(label.metadata.speed_x) ** 2
            + float(label.metadata.speed_y) ** 2
            + float(label.metadata.speed_z) ** 2
        )
        if speed <= float(speed_threshold_mps):
            continue
        dynamic_ids.add(str(label.id))
        box = label.box
        dynamic_boxes.append(
            {
                "center": np.array(
                    [box.center_x, box.center_y, box.center_z], dtype=np.float32
                ),
                "size": np.array([box.length, box.width, box.height], dtype=np.float32),
                "yaw": float(box.heading),
            }
        )
    return dynamic_ids, dynamic_boxes


def _add_waymo_box_to_mask(mask: np.ndarray, box, scale: int = 1, pad_px: int = 4):
    if mask.size == 0:
        return
    width = mask.shape[1]
    height = mask.shape[0]
    cx = float(box.center_x) / float(scale)
    cy = float(box.center_y) / float(scale)
    bw = float(box.length) / float(scale)
    bh = float(box.width) / float(scale)
    half_w = 0.5 * bw
    half_h = 0.5 * bh
    x0 = max(int(math.floor(cx - half_w)) - int(pad_px), 0)
    x1 = min(int(math.ceil(cx + half_w)) + int(pad_px), width)
    y0 = max(int(math.floor(cy - half_h)) - int(pad_px), 0)
    y1 = min(int(math.ceil(cy + half_h)) + int(pad_px), height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def _build_waymo_camera_supervision_mask(
    frame_data,
    camera_id: int,
    width: int,
    height: int,
    scale: int,
    dynamic_ids: set[str],
):
    mask = np.zeros((height, width), dtype=bool)
    for label_set in frame_data.projected_lidar_labels:
        if int(label_set.name) != int(camera_id):
            continue
        for label in label_set.labels:
            if _normalize_waymo_label_id(str(label.id)) in dynamic_ids:
                _add_waymo_box_to_mask(mask, label.box, scale=scale)
    for label_set in frame_data.camera_labels:
        if int(label_set.name) != int(camera_id):
            continue
        for label in label_set.labels:
            if _normalize_waymo_label_id(str(label.id)) in dynamic_ids:
                _add_waymo_box_to_mask(mask, label.box, scale=scale)
    return torch.from_numpy(mask)


def _build_waymo_range_dynamic_mask(
    lidar: LiDARSensor,
    frame: int,
    range_image: torch.Tensor,
    ego2world: torch.Tensor,
    dynamic_boxes: list[dict],
    box_pad_m: float = 0.15,
):
    if range_image is None:
        return None
    mask = torch.zeros(range_image.shape[:2], dtype=torch.bool)
    if not dynamic_boxes:
        return mask

    points_world = lidar.range2point(frame, range_image[..., 0]).detach().cpu()
    ego_rot = ego2world[:3, :3].float().cpu()
    ego_trans = ego2world[:3, 3].float().cpu()
    points_ego = (points_world - ego_trans.view(1, 1, 3)) @ ego_rot
    valid = range_image[..., 0] > 1.0e-3
    if not torch.any(valid):
        return mask

    mask = valid.clone()
    mask.zero_()
    for box in dynamic_boxes:
        center = torch.from_numpy(box["center"]).float()
        size = torch.from_numpy(box["size"]).float() + float(box_pad_m) * 2.0
        yaw = float(box["yaw"])
        rot = torch.tensor(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        local = (points_ego - center.view(1, 1, 3)) @ rot
        inside = (
            (local[..., 0].abs() <= size[0] * 0.5)
            & (local[..., 1].abs() <= size[1] * 0.5)
            & (local[..., 2].abs() <= size[2] * 0.5)
        )
        mask |= inside & valid
    return mask


def load_waymo_raw(base_dir, args):
    for filename in os.listdir(base_dir):
        if filename.endswith(".tfrecord"):
            fp = os.path.join(base_dir, filename)

    dataset = tf.data.TFRecordDataset(fp, compression_type="")
    dataset = list(dataset)
    lidar: LiDARSensor = None
    bboxes: Dict[str, BoundingBox] = {}  # frame * n

    # Centre all world poses on the first frame's ego position so that
    # world-frame coordinates are small (~metres from the vehicle start).
    # Waymo ego2world uses absolute UTM-like coordinates (|T| > 10 km),
    # which causes ~10 000× gradient amplification and rotation divergence.
    first_record = dataset[args.frame_length[0]]
    first_frame_data = dataset_pb2.Frame()
    first_frame_data.ParseFromString(bytearray(first_record.numpy()))
    world_origin = torch.tensor(
        first_frame_data.pose.transform, dtype=torch.float32
    ).reshape(4, 4)[:3, 3].clone()

    # Persist origin so load_waymo_cameras can use the same reference.
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(world_origin, os.path.join(cache_dir, "world_origin.pt"))

    pbar = tqdm(total=(args.frame_length[1] + 1 - args.frame_length[0]))
    for frame in range(args.frame_length[0], args.frame_length[1] + 1):
        record = dataset[frame]
        frame_data = dataset_pb2.Frame()
        frame_data.ParseFromString(bytearray(record.numpy()))
        for i in range(5):
            name = frame_data.context.laser_calibrations[i].name
            if name != 1:
                continue

            if lidar is None:
                lidar2ego = frame_data.context.laser_calibrations[i].extrinsic.transform
                lidar2ego = torch.tensor(lidar2ego, dtype=torch.float32).reshape(4, 4)
                if frame_data.context.laser_calibrations[i].beam_inclinations:
                    beam_inclination = list(
                        frame_data.context.laser_calibrations[i].beam_inclinations
                    )
                else:
                    beam_inclination_min = frame_data.context.laser_calibrations[
                        i
                    ].beam_inclination_min
                    beam_inclination_max = frame_data.context.laser_calibrations[
                        i
                    ].beam_inclination_max
                    beam_inclination = [beam_inclination_min, beam_inclination_max]
                lidar = LiDARSensor(
                    name=name,
                    sensor2ego=lidar2ego,
                    inclination_bounds=beam_inclination,
                    data_type=args.data_type,
                )

            ego2world = torch.tensor(
                frame_data.pose.transform, dtype=torch.float32
            ).reshape(4, 4)
            # Subtract world origin so all positions are relative to first frame.
            ego2world = ego2world.clone()
            ego2world[:3, 3] -= world_origin

            dynamic_ids, dynamic_boxes = _collect_waymo_dynamic_label_ids(frame_data)

            decompressed_dir = f"{base_dir}/cache"
            os.makedirs(decompressed_dir, exist_ok=True)
            decompressed_path = os.path.join(
                decompressed_dir, f"decompressed_frame_{frame}_sensor_{name}.pt"
            )
            dynamic_mask_path = os.path.join(
                decompressed_dir, f"dynamic_mask_frame_{frame}_sensor_{name}.pt"
            )
            if os.path.exists(decompressed_path):
                range_image_r1, range_image_r2 = torch.load(decompressed_path)
            else:
                for lidar_data in frame_data.lasers:
                    if lidar_data.name == name:
                        range_image_r1 = decompress_range_image(
                            lidar_data.ri_return1.range_image_compressed
                        )
                        range_image_r2 = decompress_range_image(
                            lidar_data.ri_return2.range_image_compressed
                        )
                        range_image_r1[..., 1] = torch.clamp(
                            range_image_r1[..., 1], max=1
                        )
                        range_image_r1[..., 0:2][range_image_r1[..., 0:2] == -1] = 0
                torch.save((range_image_r1, range_image_r2), decompressed_path)

            lidar.add_frame(
                frame=frame, ego2world=ego2world, r1=range_image_r1, r2=range_image_r2
            )
            if os.path.exists(dynamic_mask_path):
                dynamic_mask_r1, dynamic_mask_r2 = torch.load(dynamic_mask_path)
            else:
                dynamic_mask_r1 = _build_waymo_range_dynamic_mask(
                    lidar=lidar,
                    frame=frame,
                    range_image=range_image_r1,
                    ego2world=ego2world,
                    dynamic_boxes=dynamic_boxes,
                )
                dynamic_mask_r2 = _build_waymo_range_dynamic_mask(
                    lidar=lidar,
                    frame=frame,
                    range_image=range_image_r2,
                    ego2world=ego2world,
                    dynamic_boxes=dynamic_boxes,
                )
                torch.save((dynamic_mask_r1, dynamic_mask_r2), dynamic_mask_path)
            lidar.set_dynamic_mask(frame, return1=dynamic_mask_r1, return2=dynamic_mask_r2)

        for labels in frame_data.laser_labels:
            box = labels.box
            id, tp = labels.id, labels.type
            x, y, z, l, w, h, yaw = (
                box.center_x,
                box.center_y,
                box.center_z,
                box.length,
                box.width,
                box.height,
                box.heading,
            )
            metadata = [id, x, y, z, l, w, h, yaw, tp]
            if id not in bboxes:
                object_type = int(metadata[8])
                object_id = metadata[0]
                float_data = [float(x) for x in metadata[4:7]]
                size = torch.tensor(float_data).float().cuda()
                bboxes[id] = BoundingBox(object_type, object_id, size)
            bboxes[id].add_frame_waymo(frame, metadata, ego2world)

        pbar.update(1)
    pbar.close()
    return lidar, bboxes


def load_waymo_cameras(base_dir, args, camera_id=1, scale=4):
    """Load camera images and poses from Waymo TFRecord for camera supervision.

    Args:
        base_dir:   Directory containing the .tfrecord file (same as LiDAR loader).
        args:       Config args (needs args.frame_length).
        camera_id:  Which Waymo camera to use (1=FRONT, 2=FRONT_LEFT,
                    3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT).
        scale:      Integer downscale factor applied to images (default 4 → ¼ resolution).

    Returns:
        cameras: dict {frame_index: Camera}  – Camera objects (GPU-ready via .cuda())
        images:  dict {frame_index: torch.Tensor (H, W, 3) float32 in [0, 1]}
    """
    for filename in os.listdir(base_dir):
        if filename.endswith(".tfrecord"):
            fp = os.path.join(base_dir, filename)
    dataset = list(tf.data.TFRecordDataset(fp, compression_type=""))

    # ── Read calibration from the first requested frame ──────────────────
    rec0 = dataset[args.frame_length[0]]
    frame0_data = dataset_pb2.Frame()
    frame0_data.ParseFromString(bytearray(rec0.numpy()))

    calib = None
    for c in frame0_data.context.camera_calibrations:
        if c.name == camera_id:
            calib = c
            break
    if calib is None:
        raise ValueError(
            f"Camera ID {camera_id} not found in TFRecord. "
            f"Available: {[c.name for c in frame0_data.context.camera_calibrations]}"
        )

    W_orig, H_orig = calib.width, calib.height
    fx, fy, cx, cy = (
        calib.intrinsic[0], calib.intrinsic[1],
        calib.intrinsic[2], calib.intrinsic[3],
    )
    cam2ego = torch.tensor(
        list(calib.extrinsic.transform), dtype=torch.float32
    ).reshape(4, 4)

    # Scale intrinsics
    sx = 1.0 / scale
    W = int(W_orig * sx)
    H = int(H_orig * sx)
    fx_s = float(fx) * sx
    fy_s = float(fy) * sx
    cx_s = float(cx) * sx
    cy_s = float(cy) * sx
    FoVx = 2.0 * math.atan(W / (2.0 * fx_s))
    FoVy = 2.0 * math.atan(H / (2.0 * fy_s))
    K = np.array(
        [
            [fx_s, 0.0, cx_s],
            [0.0, fy_s, cy_s],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    cam_cache_dir = os.path.join(base_dir, f"cache_cam{camera_id}_s{scale}_v3")
    os.makedirs(cam_cache_dir, exist_ok=True)

    cameras: Dict[int, Camera] = {}
    images: Dict[int, torch.Tensor] = {}

    # Waymo camera convention: x=forward, y=left, z=up (vehicle-like frame).
    # 3DGS rasterizer uses OpenCV convention: x=right, y=down, z=forward.
    # R_wc2oc converts a point from Waymo-cam frame to OpenCV-cam frame:
    #   opencvX (right)   = -waymoCamY (-left  = right)
    #   opencvY (down)    = -waymoCamZ (-up    = down)
    #   opencvZ (forward) = +waymoCamX (forward)
    R_wc2oc = torch.tensor([[0., -1., 0.],
                             [0.,  0., -1.],
                             [1.,  0.,  0.]], dtype=torch.float32)
    # cam2world_opencv = cam2world_waymo @ R_wc2oc.T
    # (converts cam2world so that z-axis is the look direction)

    # Load the world origin saved by load_waymo_raw for consistent centering.
    # If not found (e.g. cache was cleared), compute it from the first frame.
    world_origin_path = os.path.join(base_dir, "cache", "world_origin.pt")
    if os.path.exists(world_origin_path):
        world_origin = torch.load(world_origin_path)
    else:
        ref_record = dataset[args.frame_length[0]]
        ref_frame_data = dataset_pb2.Frame()
        ref_frame_data.ParseFromString(bytearray(ref_record.numpy()))
        world_origin = torch.tensor(
            list(ref_frame_data.pose.transform), dtype=torch.float32
        ).reshape(4, 4)[:3, 3].clone()

    for frame in range(args.frame_length[0], args.frame_length[1] + 1):
        cache_path = os.path.join(cam_cache_dir, f"frame_{frame}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path)
            R, T, image_tensor = cached["R"], cached["T"], cached["image"]
            supervision_mask = cached.get("supervision_mask")
        else:
            record = dataset[frame]
            frame_data = dataset_pb2.Frame()
            frame_data.ParseFromString(bytearray(record.numpy()))

            dynamic_ids, _dynamic_boxes = _collect_waymo_dynamic_label_ids(frame_data)

            ego2world = torch.tensor(
                list(frame_data.pose.transform), dtype=torch.float32
            ).reshape(4, 4)
            # Centre on the first frame's ego position (same shift as load_waymo_raw).
            ego2world = ego2world.clone()
            ego2world[:3, 3] -= world_origin

            cam2world_waymo = ego2world @ cam2ego  # (4, 4)
            # Convert to OpenCV camera convention (z-forward)
            cam2world = cam2world_waymo.clone()
            cam2world[:3, :3] = cam2world_waymo[:3, :3] @ R_wc2oc.T

            # Camera class convention: R = cam2world rotation, T = world2cam translation
            R = cam2world[:3, :3].clone()
            T = -R.T @ cam2world[:3, 3]

            # Decode JPEG image
            img_bytes = None
            for img in frame_data.images:
                if img.name == camera_id:
                    img_bytes = bytes(img.image)
                    break
            if img_bytes is None:
                continue
            img_rgb = tf.io.decode_jpeg(img_bytes, channels=3).numpy()
            if scale != 1:
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
            image_tensor = torch.from_numpy(img_rgb).float() / 255.0  # (H, W, 3)
            supervision_mask = _build_waymo_camera_supervision_mask(
                frame_data=frame_data,
                camera_id=int(camera_id),
                width=W,
                height=H,
                scale=int(scale),
                dynamic_ids=dynamic_ids,
            )

            torch.save(
                {
                    "R": R,
                    "T": T,
                    "image": image_tensor,
                    "supervision_mask": supervision_mask,
                },
                cache_path,
            )

        cam = Camera(timestamp=frame, R=R, T=T, w=W, h=H, FoVx=FoVx, FoVy=FoVy, K=K)
        cam.supervision_mask = supervision_mask.bool() if supervision_mask is not None else None
        cameras[frame] = cam
        images[frame] = image_tensor  # (H, W, 3) float32 in [0, 1]

    return cameras, images
