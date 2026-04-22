import math
import os
import pickle
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from lib.scene import BoundingBox, LiDARSensor
from lib.scene.cameras import Camera
from lib.utils.console_utils import *
from lib.utils.kitti_utils import (
    build_kitti_range_image_from_points,
    interpolate_sensor2world_columns,
    resolve_kitti_lidar_width,
)
from lib.utils.velodyne_utils import get_kitti_hdl64e_beam_inclinations_rad

# from utils.kitti_utils import LiDAR_2_Pano_KITTI


def load_lidar2ego(base_dir, seq):
    cam2velo = np.asarray(
        [
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(4, 4)
    cam2ego = np.asarray(
        [
            0.0371783278,
            -0.0986182135,
            0.9944306009,
            1.5752681039,
            0.9992675562,
            -0.0053553387,
            -0.0378902567,
            0.0043914093,
            0.0090621821,
            0.9951109327,
            0.0983468786,
            -0.6500000000,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(4, 4)
    velo2ego = cam2ego @ np.linalg.inv(cam2velo)

    return velo2ego


def load_ego2world(file_path):
    ego2world = {}
    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.split()
        frame = int(parts[0])
        values = [float(x) for x in parts[1:]]
        mat4 = np.eye(4)
        mat4[:3, :] = np.array(values).reshape(3, 4)
        ego2world[frame] = mat4

    return ego2world


def load_lidar_point(lidar_dir, frames):
    lidar_points = {}
    for frame in range(frames[0], frames[1] + 1):
        with open(os.path.join(lidar_dir, f"{str(frame).zfill(10)}.bin"), "rb") as f:
            lidar_points[frame] = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    return lidar_points


def load_lidar_bbox(lidar_bbox_dir, full_seq, args, using_cache=True):
    bboxes: dict[str, BoundingBox] = {}
    xml_path = os.path.join(lidar_bbox_dir, full_seq + ".xml")
    if not os.path.exists(xml_path):
        return bboxes

    bbox_pickle_dir = os.path.join(lidar_bbox_dir, "cache")
    os.makedirs(bbox_pickle_dir, exist_ok=True)
    bbox_pickle_path = os.path.join(bbox_pickle_dir, f"{full_seq}.pkl")

    # read pre cached pickle file
    if os.path.exists(bbox_pickle_path) and using_cache:
        try:
            with open(bbox_pickle_path, "rb") as fp:
                bboxes = pickle.load(fp)
            print("Read pickle file from: ", bbox_pickle_path)
            return bboxes
        except:
            print(red(f"Error: Cannot read pickle file from {bbox_pickle_path}"))

    with open(xml_path, "r") as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)

    for obj in root:
        metadata = {
            "label": obj.find("label").text,
            "instanceId": obj.find("instanceId").text,
            "category": obj.find("category").text,
            "timestamp": int(obj.find("timestamp").text),
            "dynamic": int(obj.find("dynamic").text),
            "transform": {
                "rows": int(obj.find("transform/rows").text),
                "cols": int(obj.find("transform/cols").text),
                "data": [float(val) for val in obj.find("transform/data").text.split()],
            },
        }
        object_type = metadata["label"]
        object_id = metadata["instanceId"]

        if (
            metadata["timestamp"] < args.frame_length[0]
            or metadata["timestamp"] > args.frame_length[1]
        ):
            continue
        if object_type not in ["car", "truck", "bus"]:
            continue

        transform = np.array(metadata["transform"]["data"]).reshape(
            metadata["transform"]["rows"], metadata["transform"]["cols"]
        )

        if object_id not in bboxes:
            U, S, V = np.linalg.svd(transform[:3, :3])
            bboxes[object_id] = BoundingBox(1, object_id, S)

        bboxes[object_id].add_frame_kitti(metadata["timestamp"], transform)

    if not os.path.exists(bbox_pickle_path):
        try:
            with open(bbox_pickle_path, "wb") as fp:
                pickle.dump(bboxes, fp)
            print(blue(f"Save pickle file to {bbox_pickle_path}"))
        except:
            os.remove(bbox_pickle_path)
            print(red(f"Error: Cannot save pickle file to {bbox_pickle_path}"))

    return bboxes


def load_SfM_clouds(SfM_clouds_dir):
    if os.path.exists(SfM_clouds_dir):
        xyzs, rgbs = [], []
        with open(SfM_clouds_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    xyzs.append(torch.tensor(tuple(map(float, elems[1:4]))))
                    rgbs.append(torch.tensor(tuple(map(int, elems[4:7]))))
        if xyzs == []:
            return None
        else:
            return [torch.stack(xyzs, dim=0), torch.stack(rgbs, dim=0)]
    else:
        return None


def _parse_perspective_txt(calib_file: str, camera_name: str = "image_00"):
    """Parse perspective.txt for KITTI-360 perspective camera 0/1.

    Returns:
        K_raw:   (3,3) float64 – raw camera intrinsic matrix
        D:       (5,)  float64 – distortion coefficients [k1, k2, p1, p2, k3]
        K_rect:  (3,3) float64 – rectified camera intrinsic matrix (from P_rect_XX)
        W_rect:  int – rectified image width  (from S_rect_XX)
        H_rect:  int – rectified image height (from S_rect_XX)
    """
    if camera_name not in {"image_00", "image_01"}:
        raise ValueError(f"Unsupported KITTI-360 perspective camera: {camera_name}")
    cam_suffix = camera_name.split("_")[-1]
    k_key = f"K_{cam_suffix}:"
    d_key = f"D_{cam_suffix}:"
    p_key = f"P_rect_{cam_suffix}:"
    s_key = f"S_rect_{cam_suffix}:"
    K_raw = D = K_rect = W_rect = H_rect = None
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(k_key):
                vals = [float(x) for x in line.split(":")[1].split()]
                K_raw = np.array(vals).reshape(3, 3)
                K_raw[2, 2] = 1.0  # KITTI-360 perspective.txt omits the last 1
            elif line.startswith(d_key):
                D = np.array([float(x) for x in line.split(":")[1].split()])
            elif line.startswith(p_key):
                vals = [float(x) for x in line.split(":")[1].split()]
                P_rect = np.array(vals).reshape(3, 4)
                K_rect = P_rect[:3, :3]
            elif line.startswith(s_key):
                vals = [float(x) for x in line.split(":")[1].split()]
                W_rect, H_rect = int(vals[0]), int(vals[1])
    if any(x is None for x in [K_raw, D, K_rect, W_rect, H_rect]):
        raise RuntimeError(f"Could not parse all calibration from {calib_file}")
    return K_raw, D, K_rect, W_rect, H_rect


def _load_cam0_to_world(pose_file: str) -> Dict[int, np.ndarray]:
    """Parse cam0_to_world.txt → dict {frame_id: (4, 4) float64 ndarray}."""
    result: Dict[int, np.ndarray] = {}
    with open(pose_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            frame_id = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            if len(vals) == 16:
                mat = np.array(vals).reshape(4, 4)
            else:
                # 3×4 → extend to 4×4
                mat = np.eye(4)
                mat[:3, :] = np.array(vals[:12]).reshape(3, 4)
            result[frame_id] = mat
    return result


def _load_kitti360_cam_to_pose(calib_file: str) -> Dict[str, np.ndarray]:
    """Parse calib_cam_to_pose.txt → dict {'image_00': (4,4), ...}."""
    result: Dict[str, np.ndarray] = {}
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, values = line.split(":", 1)
            vals = [float(x) for x in values.split()]
            if len(vals) != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            result[name.strip()] = mat
    return result


def load_kitti360_cameras(
    base_dir: str,
    args,
    seq_num: int,
    frame_ids,
    scale: int = 4,
) -> Tuple[Dict[int, Camera], Dict[int, torch.Tensor]]:
    """Load KITTI-360 perspective camera (image_00 or image_01) for training supervision.

    Raw images have radial distortion (k1≈-0.344). Following HiGS-Calib-360,
    we undistort with cv2.undistort(img, K_raw, D) keeping K_raw as the output
    camera matrix (same image size, same FoV, distortion removed).

    Returns:
        cameras: dict {frame_id: Camera}
        images:  dict {frame_id: Tensor (H, W, 3) float32 in [0, 1]}
    """
    calib_file = os.path.join(base_dir, "calibration", "perspective.txt")
    camera_name = str(getattr(args, "kitti360_camera", "image_00"))
    pose_file = os.path.join(base_dir, "data_poses", str(seq_num), "poses.txt")
    cam_to_pose_file = os.path.join(base_dir, "calibration", "calib_cam_to_pose.txt")
    img_dir = os.path.join(base_dir, "data_2d_raw", str(seq_num), camera_name, "data_rgb")

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"poses.txt not found: {pose_file}")
    if not os.path.exists(cam_to_pose_file):
        raise FileNotFoundError(f"calib_cam_to_pose.txt not found: {cam_to_pose_file}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Camera image dir not found: {img_dir}")

    K_raw, D, K_rect, W_rect, H_rect = _parse_perspective_txt(calib_file, camera_name=camera_name)

    # Use K_raw intrinsics and raw image dimensions (1392×512).
    # cv2.undistort with K_raw keeps the same size and FoV — matches HiGS-Calib-360.
    W_orig = int(round(K_raw[0, 2] * 2))  # fallback estimate; actual size from first image
    H_orig = int(round(K_raw[1, 2] * 2))

    # Probe actual image size from first available image
    sample_imgs = sorted(os.listdir(img_dir))
    if sample_imgs:
        probe = cv2.imread(os.path.join(img_dir, sample_imgs[0]))
        if probe is not None:
            H_orig, W_orig = probe.shape[:2]

    sx = 1.0 / scale
    W = int(W_orig * sx)
    H = int(H_orig * sx)
    fx_s = float(K_raw[0, 0]) * sx
    fy_s = float(K_raw[1, 1]) * sx
    cx_s = float(K_raw[0, 2]) * sx
    cy_s = float(K_raw[1, 2]) * sx
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

    pose2world_by_frame = load_ego2world(pose_file)
    cam_to_pose_by_name = _load_kitti360_cam_to_pose(cam_to_pose_file)
    if camera_name not in cam_to_pose_by_name:
        raise KeyError(f"{camera_name} not found in {cam_to_pose_file}")
    cam_to_pose = cam_to_pose_by_name[camera_name]

    # World-centering: subtract the same ego world origin used by load_kitti_raw
    # so camera poses are in the same centered coordinate frame as the LiDAR.
    world_origin_path = os.path.join(base_dir, "cache", f"world_origin_seq{seq_num}.pt")
    if os.path.exists(world_origin_path):
        world_origin = torch.load(world_origin_path, weights_only=True).numpy()
    else:
        # Fallback: use first available pose translation
        available = sorted(pose2world_by_frame.keys())
        world_origin = pose2world_by_frame[available[0]][:3, 3].copy()
    for f in pose2world_by_frame:
        pose2world_by_frame[f][:3, 3] -= world_origin

    cache_dir = os.path.join(base_dir, f"cache_cam_kitti360_{camera_name}_s{scale}_v3")
    meta_file = os.path.join(cache_dir, "meta.json")
    os.makedirs(cache_dir, exist_ok=True)
    meta = {"W": W, "H": H, "undistorted": True, "method": "undistort_K_raw", "camera_name": camera_name}
    import json as _json
    import glob as _glob
    if os.path.exists(meta_file):
        old_meta = _json.load(open(meta_file))
        if old_meta != meta:
            print(red(f"[KITTI-360 cam] Cache outdated ({old_meta} vs {meta}), clearing cache"))
            for fp in _glob.glob(os.path.join(cache_dir, "frame_*.pt")):
                os.remove(fp)
    else:
        stale = _glob.glob(os.path.join(cache_dir, "frame_*.pt"))
        if stale:
            print(red(f"[KITTI-360 cam] No meta.json found, clearing {len(stale)} stale cache files"))
            for fp in stale:
                os.remove(fp)
    with open(meta_file, "w") as f:
        _json.dump(meta, f)

    cameras: Dict[int, Camera] = {}
    images: Dict[int, torch.Tensor] = {}

    for frame_id in frame_ids:
        cache_path = os.path.join(cache_dir, f"frame_{frame_id:010d}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            R, T, image_tensor = cached["R"], cached["T"], cached["image"]
        else:
            img_path = os.path.join(img_dir, f"{frame_id:010d}.png")
            if not os.path.exists(img_path):
                print(red(f"[KITTI-360 cam] Image not found: {img_path}, skipping frame {frame_id}"))
                continue

            # Find nearest pose entry (poses.txt may have gaps)
            if frame_id in pose2world_by_frame:
                pose2world = pose2world_by_frame[frame_id]
            else:
                available = sorted(pose2world_by_frame.keys())
                nearest = min(available, key=lambda f: abs(f - frame_id))
                pose2world = pose2world_by_frame[nearest]
            cam2world = pose2world @ cam_to_pose

            R = torch.tensor(cam2world[:3, :3], dtype=torch.float32)
            T = -R.T @ torch.tensor(cam2world[:3, 3], dtype=torch.float32)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(red(f"[KITTI-360 cam] Failed to read image: {img_path}"))
                continue
            # Undistort: remove distortion while keeping K_raw intrinsics (HiGS-Calib-360 style)
            img_undist = cv2.undistort(img_bgr, K_raw, D)
            img_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)
            if scale != 1:
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
            image_tensor = torch.from_numpy(img_rgb).float() / 255.0  # (H, W, 3)

            torch.save({"R": R, "T": T, "image": image_tensor}, cache_path)

        cam = Camera(timestamp=frame_id, R=R, T=T, w=W, h=H, FoVx=FoVx, FoVy=FoVy, K=K)
        cameras[frame_id] = cam
        images[frame_id] = image_tensor

    print(blue(f"[KITTI-360 cam] Loaded {len(cameras)} {camera_name} frames "
               f"({W}×{H} @ 1/{scale} scale)"))
    return cameras, images


def load_kitti_raw(base_dir, args):
    frames = args.frame_length
    kitti_seq = getattr(args, "kitti_seq", None)

    if kitti_seq is not None:
        # Numbered directory layout: data_3d_raw/{seq_num}/ + data_poses/{seq_num}/poses.txt
        seq_num = int(kitti_seq)
        full_seq = f"2013_05_28_drive_{seq_num:04d}_sync"
        lidar_dir = os.path.join(base_dir, "data_3d_raw", str(seq_num), "velodyne_points", "data")
        pose_file = os.path.join(base_dir, "data_poses", str(seq_num), "poses.txt")
        lidar2ego = load_lidar2ego(base_dir, None)

        # Scan directory for actual frame IDs (may be non-consecutive)
        bin_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".bin"))
        all_frame_ids = [int(f[:-4]) for f in bin_files]
        frame_ids = [fid for fid in all_frame_ids if frames[0] <= fid <= frames[1]]
        print(blue(f"[KITTI seq {seq_num}] Found {len(frame_ids)} frames in [{frames[0]}, {frames[1]}]"))

        lidar_points = {}
        for fid in frame_ids:
            with open(os.path.join(lidar_dir, f"{fid:010d}.bin"), "rb") as fp:
                lidar_points[fid] = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)
    else:
        seq = getattr(args, "seq", "0000")
        full_seq = f"2013_05_28_drive_{seq}_sync"
        lidar_dir = os.path.join(base_dir, "data_3d_raw", full_seq, "velodyne_points", "data")
        pose_file = os.path.join(base_dir, "data_pose", full_seq, "poses.txt")
        lidar2ego = load_lidar2ego(base_dir, seq)

        lidar_points = load_lidar_point(lidar_dir, frames)
        frame_ids = list(range(frames[0], frames[1] + 1))

    ego2world = load_ego2world(pose_file)

    # World-centering: subtract first frame's ego translation so all poses are
    # relative to the starting position. This prevents UTM-scale coordinates
    # (~2700 m) from causing gradient explosion during optimization.
    if kitti_seq is not None:
        sorted_ego_frames = sorted(ego2world.keys())
        world_origin = ego2world[sorted_ego_frames[0]][:3, 3].copy()
        for f in ego2world:
            ego2world[f][:3, 3] -= world_origin
        origin_cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(origin_cache_dir, exist_ok=True)
        world_origin_path = os.path.join(origin_cache_dir, f"world_origin_seq{seq_num}.pt")
        torch.save(torch.tensor(world_origin, dtype=torch.float32), world_origin_path)
        print(blue(f"[KITTI-360] World origin saved: {world_origin.tolist()}"))

    beam_inclinations_top_to_bottom = get_kitti_hdl64e_beam_inclinations_rad(
        order="top_to_bottom"
    )
    beam_inclinations_bottom_to_top = get_kitti_hdl64e_beam_inclinations_rad(
        order="bottom_to_top"
    )
    W, H = resolve_kitti_lidar_width(args), len(beam_inclinations_top_to_bottom)
    max_depth = 80.0

    lidar = LiDARSensor(
        sensor2ego=lidar2ego,
        name="velo",
        inclination_bounds=beam_inclinations_bottom_to_top.tolist(),
        data_type=args.data_type,
    )
    use_column_origin = getattr(args, "lidar_origin_mode", "center") == "column_interp"
    last_ego2world = None
    if frame_ids[0] not in ego2world.keys():
        for pre_frame in range(frame_ids[0] - 1, -1, -1):
            if pre_frame in ego2world.keys():
                last_ego2world = ego2world[pre_frame]
                break

    for frame in frame_ids:
        prev_ego2world = last_ego2world
        xyzs, intensities = lidar_points[frame][:, :3], lidar_points[frame][:, 3]

        if frame in ego2world.keys():
            last_ego2world = ego2world[frame]
        current_ego2world = last_ego2world
        if current_ego2world is None:
            raise ValueError(f"Missing ego pose for KITTI frame {frame}")
        current_sensor2world = current_ego2world @ lidar2ego

        ray_origin = None
        sensor2world_for_rays = current_sensor2world
        if use_column_origin and current_ego2world is not None:
            if prev_ego2world is not None:
                prev_sensor2world = prev_ego2world @ lidar2ego
            else:
                prev_sensor2world = None
            sensor2world_for_rays = interpolate_sensor2world_columns(
                prev_sensor2world=prev_sensor2world,
                current_sensor2world=current_sensor2world,
                width=W,
            )
            ray_origin = sensor2world_for_rays[:, :3, 3]

        range_map, intensity_map, ray_direction = build_kitti_range_image_from_points(
            xyzs=xyzs,
            intensities=intensities,
            beam_inclinations_top_to_bottom=beam_inclinations_top_to_bottom,
            width=W,
            sensor2world=sensor2world_for_rays,
            ray_origin=ray_origin,
            max_depth=max_depth,
        )
        range_image_r1 = np.stack([range_map, intensity_map], axis=-1)
        range_image_r2 = np.zeros_like(range_image_r1)
        lidar.add_frame(
            frame,
            current_ego2world,
            range_image_r1,
            range_image_r2,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            raw_points=xyzs,
            raw_intensity=intensities,
        )

    lidar_bbox = load_lidar_bbox(
        os.path.join(base_dir, "data_3d_bboxes", "train"),
        full_seq,
        args,
        using_cache=False,
    )

    return lidar, lidar_bbox
