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


def _parse_perspective_txt(calib_file: str):
    """Parse perspective.txt for KITTI-360 camera 0.

    Returns:
        K_raw:   (3,3) float64 – raw camera intrinsic matrix
        D:       (5,)  float64 – distortion coefficients [k1, k2, p1, p2, k3]
        K_rect:  (3,3) float64 – rectified camera intrinsic matrix (from P_rect_00)
        W_rect:  int – rectified image width  (from S_rect_00)
        H_rect:  int – rectified image height (from S_rect_00)
    """
    K_raw = D = K_rect = W_rect = H_rect = None
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("K_00:"):
                vals = [float(x) for x in line.split(":")[1].split()]
                K_raw = np.array(vals).reshape(3, 3)
                K_raw[2, 2] = 1.0  # KITTI-360 perspective.txt omits the last 1
            elif line.startswith("D_00:"):
                D = np.array([float(x) for x in line.split(":")[1].split()])
            elif line.startswith("P_rect_00:"):
                vals = [float(x) for x in line.split(":")[1].split()]
                P_rect = np.array(vals).reshape(3, 4)
                K_rect = P_rect[:3, :3]
            elif line.startswith("S_rect_00:"):
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


def load_kitti360_cameras(
    base_dir: str,
    args,
    seq_num: int,
    frame_ids,
    scale: int = 4,
) -> Tuple[Dict[int, Camera], Dict[int, torch.Tensor]]:
    """Load KITTI-360 perspective camera (image_00) for training supervision.

    Raw images have radial distortion (k1≈-0.344). Following HiGS-Calib-360,
    we undistort with cv2.undistort(img, K_raw, D) keeping K_raw as the output
    camera matrix (same image size, same FoV, distortion removed).

    Returns:
        cameras: dict {frame_id: Camera}
        images:  dict {frame_id: Tensor (H, W, 3) float32 in [0, 1]}
    """
    calib_file = os.path.join(base_dir, "calibration", "perspective.txt")
    pose_file = os.path.join(base_dir, "data_poses", str(seq_num), "cam0_to_world.txt")
    img_dir = os.path.join(base_dir, "data_2d_raw", str(seq_num), "image_00", "data_rgb")

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"cam0_to_world.txt not found: {pose_file}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Camera image dir not found: {img_dir}")

    K_raw, D, K_rect, W_rect, H_rect = _parse_perspective_txt(calib_file)

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
    FoVx = 2.0 * math.atan(W / (2.0 * fx_s))
    FoVy = 2.0 * math.atan(H / (2.0 * fy_s))

    cam2world_by_frame = _load_cam0_to_world(pose_file)

    # World-centering: subtract the same ego world origin used by load_kitti_raw
    # so camera poses are in the same centered coordinate frame as the LiDAR.
    world_origin_path = os.path.join(base_dir, "cache", f"world_origin_seq{seq_num}.pt")
    if os.path.exists(world_origin_path):
        world_origin = torch.load(world_origin_path, weights_only=True).numpy()
    else:
        # Fallback: use first available cam2world translation
        available = sorted(cam2world_by_frame.keys())
        world_origin = cam2world_by_frame[available[0]][:3, 3].copy()
    for f in cam2world_by_frame:
        cam2world_by_frame[f][:3, 3] -= world_origin

    cache_dir = os.path.join(base_dir, f"cache_cam_kitti360_s{scale}_v2")
    meta_file = os.path.join(cache_dir, "meta.json")
    os.makedirs(cache_dir, exist_ok=True)
    meta = {"W": W, "H": H, "undistorted": True, "method": "undistort_K_raw"}
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

            # Find nearest cam2world (cam0_to_world.txt may have gaps)
            if frame_id in cam2world_by_frame:
                cam2world = cam2world_by_frame[frame_id]
            else:
                available = sorted(cam2world_by_frame.keys())
                nearest = min(available, key=lambda f: abs(f - frame_id))
                cam2world = cam2world_by_frame[nearest]

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

        cam = Camera(timestamp=frame_id, R=R, T=T, w=W, h=H, FoVx=FoVx, FoVy=FoVy)
        cameras[frame_id] = cam
        images[frame_id] = image_tensor

    print(blue(f"[KITTI-360 cam] Loaded {len(cameras)} camera frames "
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

    W, H = 1030, 66
    inc_buttom, inc_top = math.radians(-24.9), math.radians(2.0)
    azimuth_left, azimuth_right = np.pi, -np.pi
    max_depth = 80.0
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_buttom - inc_top) / H

    lidar = LiDARSensor(
        sensor2ego=lidar2ego,
        name="velo",
        inclination_bounds=(inc_buttom, inc_top),
        data_type=args.data_type,
    )
    use_column_origin = getattr(args, "lidar_origin_mode", "center") == "column_interp"
    lidar2ego_t = torch.tensor(lidar2ego, dtype=torch.float32)
    last_ego2world = None
    if frame_ids[0] not in ego2world.keys():
        for pre_frame in range(frame_ids[0] - 1, -1, -1):
            if pre_frame in ego2world.keys():
                last_ego2world = ego2world[pre_frame]
                break

    for frame in frame_ids:
        prev_ego2world = last_ego2world
        xyzs, intensities = lidar_points[frame][:, :3], lidar_points[frame][:, 3]
        dists = np.linalg.norm(xyzs, axis=1)

        # Vectorized range-image construction (replaces slow per-point Python loop)
        valid = dists <= max_depth
        xyzs_v, ints_v, dists_v = xyzs[valid], intensities[valid], dists[valid]
        azimuth = np.arctan2(xyzs_v[:, 1], xyzs_v[:, 0])
        inclination = np.arctan2(xyzs_v[:, 2],
                                  np.sqrt(xyzs_v[:, 0]**2 + xyzs_v[:, 1]**2))
        w_idx = np.round((azimuth - azimuth_left) / h_res).astype(int)
        h_idx = np.round((inclination - inc_top) / v_res).astype(int)
        in_bounds = (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
        w_idx, h_idx = w_idx[in_bounds], h_idx[in_bounds]
        dists_v, ints_v = dists_v[in_bounds], ints_v[in_bounds]

        range_map = np.full((H, W), -1.0)
        intensity_map = np.full((H, W), -1.0)
        # Fill closest point per pixel (sort by distance ascending so last write wins
        # when we use a flat-index scatter with minimum semantics)
        order = np.argsort(dists_v)[::-1]  # farthest first → nearest overwrites
        flat = h_idx[order] * W + w_idx[order]
        range_map.flat[flat] = dists_v[order]
        intensity_map.flat[flat] = ints_v[order]

        range_image_r1 = np.stack([range_map, intensity_map], axis=-1)
        range_image_r2 = np.ones_like(range_image_r1) * -1

        if frame in ego2world.keys():
            last_ego2world = ego2world[frame]
        current_ego2world = last_ego2world

        ray_origin = None
        if use_column_origin and current_ego2world is not None:
            current_ego2world_t = torch.tensor(current_ego2world, dtype=torch.float32)
            current_center = (current_ego2world_t @ lidar2ego_t)[:3, 3]
            if prev_ego2world is not None:
                prev_ego2world_t = torch.tensor(prev_ego2world, dtype=torch.float32)
                prev_center = (prev_ego2world_t @ lidar2ego_t)[:3, 3]
                sweep = current_center - prev_center
                alpha = torch.linspace(0.0, 1.0, W, dtype=torch.float32)
                ray_origin = current_center[None, :] + alpha[:, None] * sweep[None, :]
            else:
                ray_origin = current_center[None, :].expand(W, 3)
        range_image_r1[range_image_r1 == -1] = 0
        range_image_r2[range_image_r2 == -1] = 0
        lidar.add_frame(
            frame,
            current_ego2world,
            range_image_r1,
            range_image_r2,
            ray_origin=ray_origin,
        )

    lidar_bbox = load_lidar_bbox(
        os.path.join(base_dir, "data_3d_bboxes", "train"),
        full_seq,
        args,
        using_cache=False,
    )

    return lidar, lidar_bbox
