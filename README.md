# hr-tiny

Minimal LiDAR-camera extrinsic calibration using 3DGS + OptiX ray tracing.
Derived from HR-calib; stripped to calibration-only workflow.

## Setup

```bash
conda create -n lidar-rt python=3.11.9 && conda activate lidar-rt
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-lidar-tracer
pip install submodules/simple-knn
```

## Data

Download the dataset and place it under `data/`:

```
data/
└── kitti-calibration/
    ├── calibs/
    │   └── 05.txt          # camera intrinsics (P0) + LiDAR-to-camera extrinsic (Tr)
    └── 5-50-t/
        ├── LiDAR_poses.txt
        ├── LiDAR-to-camera.json
        ├── camera-intrinsic.json
        └── 00.ply / 00.png / 00.txt / ...
```

Generate `calibs/05.txt` from the scene JSON files (run once, skips if file exists):

```bash
python3 tools/gen_kitti_calib.py --scene 5-50-t
```

Or manually:

```python
import json, numpy as np, os
base = "data/kitti-calibration"
scene = "5-50-t"
seq = int(scene.split('-')[0])
K = json.load(open(f"{base}/{scene}/camera-intrinsic.json"))
Tr4 = np.array(json.load(open(f"{base}/{scene}/LiDAR-to-camera.json"))["correct"])
P0 = np.array([[K[0][0],0,K[0][2],0],[0,K[1][1],K[1][2],0],[0,0,1,0]])
os.makedirs(f"{base}/calibs", exist_ok=True)
with open(f"{base}/calibs/{seq:02d}.txt","w") as f:
    f.write("P0: " + " ".join(map(str, P0.flatten())) + "\n")
    f.write("Tr: " + " ".join(map(str, Tr4[:3].flatten())) + "\n")
```

## Calibration

Run with `python -u` and `tee` so progress is visible in the log file in real time.
**`-u` is required** — without it, stdout is block-buffered when piped and the log stays empty.

### KITTI-calib (best recipe)

1800 cycles, Gaussian reset every 100 cycles, depth supervision disabled after cycle 10.
This is the recommended recipe: ~2° final rotation error on scene 5-50-t.

```bash
EXP=output/calib/kc_5_50_t_1800_reset100_depth10
mkdir -p $EXP

python -u tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_1800_reset100_depth10.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
  --use_gt_translation \
  --total_cycles 1800 --iters_per_cycle 150 \
  --rotation_lr 0.002 \
  --warmup_cycles 1 \
  --reset_gaussians_every 100 \
  --disable_depth_after_cycle 10 \
  --output_dir $EXP \
  2>&1 | tee $EXP/train.log
```

To resume from a cycle checkpoint (e.g., to extend a finished run):

```bash
python -u tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_1800_reset100_depth10.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
  --use_gt_translation \
  --total_cycles 2400 --iters_per_cycle 150 \
  --rotation_lr 0.002 \
  --warmup_cycles 1 \
  --reset_gaussians_every 100 \
  --disable_depth_after_cycle 10 \
  --resume_cycle_ckpt $EXP/cycle_1800.pth \
  --output_dir $EXP \
  2>&1 | tee -a $EXP/train.log
```

Quick baseline (300 cycles, no reset):

```bash
EXP=output/calib/kc_5_50_t_gtT_biasR
mkdir -p $EXP

python -u tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
  --use_gt_translation \
  --total_cycles 300 --iters_per_cycle 150 \
  --rotation_lr 0.002 \
  --warmup_cycles 1 \
  --output_dir $EXP \
  2>&1 | tee $EXP/train.log
```

Monitor progress from another terminal:

```bash
grep "Cycle" $EXP/train.log | tail -10
```

### Waymo Open Dataset

Place a `.tfrecord` file (single file per directory) under `data/waymo/<segment_name>/`.
The test config uses a 2-frame sample from the Waymo open-dataset repo:

```bash
# Download 2-frame test tfrecord (9.7 MB, no auth required)
mkdir -p data/waymo/two_frame
curl -L https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord \
     -o data/waymo/two_frame/two_frame.tfrecord
```

Run calibration:

```bash
EXP=output/calib/waymo_test
mkdir -p $EXP

python -u tools/calib.py \
  -dc configs/waymo/static/test_segment.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --init_rot_deg 5.0 --init_rot_axis 0.5774 0.5774 0.5774 \
  --use_gt_translation \
  --total_cycles 300 --iters_per_cycle 150 \
  --rotation_lr 0.002 \
  --warmup_cycles 1 \
  --reset_gaussians_every 100 \
  --disable_depth_after_cycle 10 \
  --output_dir $EXP \
  2>&1 | tee $EXP/train.log
```

For real Waymo segments, create a scene config under `configs/waymo/static/<name>.yaml`:

```yaml
parent_config: "configs/waymo/waymo_base.yaml"
source_dir: "data/waymo/<segment_dir>"
frame_length: [0, 19]
eval_frames: [0, 5, 10, 15]
scene_id: waymo_<name>
dynamic: False
waymo_camera_id: 1   # 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
camera_scale: 4
```

### Init cache

The first run estimates per-frame surface normals for all LiDAR frames
(slow, ~1 h on CPU with 50 frames). Results are cached at:

```
data/kitti-calibration/<scene>/.init_cache_fr<start>_<end>_vs<voxel>_vx<0|1>.npz
```

Subsequent runs skip warmup and load from cache automatically.
Delete the `.npz` file to force recomputation.

### Key arguments

| Argument | Description |
|---|---|
| `--init_rot_deg DEG` | Initial rotation error magnitude (degrees) |
| `--init_rot_axis X Y Z` | Fixed rotation axis; random if omitted |
| `--init_trans_xyz DX DY DZ` | Initial translation error added to GT extrinsic (m) |
| `--use_gt_translation` | Lock translation to GT; only rotation is optimised |
| `--total_cycles N` | Number of optimisation cycles |
| `--iters_per_cycle N` | Gaussian update steps per cycle |
| `--translation_start_cycle N` | Two-stage: rotation-only for N cycles, then add translation |
| `--warmup_cycles N` | Freeze pose for first N cycles (Gaussian-only warmup) |
| `--rotation_lr LR` | Learning rate for rotation quaternion |
| `--checkpoint PATH` | Load a pre-trained Gaussian `.pth` instead of init from scratch |
| `--resume_cycle_ckpt PATH` | Resume from a `cycle_NNNN.pth` checkpoint |

## RGB-D calibration with 2DGS rendered depth

Use this path when your “depth modality” is the LiDAR-supervised 2DGS rendered depth map for each RGB frame.
Both cross-modal `RGB ↔ rendered-depth` matching and intra-modal `RGB ↔ RGB` matching use `vismatch` with `matchanything-roma`.

Recommended two-stage workflow:

1. Train a LiDAR-only 2DGS scene:

```bash
cd /home/xzy/HR-calib
python train.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default_hrtiny_data.yaml \
  -ec configs/exp_kitti_3000_lidar_only_2dgs.yaml
```

2. Export per-frame rendered camera-depth maps as raw `.npy` cache from `hr-tiny`:

```bash
cd /home/xzy/hr-tiny
python tools/export_rendered_depth_cache.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default_local_data.yaml \
  -ec configs/exp_kitti_3000_lidar_only_2dgs.yaml \
  -m /home/xzy/HR-calib/output/lidar_rt/kitti_center_3000_lidar_only_2dgs/scene_kc_5_50_t_cam_single_opa_pose_higs_default_gtt_fastlr/models/model_it_3000.pth \
  -o output/depth_cache_5_50_t
```

The exported depth maps land in:

```text
/home/xzy/hr-tiny/output/depth_cache_5_50_t
```

```bash
python tools/rgbd_calib.py \
  --rgb /path/to/rgb_frames \
  --depth /path/to/rendered_depth \
  --rgb_intrinsics /path/to/rgb_intrinsics.yaml \
  --depth_intrinsics /path/to/depth_intrinsics.yaml \
  --output_dir output/rgbd_calib_run \
  --device cuda \
  --matcher_resize 832 \
  --depth_scale 1.0 \
  --depth_use_inverse \
  --save_visualizations
```

Matcher loading policy:
- prefer local Hugging Face cache first;
- if the cache is missing, fall back to `https://hf-mirror.com`;
- override the mirror with `--hf_endpoint`.

Outputs:
- `extrinsic.yaml`: calibrated `T_rgb_d`, quaternion, translation, and reprojection metrics.
- `overlays/*.png`: projected depth-over-RGB visualizations.

To inspect one frame after calibration:

```bash
python tools/warp_depth_to_rgb.py \
  --rgb /path/to/rgb_frames/000000.png \
  --depth /path/to/rendered_depth/000000.npy \
  --extrinsic output/rgbd_calib_run/extrinsic.yaml \
  --output output/rgbd_calib_run/check_000000.png
```
