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

```bash
EXP=output/calib/kc_5_50_t_gtT_biasR
mkdir -p $EXP

python -u tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
  --use_gt_translation \
  --translation_start_cycle 9999 \
  --total_cycles 300 --iters_per_cycle 200 \
  --rotation_lr 0.002 \
  --warmup_cycles 1 \
  --output_dir $EXP \
  2>&1 | tee $EXP/train.log
```

Monitor progress from another terminal:

```bash
grep "Cycle" $EXP/train.log | tail -10
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

1. Train a LiDAR-only 2DGS scene in `HR-calib`:

```bash
cd /home/xzy/HR-calib
python train.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default_hrtiny_data.yaml \
  -ec configs/exp_kitti_3000_lidar_only_2dgs.yaml
```

2. Export per-frame rendered camera-depth maps as raw `.npy`:

```bash
cd /home/xzy/HR-calib
python eval.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default_hrtiny_data.yaml \
  -ec configs/exp_kitti_3000_lidar_only_2dgs.yaml \
  -t all \
  --save_rendered_depth \
  -s output/lidar_rt/kitti_center_3000_lidar_only_2dgs/scene_kc_5_50_t_cam_single_opa_pose_higs_default_gtt_fastlr/evals/rendered_depth_export
```

The exported depth maps land in:

```text
/home/xzy/HR-calib/output/lidar_rt/kitti_center_3000_lidar_only_2dgs/scene_kc_5_50_t_cam_single_opa_pose_higs_default_gtt_fastlr/evals/rendered_depth_export/all/rendered_depth
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
