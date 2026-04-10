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

## Calibration

```bash
python tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --output_dir output/my_experiment \
  --total_cycles 300 --iters_per_cycle 200 \
  --rotation_lr 0.002 \
  --use_gt_translation \
  --translation_start_cycle 9999 \
  --warmup_cycles 1
```
