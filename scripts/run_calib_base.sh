#!/bin/bash
# Base calibration script — clean config, no extra tricks.
# Modify OUTPUT_DIR and GPU for each experiment.

OUTPUT_DIR="output/noise_inject_calib/testXX_description"
GPU=0

source /home/xzy/miniconda3/etc/profile.d/conda.sh
conda activate lidar-rt
cd /home/xzy/HR-calib

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=$GPU

exec python -u tools/reset_prim_calib.py \
    -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default.yaml \
    -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
    --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
    --init_trans_xyz 0.0718 0.1314 0.0960 \
    --total_cycles 300 --iters_per_cycle 200 \
    --translation_start_cycle 300 \
    --warmup_cycles 1 \
    --save_cycle_every 5 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
