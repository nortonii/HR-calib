#!/usr/bin/env bash
set -euo pipefail

# Repro script for PandaSet center-origin training in LiDAR-RT.
# - Uses GCC-11 toolchain to avoid chamfer_3D JIT build issues.
# - Uses correct config order: -dc (data) then -ec (experiment).

ROOT_DIR="/mnt/data16/xuzhiy/LiDAR-RT"
PYTHON_BIN="/mnt/data16/xuzhiy/HiGS-Calib/HiGS/bin/python"
GPU_ID="${GPU_ID:-1}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/pandaset_center_300_now.log}"

mkdir -p "${ROOT_DIR}/logs"

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${ROOT_DIR}"

echo "[run] python=${PYTHON_BIN}"
echo "[run] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[run] log=${LOG_PATH}"

"${PYTHON_BIN}" train.py \
  -dc configs/pandaset/static/1.yaml \
  -ec configs/pandaset/exp_300_center.yaml \
  > "${LOG_PATH}" 2>&1

echo "[done] training finished. log: ${LOG_PATH}"
