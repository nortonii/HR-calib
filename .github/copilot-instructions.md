# Copilot Instructions for LiDAR-RT

LiDAR-RT is a research codebase for LiDAR sensor re-simulation using 3D Gaussian Splatting and OptiX ray tracing, targeting autonomous driving datasets (Waymo, KITTI-360, PandaSet). Published at CVPR 2025.

## Environment & Installation

- **Python**: ≥3.10 (tested on 3.11.9), **CUDA**: 12.1, **PyTorch**: 2.3.1
- **CMake**: must be ≥3.24.1 and <3.29 (3.29+ breaks submodule builds)
- **GPU drivers**: ≥530 required; in Docker set `NVIDIA_DRIVER_CAPABILITIES=compute,graphics`

```bash
conda create -n lidar-rt python=3.11.9 && conda activate lidar-rt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-lidar-tracer
pip install submodules/simple-knn
```

## Key Commands

```bash
# Training (data config + experiment config required)
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml

# Resume from checkpoint
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml -m output/scene_wd1/models/model_it_25000.pth

# Refine U-Net only (skip Gaussian stage)
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml -m <checkpoint> -r

# Evaluation
python eval.py -dc configs/kitti360/static/3.yaml -ec configs/exp.yaml \
    --model output/scene_kd3/models/model_it_25000.pth \
    --unet output/scene_kd3/models/unet.pth \
    --type test --save_eval --save_image --save_pcd

# Point cloud visualization
python viewer.py --pcd path/to/points.ply --point_size 3.0
```

No linter or test suite is configured.

## Architecture Overview

Training is a two-stage pipeline:

1. **Gaussian stage** (`train.py`): Optimizes 3D Gaussian parameters (position, SH color, scale, rotation, opacity) using differentiable ray tracing via the custom OptiX kernel (`submodules/diff-lidar-tracer`).
2. **Refinement stage** (`train.py` with `-r`): Trains a U-Net (`lib/scene/unet.py`) on top of frozen Gaussians to improve ray drop prediction.

**Core data flow:**

```
YAML configs → lib/arguments/parse()
             → lib/dataloader/load_scene()   # dataset-specific loader
             → SceneLidar (Gaussians + LiDARSensor)
             → raytracing() per frame         # lib/gaussian_renderer/
             → losses → Adam optimizer → densify/prune every 100 iters
```

**`lib/` layout:**

| Package | Purpose |
|---|---|
| `arguments/` | Config loading; `parse()` merges YAML files into a nested `Args` object |
| `dataloader/` | `load_scene()` dispatcher → Waymo / KITTI-360 / PandaSet loaders; `SceneLidar` class |
| `gaussian_renderer/` | `raytracing()` wrapper around `OptixTracer`; returns intensity, hit mask, drop mask |
| `scene/` | `GaussianModel`, `LiDARSensor`, `UNet`, and legacy camera/COLMAP classes (unused for LiDAR) |
| `utils/` | Losses, metrics (Chamfer3D, PSNR, SSIM, LPIPS), TensorBoard logging, SH utilities |

**Submodules:**
- `diff-lidar-tracer` — C++/CUDA OptiX ray-Gaussian tracer; the render output is `[intensity, hit_prob, drop_prob]`
- `simple-knn` — CUDA KNN used during Gaussian densification

## Configuration System

Configs are hierarchical YAML files merged in order: `base.yaml` → `exp.yaml` → dataset base → scene-specific. Each level can override parent keys.

```bash
# Two flags always required:
-dc configs/<dataset>/<split>/<scene_id>.yaml   # data/scene config
-ec configs/exp.yaml                             # experiment/loss config
```

Scene configs declare `parent_config` to inherit from a dataset base. Key fields:

- `scene_id` — unique output directory name (e.g., `wd1`, `kd3`)
- `dynamic` — whether the scene has moving objects (enables bounding-box handling)
- `frame_length` — `[start, end]` frame indices to load
- `eval_frames` — specific frame indices used for evaluation
- `model.voxel_size`, `model.sh_degree`, `model.dimension` (2D or 3D Gaussians)
- `opt.iterations`, `opt.lambda_*` — training duration and loss weights

Config values are accessed as attributes: `args.model.voxel_size`, `args.opt.iterations`.

## Key Conventions

**Dataset dispatch** is path-based in `lib/dataloader/__init__.py`: if `"waymo"` in `data_dir` → Waymo loader, `"kitti"` → KITTI loader, `"pandaset"` → PandaSet loader.

**Rendering background**: a 3-channel tensor `[0, 0, 1]` (intensity=0, hit=0, drop=1) on CUDA.

**Losses** are a weighted sum defined in `exp.yaml` (`lambda_intensity_l1`, `lambda_intensity_dssim`, `lambda_raydrop_bce`, `lambda_depth_l1`, `lambda_cd`, `lambda_normal`, `lambda_reg`).

**Densification**: every 100 iterations; densify if gradient > threshold and prune if opacity < 0.003 or scale > 0.1.

**Checkpoints**: saved every 15,000 iterations to `output/<scene_id>/models/`.

**Console output**: use helpers in `lib/utils/console_utils.py` (e.g., `blue()`, `red()`) for colored terminal messages, consistent with the rest of the codebase.
