# Copilot Instructions

## Commands

### Runtime warmup / extension build

The project does not have a single package build command; native renderer setup happens through the 3DGRUT warmup step after dependencies are installed:

```bash
pip install -r requirements.txt
pip install submodules/simple-knn
python tools/warmup_3dgrut.py
```

### Tests

Tests use stdlib `unittest` discovery from the repo root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run one test file:

```bash
python -m unittest discover -s tests -p 'test_threedgrut_backend.py' -v
```

Run one named test:

```bash
python -m unittest discover -s tests -p 'test_threedgrut_backend.py' -k hybrid_training_mode_overrides_backends -v
```

### Lint / type check

`pyproject.toml` contains configurations for these commands:

```bash
pyright lib tools tests
isort --check-only lib tools tests
```

## High-level architecture

- `tools/calib.py` is the main LiDAR-camera extrinsic calibration pipeline. It loads one dataset config (`-dc`) plus one experiment config (`-ec`), builds Gaussian scene assets, runs LiDAR/camera rendering, and optimizes a shared pose correction over many cycles.
- `tools/rgbd_calib.py` is a separate RGB-D extrinsic solver for paired RGB frames and depth maps. It reuses the lower-level matching and optimization code in `lib/utils/rgbd_calibration.py`, which is also used by matcher-assisted updates inside `tools/calib.py`.
- Configs are hierarchical. `lib.arguments.parse()` follows `parent_config` chains and merges leaf overrides into base configs, so behavior is usually defined across several YAML files rather than one file.
- Dataset loaders under `lib/dataloader/` normalize KITTI-calib, KITTI-360, Waymo, and PandaSet into shared `Camera` and `LiDARSensor` objects. Loader code also owns dataset-specific cache directories and coordinate-system normalization.
- `lib/scene/gaussian_model.py` stores Gaussian parameters and checkpoint compatibility logic. `lib/scene/camera_pose_correction.py` owns the trainable camera/LiDAR extrinsic correction and is the center of shared-pose optimization.
- `lib/gaussian_renderer/__init__.py` selects the LiDAR renderer, while `lib/gaussian_renderer/camera_render.py` selects the camera renderer. `lib/gaussian_renderer/threedgrut_backend.py` bootstraps the cached 3DGRUT/3DGUT runtime under `~/.cache/hr-tiny/3dgrut/<commit>`.

## Key conventions

- Most experiments rely on YAML inheritance rather than standalone configs. When changing behavior, inspect the full `parent_config` chain before editing or assuming defaults.
- Backend names are alias-heavy and tested. Camera aliases like `3dgut`, `gut`, and `3dgrut` map to `3dgut_rasterization`; LiDAR aliases like `3dgrut`, `threedgrt`, and `3dgut` map to `3dgrt`. `model.training_render_mode: hybrid_3dgrut` overrides both camera and LiDAR backends regardless of the leaf backend strings.
- Shared extrinsic optimization usually happens through `model.pose_correction.mode: all`, which means one LiDAR-to-camera transform is optimized across all frames instead of per-frame camera corrections.
- Quaternion math uses `[w, x, y, z]` ordering in both `tools/calib.py` and `lib/scene/camera_pose_correction.py`. Keep that convention consistent when adding rotation logic.
- KITTI-calib camera poses are derived in LiDAR world coordinates as `LiDAR_pose @ inv(Tr)`. The loader cache metadata in `lib/dataloader/kitti_calib_loader` assumes that convention.
- Waymo data is explicitly re-centered to the first ego pose and persisted in `cache/world_origin.pt` so camera and LiDAR coordinates stay numerically stable. Do not reintroduce raw global coordinates in Waymo paths.
- Long calibration runs are expected to be launched with `python -u ... | tee ...`; the `-u` flag matters because buffered stdout otherwise leaves log files empty during training.
- The first 3DGRUT-backed run is expected to populate the persistent cache. If backend behavior changes, check both `tools/warmup_3dgrut.py` and `lib/gaussian_renderer/threedgrut_backend.py`.
