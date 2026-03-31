# Installations

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preperation](#data-preperation)
    - [Waymo Open Dataset](#waymo-open-dataset)
    - [KITTI-360 Dataset](#kitti-360-dataset)
    - [PandaSet Dataset](#pandaset-dataset)
- [Configuration System](#configuration-system)

## Environment Setup

Our code is tested on NVIDIA 4090 GPU, `Ubuntu 24.04` with `CUDA 12.1` and `Python 3.11.9`.

>[!Warning]
> Thanks to [@cqf7419](https://github.com/cqf7419) for [testing](https://github.com/zju3dv/LiDAR-RT/issues/19), the following two conditions are necessary:
> 
> - An appropriate version of CMake can reduce compilation errors. Version 3.24.1 was successful, whereas 3.29 and 4.0 failed.
> - Suitable GPU drivers and settings are required: GPU drivers after version 530 are functional. Additionally, since the cluster relies on Docker images, it's necessary to configure the env with NVIDIA_DRIVER_CAPABILITIES=compute,graphics

```bash
# Clone the repo
git clone https://github.com/zju3dv/LiDAR-RT --recursive
cd LiDAR-RT

# Create new conda environment (python >=3.10 recommended)
conda create -n lidar-rt python=3.11.9
conda activate lidar-rt

# Install pytorch
# Be sure you have CUDA installed, we use CUDA 12.1 in our experiments
# NOTE: you need to make sure the CUDA version used to compile the torch is the same as the version you installedv
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Data Preperation

### Waymo Open Dataset

#### Data Structure

You need to download the raw .tfrecord file from the [official website](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3) (Waymo Perception Dataset V1.4.3), or from our [Google Drive](https://drive.google.com/file/d/1Y3gfleG9Mo9ZP7WPE65-P8NnvCRUYwVl/view), then please use the following structure to organize the data.

```
$ROOT/data/waymo
├── static/
│   ├── 1/
│   │   └── segment-xxxx_with_camera_labels.tfrecord
│   ├── 2/
│   │   └── segment-xxxx_with_camera_labels.tfrecord
│   └── ...
└── dynamic/
    ├── 1/
    │   └── segment-xxxx_with_camera_labels.tfrecord
    ├── 2/
    │   └── segment-xxxx_with_camera_labels.tfrecord
    └── ...
```

#### Training Segments

We list the training scenes used in our experiments below.

<a id="waymo-segments"></a>
<details>
  <summary>🔍 Click here to toggle the segments list.</summary>
  <hr>

**Static Scene**

| Data Path | Record Name |
|:--:|:--|
| $ROOT/data/waymo/static/1 | segment-11379226583756500423_6230_810_6250_810_with_camera_labels|
| $ROOT/data/waymo/static/2 | segment-10676267326664322837_311_180_331_180_with_camera_labels |
| $ROOT/data/waymo/static/3 | segment-17761959194352517553_5448_420_5468_420_with_camera_labels |
| $ROOT/data/waymo/static/4 | segment-1172406780360799916_1660_000_1680_000_with_camera_labels |

**Dynamic Scene**

| Data Path | Record Name |
|:--:|:--|
| $ROOT/data/waymo/dynamic/1 | segment-1083056852838271990_4080_000_4100_000_with_camera_labels |
| $ROOT/data/waymo/dynamic/2 | segment-13271285919570645382_5320_000_5340_000_with_camera_labels |
| $ROOT/data/waymo/dynamic/3 | segment-10072140764565668044_4060_000_4080_000_with_camera_labels |
| $ROOT/data/waymo/dynamic/4 | segment-10500357041547037089_1474_800_1494_800_with_camera_labels |

  <hr>
</details>


>[!Note]
> If you want to use other training segments from the *Waymo Open Dataset*, just download the .tfrecord file and put it into the correct data path mentioned above.
> Then, you need to write a config file similar to our provided.
>
> ```yaml
> parent_config: "configs/waymo/waymo_base.yaml"
> source_dir: "data/waymo/dynamic/1"      # change to your data path
>
> frame_length: [148, 197]                # change the frame range
> eval_frames: [158, 168, 178, 197]       # change the selected frames for evaluation
>
> scene_id: wd1                           # change the scene name
> dynamic: True                           # whether this scene is dynamic scene
> ```

### KITTI-360 Dataset

#### Data Structure

For **KITTI-360** dataset, we follow [kitti360 document](https://www.cvlibs.net/datasets/kitti-360/documentation.php) to organize data. You can download the data from the [official website](https://www.cvlibs.net/datasets/kitti-360/).

```
$ROOT/data/kitti360
├── data_pose/
│   ├── 2013_05_28_drive_0000_sync/
│   │   ├── cam0_to_world.txt
│   │   └── poses.txt
├── data_3d_raw/
│   ├── 2013_05_28_drive_0000_sync/
│   │   ├── velodyne_points/
│   │   │   ├── data/
│   │   │   │   ├── 0000000000.bin
│   │   │   │   └── ...
│   │   │   └── timestamps.txt
├── data_3d_bboxes/
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync.xml
│   │   └── ...
...
```

#### Training Segments

The detailed training segments used in our experiments can be found in the config files, here is one example:

```yaml
# configs/kitti360/dynamic/3.yaml
parent_config: "configs/kitti360/kitti_base.yaml"

frame_length: [8121, 8170]
eval_frames: [8130, 8140, 8150, 8160]

scene_id: kd3
dynamic: True
```

### PandaSet Dataset

#### Data Structure

We support PandaSet raw sequences through the official `pandaset` devkit. Please organize the extracted data as:

```
$ROOT/data/pandaset
├── 001/
│   ├── lidar/
│   ├── meta/
│   ├── camera/            # optional
│   └── annotations/       # optional
├── 002/
│   └── ...
└── ...
```

The loader reads the sequence-level LiDAR timestamps and poses from the PandaSet devkit, and can also use point-wise timestamps for the timestamp-based column-origin mode.
You can select the origin tracing mode through `lidar_origin_mode`:

- `center`
- `column_interp`
- `timestamp_column`
- `timestamp_point`

#### Training Example

```yaml
# configs/pandaset/static/1.yaml
parent_config: "configs/pandaset/pandaset_base.yaml"

frame_length: [0, 79]
eval_frames: [10, 20, 30, 40, 50, 60, 70]

scene_id: ps1
dynamic: False
```

## Configuration System

Our configuration system is structured as:

```
$ROOT/configs
├── waymo/
│   ├── dynamic/
│   │   ├── 1.yaml        # Specific scene config
│   │   ├── 2.yaml
│   │   └── ...
│   ├── static/
│   │   ├── 1.yaml
│   │   ├── 2.yaml
│   │   └── ...
│   └── waymo_base.yaml   # Scene base config
├── pandaset/
│   ├── static/
│   │   ├── 1.yaml
│   │   └── 1_timestamp.yaml
│   │   └── 1_timestamp_point.yaml
│   └── pandaset_base.yaml
├── kitti360/
│   ├── dynamic/
│   │   ├── 1.yaml
│   │   ├── 2.yaml
│   │   └── ...
│   ├── static/
│   │   ├── 1.yaml
│   │   ├── 2.yaml
│   │   └── ...
│   └── kitti_base.yaml
├── base.yaml              # Gaussian Splatting configs
└── exp.yaml               # LiDAR-RT configs
```
