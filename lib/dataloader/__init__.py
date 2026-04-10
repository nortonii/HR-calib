import os
import torch
from lib.dataloader import kitti_calib_loader
from lib.dataloader.gs_loader import SceneLidar
from lib.utils.console_utils import *


def load_scene(data_dir, args, test=False):
    if "kitti-calibration" in data_dir or "kitti_calib" in data_dir:
        print(blue("\n====== [Loading] KITTI-Calibration Dataset ======"))
        lidars, bboxes = kitti_calib_loader.load_kitti_calib_raw(data_dir, args)
    else:
        raise ValueError(f"Unsupported dataset: {data_dir}")

    print(blue("------------"))
    scene = SceneLidar(args, (lidars, bboxes), test=test)
    return scene
