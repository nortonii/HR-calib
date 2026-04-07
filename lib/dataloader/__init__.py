import os

import numpy
import torch
from lib.dataloader import kitti_loader
from lib.dataloader import pandaset_loader
from lib.dataloader import kitti_calib_loader
from lib.dataloader.gs_loader import SceneLidar
from lib.utils.console_utils import *


def load_scene(data_dir, args, test=False):
    if "waymo" in data_dir:
        from lib.dataloader import waymo_loader
        print(blue("\n====== [Loading] Waymo Open Dataset ======"))
        lidars, bboxes = waymo_loader.load_waymo_raw(data_dir, args)
    elif "kitti-calibration" in data_dir or "kitti_calib" in data_dir:
        print(blue("\n====== [Loading] KITTI-Calibration Dataset ======"))
        lidars, bboxes = kitti_calib_loader.load_kitti_calib_raw(data_dir, args)
    elif "kitti" in data_dir:
        print(blue("\n====== [Loading] KITTI Dataset ======"))
        lidars, bboxes = kitti_loader.load_kitti_raw(data_dir, args)
    elif "pandaset" in data_dir:
        print(blue("\n====== [Loading] PandaSet Dataset ======"))
        lidars, bboxes = pandaset_loader.load_pandaset_raw(data_dir, args)
    else:
        raise ValueError("Error: invalid dataset")

    print(blue("------------"))
    scene = SceneLidar(args, (lidars, bboxes), test=test)
    return scene
