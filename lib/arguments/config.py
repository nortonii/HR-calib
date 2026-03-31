import argparse
import os
import numpy as np
from . import yacs
from .yacs import CfgNode as CN
from lib.utils.cfg_utils import *

cfg = CN()
cfg.workspace = os.environ['PWD']
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train'
cfg.task = 'hello'
cfg.exp_name = 'test'
cfg.gpus = [0]
cfg.debug = False
cfg.resume = True

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

cfg.eval = CN()
cfg.eval.skip_train = False
cfg.eval.skip_test = False
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None

cfg.optim = CN()
cfg.optim.position_lr_init = 0.00016
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
cfg.optim.percent_dense = 0.01
cfg.optim.lambda_dssim = 0.2
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002
cfg.optim.densify_grad_source = "lidar"
cfg.optim.depth_supervision_interval = 1
cfg.optim.lambda_rgb_phase = 0.0

# introduced by myself
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1

cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.semantic_mode = 'logits'
cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True
cfg.model.nsg.include_obj = True
cfg.model.nsg.include_sky = False
cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True

cfg.model.use_color_correction = False
cfg.model.color_correction = CN()
cfg.model.color_correction.mode = 'sensor' # [image, sensor]
cfg.model.color_correction.use_mlp = False

cfg.model.use_pose_correction = False
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'frame' # [image, frame, all]
cfg.model.pose_correction.accumulate_steps = 1
cfg.model.pose_correction.stage_translation = False
cfg.model.pose_correction.translation_warmup_rotation_error_deg = 3.0
cfg.model.pose_correction.translation_warmup_patience = 5
cfg.model.pose_correction.use_gt_translation = False
cfg.model.pose_correction.sdf_loss_weight = 0.0
cfg.model.pose_correction.sdf_bootstrap_iterations = 0
cfg.model.pose_correction.sdf_max_points = 20000
cfg.model.pose_correction.sdf_canny_threshold1 = 100
cfg.model.pose_correction.sdf_canny_threshold2 = 200
cfg.model.pose_correction.sdf_distance_clip = 64.0
cfg.model.pose_correction.sdf_depth_weight_power = 1.0
cfg.model.pose_correction.update_method = "gradient"
cfg.model.pose_correction.matcher_name = "xoftr"
cfg.model.pose_correction.matcher_max_num_keypoints = 2048
cfg.model.pose_correction.matcher_min_matches = 20
cfg.model.pose_correction.matcher_min_depth_matches = 12
cfg.model.pose_correction.matcher_min_pnp_inliers = 8
cfg.model.pose_correction.matcher_ransac_reproj_thresh = 3.0
cfg.model.pose_correction.matcher_pnp_reproj_error = 4.0
cfg.model.pose_correction.matcher_pnp_iterations = 1000
cfg.model.pose_correction.matcher_depth_min = 0.1
cfg.model.pose_correction.matcher_depth_max = 80.0
cfg.model.pose_correction.matcher_depth_use_inverse = True
cfg.model.pose_correction.matcher_depth_percentile_low = 5.0
cfg.model.pose_correction.matcher_depth_percentile_high = 95.0
cfg.model.pose_correction.matcher_update_interval = 1
cfg.model.pose_correction.matcher_update_blend = 1.0


cfg.data = CN()
cfg.data.white_background = False
cfg.data.shuffle = True
cfg.data.split_test = -1
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_colmap = True
cfg.data.use_lidar = True
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False


cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.save_box = False
cfg.render.edit = False
cfg.render.novel_view = CN()
cfg.render.novel_view.name = 'test'
cfg.render.novel_view.mode = 'shift' # shift, rotate
cfg.render.novel_view.frame = None
cfg.render.novel_view.shift = [0, 0, 0] # shift for three axis in meters
cfg.render.novel_view.rotate = 0 # yaw offset

cfg.render.use_virtual_wrap = False
cfg.render.virtual_wrap = CN()
cfg.render.virtual_wrap.cam = [0] # first cam by default
cfg.render.virtual_wrap.min_shift = [0, 0, 0]
cfg.render.virtual_wrap.max_shift = [0, 0, 0]
cfg.render.virtual_wrap.min_rotate = 0
cfg.render.virtual_wrap.max_rotate = 0

cfg.viewer = CN()
cfg.viewer.frame_id = 0


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(cfg, args)
