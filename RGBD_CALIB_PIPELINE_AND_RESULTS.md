# RGB-D 标定流程与实验结果

## 1. 目标

在 `hr-tiny` 中补齐一条独立的 RGB-D 外参标定链路：

- **RGB**：真实相机图像
- **Depth**：由 LiDAR 监督训练后的 2DGS 渲染深度
- **跨模态匹配**：`vismatch / matchanything-roma`
- **模态内时序匹配**：同样统一使用 `matchanything-roma`

这条链路与原来的 3DGS 标定主训练解耦，便于单独做多帧外参优化、时序策略对比和闭环实验。

## 2. 当前实现

### 2.1 主要工具

| 文件 | 作用 |
| --- | --- |
| `tools/rgbd_calib.py` | 多帧 RGB-D 外参标定主入口 |
| `lib/utils/rgbd_calibration.py` | matcher 封装、深度编码、2D-3D 对应、PnP 初值、全局优化 |
| `tools/export_depth_cache_from_extrinsic.py` | 给定外参重新渲染 depth cache，用于 closed-loop 实验 |
| `tools/calib.py` | 加入了 cycle 式 matcher pose update 实验入口 |

### 2.2 标定流程

1. 用 LiDAR-only 2DGS checkpoint 渲染每帧深度。
2. 把深度编码成 matcher 可用的三通道图。
3. 用 `matchanything-roma` 做 RGB-depth 匹配，得到 `rgb_points / depth_points`。
4. 在 depth 图上取样深度并反投影为 3D 点。
5. 先做共享外参 PnP 初值，再做多帧联合鲁棒优化。
6. 可选加入 RGB-RGB 时序匹配，做 frame support / pairwise residual / staged refinement。

## 3. 多场景策略对比

评测场景：`6-50-t`、`10-50-t`、`7-200-t`  
每个场景测试两种初始深度偏置：`initial1`、`mild_bias`

### 3.1 策略平均结果

| 策略 | 平均旋转误差(°) | 平均平移误差(m) | 平均重投影(px) | 结论 |
| --- | ---: | ---: | ---: | --- |
| `pairwise025` | 1.1485 | 0.1517 | 16.6210 | **综合最优，最稳** |
| `prior01` | 1.0904 | 0.1923 | 19.4854 | 旋转略好，但平移和重投影更差 |
| `support_filter_weight` | 1.5536 | 0.4385 | 19.8568 | 场景敏感，稳定性最差 |

### 3.2 按场景平均

| 场景 | 平均旋转误差(°) | 平均平移误差(m) | 平均重投影(px) | 结论 |
| --- | ---: | ---: | ---: | --- |
| `7-200-t` | 1.1689 | 0.1322 | 14.0671 | 最稳定 |
| `6-50-t` | 1.2202 | 0.2194 | 17.9444 | 中等 |
| `10-50-t` | 1.4035 | 0.4309 | 23.9516 | **最差场景，明显 outlier** |

结论：后续默认优先用 `pairwise025`，并重点关注 `10-50-t` 这类困难场景。

## 4. pairwise025 迭代实验

### 4.1 不重渲染 depth，只把上一轮外参作为下一轮初始化

| 轮次 | 平均旋转误差(°) | 平均平移误差(m) | 平均重投影(px) |
| --- | ---: | ---: | ---: |
| `round1` | 1.1485 | 0.1517 | 16.6210 |
| `round2` | 1.1420 | 0.1508 | 2.3524 |
| `round3` | 1.1285 | 0.1511 | 2.3471 |

结论：

- `round2` 有小幅收益；
- `round3` 继续提升，但幅度已经很小；
- 这条路是**低风险、小收益**的 refinement。

### 4.2 每轮都按上一轮外参重新渲染 depth（closed-loop）

| 轮次 | 平均旋转误差(°) | 平均平移误差(m) | 平均重投影(px) | 平均有效帧数 | 平均匹配数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `round1` | 1.1485 | 0.1517 | 16.6210 | 33.5 | 20252.5 |
| `round2_closedloop` | 1.3075 | 0.1904 | 2.3199 | 13.7 | 759.7 |
| `round3_closedloop` | 1.4263 | 0.2158 | 2.2853 | 15.5 | 1004.2 |

结论：

- closed-loop 会让**重投影误差很好看**；
- 但 GT 相对误差反而持续变差；
- 同时有效帧和匹配数大幅塌缩。

### 4.3 “不能兼得”的原因

当前 closed-loop 的问题本质上是：

1. **优化目标变成“对自己的重渲染 depth 自洽”**，而不是对 GT 外参更准。
2. 一旦外参有偏，新的渲染 depth 也带偏，matcher 会继续强化这套偏差。
3. 支撑帧和匹配数变少后，优化更容易落入局部自洽解。

所以它能同时做到：

- 更低的 reprojection；
- 更差的 GT-relative 外参。

现阶段不建议把 closed-loop rerender 当默认方案。

## 5. 新增 cycle 实验：每 150 iter 用 matcher 更新一次 pose

## 5.1 实验设计

目标是验证这组约束是否可行：

- **相机分支只更新 pose**
- **LiDAR 分支只更新 opacity / covariance 相关参数**
- 每个 cycle 跑 `150` iter
- 每个 cycle 结束时，用 `matchanything-roma` 在 **GT RGB vs 当前 pose 下的 2DGS 渲染深度** 上做一次外参更新

为此在 `tools/calib.py` 中新增了：

- `--matcher_pose_update`
- `--matcher_update_interval`
- `--matcher_update_blend`
- `--matcher_name`
- `--lidar_updates_opacity_covariance_only`

并把 matcher 更新接到了 `pose_correction.apply_relative_camera_transform(...)`。

## 5.2 一个关键实现细节

相机渲染这条链如果走默认 `rasterization`，`render_camera(...)[\"depth\"]` 会是全 0，不能直接做 matcher 反投影。

所以在 cycle matcher update 内部，实际强制切到 **`surfel_rasterization`** 取 metric depth；否则虽然能 match 到 2D 点，但拿不到有效 3D 点。

## 5.3 5-50-t 实验结果

实验命令：

```bash
/home/xzy/miniconda3/envs/lidar-rt/bin/python tools/calib.py \
  -dc configs/kitti_calib/static/5_50_t_cam_single_opa_pose_higs_default_local_data.yaml \
  -ec configs/exp_kitti_10000_cam_single_opa_pose_higs_default.yaml \
  --checkpoint /home/xzy/HR-calib/output/lidar_rt/kitti_center_3000_lidar_only_2dgs/scene_kc_5_50_t_cam_single_opa_pose_higs_default_gtt_fastlr/models/model_it_3000.pth \
  --init_rot_deg 9.9239 --init_rot_axis 0.5774 0.5774 0.5774 \
  --total_cycles 5 --iters_per_cycle 150 \
  --matcher_pose_update --matcher_update_interval 1 \
  --matcher_name matchanything-roma \
  --lidar_updates_opacity_covariance_only \
  --save_cycle_every 1 \
  --output_dir output/calib/cycle_matcher_poseonly_lidaropacov_5_50_t
```

### 5.3.1 总结果

| 指标 | 初始 | 最终 |
| --- | ---: | ---: |
| 旋转误差(°) | 9.9238 | **1.0009** |
| 平移误差(m) | 0.1801 | **0.1072** |

### 5.3.2 每个 cycle 的 matcher 更新情况

| Cycle | rot_err(°) | T_err(m) | matcher frames | matcher matches | matcher reproj(px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.1043 | 0.1136 | 47 | 56561 | 5.527 |
| 2 | 1.0317 | 0.1195 | 50 | 78173 | 4.005 |
| 3 | 0.9860 | 0.1262 | 49 | 75437 | 3.918 |
| 4 | 1.0009 | 0.1185 | 50 | 77025 | 4.887 |
| 5 | 1.0009 | 0.1072 | 49 | 74037 | 4.704 |

结论：

- 这条“**matcher 更新 pose + LiDAR 只调 opacity/covariance**”的实验链已经能稳定工作；
- 第一个 cycle 就能把大误差快速拉回；
- 后续 cycle 是小步 refinement，波动不大；
- 至少在 `5-50-t` 上，这个思路是可行的。

### 5.3.3 继续跑到 10 cycles

后续又从第 5 轮继续跑到了第 10 轮，结果如下：

| Cycle | rot_err(°) | T_err(m) | matcher frames | matcher matches | matcher reproj(px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.9190 | 0.1010 | 49 | 73950 | 4.122 |
| 7 | **0.9031** | 0.1059 | 49 | 72220 | 4.091 |
| 8 | 0.9342 | 0.1041 | 50 | 71212 | 4.324 |
| 9 | 0.9500 | 0.1074 | 50 | 73406 | 4.287 |
| 10 | 0.9606 | 0.1182 | 50 | 73069 | 6.049 |

结论：

- 第 6~7 轮还有一点收益；
- 之后基本进入平台期，并开始轻微震荡；
- 这组设置下推荐轮数大约在 **6~7 cycles**。

## 6. 新尝试：match 监督颜色，整图 RGB 只更新 pose，LiDAR 监督几何

这次按照新的目标额外实现了一条训练分支：

1. **LiDAR depth loss**：继续监督几何。
2. **整图相机 L1/SSIM**：只更新相机位姿，不更新 Gaussians。
3. **match color supervision**：用 `matchanything-roma` 在 `GT RGB ↔ rendered depth` 上取对应点，把 GT 颜色映射到 depth 像素，再对渲染 RGB 做稀疏颜色监督。

对应新增参数：

- `--matcher_color_supervision`
- `--matcher_color_weight`
- `--camera_rgb_pose_only`

### 6.1 实现细节

- depth 匹配仍然使用 `surfel_rasterization` 生成的 metric depth；
- 颜色监督的梯度不再走 surfel RGB 分支，而是改用普通 camera render 的 RGB，
  避免 2DGS surfel backward 出现 shape mismatch；
- 因此这条分支实际是：
  - **depth render**：负责 matcher 建立跨模态对应；
  - **rgb render**：负责把 sparse GT color loss 反向传到颜色参数。

### 6.2 从大初值直接开始的结果（负结果）

实验目录：

- `output/calib/matcher_color_poseonly_lidargeom_5_50_t/`

配置：

- 初始旋转误差：`9.9238°`
- `5 cycles × 150 iter`

结果：

| Cycle | rot_err(°) | T_err(m) | loss_depth | loss_rgb | loss_match_color |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.7662 | 0.3075 | 0.6323 | 0.3438 | 0.2733 |
| 2 | 43.6323 | 0.4423 | 0.6345 | 0.3332 | 0.2629 |
| 3 | 47.1749 | 0.5312 | 0.6186 | 0.3321 | 0.2776 |
| 4 | 50.6894 | 0.5932 | 0.5903 | 0.3332 | 0.2321 |
| 5 | 50.5779 | 0.6368 | 0.5727 | 0.3349 | 0.2083 |

结论：

- 这条设置会**明显破坏 pose**；
- 虽然 photometric / sparse color loss 在下降，但外参持续漂走；
- 所以它目前**不能直接替代**之前的 matcher pose update。

### 6.3 从较好 pose 开始的结果（仍然会拉坏旋转）

实验目录：

- `output/calib/matcher_color_poseonly_lidargeom_from_cycle8_5_50_t/`

起点：

- 从前一组较好实验的 `cycle_0008.pth` 恢复，再切到这条新监督方式。

结果：

| Cycle | rot_err(°) | T_err(m) | loss_depth | loss_rgb | loss_match_color |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 19.4898 | 0.0970 | 0.5329 | 0.3505 | 0.2564 |
| 10 | 26.0002 | 0.2223 | 0.5270 | 0.3402 | 0.2705 |

结论：

- 即便从较好的 pose 出发，这条分支仍会把**旋转明显拉坏**；
- 说明问题不只是“大初值不稳”，而是这套目标组合本身对 pose 约束不一致。

### 6.4 当前判断

这条新实验之所以不稳，主要原因是：

1. **整图 photometric loss 在 pose-only 设置下仍然过强**，会推动 pose 去解释颜色差异；
2. **matcher color loss 只约束颜色，不直接约束几何一致性**；
3. LiDAR depth 负责的是几何，但不足以抵消相机侧颜色目标带来的错误 pose 梯度。

当前更合理的下一步，不是直接继续加轮数，而是改成：

- 对整图 camera pose loss 加强正则 / 降低权重；
- 或者只在 matcher inlier 区域内对 pose 做 photometric；
- 或者把 matcher color supervision 完全限制为 **颜色分支 + 固定 pose** 的 refinement。

## 7. 目前建议

1. RGB-D 多帧标定默认继续以 **`pairwise025`** 为主。
2. closed-loop rerender 作为负结果保留，不建议默认启用。
3. cycle matcher pose-update 的可用轮数建议控制在 **6~7 cycles**。
4. “match 颜色 + pose-only camera + LiDAR 几何” 当前作为**负结果**保留，暂不建议直接扩场景。
5. cycle 实验下一步建议直接扩到：
   - `6-50-t`
   - `10-50-t`
   - `7-200-t`
6. 如果要更稳，可以继续试：
   - `matcher_update_blend < 1`
   - 只在高 support frame 上做 cycle update
   - 每轮 matcher update 后做一次 inlier/support 过滤

## 8. 结果文件

- 多场景策略汇总：`output/multiscene_strategy_eval/summary.json`
- 迭代不重渲染：`output/multiscene_pairwise025_iter3/summary.json`
- closed-loop 重渲染：`output/multiscene_pairwise025_closedloop/summary.json`
- 新 cycle 实验：`output/calib/cycle_matcher_poseonly_lidaropacov_5_50_t/`
- match 颜色监督实验：`output/calib/matcher_color_poseonly_lidargeom_5_50_t/`
- 从较好 pose 启动的 match 颜色监督实验：`output/calib/matcher_color_poseonly_lidargeom_from_cycle8_5_50_t/`
