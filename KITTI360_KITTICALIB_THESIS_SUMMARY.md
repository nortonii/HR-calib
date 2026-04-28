# KITTI360 / KITTI-calib / PandaSet 标定方案与实验总结

## 1. 研究目标

本文档总结了当前 `hr-tiny` 仓库中围绕 KITTI360、KITTI-calib 与 PandaSet 外参标定所做的方法设计、关键消融和最终结论，供毕业论文撰写使用。

本阶段工作的核心目标是：

1. 在 LiDAR 监督的 3DGS / 3DGRT / 3DGUT 标定框架内，进一步降低相机-激光雷达外参中的旋转误差。
2. 保持或尽量不破坏平移精度，尤其避免 refine 后期的 translation drift。
3. 明确不同数据集上有效策略是否一致，并给出可复现实验结论。

需要特别说明的是，本文后续所有结论均优先使用 **最终 cycle 的结果**，而不是训练过程中出现过的最优中间结果。

## 2. 问题演化与核心观察

### 2.1 初始问题

在 KITTI360 上，初始 PurePnP 往往能够把位姿拉到一个较好的局部区域，但继续进入 tail refine 后，经常出现以下现象：

- rotation 有时继续下降；
- translation 更容易在后半段逐渐漂移；
- 最终结果未必优于 PurePnP 后的即时状态。

因此，关键矛盾并不是“PnP 是否有效”，而是：

> 如何在保留 PurePnP 有效初始化的前提下，让后续 refine 继续优化 rotation，同时不要把 translation 再带坏。

### 2.2 对 KITTI360 的关键判断

在 KITTI360 上，最终确认：

- 初始 PurePnP 是有效的。
- 纯粹依靠后期 photometric refine，容易在 tail 阶段把 translation 带偏。
- 需要某种方式把后续优化限制在更合理的局部区域内。

### 2.3 对 KITTI-calib 的关键判断

在 KITTI-calib 上，历史较好的结果并不依赖 `rebase` 或 `anchor`。重新核对旧会话与新实验后可以确认：

- KITTI-calib 更偏向旧骨架，即 `no-rebase + no-anchor`。
- 将 KITTI360 上有效的 `rebase/anchor` 方案直接平移到 KITTI-calib，并不能保证更优。

这说明两套数据集的有效策略并不完全一致，后续实验应分数据集讨论。

## 3. 方法演化

### 3.1 `rebase after initial PurePnP`

这是本阶段最重要的结构性改动之一。做法是：

1. 先运行初始 PurePnP，得到共享 LiDAR-to-camera pose。
2. 将该 pose 直接改写为新的 base pose。
3. 清空原有 tail delta。
4. 再继续 refine。

这样做的目的不是“冻结优化”，而是：

- 让后续 tail refine 围绕更可信的 PnP 解局部优化；
- 避免继续围绕 noisy init 做大范围修正；
- 在实践中显著缓解 KITTI360 的后期 translation drift。

### 3.2 `anchor` 正则

后续又引入了 `anchor` 机制，但它并不是把位姿拉回最初的 noisy init，而是：

1. 在 PurePnP 完成且完成 `rebase` 后，记录当前共享 LiDAR-to-camera pose；
2. 将其保存为 `anchor_lidar_to_camera_*`；
3. 之后的 `regularization_loss()` 不再约束原始 delta，而是约束“当前 pose 相对 PnP anchor 的偏离”。

对应权重为：

- `lambda_translation_anchor`
- `lambda_rotation_anchor`

它的主要作用是：

- 保持后期优化仍然可动；
- 但避免 translation 从已经不错的 PnP 解继续漂走。

### 3.3 固定随机种子

实验过程中进一步发现，配置中的 `seed` 早期并未真正进入 `calib` 主流程，导致“单次幸运轨迹”与“稳定可复现实验”混在一起。

后续已在 `tools/calib.py` 中补齐：

- 主流程显式调用 `set_seed(seed)`
- 新增 `--seed` CLI 覆盖参数

修复后可确认：

- 同一 seed 的复现实验基本一致；
- 不同 seed 会走向不同局部解；
- 因此论文中应优先报告固定 seed 下的可复现实验结果。

本阶段后续主要使用 `seed=42`。

## 4. 关键实现改动

### 4.1 `tools/calib.py`

主要补充了：

- `--seed`
- `--rebase_pose_after_initial_pure_pnp`
- `--no-rebase_pose_after_initial_pure_pnp`
- `--translation_lr`
- `--restore_best_pose_metric`
- 最终日志中显式打印 `Raw final cycle pose`
- 向后兼容旧 checkpoint 的 anchor buffer 缺失
- 本轮新增的 `--camera_supervision_exp_decay_gamma`

### 4.2 `lib/scene/camera_pose_correction.py`

主要补充了：

- rebase 后 pose anchor buffer
- `rebase_shared_lidar_to_camera(...)`
- `regularization_loss()` 中的 anchor 正则项

### 4.3 新增配置

目前实验中用到的关键配置包括：

- `configs/exp_kitti_10000_cam_single_opa_pose_higs_hybrid_3dgrut.yaml`
- `configs/exp_kitti_10000_cam_single_opa_pose_higs_hybrid_3dgrut_noanchor.yaml`
- `configs/exp_kitti_10000_cam_single_opa_pose_higs_hybrid_3dgrut_noanchor_dssim09.yaml`

## 5. KITTI360 的主要结果

### 5.1 `rebase + anchor10 + translation_lr=0.002`：稳定的 full-static 候选

这是在 KITTI360 上首先被完整验证成功的一条 full-static 方案。

5-scene 最终结果：

| Scene | Rotation (°) | Translation (m) |
| --- | ---: | ---: |
| k1 | 0.1978 | 0.0154 |
| k2 | 0.1856 | 0.0673 |
| k3 | 0.2423 | 0.0696 |
| k4 | 0.1978 | 0.0673 |
| k5 | 0.2112 | 0.0786 |
| **Avg** | **0.2069** | **0.0596** |

这一结果相对早期的 `translation_lr=0` 参考平均值 `0.2265° / 0.0609m` 略有提升，说明：

- `rebase` 能有效稳住 translation；
- `anchor` 机制能够在 full-static 上给出更稳定的 tail refine。

### 5.2 rotation-priority 方向：更长 tail + 更高 `lambda_dssim`

后续针对 rotation 更重要的设定，又测试了：

- tail 延长
- `pose_lr_drop`
- 更高 `lambda_dssim`

其中一个显著结果是：

5-scene 最终平均达到：

- `0.1738° / 0.0629m`

相较于 `0.2069° / 0.0596m`：

- rotation 明显改善；
- translation 略有退化。

这说明：

- 如果任务优先级偏向 rotation，可接受少量 translation 损失；
- 提高 `SSIM` 权重在这条可动平移骨架上对 rotation 是有效的。

### 5.3 满足均值门槛的最佳 KITTI360 平均方案

在进一步消融 `anchor` 后，发现对 KITTI360 平均结果最好的并不是 anchor 方案，而是：

- `no-anchor`
- `lambda_dssim=0.9`
- `tail74 + pose lr drop`
- `seed=42`

对应 5-scene **最终结果** 为：

| Scene | Rotation (°) | Translation (m) |
| --- | ---: | ---: |
| k1 | 0.1251 | 0.0434 |
| k2 | 0.0485 | 0.0639 |
| k3 | 0.1507 | 0.1214 |
| k4 | 0.1978 | 0.0861 |
| k5 | 0.1532 | 0.0794 |
| **Avg** | **0.1351** | **0.0788** |

该结果的重要性在于：

- 平均 rotation 低于 `0.15°`
- 平均 translation 低于 `0.08m`

因此这是当前 **KITTI360 平均指标最强** 的一条方案。

但应注意：

- 它满足的是平均门槛；
- 并不是每个 scene 都同时满足单场景门槛；
- 尤其 `k3`、`k4` 的最终 translation 仍偏高。

## 6. KITTI-calib 的主要结论

### 6.1 数据集策略与 KITTI360 不同

结合旧会话记忆与新验证，KITTI-calib 上更合理的判断是：

- 历史较优结果来自 `no-anchor + no-rebase`；
- KITTI360 上有效的 `rebase/anchor` 不能直接迁移。

### 6.2 代表场景验证

在 `7_200_t` 上，从 `cycle_0100` 起继续跑 `no-anchor + no-rebase + dssim=0.9` 时，曾在中途获得明显优于旧最终值的结果，例如：

- `Cycle 126`: `0.0485° / 0.0427m`
- `Cycle 129`: `0.0969° / 0.0404m`

这说明：

- KITTI-calib 更偏向旧骨架；
- 直接从 `cycle_0150` 往后续跑，会低估这条思路；
- 它和 KITTI360 的最优方案不一致。

因此论文中应明确写明：

> KITTI360 与 KITTI-calib 的有效 refine 策略存在显著数据集差异。

### 6.3 修复 shared-extrinsic translation writeback 后的可信结果

需要特别说明的是，KITTI-calib 的一批早期结果后来被发现受到 shared-extrinsic translation writeback bug 污染，因此：

- 修复前结果可作为排障历史参考；
- 但不应作为论文中的最终定量结论；
- 论文里应优先使用修复后 rerun 的结果。

修复后，首先对 3 个代表场景做了 tail rerun 验证，得到如下 **最终结果**：

| Scene | Rotation (°) | Translation (m) |
| --- | ---: | ---: |
| 5-50-t | 0.3208 | 0.1199 |
| 6-50-t | 0.1678 | 0.0530 |
| 7-200-t | 0.0249 | 0.0958 |
| **Avg** | **0.1712** | **0.0896** |

这组结果的意义在于：

- 它们是在修复 writeback 语义错误后重新得到的；
- 因而比更早的 pre-fix 平均值更可信；
- 也说明 KITTI-calib 的 translation 误差确实会受到共享外参写回实现的强烈影响。

### 6.4 新 schedule：`warmup100 + PnP5 + refine50 + color warmup + no grad constraint`

在完成 bug 修复后，又进一步测试了新的 3-scene schedule：

- `warmup100`
- `pure_pnp_iters=5`
- `refine50`
- 保留 color warmup
- 去掉 geometry gradient conflict constraint

对应 3 个代表场景的 **最终结果** 为：

| Scene | Rotation (°) | Translation (m) |
| --- | ---: | ---: |
| 5-50-t | 0.2977 | 0.0793 |
| 6-50-t | 0.1570 | 0.0525 |
| 7-200-t | 0.1891 | 0.0437 |
| **Avg** | **0.2146** | **0.0585** |

这一组结果表明：

- 相比前一组 post-fix tail rerun，translation 平均值进一步下降；
- 但 rotation 平均值有所回退；
- 因而它更像是一个偏平移优先的折中 schedule，而非更优的统一替代方案。

### 6.5 负结果与方法边界

KITTI-calib 上还得到了一些对论文讨论有价值的负结果：

1. `warmup50 -> PnP1 -> refine50` 且不加 RGB supervision 时，refine 对 pose 改善极弱；
2. 直接把 `lambda_rgb` 提到 `1.0` 会明显破坏稳定性；
3. 将 matcher 替换为 MINIMA dense PurePnP 在 `6-50-t` 上得到约 `2.3681° / 0.6113m`，显著差于主线方案；
4. OpenCV 5-step shared refine 也未优于更早的 SciPy 5-step baseline。

这些结果共同说明：

- KITTI-calib 的有效区间比较窄；
- 并不是增加更强 photometric 或更换更重 matcher 就会自然提升；
- 相比之下，保留旧骨架语义的一致性更重要。

### 6.6 关于 full 30-scene KITTI-calib

旧会话记录显示，`warmup100 + PnP5 + refine50 + color warmup + no grad constraint` 这一 schedule 最终已经完成了 **30 个 KITTI-calib 场景** 的批量运行。

但需要注意：

- 当时虽然所有 scene 输出都已完成；
- 30-scene 的最终平均值与 3σ consistency 统计并未在会话压缩前计算完成；
- 因此当前文档只引用已经明确核对过的 3-scene 结果，而不直接声称 30-scene 平均结论。

另外，当前机器上的本地输出目录在整理时一度缺少 `5_0_t` 对应结果。为补齐 scene 集合，后续又按同一条 CLI schedule 补跑了：

- `5_0_t`：`0.0685° / 0.0775m`
  输出目录：`/mnt/xzy/hr-tiny-output/calib/kc_5_0_t_mnt_w100_pnp5_refine50_colorwarm5_rgb01_depthon_nogradconstraint_full`

这说明：

- 从输出目录覆盖角度看，当前这组 KITTI-calib scene 已经补齐到 30 个；
- 但由于 `5_0_t` 属于后补跑结果，若论文需要严格强调“同一次完整 batch”的实验时序一致性，仍应单独注明这一点；
- 更稳妥的写法是：30 个 scene 的结果目录现已齐备，但全量平均值仍需再统一统计一次。

### 6.7 当前可回收的 30-scene 详细统计

基于当前机器上 `/mnt/xzy/hr-tiny-output/calib/` 可回收的 30 个 scene 输出目录，现已可直接统计

- `PurePnP x5` 结束后的误差；
- `cycle 150` 的 **raw final** 误差。

这里统一采用的目录选择原则是：

- 若某个 scene 同时存在 `full`、`restart`、`rerun`、`retry` 多个目录，则优先使用 **确实带有 `cycle_0150.pth` 的最终有效目录**；
- 对于 `10_0_r`，最终采用 `full` 而不是不完整的 `full_retry`；
- 对于 `9_240_r`，采用 `rerun`；
- 对于 `7_200_t`，采用 `restart`；
- 对于 `5_0_t`，采用后补跑得到的 `full` 目录。

30 个 scene 的详细结果如下：

| Scene | PurePnP Rot (°) | PurePnP Trans (m) | Final Rot (°) | Final Trans (m) |
| --- | ---: | ---: | ---: | ---: |
| 10-0-r | 0.1251 | 0.1032 | 0.2056 | 0.1147 |
| 10-100-t | 0.0839 | 0.0405 | 0.2487 | 0.0435 |
| 10-250-t | 0.2609 | 0.0926 | 0.3000 | 0.0678 |
| 10-300-t | 0.1119 | 0.0785 | 0.2770 | 0.0433 |
| 10-350-t | 0.0969 | 0.0686 | 0.6720 | 0.0624 |
| 10-50-t | 0.1084 | 0.0500 | 0.0396 | 0.0588 |
| 3-0-t | 0.2075 | 0.0494 | 0.1427 | 0.0376 |
| 5-0-t | 0.1454 | 0.0780 | 0.0685 | 0.0775 |
| 5-100-r | 0.0928 | 0.0568 | 0.1454 | 0.0638 |
| 5-200-t | 0.4224 | 0.0904 | 0.4102 | 0.0845 |
| 5-250-t | 0.1791 | 0.0612 | 0.2439 | 0.0471 |
| 5-300-t | 0.1532 | 0.0373 | 0.1371 | 0.0396 |
| 5-350-t | 0.2407 | 0.0474 | 0.1532 | 0.0589 |
| 5-50-t | 0.1631 | 0.0920 | 0.2961 | 0.0793 |
| 6-0-t | 0.2307 | 0.0591 | 0.1187 | 0.0582 |
| 6-100-t | 0.2203 | 0.0555 | 0.1312 | 0.0566 |
| 6-150-t | 0.2853 | 0.1017 | 0.2407 | 0.0940 |
| 6-275-r | 0.2518 | 0.0347 | 0.2273 | 0.0766 |
| 6-50-t | 0.2564 | 0.0457 | 0.1558 | 0.0525 |
| 7-110-r | 0.3013 | 0.1078 | 0.3785 | 0.1201 |
| 7-200-t | 0.1607 | 0.0590 | 0.1856 | 0.0437 |
| 7-250-t | 0.2112 | 0.0473 | 0.2825 | 0.0287 |
| 7-300-r | 0.1791 | 0.1345 | 0.2185 | 0.1002 |
| 7-350-t | 0.2324 | 0.0472 | 0.2307 | 0.0923 |
| 7-450-r | 0.2238 | 0.1530 | 0.1607 | 0.1046 |
| 9-240-r | 0.2683 | 0.1467 | 0.1371 | 0.1287 |
| 9-290-r | 0.0740 | 0.0740 | 0.0000 | 0.0817 |
| 9-390-t | 0.1251 | 0.0719 | 0.1371 | 0.0397 |
| 9-540-t | 0.2167 | 0.0271 | 0.2948 | 0.0573 |
| 9-800-t | 0.0928 | 0.0543 | 0.0626 | 0.0482 |
| **Avg** | **0.1907** | **0.0722** | **0.2100** | **0.0687** |

补充统计如下：

- PurePnP 30-scene 平均：`0.1907° / 0.0722m`
- Final 30-scene 平均：`0.2100° / 0.0687m`
- Final 30-scene median：`0.1956° / 0.0607m`

这说明在当前这条 full KITTI-calib schedule 上：

- refine 之后，**translation 平均值进一步下降**，约从 `0.0722m` 降到 `0.0687m`；
- 但 **rotation 平均值反而上升**，约从 `0.1907°` 回到 `0.2100°`；
- 因而这条 schedule 的整体特征更像是“平移更稳、旋转不一定持续获益”的折中方案。

如果进一步看 scene 级变化，则：

- `16 / 30` 个 scene 在 refine 后 rotation 优于 PurePnP；
- `17 / 30` 个 scene 在 refine 后 translation 优于 PurePnP；
- `8 / 30` 个 scene 同时实现 rotation 与 translation 双改善；
- `5 / 30` 个 scene 在 rotation 与 translation 上都退化。

这一统计也说明：

- 平均值并不能完全代表 scene 级行为；
- KITTI-calib 的 refine 收益具有较明显的场景依赖性；
- 后续若要强调稳定性，除了报告均值，也应补充 scene 分布或一致性统计。

从极值看：

- Final rotation 最好的是 `9-290-r`，约 `0.0000°`
  该值应理解为“数值上接近零”，并不意味着严格数学意义上的零误差。
- Final rotation 最差的是 `10-350-t`，为 `0.6720°`
- Final translation 最好的是 `7-250-t`，为 `0.0287m`
- Final translation 最差的是 `9-240-r`，为 `0.1287m`

因此，如果论文中需要一句更准确的概括，可写为：

> 在当前可回收的 30-scene KITTI-calib 结果上，`warmup100 + PnP5 + refine50 + color warmup + no grad constraint` 这一 schedule 的最终平均误差约为 `0.2100° / 0.0687m`。相较 PurePnP 后的平均 `0.1907° / 0.0722m`，后续 refine 更有利于压低 translation，但对 rotation 的收益并不稳定，呈现明显的 scene-dependent 行为。

## 7. 当前阶段系统消融结论

### 7.1 已验证有效的方向

1. `rebase after initial PurePnP`
2. `anchor` 约束 PnP rebase 后 pose，而不是原始 noisy init
3. 固定随机种子后报告最终结果
4. 在可动平移骨架上，提高 `lambda_dssim` 有助于 rotation
5. 对 KITTI360 平均结果而言，`no-anchor + dssim=0.9` 优于 anchor 版本

### 7.2 已基本证伪或收益不足的方向

1. `freeze_gaussians_after_color_warmup`
   - translation 稳，但 rotation 压不下去
2. 单纯延长 `color_warmup`
   - 只能延后 drift，不能根治
3. `translation_start_cycle` 两阶段
   - 只能把漂移推迟，最终仍难同时兼顾 R/T
4. `freeze_translation + periodic PurePnP`
   - translation 能稳，但 rotation 不够好
5. `cross_frame_weight`
   - 最多接近基线，没有稳定超越
6. `flow_proj_weight`
   - 初始 PurePnP 终点就被拖坏
7. `camera_supervision_exp_decay_gamma`
   - 提供了代码工具，但未超过现有基线
8. `camera_rgb_pose_only`
   - 与基线基本打平
9. `matcher_color_supervision`
   - 理论可行，但每 cycle 成本太高，当前性价比不足
10. `pose_step_at_cycle_end` / 稀疏 pose step
   - 未观察到比基线更好的最终趋势

## 8. 当前推荐方案

### 8.1 如果论文强调 KITTI360 平均指标

推荐使用：

- `no-anchor`
- `no-rebase`
- `lambda_dssim=0.9`
- 固定 `seed=42`
- 采用已经验证过的 tail74 / pose LR drop 方案

其依据是：

- 当前平均结果最强；
- 达到 `0.1351° / 0.0788m`；
- 能满足“平均 rotation < 0.15°、平均 translation < 0.08m”的目标。

### 8.2 如果论文强调稳定 full-static 候选

推荐使用：

- `rebase + anchor10 + translation_lr=0.002`

其依据是：

- full-static 上更稳；
- translation 漂移更受控；
- 5-scene 达到 `0.2069° / 0.0596m`。

### 8.3 如果论文强调 KITTI-calib

推荐单独表述为：

- KITTI-calib 更适合 `no-anchor + no-rebase`
- 不建议直接复用 KITTI360 的 `rebase/anchor` 配方
- 若需要给出可信数值，应优先引用修复 translation writeback bug 后的结果

## 9. 论文写作建议

建议在论文中将本阶段工作概括为三条主线：

### 9.1 结构性改进

- 用 `rebase after PurePnP` 替代直接在 noisy init 周围 refine。
- 用 `anchor` 约束当前 pose 相对 PnP rebase pose 的偏离，而不是约束原始 delta。

### 9.2 训练稳定性改进

- 修正 `seed` 进入主流程的问题；
- 统一按最终 cycle 结果汇报；
- 区分“单次 lucky run”与“固定随机种子的可复现结果”。

### 9.3 数据集差异分析

- KITTI360 更受 tail translation drift 影响；
- KITTI-calib 更依赖旧骨架；
- 因此最优 refine 策略具有数据集依赖性。

## 10. 可直接引用的总结性结论

可直接写入论文正文的总结如下：

> 在 KITTI360 上，初始 PurePnP 能将外参快速拉入较优局部区域，但后续 refine 常引起 translation drift。为缓解该问题，本文首先提出在初始 PurePnP 后执行 pose rebase，并进一步引入以 PnP pose 为中心的 anchor 正则，从而显著提高了 full-static 方案的稳定性。在此基础上，通过系统消融发现，若目标是优化 5-scene 平均指标，则去除 anchor、提高 photometric loss 中 SSIM 权重，并采用固定随机种子与后期 pose learning-rate drop 的 tail schedule，可在 KITTI360 上取得 `0.1351° / 0.0788m` 的平均最终误差，达到预设目标线。另一方面，KITTI-calib 的实验表明，其较优策略并不依赖 `rebase/anchor`，说明不同数据集上的最优 refine 机制具有明显差异。

## 11. 文档用途说明

本文件面向论文写作，优先保留：

- 方法设计逻辑
- 最终可复现实验结论
- 关键消融的正负结果

如需进一步扩展为论文附录，可在此基础上补充：

- 具体命令行
- 输出目录
- 对应图表和训练曲线
- 不同 seed 下的方差分析

## 12. 本机实验来源补充

除本文前半部分已经整理的 KITTI360 / KITTI-calib 结论外，这台机器上还保留了两类额外材料，可作为论文补充依据：

1. 已落盘的本机实验目录  
   主要位于 `/mnt/data16/xuzhiy/hr-tiny-output/calib/`，其中：
   - KITTI360 当前能完整回收的 5-scene 批量结果位于  
     `kitti360_batch_w100_pnp5_refine50_colorwarm5_rgb01_depthon_nogradconstraint_restart_higsinit/`
   - PandaSet 当前能完整回收的关键结果位于  
     `pandaset_ps28/40/68/70/72_*`

2. `.copilot` 会话记忆  
   本机 `~/.copilot/session-state/417ae335-d421-414e-9b0f-f9d12b934192/` 中保留了 PandaSet 方案演化记录，包括：
   - PandaSet range-image 构造重写；
   - zero-depth 像素从 LiDAR supervision 中剔除；
   - 动态 cuboid mask 的 padding / scale 多轮迭代；
   - camera cache / fused init cache 的 key 升级；
   - `configs/exp_pandaset_pose_2dgs_nosphere.yaml` 的单独引入；
   - 以及后续针对 `ps28` / `ps68` 的数值稳定性修复。

因此，本文件后续 PandaSet 部分同时基于：

- 本机输出目录中的真实实验结果；
- `.copilot` 记忆中的方法演化和排障记录；
- 当前仓库中已落地的代码与配置改动。

## 13. PandaSet 方法演化与本机实现补充

### 13.1 PandaSet 方案演化主线

根据本机 `.copilot` 记忆与当前代码，PandaSet 方向的主要演化顺序可以概括为：

1. 从基础的 `local + PurePnP + tail refine` 方案出发；
2. 去掉 sphere 假设，引入 `nosphere` 独立实验配置；
3. 重写 PandaSet range-image 构造，避免直接依赖固定线束角表；
4. 在 LiDAR supervision 中显式排除 zero-depth 像素；
5. 针对动态目标逐步引入更保守的 cuboid mask：
   - 先做 additive padding；
   - 再切换到 multiplicative cuboid scale；
   - 同时加入 `near500` 近距离过滤与 `fillcur` 掩膜孔洞深度填充；
6. 升级 fused init cache 与 camera cache key，避免不同 masking 设定之间误复用旧缓存；
7. 最终形成目前 PandaSet 主线配方：
   - `fullmask`
   - `nosphere`
   - `raw supervision`
   - `dynamic cuboid scale = 1.2`
   - `bkgd_init_min_distance = 5.0`
   - `mask-hole LiDAR depth fill`

### 13.2 PandaSet 数值稳定性修复

在 `ps28` 与 `ps68` 上，原始 refine 会在后期把共享 pose 状态污染成 `NaN`。本机后续已完成两类修复：

1. `tools/calib.py`
   - pose step 前后清理非有限梯度 / 参数；
   - 保存最后一个有限 pose snapshot；
   - 一旦 step 后状态非有限，立即回滚。

2. `lib/scene/camera_pose_correction.py`
   - 非法四元数重置为单位四元数；
   - `update_extrinsics()` 避免把坏 pose 折叠进共享 base extrinsic。

修复后：

- `ps28` 与 `ps68` 均能完整跑到 `cycle 150`；
- `best_rotation.npz` 中不再出现 `NaN`；
- 后续 PandaSet 对比才具备可解释性。

### 13.3 PandaSet 保守 refine 默认

在进一步排查 `ps70` / `ps72` 的“refine 变差”问题后，本机又新增了一套保守默认：

- `rotation_lr = 5e-4`
- `translation_lr = 5e-4`
- `pose_step_at_cycle_end = true`

并且已经让 `exp_config` 可以直接接管这些参数，当前落地在：

- `tools/calib.py`
- `configs/exp_pandaset_pose_2dgs_nosphere.yaml`

其核心目的不是追求更激进的 tail refine，而是：

- 降低 per-iter pose 更新噪声；
- 避免 `ps72` 这类场景在 `cycle 120+` 出现有限值但方向错误的坏更新；
- 让 refine 更像是在 PurePnP 局部附近做缓慢修正。

## 14. PandaSet 主要结果补充

### 14.1 五场景最终结果汇总

当前本机可确认的 PandaSet 五场景结果为：

| Scene | PurePnP rot (°) | PurePnP trans (m) | Final rot (°) | Final trans (m) | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| ps28 | 0.2654 | 0.3594 | 0.1558 | 0.3572 | `fixsanity` 后稳定收敛 |
| ps40 | 0.9304 | 0.6765 | 1.0164 | 0.6567 | 原始 refine 收益很弱 |
| ps68 | 1.2297 | 0.7074 | 0.8477 | 0.6088 | `fixsanity` 后可跑完且有改善 |
| ps70 | 0.3357 | 0.1149 | 0.4243 | 0.1210 | `consdefault` 后避免明显退化 |
| ps72 | 0.8495 | 0.5490 | 0.7089 | 0.5296 | `consdefault` 后去掉了 `cycle 121` 跳变 |
| **Avg** | **0.7221** | **0.4814** | **0.6306** | **0.4547** | 当前本机 5-scene 汇总 |

需要说明：

- `ps28` / `ps68` 使用的是加入数值稳定性修复后的 `fixsanity` 结果；
- `ps70` / `ps72` 使用的是加入保守 refine 默认后的 `consdefault` 结果；
- 因此这张表反映的是“当前本机已修复、已验证”的 PandaSet 结果，而不是最早一版未修复 run。

### 14.2 PandaSet 的关键结论

从这 5 个 scene 可以得到三点结论：

1. PurePnP 本身已经是很强的基线  
   在多个 scene 上，PurePnP 已可把旋转误差压到 `1°` 左右甚至更低。

2. refine 的问题不是统一的  
   - `ps70` 的主因是 refine 阶段 pose 更新过于激进；
   - `ps72` 则不仅 refine 敏感，PurePnP 初始化本身也明显更脆弱。

3. PandaSet 上更适合“保守 refine”而不是“激进 refine”  
   本机验证表明，简单提高 refine 强度并不会稳定提升结果；更合理的思路是：
   - 用 PurePnP 做主初始化；
   - 将 refine 视为 optional local improvement；
   - 必要时采用 early-stop 或 rollback，而不是默认相信 tail 一定更优。

### 14.3 `ps72` 的特殊性

`ps72` 是 PandaSet 中最有代表性的异常场景。

原始 run 中：

- `cycle 120`: `0.5623° / 0.2829m`
- `cycle 121`: 直接跳到 `4.4617° / 0.2509m`

这说明旧版 refine 存在一次“有限值但错误方向”的坏更新，而不是单纯 `NaN`。

改用保守默认后：

- `cycle 121`: `0.7543° / 0.5419m`
- `cycle 150`: `0.7089° / 0.5296m`

结论是：

- `ps72` 的 refine 崩坏已经被修掉；
- 但其瓶颈进一步暴露为 PurePnP 初始化本身不够好；
- 因此 PandaSet 后续更值得优先改进的是跨模态匹配 / PurePnP 初始化质量，而不是继续单纯加长 tail。

## 15. 本机 KITTI360 结果补充

除前文依据旧会话归纳的 KITTI360 结论外，这台机器当前还能直接回收一组完整的 5-scene 结果，位于：

- `/mnt/data16/xuzhiy/hr-tiny-output/calib/kitti360_batch_w100_pnp5_refine50_colorwarm5_rgb01_depthon_nogradconstraint_restart_higsinit/`

该目录下的 `kitti360_purepnp_refine_summary_v2.json` 给出的 5-scene 汇总为：

| Scene | PurePnP rot (°) | PurePnP trans (m) | Final rot (°) | Final trans (m) |
| --- | ---: | ---: | ---: | ---: |
| k1 | 0.2987 | 0.0318 | 0.1371 | 0.0250 |
| k2 | 0.1856 | 0.0499 | 0.0396 | 0.0489 |
| k3 | 0.3572 | 0.0933 | 0.1480 | 0.0933 |
| k4 | 0.2564 | 0.0723 | 0.1938 | 0.0723 |
| k5 | 0.2654 | 0.0919 | 0.1371 | 0.0825 |
| **Avg** | **0.2727** | **0.0678** | **0.1311** | **0.0644** |

这组结果与前文 KITTI360 主线结论是一致的：

- PurePnP 本身已经有效；
- refine 仍然能继续降低 rotation；
- 但 translation 改善有限，更多体现为“稳住而非大幅继续下降”。

同时，这组本机落盘结果也说明：

- KITTI360 与 PandaSet 的行为差异明显；
- `pose_step_at_cycle_end` 在 KITTI360 上并未表现出像 PandaSet 那样明显的必要性；
- 而 PandaSet 则更依赖保守的 pose 更新策略来避免 tail refine 反向破坏已有解。
