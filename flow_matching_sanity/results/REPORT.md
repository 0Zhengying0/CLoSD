# VP-CFM Two Moons Sanity Check

**结论：通过。**

## 设置

- 独立实验脚本：`run.py`；CPU，随机种子 20261017，Two Moons 训练集 4096 点、验证集 1024 点，噪声 0.08，并按训练集每个坐标维度标准化。
- 两个模型均为 3 层宽度 64 的 SiLU MLP；使用相同初始权重、每步同一批样本/高斯源/离散时间；Adam 学习率 1e-3，batch 256，训练 6000 步。
- CLoSD 分支直接执行原文件中的 `FlowMatching` 类，仅将动作数据专用的 `masked_l2` 换成二维均值 MSE，并设 `lambda_target_loc=0`。TorchCFM 分支使用官方 `VariancePreservingConditionalFlowMatcher(sigma=0)`，版本 1.0.7。时间在两边均为 `t/1000`。
- 从同一 1024 个高斯点出发，两个模型都用 100 步固定步长 midpoint ODE 采样。

## 结果

| 指标 | CLoSD | TorchCFM |
|---|---:|---:|
| 路径 x_t 最大绝对差 | 0 | 基准 |
| 目标速度最大绝对差 | 0 | 基准 |
| 最后 100 步平均 MSE | 2.24347 | 2.24347 |
| 生成 vs 验证集 MMD² | 0.00298 | 0.00298 |
| 轨迹相邻步反向比例 | 0.00020 | 0.00020 |
| 相邻步相对变化量 | 0.03163 | 0.03163 |

两条训练 Loss 的最大绝对差为 0；完整采样轨迹的最大绝对差为 0。高斯源与验证集的 MMD² 为 0.02663。相同初值的轨迹只有极少量反向步（约 0.020%），图中未见密集折返或异常抖动。生成点已呈双月形，但比真实数据更分散，部分点落在两条月牙之间；本结论是算法一致性与基本生成行为通过，不代表高保真生成质量。

判定条件：路径和速度差均小于 1e-5；配对训练 Loss 差小于 1e-5；两种生成分布的 MMD² 均低于高斯源的 50%；轨迹反向步比例小于 1%，且所有状态为有限值。各项结果：{"path_and_velocity_match": true, "paired_training_matches": true, "generated_distribution_improves": true, "rare_trajectory_reversals": true, "finite_trajectories": true}。

## 图与复现

- [Loss 曲线](loss.png) · [二维点分布](distributions.png) · [采样轨迹](trajectories.png)
- [完整指标](metrics.json) · [逐步 Loss](loss.csv)
- 运行：`envs/closd/bin/python flow_matching_sanity/run.py`（从工作区根目录）。

TorchCFM 原版源码固定于提交 `7c653857de4c25b979a740ef8b6a5d5ddb09b6c1`，位于 `vendor/torchcfm/`，保留 MIT 许可证。官方模块导入时会引用未在本机安装的 POT；实验只使用 VP-CFM，不调用 OT 类，因此在脚本中为 OT 导入提供了显式占位类，官方 VP-CFM 源码未改动。此实验验证二维路径与小型 MLP 的行为，不代替完整动作模型评测。
