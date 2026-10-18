# VP-CFM 与 OT-CFM：CLoSD 300k 对比

训练：相同 MDM、条件、CFG、目标位置损失权重与 300k steps；OT-CFM 使用精确 minibatch OT 噪声配对和线性路径。
评测：相同 CLoSD 控制器，seed=0，4096 环境，500 帧，CFG=7.5，RK4 三步（12 NFE），每任务至少 1000 个结束回合。

| 模型 | REACH | KICK | PUNCH | SIT | GET_UP | 五任务平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VP-CFM | 70.58% | 87.87% | 87.87% | 96.44% | 76.34% | 83.82% |
| OT-CFM | 86.62% | 71.73% | 71.73% | 95.94% | 83.72% | 81.95% |
| OT-CFM − VP-CFM | +16.04 pp | -16.14 pp | -16.14 pp | -0.50 pp | +7.38 pp | -1.87 pp |

成功率来自完整任务日志中的 success/resets 计数；五任务平均为五个标签等权平均。
**评测器限制：KICK 与 PUNCH 共用同一击打成功/重置计数，不能解释成两项独立成功率。**
合并击打只计一次时，四个独立状态的等权平均：VP-CFM 82.81%，OT-CFM 84.50%。
两次评测来自不同代码修订；已核对固定步 RK4 分支与成功计数公式未变，但代码版本仍是比较限制。

## 生成动作质量

两模型复用同 20 个真实 CLoSD 条件与初始噪声，每任务 4 个案例。以下为平均值；目标误差越低越好。平滑度指标仅作描述。

| 模型 | 终点目标误差(m) | 最近目标距离(m) | 历史衔接 root 跳变(m) | 历史衔接姿态跳变(m) | root 加速度(m/s²) | 关节加速度(m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VP-CFM | 0.8950 | 0.7769 | 0.0189 | 0.0404 | 19.3250 | 33.6853 |
| OT-CFM | 0.8564 | 0.6707 | 0.0147 | 0.0301 | 15.5245 | 28.3086 |

### 分任务的动作质量

每项四个相同条件案例；距离和动作连续性是描述性指标。

| 任务 | 模型 | 终点目标误差(m) | 最近目标距离(m) | root 加速度(m/s²) | 关节加速度(m/s²) |
| --- | --- | ---: | ---: | ---: | ---: |
| REACH | VP-CFM | 1.6022 | 1.6022 | 16.9611 | 28.9002 |
| REACH | OT-CFM | 1.5498 | 1.5498 | 10.7979 | 18.7343 |
| STRIKE_KICK | VP-CFM | 0.7066 | 0.5561 | 19.5903 | 34.6388 |
| STRIKE_KICK | OT-CFM | 0.7222 | 0.3214 | 20.9350 | 39.7395 |
| STRIKE_PUNCH | VP-CFM | 0.6705 | 0.6643 | 20.1095 | 34.5864 |
| STRIKE_PUNCH | OT-CFM | 1.0178 | 0.5954 | 14.3662 | 28.4424 |
| SIT | VP-CFM | 0.9456 | 0.8970 | 17.7323 | 32.6710 |
| SIT | OT-CFM | 0.7032 | 0.7031 | 14.0048 | 25.4683 |
| GET_UP | VP-CFM | 0.5501 | 0.1651 | 22.2316 | 37.6304 |
| GET_UP | OT-CFM | 0.2890 | 0.1840 | 17.5184 | 29.1585 |

动作质量是规划器直接输出的描述性指标，不等于控制器执行成功率。SIT 与击打目标距离也不等于官方碰撞判定。

## 原始记录

- OT-CFM 300k checkpoint：`reproduction/runs/OTCFM_DIMP_FINAL_seed10_300k/model000300000.pt`
- OT-CFM 训练日志：`logs/ot_cfm_300k_5220077.out`
- VP-CFM 评测日志：`reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/logs/FM-R12_5185912.out`
- OT-CFM 评测日志：`reproduction/experiments/ot_cfm/runs/ot_cfm_20261018_01/logs/OT-R12_5223552.out`
- VP-CFM 20 案例：`reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/results/results.csv`
- OT-CFM 20 案例：`reproduction/experiments/ot_cfm/runs/ot_cfm_20261018_01/quality/results.csv`
- OT-CFM 案例轨迹：`reproduction/experiments/ot_cfm/runs/ot_cfm_20261018_01/quality`
