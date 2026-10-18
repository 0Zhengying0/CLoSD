# DiP vs Flow Matching：规划器直接输出的动作质量

状态：**SMOKE COMPLETE**；1 个案例、2 条模型结果。
比较使用同一真实 CLoSD 条件快照及相同初始噪声；DiP 后续扩散随机过程与 FM 不同。未运行控制器。
主目标误差对环境原始目标计算；另在 CSV 中报告对规划器截断目标的误差。REACH 为 pelvis 的水平距离；GET_UP 为 pelvis 的三维距离。SIT 与击打类距离不等于官方任务成功。
所有位置在同一局部 HumanML 坐标系，以米计。历史 20 帧，生成 40 帧，20 fps。无严格匹配 GT，因此不报告 GT 误差。

| 案例 | 任务 | 难度 | 模型 | 终点误差(m) | 最小距离(m) | 衔接 root 跳变(m) | 衔接关节姿态跳变(m) | 衔接朝向跳变(°) | root 加速度均值(m/s²) | 关节加速度均值(m/s²) | 距离曲线 | MP4 | 并排 MP4 | GIF |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| REACH_00_normal | REACH | normal | DIP | 0.6584 | 0.6584 | 0.0133 | 0.0208 | 0.1837 | 0.8308 | 1.6317 | [PNG](REACH_00_normal/target_distance.png) | [video](REACH_00_normal/dip.mp4) | [video](REACH_00_normal/side_by_side.mp4) | [GIF](REACH_00_normal/side_by_side.gif) |
| REACH_00_normal | REACH | normal | FM | 1.1338 | 1.1338 | 0.0399 | 0.0659 | 3.8229 | 16.8752 | 29.0594 | [PNG](REACH_00_normal/target_distance.png) | [video](REACH_00_normal/fm.mp4) | [video](REACH_00_normal/side_by_side.mp4) | [GIF](REACH_00_normal/side_by_side.gif) |

## 按任务分组的均值与中位数

| 任务 | 模型 | N | 终点误差 mean/median (m) | 最小距离 mean/median (m) | root 加速度 mean/median (m/s²) | 关节加速度 mean/median (m/s²) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| REACH | DIP | 1 | 0.6584 / 0.6584 | 0.6584 / 0.6584 | 0.8308 / 0.8308 | 1.6317 / 1.6317 |
| REACH | FM | 1 | 1.1338 / 1.1338 | 1.1338 / 1.1338 | 16.8752 / 16.8752 | 29.0594 / 29.0594 |

模型、checkpoint、`args.json` SHA256、Git commit 和 Slurm Job ID 见 `provenance.json`；逐帧距离见各案例的 `*_target_distance.csv`。
