# DiP vs Flow Matching：规划器直接输出的动作质量

状态：**FORMAL COMPLETE**；20 个案例、40 条模型结果。
比较使用同一真实 CLoSD 条件快照及相同初始噪声；DiP 后续扩散随机过程与 FM 不同。未运行控制器。
主目标误差对环境原始目标计算；另在 CSV 中报告对规划器截断目标的误差。REACH 为 pelvis 的水平距离；GET_UP 为 pelvis 的三维距离。SIT 与击打类距离不等于官方任务成功。
所有位置在同一局部 HumanML 坐标系，以米计。历史 20 帧，生成 40 帧，20 fps。无严格匹配 GT，因此不报告 GT 误差。

| 案例 | 任务 | 难度 | 模型 | 终点误差(m) | 最小距离(m) | 衔接 root 跳变(m) | 衔接关节姿态跳变(m) | 衔接朝向跳变(°) | root 加速度均值(m/s²) | 关节加速度均值(m/s²) | 距离曲线 | MP4 | 并排 MP4 | GIF |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| GET_UP_00_normal | GET_UP | normal | DIP | 0.0413 | 0.0413 | 0.0213 | 0.0340 | 0.0612 | 1.5396 | 2.0229 | [PNG](GET_UP_00_normal/target_distance.png) | [video](GET_UP_00_normal/dip.mp4) | [video](GET_UP_00_normal/side_by_side.mp4) | — |
| GET_UP_00_normal | GET_UP | normal | FM | 0.4922 | 0.1869 | 0.0263 | 0.0483 | 5.5920 | 23.2443 | 38.5774 | [PNG](GET_UP_00_normal/target_distance.png) | [video](GET_UP_00_normal/fm.mp4) | [video](GET_UP_00_normal/side_by_side.mp4) | — |
| GET_UP_01_moderate | GET_UP | moderate | DIP | 0.0645 | 0.0645 | 0.0146 | 0.0341 | 1.5083 | 1.2276 | 1.8305 | [PNG](GET_UP_01_moderate/target_distance.png) | [video](GET_UP_01_moderate/dip.mp4) | [video](GET_UP_01_moderate/side_by_side.mp4) | — |
| GET_UP_01_moderate | GET_UP | moderate | FM | 0.5679 | 0.0637 | 0.0246 | 0.0492 | 8.1174 | 19.7965 | 33.4270 | [PNG](GET_UP_01_moderate/target_distance.png) | [video](GET_UP_01_moderate/fm.mp4) | [video](GET_UP_01_moderate/side_by_side.mp4) | — |
| GET_UP_02_challenging | GET_UP | challenging | DIP | 0.1133 | 0.1133 | 0.0530 | 0.0335 | 1.9157 | 1.4178 | 1.8778 | [PNG](GET_UP_02_challenging/target_distance.png) | [video](GET_UP_02_challenging/dip.mp4) | [video](GET_UP_02_challenging/side_by_side.mp4) | — |
| GET_UP_02_challenging | GET_UP | challenging | FM | 0.5289 | 0.1981 | 0.0049 | 0.0402 | 1.5617 | 22.8175 | 40.0197 | [PNG](GET_UP_02_challenging/target_distance.png) | [video](GET_UP_02_challenging/fm.mp4) | [video](GET_UP_02_challenging/side_by_side.mp4) | — |
| GET_UP_03_hard | GET_UP | hard | DIP | 0.1375 | 0.1375 | 0.0073 | 0.0333 | 0.0322 | 1.1582 | 1.6028 | [PNG](GET_UP_03_hard/target_distance.png) | [video](GET_UP_03_hard/dip.mp4) | [video](GET_UP_03_hard/side_by_side.mp4) | [GIF](GET_UP_03_hard/side_by_side.gif) |
| GET_UP_03_hard | GET_UP | hard | FM | 0.6113 | 0.2115 | 0.0620 | 0.0697 | 2.6856 | 23.0680 | 38.4973 | [PNG](GET_UP_03_hard/target_distance.png) | [video](GET_UP_03_hard/fm.mp4) | [video](GET_UP_03_hard/side_by_side.mp4) | [GIF](GET_UP_03_hard/side_by_side.gif) |
| REACH_00_normal | REACH | normal | DIP | 0.6584 | 0.6584 | 0.0133 | 0.0208 | 0.1837 | 0.8308 | 1.6317 | [PNG](REACH_00_normal/target_distance.png) | [video](REACH_00_normal/dip.mp4) | [video](REACH_00_normal/side_by_side.mp4) | — |
| REACH_00_normal | REACH | normal | FM | 1.1338 | 1.1338 | 0.0399 | 0.0659 | 3.8229 | 16.8752 | 29.0594 | [PNG](REACH_00_normal/target_distance.png) | [video](REACH_00_normal/fm.mp4) | [video](REACH_00_normal/side_by_side.mp4) | — |
| REACH_01_moderate | REACH | moderate | DIP | 0.9450 | 0.9450 | 0.0104 | 0.0218 | 4.2660 | 1.4502 | 2.2811 | [PNG](REACH_01_moderate/target_distance.png) | [video](REACH_01_moderate/dip.mp4) | [video](REACH_01_moderate/side_by_side.mp4) | — |
| REACH_01_moderate | REACH | moderate | FM | 1.4682 | 1.4682 | 0.0175 | 0.0358 | 4.7935 | 18.1526 | 30.2925 | [PNG](REACH_01_moderate/target_distance.png) | [video](REACH_01_moderate/fm.mp4) | [video](REACH_01_moderate/side_by_side.mp4) | — |
| REACH_02_challenging | REACH | challenging | DIP | 1.1646 | 1.1646 | 0.0092 | 0.0207 | 1.1400 | 1.4790 | 2.3768 | [PNG](REACH_02_challenging/target_distance.png) | [video](REACH_02_challenging/dip.mp4) | [video](REACH_02_challenging/side_by_side.mp4) | — |
| REACH_02_challenging | REACH | challenging | FM | 1.5031 | 1.5031 | 0.0190 | 0.0350 | 1.8717 | 16.8824 | 28.1056 | [PNG](REACH_02_challenging/target_distance.png) | [video](REACH_02_challenging/fm.mp4) | [video](REACH_02_challenging/side_by_side.mp4) | — |
| REACH_03_hard | REACH | hard | DIP | 1.8693 | 1.8693 | 0.0112 | 0.0210 | 0.1272 | 1.4996 | 2.3138 | [PNG](REACH_03_hard/target_distance.png) | [video](REACH_03_hard/dip.mp4) | [video](REACH_03_hard/side_by_side.mp4) | [GIF](REACH_03_hard/side_by_side.gif) |
| REACH_03_hard | REACH | hard | FM | 2.3037 | 2.3037 | 0.0101 | 0.0343 | 0.1946 | 15.9344 | 28.1432 | [PNG](REACH_03_hard/target_distance.png) | [video](REACH_03_hard/fm.mp4) | [video](REACH_03_hard/side_by_side.mp4) | [GIF](REACH_03_hard/side_by_side.gif) |
| SIT_00_normal | SIT | normal | DIP | 0.1750 | 0.1750 | 0.0013 | 0.0219 | 0.5212 | 1.2895 | 1.6379 | [PNG](SIT_00_normal/target_distance.png) | [video](SIT_00_normal/dip.mp4) | [video](SIT_00_normal/side_by_side.mp4) | — |
| SIT_00_normal | SIT | normal | FM | 0.7494 | 0.7291 | 0.0073 | 0.0320 | 7.7239 | 13.4433 | 32.8393 | [PNG](SIT_00_normal/target_distance.png) | [video](SIT_00_normal/fm.mp4) | [video](SIT_00_normal/side_by_side.mp4) | — |
| SIT_01_moderate | SIT | moderate | DIP | 0.3687 | 0.3687 | 0.0030 | 0.0255 | 3.6294 | 1.6508 | 2.2634 | [PNG](SIT_01_moderate/target_distance.png) | [video](SIT_01_moderate/dip.mp4) | [video](SIT_01_moderate/side_by_side.mp4) | — |
| SIT_01_moderate | SIT | moderate | FM | 0.8374 | 0.8374 | 0.0024 | 0.0309 | 1.1683 | 21.6683 | 31.6741 | [PNG](SIT_01_moderate/target_distance.png) | [video](SIT_01_moderate/fm.mp4) | [video](SIT_01_moderate/side_by_side.mp4) | — |
| SIT_02_challenging | SIT | challenging | DIP | 0.2644 | 0.2644 | 0.0019 | 0.0256 | 2.7816 | 1.8618 | 2.8262 | [PNG](SIT_02_challenging/target_distance.png) | [video](SIT_02_challenging/dip.mp4) | [video](SIT_02_challenging/side_by_side.mp4) | — |
| SIT_02_challenging | SIT | challenging | FM | 0.9439 | 0.9439 | 0.0144 | 0.0291 | 0.1409 | 17.6826 | 30.4660 | [PNG](SIT_02_challenging/target_distance.png) | [video](SIT_02_challenging/fm.mp4) | [video](SIT_02_challenging/side_by_side.mp4) | — |
| SIT_03_hard | SIT | hard | DIP | 0.3058 | 0.3058 | 0.0014 | 0.0234 | 0.2751 | 1.7199 | 1.9300 | [PNG](SIT_03_hard/target_distance.png) | [video](SIT_03_hard/dip.mp4) | [video](SIT_03_hard/side_by_side.mp4) | [GIF](SIT_03_hard/side_by_side.gif) |
| SIT_03_hard | SIT | hard | FM | 1.2516 | 1.0778 | 0.0096 | 0.0333 | 4.9041 | 18.1350 | 35.7045 | [PNG](SIT_03_hard/target_distance.png) | [video](SIT_03_hard/fm.mp4) | [video](SIT_03_hard/side_by_side.mp4) | [GIF](SIT_03_hard/side_by_side.gif) |
| STRIKE_KICK_00_normal | STRIKE_KICK | normal | DIP | 0.1547 | 0.1547 | 0.0118 | 0.0206 | 0.0836 | 3.9416 | 6.2302 | [PNG](STRIKE_KICK_00_normal/target_distance.png) | [video](STRIKE_KICK_00_normal/dip.mp4) | [video](STRIKE_KICK_00_normal/side_by_side.mp4) | — |
| STRIKE_KICK_00_normal | STRIKE_KICK | normal | FM | 0.5509 | 0.0665 | 0.0089 | 0.0330 | 5.5280 | 14.1439 | 31.2748 | [PNG](STRIKE_KICK_00_normal/target_distance.png) | [video](STRIKE_KICK_00_normal/fm.mp4) | [video](STRIKE_KICK_00_normal/side_by_side.mp4) | — |
| STRIKE_KICK_01_moderate | STRIKE_KICK | moderate | DIP | 0.2926 | 0.2926 | 0.0113 | 0.0218 | 2.4970 | 3.3639 | 6.8879 | [PNG](STRIKE_KICK_01_moderate/target_distance.png) | [video](STRIKE_KICK_01_moderate/dip.mp4) | [video](STRIKE_KICK_01_moderate/side_by_side.mp4) | — |
| STRIKE_KICK_01_moderate | STRIKE_KICK | moderate | FM | 0.4199 | 0.3022 | 0.0205 | 0.0340 | 4.6389 | 26.3372 | 39.9923 | [PNG](STRIKE_KICK_01_moderate/target_distance.png) | [video](STRIKE_KICK_01_moderate/fm.mp4) | [video](STRIKE_KICK_01_moderate/side_by_side.mp4) | — |
| STRIKE_KICK_02_challenging | STRIKE_KICK | challenging | DIP | 0.8228 | 0.6693 | 0.0139 | 0.0206 | 0.6888 | 6.3172 | 9.0789 | [PNG](STRIKE_KICK_02_challenging/target_distance.png) | [video](STRIKE_KICK_02_challenging/dip.mp4) | [video](STRIKE_KICK_02_challenging/side_by_side.mp4) | — |
| STRIKE_KICK_02_challenging | STRIKE_KICK | challenging | FM | 0.9108 | 0.9108 | 0.0027 | 0.0416 | 5.1901 | 19.4335 | 31.0322 | [PNG](STRIKE_KICK_02_challenging/target_distance.png) | [video](STRIKE_KICK_02_challenging/fm.mp4) | [video](STRIKE_KICK_02_challenging/side_by_side.mp4) | — |
| STRIKE_KICK_03_hard | STRIKE_KICK | hard | DIP | 0.9151 | 0.6188 | 0.0138 | 0.0226 | 4.6957 | 2.2814 | 4.0054 | [PNG](STRIKE_KICK_03_hard/target_distance.png) | [video](STRIKE_KICK_03_hard/dip.mp4) | [video](STRIKE_KICK_03_hard/side_by_side.mp4) | [GIF](STRIKE_KICK_03_hard/side_by_side.gif) |
| STRIKE_KICK_03_hard | STRIKE_KICK | hard | FM | 0.9447 | 0.9447 | 0.0028 | 0.0306 | 1.4180 | 18.4468 | 36.2559 | [PNG](STRIKE_KICK_03_hard/target_distance.png) | [video](STRIKE_KICK_03_hard/fm.mp4) | [video](STRIKE_KICK_03_hard/side_by_side.mp4) | [GIF](STRIKE_KICK_03_hard/side_by_side.gif) |
| STRIKE_PUNCH_00_normal | STRIKE_PUNCH | normal | DIP | 0.2900 | 0.2900 | 0.0043 | 0.0228 | 3.5837 | 4.4633 | 8.3699 | [PNG](STRIKE_PUNCH_00_normal/target_distance.png) | [video](STRIKE_PUNCH_00_normal/dip.mp4) | [video](STRIKE_PUNCH_00_normal/side_by_side.mp4) | — |
| STRIKE_PUNCH_00_normal | STRIKE_PUNCH | normal | FM | 0.3935 | 0.3935 | 0.0006 | 0.0269 | 1.6128 | 19.6500 | 37.0752 | [PNG](STRIKE_PUNCH_00_normal/target_distance.png) | [video](STRIKE_PUNCH_00_normal/fm.mp4) | [video](STRIKE_PUNCH_00_normal/side_by_side.mp4) | — |
| STRIKE_PUNCH_01_moderate | STRIKE_PUNCH | moderate | DIP | 0.1581 | 0.1344 | 0.0062 | 0.0233 | 4.4216 | 1.5569 | 4.0387 | [PNG](STRIKE_PUNCH_01_moderate/target_distance.png) | [video](STRIKE_PUNCH_01_moderate/dip.mp4) | [video](STRIKE_PUNCH_01_moderate/side_by_side.mp4) | — |
| STRIKE_PUNCH_01_moderate | STRIKE_PUNCH | moderate | FM | 0.5132 | 0.5132 | 0.0163 | 0.0362 | 0.0793 | 18.3978 | 32.6907 | [PNG](STRIKE_PUNCH_01_moderate/target_distance.png) | [video](STRIKE_PUNCH_01_moderate/fm.mp4) | [video](STRIKE_PUNCH_01_moderate/side_by_side.mp4) | — |
| STRIKE_PUNCH_02_challenging | STRIKE_PUNCH | challenging | DIP | 0.3877 | 0.3877 | 0.0073 | 0.0216 | 0.7556 | 3.6059 | 7.6384 | [PNG](STRIKE_PUNCH_02_challenging/target_distance.png) | [video](STRIKE_PUNCH_02_challenging/dip.mp4) | [video](STRIKE_PUNCH_02_challenging/side_by_side.mp4) | — |
| STRIKE_PUNCH_02_challenging | STRIKE_PUNCH | challenging | FM | 0.8742 | 0.8494 | 0.0418 | 0.0521 | 1.0370 | 16.9388 | 29.4970 | [PNG](STRIKE_PUNCH_02_challenging/target_distance.png) | [video](STRIKE_PUNCH_02_challenging/fm.mp4) | [video](STRIKE_PUNCH_02_challenging/side_by_side.mp4) | — |
| STRIKE_PUNCH_03_hard | STRIKE_PUNCH | hard | DIP | 0.7173 | 0.7019 | 0.0049 | 0.0220 | 5.8815 | 1.1409 | 3.6201 | [PNG](STRIKE_PUNCH_03_hard/target_distance.png) | [video](STRIKE_PUNCH_03_hard/dip.mp4) | [video](STRIKE_PUNCH_03_hard/side_by_side.mp4) | [GIF](STRIKE_PUNCH_03_hard/side_by_side.gif) |
| STRIKE_PUNCH_03_hard | STRIKE_PUNCH | hard | FM | 0.9012 | 0.9012 | 0.0459 | 0.0506 | 0.8346 | 25.4513 | 39.0829 | [PNG](STRIKE_PUNCH_03_hard/target_distance.png) | [video](STRIKE_PUNCH_03_hard/fm.mp4) | [video](STRIKE_PUNCH_03_hard/side_by_side.mp4) | [GIF](STRIKE_PUNCH_03_hard/side_by_side.gif) |

## 按任务分组的均值与中位数

| 任务 | 模型 | N | 终点误差 mean/median (m) | 最小距离 mean/median (m) | root 加速度 mean/median (m/s²) | 关节加速度 mean/median (m/s²) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| REACH | DIP | 4 | 1.1593 / 1.0548 | 1.1593 / 1.0548 | 1.3149 / 1.4646 | 2.1508 / 2.2974 |
| REACH | FM | 4 | 1.6022 / 1.4856 | 1.6022 / 1.4856 | 16.9611 / 16.8788 | 28.9002 / 28.6013 |
| STRIKE_KICK | DIP | 4 | 0.5463 / 0.5577 | 0.4339 / 0.4557 | 3.9760 / 3.6528 | 6.5506 / 6.5590 |
| STRIKE_KICK | FM | 4 | 0.7066 / 0.7309 | 0.5561 / 0.6065 | 19.5903 / 18.9401 | 34.6388 / 33.7653 |
| STRIKE_PUNCH | DIP | 4 | 0.3883 / 0.3389 | 0.3785 / 0.3389 | 2.6918 / 2.5814 | 5.9168 / 5.8386 |
| STRIKE_PUNCH | FM | 4 | 0.6705 / 0.6937 | 0.6643 / 0.6813 | 20.1095 / 19.0239 | 34.5864 / 34.8829 |
| SIT | DIP | 4 | 0.2785 / 0.2851 | 0.2785 / 0.2851 | 1.6305 / 1.6854 | 2.1644 / 2.0967 |
| SIT | FM | 4 | 0.9456 / 0.8906 | 0.8970 / 0.8906 | 17.7323 / 17.9088 | 32.6710 / 32.2567 |
| GET_UP | DIP | 4 | 0.0892 / 0.0889 | 0.0892 / 0.0889 | 1.3358 / 1.3227 | 1.8335 / 1.8542 |
| GET_UP | FM | 4 | 0.5501 / 0.5484 | 0.1651 / 0.1925 | 22.2316 / 22.9427 | 37.6304 / 38.5374 |

模型、checkpoint、`args.json` SHA256、Git commit 和 Slurm Job ID 见 `provenance.json`；逐帧距离见各案例的 `*_target_distance.csv`。
