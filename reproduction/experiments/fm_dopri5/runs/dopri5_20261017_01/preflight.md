# FM 300k：DOPRI5 与 RK4 闭环评测

五项任务采用官方成功率口径，每项至少 1000 个结束回合；平均值为五项等权平均。
DOPRI5 NFE 是每次规划真实速度场调用数，CFG=7.5 时每 NFE 对应两次 MDM 前向。

| 求解器 | 状态 | REACH | KICK | PUNCH | SIT | GET_UP | 等权平均 | NFE 均值 [最小,最大] | MDM 前向均值 | Planner 延迟均值/中位数 (s) | Job ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RK4 3 steps | COMPLETE | 70.58% (981/1390) | 87.87% (1152/1311) | 87.87% (1152/1311) | 96.44% (1354/1404) | 76.34% (1026/1344) | 83.82% | 12.0 [12,12] | 24.0 | 24.070/23.378 | 5185912 |
| DOPRI5 | MISSING_LOG | — | — | — | — | — | — | — | — | — | — |

## 分析

- 正式评测尚未完整结束；不填补缺失的成功率或耗时。DOPRI5 状态：MISSING_LOG。
- GPU 冒烟采样：REACH_00_normal 80 NFE、160 MDM 前向、1.277s；GET_UP_00_normal 68 NFE、136 MDM 前向、0.849s；只证明规划器离线输出有效，不是闭环成功率。
- MDM 仍将 t∈[0,1] 四舍五入到 0–999；速度场随时间呈阶梯变化，DOPRI5 的误差估计与高阶收敛保证受此限制。

## 复现信息

- 配置：`{"checkpoint": "/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt", "case_manifest": "/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/conditions/manifest.json", "rtol": 0.001, "atol": 0.0001, "max_num_steps": 128, "max_nfe": 256, "cfg": 7.5, "seed": 0, "num_envs": 4096, "episode_length": 500, "samples_per_task": 1000, "solver": "dopri5", "time_encoding": "round(t * 999)"}`
- DOPRI5 运行元数据：`{}`
- RK4 历史日志：`/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/logs/FM-R12_5185912.out`
- DOPRI5 日志：`尚未提交`
