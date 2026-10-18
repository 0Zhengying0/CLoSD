# Flow Matching 300k：ODE 积分器对照

运行目录：`reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01`。只在日志包含完整五项计数、评测完成标志和正确的 NFE/CFG 前向次数时显示成功率。
每项成功率由原始成功/结束回合计数计算，macro 是五项等权平均；延迟为日志中所有 planner 调用的均值与中位数。

```
高阶积分器不保证更好。 在 12 NFE 下 RK4 最好，20、40 NFE 下则是 Heun 最好。
更多积分步不保证更好。 三种方法在 40 NFE 的总体成功率都低于其 12 NFE 结果。
性能差距没有消失。 不论选择哪种积分器，FM 都没有接近 DiP 的 89.4%。

进一步对比fm(12 NFE 下 RK4)和DiP：
1.FM 的主要短板是 REACH 和 GET_UP， SIT显著优于DiP
2.增加积分次数，REACH成功率下降，而GET_UP成功率上升

- 积分准确性： 数值方法是否准确求解模型预测的速度场。
- 动作生成质量： 模型预测的速度场是否能够产生满足任务目标的合理动作。
RK4 可能比 Euler 更准确地积分一个速度场，但这不等于这个速度场本身就是正确的。
目前的 FM 模型是根据训练数据学习速度场的。如果它对某些动作、目标位置或状态条件的预测存在偏差，那么提高积分精度并不会自动消除这些偏差。
```

| 实验 | 状态 | 求解器 | 步数 | NFE/规划 | 实际前向/规划 | 总前向 | Planner 均值/中位数(s) | REACH | KICK | PUNCH | SIT | GET_UP | Macro | Job ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FM-E12 | COMPLETE | euler | 12 | 12 | 24 | 240 | 22.893 / 22.879 | 72.73% | 81.95% | 81.95% | 95.53% | 75.24% | 81.48% | 5185910 |
| FM-H12 | COMPLETE | heun | 6 | 12 | 24 | 240 | 22.780 / 22.769 | 74.60% | 87.43% | 87.43% | 93.35% | 69.93% | 82.55% | 5185911 |
| FM-R12 | COMPLETE | rk4 | 3 | 12 | 24 | 240 | 24.070 / 23.378 | 70.58% | 87.87% | 87.87% | 96.44% | 76.34% | 83.82% | 5185912 |
| FM-E20 | COMPLETE | euler | 20 | 20 | 40 | 400 | 38.053 / 38.030 | 63.17% | 79.28% | 79.28% | 97.43% | 80.72% | 79.97% | 5185913 |
| FM-H20 | COMPLETE | heun | 10 | 20 | 40 | 400 | 37.785 / 37.768 | 66.98% | 86.94% | 86.94% | 97.36% | 79.25% | 83.49% | 5185914 |
| FM-R20 | COMPLETE | rk4 | 5 | 20 | 40 | 400 | 37.989 / 37.960 | 63.02% | 82.99% | 82.99% | 97.50% | 85.15% | 82.33% | 5185915 |
| FM-E40 | COMPLETE | euler | 40 | 40 | 80 | 800 | 76.317 / 76.329 | 59.93% | 76.58% | 76.58% | 97.43% | 83.90% | 78.88% | 5185916 |
| FM-H40 | COMPLETE | heun | 20 | 40 | 80 | 800 | 76.128 / 76.110 | 59.50% | 77.91% | 77.91% | 97.57% | 83.84% | 79.35% | 5185917 |
| FM-R40 | COMPLETE | rk4 | 10 | 40 | 80 | 800 | 75.742 / 75.736 | 55.47% | 76.14% | 76.14% | 96.50% | 85.28% | 77.91% | 5185918 |

## 运行元数据

| 实验 | checkpoint | Git commit | Dirty | 源码 SHA256 | Seed | 配置 | Planner 调用数 |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| FM-E12 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-H12 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-R12 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-E20 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-H20 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-R20 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-E40 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-H40 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |
| FM-R40 | /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt | ac509d6afab7f90cda5df1ead027377367e7959c | 1 | d81aca18c2dc26fd19a7e444920ecd5aaa6248b7d1884219c9c609f11a5a4a97 | 0 | learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False | 10 |

注：MDM 使用最近整数时间索引；RK4 的中间时刻也经相同取整。速度场随时间索引呈阶梯变化，不能保证经典 RK4 的理论四阶收敛率。
