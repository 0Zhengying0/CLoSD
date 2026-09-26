# 单次评测结果

| 模型 | REACH | KICK | PUNCH | SIT | GET_UP | 作业 |
|---|---|---|---|---|---|---|
| official_300k | 1.00 | 0.93 | 0.93 | 0.88 | 0.97 | [5100756](../logs/dip_eval_official_5100756.out) |
| reproduction_300k | 1.00 | 0.88 | 0.88 | 0.47 | 0.91 | [5114966](../logs/dip_eval_repro300_5114966.out) |
| reproduction_600k | 0.99 | 0.89 | 0.89 | 0.76 | 0.93 | [5114967](../logs/dip_eval_repro600_5114967.out) |

使用官方 CLoSD_multitask_finetune 控制器；每类任务 1000 个样本，4096 个并行环境，episode length 500，CFG 7.5。数值直接取日志的两位小数；不是多随机种子的均值或论文结果声明。完整 CSV 见 [results.csv](results.csv)。
