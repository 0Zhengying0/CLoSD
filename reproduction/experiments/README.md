# 实验记录

## 主实验

- 完整训练：作业 **5099223**，seed 10，目录 `DiP_multi-target_repro_seed10`；日志记录完成 600000 步。
- 正式评测：官方 300k 作业 **5100756**、复现 300k 作业 **5114966**、复现 600k 作业 **5114967**。结果见 [results.md](results.md)。
- 主训练提交为 `4af01718653dbabf1de1dd9a2600aa54c91acc38`；训练日志记录当时工作区无修改。
- 实际配置为 batch size 64、10 diffusion steps、20 context frames、40 predicted frames、EMA、BERT text encoder、target conditioning。训练期间 `eval_during_training=false`、`gen_during_training=false`；正式 CLoSD 评测在训练后单独进行。

## 独立短时长续训实验

目录 `DiP_multi-target_repro_seed10_short`：6 小时作业 **5100766** 到 120k；2 小时作业 **5105097** 从 120k 续至 160k，**5106709** 从 160k 续至 200k，**5108318** 从 200k 续至 240k。它没有用于本次发布的 300k/600k 权重。

训练 smoke 使用 `DiP_multi-target_smoke`，是独立验证任务。上述三种实验分别保存 `args.original.json`，其中旧绝对路径保持原样。

## 环境测试与历史尝试

以下状态根据 stdout 中明确的结束标记整理，不替代 Slurm accounting。stderr 可能存在非致命告警或 NFS 清理错误；完整文件均保留。环境规模测试用于运行可行性检查，不计入正式结果。

| 日志 | 分类 | stdout 完成标记 |
|---|---|---|
| [closd_real_smoke_5099144.out](../logs/closd_real_smoke_5099144.out) | smoke / 环境验证 | === REAL CLoSD SMOKE PASSED === |
| [closd_smoke_5099017.out](../logs/closd_smoke_5099017.out) | smoke / 环境验证 | 未发现明确完成标记；查看原始日志 |
| [closd_smoke_5099078.out](../logs/closd_smoke_5099078.out) | smoke / 环境验证 | 未发现明确完成标记；查看原始日志 |
| [closd_smoke_5099128.out](../logs/closd_smoke_5099128.out) | smoke / 环境验证 | === SMOKE TEST PASSED === |
| [dip_envtest_5100447.out](../logs/dip_envtest_5100447.out) | 环境规模测试 | Process exit code: 1；=== EVAL ENV TEST FAILED === |
| [dip_envtest_5100449.out](../logs/dip_envtest_5100449.out) | 环境规模测试 | Process exit code: 1；=== EVAL ENV TEST FAILED === |
| [dip_envtest_5100460.out](../logs/dip_envtest_5100460.out) | 环境规模测试 | Process exit code: 0；=== EVAL ENV TEST PASSED === |
| [dip_envtest_5100524.out](../logs/dip_envtest_5100524.out) | 环境规模测试 | Process exit code: 0；=== EVAL ENV TEST PASSED === |
| [dip_envtest_5100535.out](../logs/dip_envtest_5100535.out) | 环境规模测试 | Process exit code: 0；=== EVAL ENV TEST PASSED === |
| [dip_envtest_5100559.out](../logs/dip_envtest_5100559.out) | 环境规模测试 | Process exit code: 0；=== EVAL ENV TEST PASSED === |
| [dip_eval_official_5100596.out](../logs/dip_eval_official_5100596.out) | 正式评测的历史尝试 | 未发现明确完成标记；查看原始日志 |
| [dip_mt_2h_5105097.out](../logs/dip_mt_2h_5105097.out) | 独立短时长训练 | 未发现明确完成标记；查看原始日志 |
| [dip_mt_2h_5106709.out](../logs/dip_mt_2h_5106709.out) | 独立短时长训练 | 未发现明确完成标记；查看原始日志 |
| [dip_mt_2h_5108318.out](../logs/dip_mt_2h_5108318.out) | 独立短时长训练 | 未发现明确完成标记；查看原始日志 |
| [dip_mt_6h_5100766.out](../logs/dip_mt_6h_5100766.out) | 独立短时长训练 | 未发现明确完成标记；查看原始日志 |
| [dip_train_smoke_5100032.out](../logs/dip_train_smoke_5100032.out) | smoke / 环境验证 | === TRAINING SMOKE PASSED === |

已确认的早期失败包括缺少 `libgthread-2.0.so.0`、Python `learning` 导入路径，以及 `libpython3.8.so.1.0` 动态链接路径；后续日志与当前脚本保留了修正后的路径设置。没有为没有完成标记的尝试补造成功结论。

## 本地迁移

迁移清单记录 399 个条目；大文件采用同文件系统重命名并核对 inode/大小，关键权重及小文件另外记录 SHA-256。原始脚本先逐字节归档再移除外部重复项。官方资源软链接改为指向仓库内专属缓存。没有需要保留的旧外部路径兼容链接。

W&B 完整二进制运行保留在 `runs/wandb`，文本日志和配置另复制到 `reproduction/logs/wandb`。可确认的训练 logger 文件从共享 tmp 移至 `reproduction/logs/training_logger`；未能确定归属的通用临时目录仍留原地。
