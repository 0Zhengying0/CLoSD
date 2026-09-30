# DiP multi-target 从零训练及其在 CLoSD 中的评测

本目录记录 Zhengying 在 Karolina Slurm 集群上的复现实验。训练的是 DiP multi-target；评测的运动控制器来自官方 `CLoSD_multitask_finetune` 权重。没有重新从零训练整套 CLoSD 控制器。

## 材料入口

- [实验结果和原始日志来源](experiments/results.md)：官方 300k 基线、复现 300k 和复现 600k。
- [实验时间线与失败尝试](experiments/README.md)：完整训练和 `_short` 续训分别记录。
- [环境与官方资源恢复](environment/README.md)：实际依赖清单、容器身份、固定资源版本。
- [完整日志](logs/)：Slurm stdout/stderr、GPU CSV、训练 logger 与 W&B 文本记录。
- [运行脚本](scripts/hpc/)和[工作笔记](../note.txt)：脚本使用当前目录；笔记保留你的最新内容，其中历史命令可能仍含旧路径。
- [迁移清单](experiments/migration-manifest.json)：记录首次整理时的迁移快照。随后已将笔记移至根目录，并删除有对应运行入口的历史脚本副本；清单中的历史路径不代表当前布局。

训练源码提交为 `4af01718653dbabf1de1dd9a2600aa54c91acc38`；本次整理后的提交只调整复现入口与材料组织，不改模型或训练算法。历史 `args.original.json` 保留原始路径。当前运行配置直接写在 `reproduction/scripts/hpc/*.slurm` 内。旧脚本从提交 `bfe9777` 的原始归档恢复，仅适配目录路径；旧训练 smoke 使用作业号后缀输出目录，保留已有实验。

## 使用入口

所有 Slurm 脚本内部写全环境、容器及执行命令。从 **Zhengying** 目录直接提交：

```bash
# 官方权重与两份历史复现权重的任务成功率评测。
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_official.slurm
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_repro_300k.slurm
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_repro_600k.slurm

# 本次新训练的 DIMP_FINAL 300k 权重。
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_recipe_300k.slurm

# 环境测试。
sbatch --export=ALL,NUM_ENVS=128 CLoSD/reproduction/scripts/hpc/dip_eval_envtest.slurm
sbatch CLoSD/reproduction/scripts/hpc/closd_smoke.slurm
sbatch CLoSD/reproduction/scripts/hpc/closd_real_smoke.slurm
```

旧训练脚本 `dip_train.slurm` 为完整训练，`dip_train_{2h,4h,6h,12h}.slurm` 使用 `_short` 目录，`dip_train_smoke.slurm` 为历史训练流程 smoke。它们保留原始 DIMP_FULL 参数及自动查找最新 checkpoint 的原项目逻辑；旧训练没有开启训练中评测。要新建实验，修改文件内外两处 `SAVE_DIR` 为同一个新目录；恢复历史训练须有相邻 `args.json` 和对应优化器状态。下载的 Release 只有权重与参数，不能当作完整续训备份。

旧 Slurm 文件不支持 `--check`、`--resume`、`--save-dir` 或 `--model` 参数，应直接阅读、修改脚本。静态语法检查可使用 `bash -n <文件>`，它不执行作业。新 `dip_eval_recipe_300k.slurm` 自带 `--check` 和 `--model PATH`，检查模式只输出资源路径和评测命令，不启动容器。

## 路径与资源配置

旧完整脚本保留显式 `ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying`，不依赖提交目录或 Slurm spool 路径。仓库在 `$ROOT/CLoSD`，产物在 `CLoSD/reproduction/runs`，日志在 `CLoSD/reproduction/logs`，专属缓存在 `CLoSD/.local/cache`；Conda、Isaac Gym、容器与通用缓存留在共享根目录。

旧脚本的路径与参数均可直接编辑；Slurm 资源可通过 `sbatch --time=... --partition=... <文件>` 覆盖。新评测入口另支持 `CLOSD_ROOT`、`CLOSD_SHARED_ROOT`、`CLOSD_ENV`、`CLOSD_ISAACGYM`、`CLOSD_CONTAINER`、`CLOSD_CACHE`、`CLOSD_LOGS` 路径覆盖。

`run_job.py`、`jobs.json`、`submit.sh` 等先前整理工具暂留；当前 Slurm 不调用它们，本文以直接 sbatch 完整文件为提交方式。它们的旧注册配置不能代替当前 Slurm 内容。

## 权重与本地存储

[Release repro-dip-seed10-v1](https://github.com/0Zhengying0/CLoSD/releases/tag/repro-dip-seed10-v1) 发布原始 `model000300000.pt`、`model000600000.pt`、相邻 `args.json` 和 SHA-256 清单。下载器依据仓库内的校验值验证，不覆盖内容不同的已有文件。

```bash
python3 reproduction/scripts/hpc/download_release.py
python3 reproduction/scripts/hpc/download_release.py --check
```

默认恢复到 `reproduction/runs/DiP_multi-target_repro_seed10/`。其它 checkpoint、优化器和 W&B 二进制保留在本地 `reproduction/runs`；专属缓存保留在 `.local`，均不提交。共享 Isaac Gym、Miniconda、现有环境和容器保持原路径。归属不明的 `pymp-*` / `torch-shm-*` 临时目录留在共享 `tmp`，未删除。

本次没有重新训练、重跑正式评测或声称实现逐位确定性复现。具体检查范围见 [验证记录](VALIDATION.md)。

## DIMP_FINAL 新训练与完整流程 smoke

2026-09-27 起，复现脚本统一放在 `reproduction/scripts/hpc/`，训练输出统一放在 `reproduction/runs/`。共享环境、原项目代码和 `.local/cache` 位置保持不变。

作业 5115268 在 step 0 的训练中评测因 `DIMP_FULL` 断言失败。其失败输出目录已删除，stdout/stderr 保留在 `logs/`；正式脚本已改为 `DIMP_FINAL` 以适配当前评测实现。**这会改变目标条件配置，因此新实验不能标为与历史官方 DIMP_FULL 参数完全一致。** 既有 300k/600k 历史实验及权重未改动。

新增的两个独立 Slurm 文件直接通过 `sbatch` 使用，不经过 `submit.sh` 的旧实验注册表。以下从 **Zhengying 目录**执行：

```bash
# 先测试：训练 + checkpoint + HumanML 评测 + 动作视频生成。
sbatch CLoSD/reproduction/scripts/hpc/dip_train_official_recipe_smoke.slurm

# smoke 通过后再提交正式训练。
sbatch CLoSD/reproduction/scripts/hpc/dip_train_official_recipe_300k.slurm
```

Smoke 使用 `qgpu_exp`、1 张 A100、8 CPU、30 分钟时限，输出目录为 `reproduction/runs/DiP_multi-target_official_recipe_smoke_<jobid>`，日志为 `logs/dip_recipe_smoke_<jobid>.out/.err`。每个新作业使用自己的目录，不删除既有 smoke 结果。

与正式版相同：`DIMP_FINAL`、网络与优化器参数、batch size 64、评测和生成开关。Smoke 参数缩为 `num_steps=20`、`save_interval=10`、`log_interval=1`、`eval_num_samples=320`、`eval_rep_times=1`、`gen_num_samples=2`、`gen_num_repetitions=1`。评测样本保持大于 300，以满足现有 diversity 指标检查；这些指标仅用于验证流程，不作为正式结果。生成样本至少 2 个，以避开上游可视化代码 `squeeze()` 删除单样本维度的问题（5121606 在视频生成时触发）。

Smoke 与正式版共用 `scripts/hpc/train_until_checkpoint.py`，分别传入 `--stop-after-checkpoint 10` 和 `300000`。Smoke 在 step 0 和 step 10 执行保存、评测、生成，step 10 全部完成后正常返回。成功标记为 `TRAIN/EVAL/GEN SMOKE PASSED`。30 分钟是申请的上限，不是保证完成时长。

正式版在 `model000300000.pt` 对应的保存、评测及全部生成完成后正常退出，随后关闭 W&B；Slurm 前台等待 Python 并保留错误退出码。两版均清除 `DIFFUSION_TRAINING_TEST`，不再轮询文件或发送终止信号。包装入口仅控制结束时机，不改变原项目源码、优化器、学习率或训练计算；保留正式版 `num_steps=600000`，另行指定停止点。

停止点沿用上游从 0 开始的 checkpoint 编号：从零训练到文件编号 300000 包含 300001 次训练循环；smoke 的 10 包含 11 次。目标必须是保存间隔的整数倍；目标周期失败或未到达时返回错误，不报告成功。当前官方 README 的目标配置默认值也是 `DIMP_FINAL`；本脚本的生成参数仍沿用已发布权重的参数，因此不要将全部参数标为与当前 README 完全一致。

## 新训练 300k 权重的 CLoSD 任务评测

训练 5121622 的日志已记录 `CHECKPOINT CYCLE COMPLETE: step 300000` 和 `TARGET 300k RUN FINISHED`。评测入口为 `scripts/hpc/dip_eval_recipe_300k.slurm`，默认模型为 `runs/DiP_multi-target_official_recipe_seed10/model000300000.pt`（以上路径相对于 reproduction）。原 `dip_eval_repro_300k` 继续指向历史复现权重，`dip_eval_official` 继续指向官方发布权重。

三者使用相同 CLoSD 多任务评测参数：官方 `CLoSD_multitask_finetune` 控制器、4096 并行环境、episode length 500、DiP CFG 7.5；申请 qgpu / A100 1 张 / 8 CPU / 4 小时。评测 REACH、KICK、PUNCH、SIT、GET_UP 的成功率；这是 CLoSD 仿真任务评测，与训练中的 HumanML 指标不同。

从 **Zhengying** 目录执行：

```bash
# 只检查路径和命令，不提交。
bash CLoSD/reproduction/scripts/hpc/dip_eval_recipe_300k.slurm --check
# 正式提交，脚本内部直接配置环境并启动评测。
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_recipe_300k.slurm
```

日志保存为 `reproduction/logs/dip_eval_recipe300_<jobid>.out/.err`。新入口只是更换被测 DiP 权重与作业名称，不覆盖已有基线结果。

### 30 分钟排队副本

已完成的官方 300k 作业 5100756 用时 6 分 32 秒，旧复现 300k 作业 5114966 用时 11 分 34 秒（Slurm Elapsed）。为尝试较短作业进入调度回填，复制新训练评测脚本为 `dip_eval_recipe_300k_30m.slurm`，只把时间上限从 4 小时改为 30 分钟，并改作业名称；权重和完整评测设置不变。30 分钟是上限，不能保证更早启动或本次用时相同。

从 Zhengying 目录提交：

```bash
sbatch CLoSD/reproduction/scripts/hpc/dip_eval_recipe_300k_30m.slurm
```

短版日志为 `reproduction/logs/dip_eval_recipe30m_<jobid>.out/.err`。原 4 小时作业 5133827 在创建副本时仍为 PENDING，未取消；如两个作业都进入运行，会执行相同评测。
