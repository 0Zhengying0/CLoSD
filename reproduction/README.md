# DiP multi-target 从零训练及其在 CLoSD 中的评测

本目录记录 Zhengying 在 Karolina Slurm 集群上的复现实验。训练的是 DiP multi-target；评测的运动控制器来自官方 `CLoSD_multitask_finetune` 权重。没有重新从零训练整套 CLoSD 控制器。

## 材料入口

- [实验结果和原始日志来源](experiments/results.md)：官方 300k 基线、复现 300k 和复现 600k。
- [实验时间线与失败尝试](experiments/README.md)：完整训练和 `_short` 续训分别记录。
- [环境与官方资源恢复](environment/README.md)：实际依赖清单、容器身份、固定资源版本。
- [完整日志](logs/)：Slurm stdout/stderr、GPU CSV、训练 logger 与 W&B 文本记录。
- [历史脚本](history/scripts/)和[原始笔记](history/note.txt)：保留原始内容，仅作证据，包含迁移前绝对路径；不作为新运行入口。
- [迁移清单](experiments/migration-manifest.json)：文件原址、新址、大小、inode、关键文件校验和与软链接变更。

训练源码提交为 `4af01718653dbabf1de1dd9a2600aa54c91acc38`；本次整理后的提交只调整复现入口与材料组织，不改模型或训练算法。历史 `args.original.json` 保留原始路径。当前执行配置见 `scripts/hpc/jobs.json` 和各入口 `--check` 输出。

## 使用入口

从仓库根目录执行；提交脚本自身可从任意工作目录调用。先检查，再去掉 `--check` 提交。

```bash
# 新训练必须使用空目录；默认目录已有历史权重，所以示例指定新目录。
bash scripts/hpc/submit.sh dip_train --save-dir "$PWD/runs/my_seed10" --check

# 短时训练的新实验与续训必须使用相同目录。
bash scripts/hpc/submit.sh dip_train_6h --save-dir "$PWD/runs/my_short" --check
bash scripts/hpc/submit.sh dip_train_2h --resume --check

# 复现权重与官方基线评测（各入口保留原始评测参数）。
bash scripts/hpc/submit.sh dip_eval_official --check
bash scripts/hpc/submit.sh dip_eval_repro_300k --check
bash scripts/hpc/submit.sh dip_eval_repro_600k --check

# 环境规模测试和两类 smoke。
bash scripts/hpc/submit.sh dip_eval_envtest --num-envs 128 --check
bash scripts/hpc/submit.sh closd_smoke --check
bash scripts/hpc/submit.sh closd_real_smoke --check
bash scripts/hpc/submit.sh dip_train_smoke --save-dir "$PWD/runs/new_smoke" --check
```

另有 `dip_train_4h`、`dip_train_12h`，使用 `_short` 默认输出目录。`--resume` 选择该目录步数最大的 `model*.pt` 并要求相应优化器文件及 `args.json`，与原训练器自动查找规则一致。下载的 Release 仅用于评测，不含优化器，不能当作完整续训备份。原 smoke 会删除已有目录，现在改为拒绝覆盖。

`--check` 不写文件、不提交作业、不启动容器。实际提交会先创建日志目录，并显式设置 Slurm 工作目录、输出路径及 `CLOSD_ROOT`。不要在其它目录直接 `sbatch` 这些包装脚本；统一使用 `submit.sh`。

## 路径与资源配置

默认仓库根目录由脚本定位，共享根目录是其父目录。可通过环境变量覆盖：

| 变量 | 默认值（相对仓库） |
|---|---|
| `CLOSD_SHARED_ROOT` | `..` |
| `CLOSD_ENV` | `../envs/closd` |
| `CLOSD_ISAACGYM` | `../isaacgym` |
| `CLOSD_CONTAINER` | `../containers/closd_cuda121_ubuntu20_devel.sif` |
| `CLOSD_CACHE` | `.local/cache` |
| `CLOSD_LOGS` | `reproduction/logs` |

Slurm 默认为 `EU-26-37`、A100、8 CPU；分区和时长保留各原始脚本设置。可用 `CLOSD_SLURM_ACCOUNT`、`CLOSD_SLURM_PARTITION`、`CLOSD_SLURM_GRES`、`CLOSD_SLURM_CPUS_PER_TASK`、`CLOSD_SLURM_TIME` 覆盖。`--save-dir` 和 `--model` 接受相对调用目录或绝对路径。

## 权重与本地存储

[Release repro-dip-seed10-v1](https://github.com/0Zhengying0/CLoSD/releases/tag/repro-dip-seed10-v1) 发布原始 `model000300000.pt`、`model000600000.pt`、相邻 `args.json` 和 SHA-256 清单。下载器依据仓库内的校验值验证，不覆盖内容不同的已有文件。

```bash
python3 scripts/hpc/download_release.py
python3 scripts/hpc/download_release.py --check
```

默认恢复到 `runs/DiP_multi-target_repro_seed10/`。其它 checkpoint、优化器和 W&B 二进制保留在本地 `runs`；专属缓存保留在 `.local`，均不提交。共享 Isaac Gym、Miniconda、现有环境和容器保持原路径。归属不明的 `pymp-*` / `torch-shm-*` 临时目录留在共享 `tmp`，未删除。

本次没有重新训练、重跑正式评测或声称实现逐位确定性复现。具体检查范围见 [验证记录](VALIDATION.md)。
