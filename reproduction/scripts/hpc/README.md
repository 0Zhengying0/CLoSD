# HPC reproduction scripts

完整中文说明见 [reproduction/README.md](../../README.md)。

当前 `.slurm` 文件直接包含资源、路径、环境设置、容器启动和训练/评测命令。旧 12 份脚本从 Git 提交 `bfe9777` 的 `reproduction/history/scripts` 原文恢复，只适配迁移后的路径；旧训练 smoke 使用新的作业号目录，避免删除已有历史产物。

从 Zhengying 目录执行 `sbatch CLoSD/reproduction/scripts/hpc/<文件>.slurm`。旧文件的参数直接在文件里编辑；语法检查用 `bash -n`，不要用 `bash <旧文件> --check`，旧脚本没有该检查选项。

新 `dip_eval_recipe_300k.slurm` 内部写全新训练的 300k 评测配置，支持 `--check` 和 `--model PATH`。旧 `dip_eval_official`、`dip_eval_repro_300k`、`dip_eval_repro_600k` 分别保留各自原权重路径和原始评测参数。

新增的 `dip_train_official_recipe_{300k,smoke}.slurm` 保留完整训练命令，仅共用获授权的 `train_until_checkpoint.py` 正常停止机制，目标分别为 300000 和 10；不调用 run_job.py。其参数只属于复现入口，原项目 Python 接口不变。

之前的 `run_job.py`、`jobs.json`、`submit.sh` 等整理工具保留，但不再是当前 Slurm 的运行依赖；以直接 sbatch 和 Slurm 内的配置为准。
