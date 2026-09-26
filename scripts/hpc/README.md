# HPC reproduction entrypoints

完整中文说明见 [reproduction/README.md](../../reproduction/README.md)。

使用 `bash scripts/hpc/submit.sh <job> --check` 检查资源、最终 Python 命令和 Slurm 命令；去掉 `--check` 后提交。已有训练目录需显式 `--resume`，新训练可用 `--save-dir` 指定空目录。

`jobs.json` 保留历史实验参数；`run_job.py` 统一处理路径、容器、训练保护和 GPU 监控。`.slurm` 文件保留各入口资源设置，调用同一个执行器。历史原文在 `reproduction/history/scripts`，不可作为迁移后的运行入口。
