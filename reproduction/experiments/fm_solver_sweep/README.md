# Flow Matching 300k 积分器对照实验

本目录只准备实验，不含新评测结果。九个任务复用 [`closd_velocity_eval.slurm`](../../scripts/hpc/closd_velocity_eval.slurm)；使用现有 Flow Matching 300k checkpoint，不训练或改动权重。正式评测沿用当前闭环设置：Karolina A100、评测随机种子 **0**（checkpoint 的训练种子是 **10**）、4096 个并行环境、每任务至少 1000 个结束回合、episode length 500、CFG 7.5、官方 `CLoSD_multitask_finetune` 控制器。五项成功率按[现有评测规则](../closd_evaluation_success_criteria.md)统计。

| 编号 | 积分器 | 步数 | NFE | CFG 开启时预期 MDM 前向/每次规划 |
| --- | --- | ---: | ---: | ---: |
| FM-E12 | Euler | 12 | 12 | 24 |
| FM-H12 | Heun | 6 | 12 | 24 |
| FM-R12 | RK4 | 3 | 12 | 24 |
| FM-E20 | Euler | 20 | 20 | 40 |
| FM-H20 | Heun | 10 | 20 | 40 |
| FM-R20 | RK4 | 5 | 20 | 40 |
| FM-E40 | Euler | 40 | 40 | 80 |
| FM-H40 | Heun | 20 | 40 | 80 |
| FM-R40 | RK4 | 10 | 40 | 80 |

已有 [Euler 10 步正式结果](../../../logs/fm_10step_5153280.out)，不重复提交。NFE 是速度场计算次数；实际 MDM 前向由网络前向钩子在每次规划时计数，当前 CFG 7.5 对每次速度场计算执行条件与无条件两次 MDM 前向。延迟使用已有 `PLANNER_LATENCY_SEC` 口径，即同步计时一次采样器调用，不含调用前的文本编码和调用后的动作坐标转换。

## 提交前检查与正式提交

在仓库根目录选择一个新标签，并对**同一标签**运行检查和提交。`--check` 核对 checkpoint、相邻 `args.json`、控制器、路径、矩阵参数、Python 与 Shell 语法，运行无 GPU 单元测试，并列出全部九个 `sbatch` 命令。它不会调用 `sbatch` 或创建运行目录。

```bash
export FM_SWEEP_TAG=fm_solver_20261012
bash reproduction/scripts/hpc/submit_fm_solver_sweep.sh --check --run-tag "$FM_SWEEP_TAG"
bash reproduction/scripts/hpc/submit_fm_solver_sweep.sh --submit --run-tag "$FM_SWEEP_TAG"
```

请在提交时把标签换成一个尚未使用的值。`--submit` 会**重新执行全部检查和测试**，然后原子创建新的 `runs/<标签>` 目录；目录已存在时拒绝提交，避免覆盖历史结果。每项提交的 Job ID 与状态写入 `submissions.tsv`。若某项 `sbatch` 失败，脚本继续尝试其余项，明确标为 `submit_failed`，最后以非零状态退出；不会取消已提交任务。

查询状态：

```bash
squeue -u "$USER" -o "%.18i %.36j %.2t %.10M %.20R"
sacct -u "$USER" --format=JobID,JobName,State,ExitCode,Elapsed -P
```

全部任务结束后生成汇总表；输出文件已存在时脚本拒绝覆盖。也可省略 `--output`，直接在终端预览当前结果，未完成任务会显示状态而不会填入猜测值。

```bash
FM_SWEEP_RUN="reproduction/experiments/fm_solver_sweep/runs/$FM_SWEEP_TAG"
python3 reproduction/scripts/hpc/summarize_fm_solver_sweep.py \
    --run-dir "$FM_SWEEP_RUN" --output "$FM_SWEEP_RUN/summary.md"
```

`summary.md` 使用新增的 `EVAL_RESULT` 原始成功/结束计数求五项等权平均，不用两位小数的屏幕输出反推；同时列出每个任务的积分步数、NFE、实际 MDM 前向数、planner 延迟均值/中位数、Job ID、checkpoint、运行 Git commit、源码哈希、种子和配置。没有完成标记、五项计数或正确 CFG 前向次数的日志不会被标为成功。

## 时间离散化与并发输出

MDM 的时间编码仍要求整数索引。Euler、Heun 和 RK4 共用 `round(t × (T−1))`，其中 `T=1000`、`t` 从 0 增至 1；RK4 的两个中间阶段取各步的中点并分别映射到整数索引。取整使速度场对时间呈阶梯变化，**不能保证经典 RK4 的理论四阶收敛率**；此次对照测量的是实际成功率与延迟。

每项 Slurm 作业的 stdout/stderr 含 Job ID，并有独立的 Hydra 目录、控制器权重只读链接、临时目录、Torch 扩展缓存、XDG/W&B 缓存和 Isaac Gym VHACD 缓存。`multiprocessing.Manager()` 的 Unix socket 使用作业内新建的短路径 `/tmp/cfm.*`，启动时实际测试 socket 绑定；其他缓存仍在独立工作目录。Hugging Face 预下载的 18 GB 数据缓存保持共享，以离线模式读取。运行时记录 Git commit、工作区是否有未提交内容，以及本次相关源码的 SHA256；commit 单独不能表示未提交的本地修改。不要在九项作业运行期间改动采样器、评测器或控制代码。

`fm_solver_20261012_01` 的九项作业均在加载动作库时因 `AF_UNIX path too long` 退出，没有形成评测结果。其日志和 `submissions.tsv` 保留原样。修复后必须使用新的运行标签；本地检查已通过 `fm_solver_20261012_02`。
