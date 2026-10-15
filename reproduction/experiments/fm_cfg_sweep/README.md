# Flow Matching CFG 扫描

使用同一个 FM 300k EMA checkpoint、RK4 三步（12 NFE）、官方 CLoSD 成功率统计，独立评测 CFG=1、2.5、5、7.5、10、15。每组 seed=0、4096 环境、episode length=500，五项任务各至少 1000 个结束回合。控制器与任务判定保持原样。

| 实验 | CFG | 引导包装 | 每次规划 NFE | 每次规划 MDM 前向 |
| --- | ---: | --- | ---: | ---: |
| FM-CFG1 | 1 | 无 | 12 | 12 |
| FM-CFG2p5 | 2.5 | 仅文字 CFG | 12 | 24 |
| FM-CFG5 | 5 | 仅文字 CFG | 12 | 24 |
| FM-CFG7p5 | 7.5 | 仅文字 CFG | 12 | 24 |
| FM-CFG10 | 10 | 仅文字 CFG | 12 | 24 |
| FM-CFG15 | 15 | 仅文字 CFG | 12 | 24 |

从仓库根目录执行：

```bash
RUN_TAG=fm_cfg_20261014_01
bash reproduction/scripts/hpc/submit_fm_cfg_sweep.sh --check --run-tag "$RUN_TAG"
bash reproduction/scripts/hpc/submit_fm_cfg_sweep.sh --submit --run-tag "$RUN_TAG"
RUN_DIR="$PWD/reproduction/experiments/fm_cfg_sweep/runs/$RUN_TAG"
squeue -u "$USER" -o '%.18i %.50j %.12T %.10M %.6D %R'
python3 reproduction/scripts/hpc/summarize_fm_cfg_sweep.py --run-dir "$RUN_DIR" --output "$RUN_DIR/summary.md"
```

`--check` 只执行静态检查、CPU 测试和六组 Slurm 命令预检；它不会创建运行目录或提交作业。`--submit` 再次预检，然后用新的 run tag 原子创建目录并提交。若已存在同名目录，脚本拒绝运行；不会覆盖历史结果或取消任何任务。提交清单为 `submissions.tsv`，其 `job_id` 仅说明提交成功，不代表评测成功。

汇总脚本只对完整且计数、NFE、前向次数、运行时 CFG、profile 都一致的作业显示成功率。输出文件必须是新路径；已有 `summary.md` 不会覆盖，可直接不带 `--output` 在终端查看最新状态。`MISSING_LOG`、`INCOMPLETE`、`FAILED`、`INVALID_METRICS`、`SUBMIT_FAILED` 均不会填入猜测值。它还读取原积分器扫描的 FM-R12 Job 5185912 作为 83.82% 的 CFG=7.5 基线，比较 REACH、GET_UP 和 Macro。各组实际分母可能超过 1000。

所有组使用同一文字 CFG 路径；CFG=1 直接调用原始 MDM。目标位置条件等其余输入依照现有 CLoSD 代码传入。模型时间编码依然是最近整数索引，RK4 中间时间也会取整。
