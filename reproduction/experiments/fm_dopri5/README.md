# FM 300k：DOPRI5 与 Euler / Heun / RK4 的同预算对照

历史积分器实验按 **12、20、40 NFE** 比较 Euler、Heun、RK4。DOPRI5 的步长自适应，设置 `ODE_STEPS=0`，不能直接指定固定 NFE。已完成的两案例初始冒烟在 `rtol=1e-3, atol=1e-4` 下测得 68、80 NFE；因此**不要将这一设置直接标作与历史 RK4 12 NFE 等预算**。

新实验先在同一个 batch 内重放 20 个真实 CLoSD 条件快照，测得当前 DOPRI5 容差的 NFE，再取最近的 **4 的倍数**作为公共预算。随后用这一预算设置三个固定步求解器：

| 求解器 | 正式积分设置 | 每次规划目标 NFE |
| --- | --- | ---: |
| Euler | `steps=公共预算` | 公共预算 |
| Heun | `steps=公共预算/2` | 公共预算 |
| RK4 | `steps=公共预算/4` | 公共预算 |
| DOPRI5 | `rtol=1e-3, atol=1e-4`，自适应 | 实测值；校准值接近公共预算 |

四组共用 FM 300k EMA checkpoint、CFG 7.5、seed 0、4096 环境、episode length 500、每项至少 1000 个结束回合与官方成功率判定。CFG 下每 NFE 对应两次实际 MDM 前向。Planner 延迟和 NFE 都从每次正式规划调用中读取。**离线 20 案例校准不保证 4096 环境闭环时 NFE 不变**：汇总器会列出正式运行的 NFE 均值、最小、最大值；若 DOPRI5 平均值偏离公共预算超过 20%（至少 2 NFE），状态标为 `OUT_OF_BAND`，不解读为等计算量比较。

MDM 的时间编码为整数位置编码下标。四种方法共用 `round(t × 999)`，其中 `t` 从 0 到 1。时间取整令速度场随时间跳变，限制 DOPRI5 的高阶误差估计。DOPRI5 使用 [`torchdiffeq==0.2.5`](requirements.txt)，最大 128 个内部积分步、256 次速度场调用；触及上限会报错。

## 运行顺序

从 CLoSD 仓库根目录执行。`dopri5_20261017_01` 的两案例冒烟已完成，Job ID `5220070`，可以直接从**校准**开始：

```bash
cd /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD
RUN_TAG=dopri5_20261017_01
RUN_DIR="$PWD/reproduction/experiments/fm_dopri5/runs/$RUN_TAG"
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --check --run-tag "$RUN_TAG"
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --calibrate --run-tag "$RUN_TAG"
CAL_JOB=$(cat "$RUN_DIR/calibration_job_id.txt")
squeue -j "$CAL_JOB" -o '%.18i %.24j %.2t %.10M %.2D %R'
```

校准任务完成后查看 `$RUN_DIR/calibration_$CAL_JOB/calibration.json` 中的 `default_dopri5.nfe` 和 `recommended_common_nfe`。校准同时记录不同容差在 12、20、40 NFE 附近的可达性，但**正式四组**使用相同的 DOPRI5 默认容差和新公共预算。

```bash
cat "$RUN_DIR/calibration_$CAL_JOB/calibration.json"
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --check --run-tag "$RUN_TAG"
```

此时 `--check` 打印四组确切步数、NFE 与 Slurm 命令，运行 CPU 测试和 Slurm `--test-only`，不会提交任务。确认后由你亲自提交四组：

```bash
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --submit --run-tag "$RUN_TAG"
column -ts $'\t' "$RUN_DIR/matched_submissions.tsv"
squeue -u "$USER" -o '%.18i %.32j %.2t %.10M %.2D %R'
```

若要完全由你新建冒烟任务，用未使用的 `RUN_TAG` 运行 `--smoke`，确认 `$RUN_DIR/smoke_<JobID>/smoke.json` 与 `logs/smoke_<JobID>.out` 完整成功后，再运行 `--calibrate`。

全部四组结束后生成报告和 CSV；输出已存在时拒绝覆盖：

```bash
python3 reproduction/scripts/hpc/summarize_fm_dopri5.py \
    --run-dir "$RUN_DIR" --output "$RUN_DIR/matched_summary.md"
```

日志、校准 JSON、提交 manifest 和报告都在独立的 `reproduction/experiments/fm_dopri5/runs/<RUN_TAG>/` 中。历史九组日志、checkpoint 和模型权重均不修改。

## 2026-10-17 四组启动失败后的重提

`5220246`、`5220247`、`5220248`、`5220249` 都因运行目录缺少 `work/`，在进入模型前失败。共享评测脚本现会创建 `work/`，提交器也会在提交前创建它。保留原 `matched_submissions.tsv` 和原日志，重提使用新清单及带后缀的日志；已完成的 20 案例校准直接复用。

```bash
cd /mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD
RUN_TAG=dopri5_20261017_01
RUN_DIR="$PWD/reproduction/experiments/fm_dopri5/runs/$RUN_TAG"
RETRY_TAG=retry01
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --check --run-tag "$RUN_TAG" --retry-tag "$RETRY_TAG"
bash reproduction/scripts/hpc/submit_fm_dopri5.sh --submit --run-tag "$RUN_TAG" --retry-tag "$RETRY_TAG"
column -ts $'\t' "$RUN_DIR/matched_submissions_${RETRY_TAG}.tsv"
```

四组完成后，用重提清单汇总；该命令不会覆盖已有报告：

```bash
python3 reproduction/scripts/hpc/summarize_fm_dopri5.py \
    --run-dir "$RUN_DIR" \
    --manifest "$RUN_DIR/matched_submissions_${RETRY_TAG}.tsv" \
    --output "$RUN_DIR/matched_summary_${RETRY_TAG}.md"
```
