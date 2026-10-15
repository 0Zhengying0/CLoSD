# Flow Matching RK4 3 步与原始 DiP：闭环逐回合诊断

两个任务使用同一 CLoSD 正式闭环评测入口和控制器，均为 Karolina A100、随机种子 0、4096 个环境、episode length 500、CFG 7.5、每项官方指标至少 1000 个结束回合。FM 使用现有 300k checkpoint、RK4 3 积分步（12 NFE）；DiP 使用官方 `DiP_multi-target_10steps_context20_predict40/model000300000.pt` 与原始 Gaussian diffusion `p_sample_loop`（模型配置的 10 步）。不重新训练、不修改权重和官方计数公式。

诊断是可选的：只有设置 `CLOSD_EPISODE_DIAG_DIR` 才启用。它在官方 `_post_step` 用**同一组布尔成功掩码**记录每个结束回合，并在评测结束时把逐回合计数与官方总计数逐项核对；不改变 `done`、`terminate`、`task_done`、状态转换或停止条件。评测停止时尚未结束的轨迹标为 `evaluation_stopped`，其官方成功字段为 `null`，不进入官方分母。

## 提交

从 `CLoSD` 仓库根目录执行，运行标签须未使用过。先检查，再由你亲自提交两个 GPU 作业：

```bash
DIAG_TAG=rk4_vs_dip_20261014_01
bash reproduction/scripts/hpc/submit_closd_episode_diagnostics.sh --check --run-tag "$DIAG_TAG"
bash reproduction/scripts/hpc/submit_closd_episode_diagnostics.sh --submit --run-tag "$DIAG_TAG"
```

`--check` 验证两个 checkpoint 及相邻 `args.json`、控制器、求解器、语法和无 GPU 测试，打印两条完整 `sbatch` 命令，不提交作业。`--submit` 再次检查并创建唯一运行目录；已存在目录会被拒绝，不覆盖历史结果。`submissions.tsv` 记录两个 Job ID 和提交状态。两个作业各有独立的 stdout/stderr、Hydra、控制器链接、Torch/W&B/XDG/VHACD 缓存，以及短路径 `/tmp` socket 目录。

```bash
squeue -u "$USER" -o "%.18i %.36j %.2t %.10M %.20R"
sacct -u "$USER" --format=JobID,JobName,State,ExitCode,Elapsed -P
```

## 结果文件

```text
reproduction/experiments/episode_diagnostics/runs/<DIAG_TAG>/
├── submissions.tsv
├── fm_rk4_3/
│   ├── logs/                 # 原始官方评测 stdout/stderr
│   └── diagnostics/
│       ├── manifest.json     # 完成数、官方原始计数、字段/单位说明
│       ├── episodes.jsonl    # 一行一个已结束回合或停止时的未结束片段
│       └── trajectories/part-*.npz
└── dip_original/            # 相同结构
```

`episodes.jsonl` 含 `initial_task`、初始/最终目标与状态、首次物理达标步及秒数、首次官方 `task_done` 锁存步、最小目标距离、跌倒与时间、`terminal_reason`、完整 `state_transitions`、SIT→GET_UP 切换时间、GET_UP 首次达标及结束时骨盆高度，以及逐项官方 `counted`/`success`。`official_success_for_initial_task` 是按**原始计数公式**对此初始任务得到的结果。KICK/PUNCH 的官方两列原本使用相同的混合击打掩码；逐回合的 `initial_task` 才能区分踢和拳。GET_UP 只来自 SIT 后的切换。

`first_initial_goal_step` 读取物理步后的即时目标条件；`first_official_task_done_step` 记录步后可见的官方锁存标志，可能晚一个控制步。KICK/PUNCH 达标后会立刻切到 HALT 并清除 `task_done`，所以该锁存字段可能为 `null`，此时以即时达标字段和状态转换为准。状态转换在物理步前发生，因此 `sit_to_get_up_step` 是切换时的**物理步前** progress。`fall_step` 是官方 `_terminate_buf` 首次为真时的物理步后 progress；`terminal_reason=fall`、`time_limit` 或 `other_reset`。每步 `progress_step × control_dt_s` 为仿真秒数（当前 30 Hz），不是运行耗时。

`min_initial_target_distance_m` 对 REACH 是骨盆水平距离、GET_UP 是骨盆三维距离、SIT 是骨盆到有效坐姿体积的距离。KICK/PUNCH 为当前指定脚/腕到箱体中心的距离，**仅作轨迹诊断**；击打成功仍由 `box_up_dot <= 0.25` 判定。目标位置与状态在轨迹中逐步记录，便于检查切换后的目标变化。`controller_action` 是每个控制步的策略动作向量；`pd_target` 是实际送给 PD 控制器的关节位置目标；`dof_pos` 和 `body_pos_xyz` 分别是物理步后实现的关节位置、人体各刚体世界坐标，均保存为原始 `float32`；压缩 NPZ 文件可能较大。

## 汇总和查看某个回合

两个作业完成后执行。汇总脚本对缺失任务显示 `MISSING/FAILED`，不填猜测结果；输出文件已存在时拒绝覆盖。

```bash
DIAG_RUN="reproduction/experiments/episode_diagnostics/runs/$DIAG_TAG"
python3 reproduction/scripts/hpc/analyze_episode_diagnostics.py summary \
  --run-dir "$DIAG_RUN" \
  --output "$DIAG_RUN/comparison.md" \
  --episodes-csv "$DIAG_RUN/episodes.csv"
```

`comparison.md` 先给官方五项指标原始分子/分母，再按**初始任务**比较首次达标、达标后跌倒、最小距离、最终官方成功，并单列 SIT→GET_UP。`episodes.csv` 可按 `initial_task`、`official_success_for_initial_task`、`fell`、`terminal_reason` 等列筛选。两次运行使用相同评测种子，但生成器可能消耗不同随机数，不能把相同 `env_id` 当成严格配对样本。

从 `episodes.csv` 选择一个 `episode_id`，导出它的每步人体轨迹、目标和控制动作：

```bash
python3 reproduction/scripts/hpc/analyze_episode_diagnostics.py trajectory \
  --diagnostics-dir "$DIAG_RUN/fm_rk4_3/diagnostics" \
  --episode-id env0000-ep0000 \
  --output "$DIAG_RUN/fm_env0000_ep0000.csv"
```

导出的 CSV 包含 `progress_step`、`time_s`、状态编号、目标距离、箱体上方向点积、达标/终止布尔值、目标 XYZ、各人体刚体 XYZ、策略动作、PD 目标和物理步后关节位置。状态编号对应 `closd/utils/closd_util.py` 中的 `STATES`。
