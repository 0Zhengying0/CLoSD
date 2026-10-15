# DiP 与 Flow Matching 规划器动作质量对照

这是**离线规划器输出**对照，不重新运行 4096 环境成功率评测，不把生成的动作称为控制器已执行的轨迹。DiP 使用 `DiP_multi-target_10steps_context20_predict40/model000300000.pt` 相邻的 `args.json`（扩散 10 步、DIMP_FULL）；FM 使用 `FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt` 相邻的 `args.json`（训练时间离散步 1000、DIMP_FINAL），RK4 三步、12 NFE。两者均使用 checkpoint 的 EMA 权重和 CFG=7.5、文字 CFG。程序会拒绝缺失 EMA 或参数不匹配的模型，不混用 `args.json`。

**视频含义：**`results/` 或 `smoke_<jobid>/` 中原有的固定相机骨架视频，仍是离线规划器的 22 关节输出。新增的 `official_skeleton_*` 使用仓库原有 `plot_3d_motion`（`sample/generate.py` 同样调用）来显示**同一份已保存轨迹**；该函数会逐帧移动显示中心并按 HumanML3D 常量缩放，因此外观与官方骨架演示一致，但公平的空间尺度比较仍应看原固定相机视频。新增的 `execution/` 和 `official_execution_*` 则由真实闭环仿真记录，经仓库的 `HumanoidIm._record_states`、`record2anim.py`、MJCF 网格与 FBX 目标生成，是控制器实际执行动作。两类视频不能相互当作同一条轨迹。仓库的 `empty.blend` 在 Karolina Blender 4.0.2 下没有相机，离屏 MP4 导出时脚本仅补固定相机、灯光和棋盘地面；人体网格和关键帧仍使用官方代码。

## 实验阶段

**以下命令由用户手动执行；`--check` 不提交 GPU 任务。** 使用绝对路径设置运行目录，避免当前目录是 `Zhengying` 时拼错路径。采集 Job 5199654 已完成；首次冒烟 Job 5199655 因脚本错误失败，修复后的冒烟 Job 5201637 已成功生成一个案例的指标与视频。旧目录均保留。若已确认 5201637 的输出，可直接提交 `--compare`，无需再次提交 `--capture` 或 `--smoke`。

```bash
RUN_TAG=dip_fm_quality_20261014_02
CLOSD_ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD
RUN_DIR="$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/$RUN_TAG"
cd "$CLOSD_ROOT"

# 02 的采集作业 5199654 已完成，已有 20 个真实案例；请勿再次运行 --capture
test -s "$RUN_DIR/conditions/manifest.json"
bash reproduction/scripts/hpc/submit_dip_fm_quality.sh --check --run-tag "$RUN_TAG"
# 已完成的冒烟 Job 5201637：检查产物
SMOKE_JOB=$(awk -F '\t' '$1=="smoke"{id=$2} END{print id}' "$RUN_DIR/jobs.tsv")
sacct -j "$SMOKE_JOB" --format=JobID,State,ExitCode,Elapsed
test -s "$RUN_DIR/smoke_$SMOKE_JOB/summary.md" && echo "SMOKE_SUMMARY=OK"
test -s "$RUN_DIR/smoke_$SMOKE_JOB/results.csv" && echo "SMOKE_CSV=OK"
# 确认冒烟成功后提交全部 20 个案例：
bash reproduction/scripts/hpc/submit_dip_fm_quality.sh --compare --run-tag "$RUN_TAG"

squeue -u "$USER" -o '%.18i %.50j %.12T %.10M %R'
cat "$RUN_DIR/jobs.tsv"
# 正式任务成功后查看自动生成的结果，无需再次运行汇总脚本
cat "$RUN_DIR/results/summary.md"
```

每次成功提交的 Job ID 也追加记录在 `$RUN_DIR/jobs.tsv`；Slurm 日志在 `$RUN_DIR/logs/`，按阶段和 Job ID 命名。

采集 Slurm 作业使用真实 CLoSD 控制器和 128 个仿真环境，只采集规划器输入，不统计正式成功率。每个任务状态最多收集来自不同环境的 16 个候选条件；若某状态的环境数不足 16，在五类各有至少 4 个候选且连续 5 次规划没有新增候选后结束采集。清单记录每类实际候选数。再按捕获时到**环境原始目标**的距离从近到远选 4 个，形成每类正常到困难的相对难度案例，总计 20 个。GET_UP 必须在真实 SIT→GET_UP 转换后被采集。若 400 次规划仍没有采齐五类，任务失败，不会伪造案例。采集跳过首次规划，因为那时 CLoSD 可能用 HumanML 数据替换 prefix；保存的历史均来自真实仿真 `pose_buffer`，且是实际送入规划器的非零张量。采集要求关闭 context switching，以保持目标与历史的坐标对齐。

案例文件在 `conditions/cases/<case_id>.json` 和 `.npz`。保存文字、任务和目标关节名称、heading 开关、局部 HML `target_cond`、原始环境目标及条件截断目标的世界位置、原始环境目标在局部 HML 中的位置、归一化 prefix、有效帧 mask、HumanML mean/std、恢复坐标变换、仿真人体历史、固定高斯初始噪声及 seed。两模型使用**同一份案例文件**，每个结果记录输入 SHA256。DiP 还会在其扩散步骤内部取随机数，不能声称两个过程的所有内部随机性相同。

`--smoke` 使用同一个 manifest 的第一个案例，输出到独立的 `smoke_<jobid>/`，生成两段单模型 MP4、并排 MP4、并排 GIF 和指标。程序解码检查 MP4/GIF 第一帧；正式提交入口要求已存在成功的冒烟汇总。`--compare` 处理全部 20 个案例，写到新的 `results/`；路径已存在时拒绝覆盖。两阶段均不修改 checkpoint 或历史评测产物。

## 输出与指标

- `results/<case_id>/dip_trajectory.npz`、`fm_trajectory.npz`：原始归一化 HML 动作 `[263,1,60]`、按 prefix 末帧对齐的局部 HML 骨骼 `[60,22,3]`（米）和目标距离曲线。对齐沿用 CLoSD 的“扣除 prefix 末帧水平位移，再按末帧根部旋转”的流程，使骨骼与目标条件处于同一空间。
- `results/<case_id>/dip_target_distance.csv`、`fm_target_distance.csv`、`target_distance.png`：预测 40 帧到环境原始目标及规划器条件目标的逐帧距离和对照曲线。
- `results/<case_id>/dip.mp4`、`fm.mp4`、`side_by_side.mp4`：统一相机、坐标轴范围、时间轴和 20 fps。灰色为已观察历史，橙色/青色为各模型生成段，洋红星号为环境原始目标，蓝色菱形为规划器条件目标。每类最困难案例额外有 `side_by_side.gif`；MP4 保留。
- `results/<case_id>/*_metrics.json`、`results/results.csv`、`results/group_summary.csv`、`results/summary.md`：逐案例与任务分组的均值/中位数（分组 CSV 覆盖全部数值指标），以及视频相对路径。`results/provenance.json` 保存 checkpoint、各自 `args.json` SHA256、Git commit、Slurm Job ID 和案例 manifest SHA256。

REACH 对局部 HML pelvis 和原始环境目标计算水平 XZ 距离；GET_UP 对 pelvis 计算三维距离。两个坐标量都先恢复到同一局部 HML 米制空间。`results.csv` 另列对规划器截断目标的误差，便于区分长期目标与单次规划视野。SIT 为 pelvis 到座椅目标点的诊断距离，不能代替官方座椅体积成功判定；KICK/PUNCH 为目标脚/腕到箱体目标点距离，不能代替箱子倾倒判定。无严格匹配的真实未来动作时，GT 误差留空。平滑性是生成段骨骼速度/加速度的描述统计，不表示动力学可部署性。

若需要基于已有结果重新生成 Markdown（不会覆盖原文件）：

```bash
python3 reproduction/scripts/hpc/summarize_dip_fm_quality.py \
  --run-dir "$RUN_DIR/results" --no-csv \
  --output "$RUN_DIR/results/summary_recheck.md"
```

生成程序也支持指定单案例的 GPU 冒烟入口：`python3 reproduction/scripts/hpc/run_dip_fm_quality.py --cases "$RUN_DIR/conditions/manifest.json" --case-id REACH_00_normal --output "$RUN_DIR/manual_smoke"`。该命令需要在 A100 GPU 环境中执行，输出目录必须尚不存在。

## 两种官方代码可视化

以下从已经成功的 5201637 单案例轨迹，使用**官方骨架绘图函数**产生的独立产物已经验证：

- `official_skeleton_smoke_5201637_v2/REACH_00_normal/dip_official.mp4`
- `official_skeleton_smoke_5201637_v2/REACH_00_normal/fm_official.mp4`
- `official_skeleton_smoke_5201637_v2/REACH_00_normal/side_by_side_official.mp4` 和 `.gif`

文件已经过首帧解码检查。`official_skeleton_smoke_5201637_v2/index.csv` 给出相对路径。若要为**正式 20 个案例**制作同风格视频，先等 `results/summary.md` 存在，再由用户提交渲染作业。当前用户的 `EU-26-37` Slurm 关联只包含 `qgpu` 等 GPU 分区；`qcpu` 会报 `Invalid account or account/partition combination`，默认 CPU 账户的 `--test-only` 则报 `AssocMaxSubmitJobLimit`。因此脚本在 `qgpu` 申请 1 块 A100，虽然骨架/Blender 渲染主要使用 CPU，但会占用 GPU 配额。`--check` 现在调用 `sbatch --test-only` 核对账户与分区，不会真正提交。四个质量实验 Slurm 脚本均通过 `zy.xu@tum.de` 发送 BEGIN、END、FAIL 通知。

```bash
cd "$CLOSD_ROOT"
test -s "$RUN_DIR/results/summary.md"
QUALITY_RUN_ROOT="$RUN_DIR" QUALITY_RENDER_KIND=skeleton \
  QUALITY_SOURCE_DIR="$RUN_DIR/results" \
  bash reproduction/scripts/hpc/dip_fm_quality_official_render.slurm --check
sbatch --chdir="$CLOSD_ROOT" \
  --job-name="quality_skeleton_${RUN_TAG}" \
  --output="$RUN_DIR/logs/official_skeleton_%j.out" \
  --error="$RUN_DIR/logs/official_skeleton_%j.err" \
  --export="ALL,QUALITY_RUN_ROOT=$RUN_DIR,QUALITY_RENDER_KIND=skeleton,QUALITY_RENDER_MODE=formal,QUALITY_SOURCE_DIR=$RUN_DIR/results" \
  reproduction/scripts/hpc/dip_fm_quality_official_render.slurm
```

官方项目页那种**实体角色加目标物和棋盘地面**要先采集闭环仿真。下面的 `--check` 不提交任务；`--submit` 才提交两个 A100 作业，每个运行 128 个环境、seed 0、episode length 500、CFG 7.5，分别加载各自的 300k checkpoint，记录 500 个仿真帧。输出在 `execution/dip/episodes.pkl` 和 `execution/fm/episodes.pkl`，每个记录含人体姿态、目标和任务状态。它与离线 20 案例不是相同输入快照，因此该视频用于观察控制器执行外观，不用于代替前面的离线定量误差。

```bash
bash reproduction/scripts/hpc/submit_dip_fm_execution.sh --check --run-tag "$RUN_TAG"
# 用户确认需要采集时手动提交；本仓库准备工作不会自行提交
bash reproduction/scripts/hpc/submit_dip_fm_execution.sh --submit --run-tag "$RUN_TAG"
cat "$RUN_DIR/execution_jobs.tsv"
squeue -u "$USER" -o '%.18i %.50j %.12T %.10M %R'
```

两个录制任务完成后，先提交仅 3 帧的技术冒烟。正式渲染会寻找两份记录中具有**相同环境回合键和相同初始任务**、且都出现对应任务状态的 REACH、KICK、PUNCH、SIT、GET_UP 回合；找不到就明确失败，不拼凑不同条件。生成的 `*_prepared.npz` 只是让 Blender 4.0.2 不需要安装 PyTorch 的适配中间文件，保留原始 `episodes.pkl`。正式输出中每类有 DiP、FM、并排 MP4，GET_UP 有并排 GIF；`index.csv` 指向视频。两个模型之后的闭环状态允许自然分化。

本次 5201946/5201947 录制各约 16 GB。已经读出匹配回合：REACH=`0_0`、KICK=`25_0`、PUNCH=`3_0`、SIT=`2_0`、GET_UP=`2_0`。不建议在登录节点重复运行 `render_quality_execution.py --check`，因为它要完整读取两份大型 pickle；渲染作业本身会再次核对并在计算节点上处理。读取时将官方记录中的 CUDA 张量映射到 CPU，不修改原始文件；每个 pickle 只加载一次，提取选中回合后再运行 Blender。

```bash
test -s "$RUN_DIR/execution/dip/episodes.pkl"
test -s "$RUN_DIR/execution/fm/episodes.pkl"
QUALITY_RUN_ROOT="$RUN_DIR" QUALITY_RENDER_KIND=execution \
  bash reproduction/scripts/hpc/dip_fm_quality_official_render.slurm --check
sbatch --chdir="$CLOSD_ROOT" \
  --job-name="quality_exec_smoke_${RUN_TAG}" \
  --output="$RUN_DIR/logs/official_execution_smoke_%j.out" \
  --error="$RUN_DIR/logs/official_execution_smoke_%j.err" \
  --export="ALL,QUALITY_RUN_ROOT=$RUN_DIR,QUALITY_RENDER_KIND=execution,QUALITY_RENDER_MODE=smoke" \
  reproduction/scripts/hpc/dip_fm_quality_official_render.slurm
# 检查冒烟日志中的 QUALITY_OFFICIAL_RENDER_COMPLETE=1，再手动提交正式渲染：
sbatch --chdir="$CLOSD_ROOT" \
  --job-name="quality_exec_formal_${RUN_TAG}" \
  --output="$RUN_DIR/logs/official_execution_formal_%j.out" \
  --error="$RUN_DIR/logs/official_execution_formal_%j.err" \
  --export="ALL,QUALITY_RUN_ROOT=$RUN_DIR,QUALITY_RENDER_KIND=execution,QUALITY_RENDER_MODE=formal" \
  reproduction/scripts/hpc/dip_fm_quality_official_render.slurm
```

所有新输出均带 Slurm Job ID 且在独立目录，已有 MP4/GIF、checkpoint 和历史评测结果不会被覆盖。`execution` 作业是额外的闭环仿真录制，不是 4096 环境官方成功率评测。
