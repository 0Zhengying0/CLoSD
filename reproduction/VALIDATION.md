# 整理与验证记录

验证日期：2026-09-26。没有提交新的 GPU 作业，没有重新训练或正式评测。

已完成：

- 迁移前检查相关 Slurm 作业，未发现 CLoSD/DiP 活动或排队任务。
- 12 个作业入口通过 `--check`，原训练与评测超参数保留；正式评测仍为 4096 环境、500 帧、CFG 7.5。
- 所有活动 Shell/Slurm 脚本通过 `bash -n`，Python 脚本通过语法检查。
- 4 个 CPU 回归测试通过：检查模式不写文件、非空训练目录保护、最新 checkpoint 与优化器校验、显式 Slurm 路径及 spool 目录处理。
- 399 个迁移条目核对通过；文件大小及 inode/软链接一致；记录校验和的文件重新计算 SHA-256 一致。
- 6 个官方资源链接有效，官方 CLoSD 与 DistilBERT 缓存 revision 固定；在现有容器内用 `snapshot_download(local_files_only=True)` 成功解析相同 revision。
- 两份发布权重在现有容器内由 PyTorch 用 CPU 加载成功，每份包含 160 个模型 state_dict 项和 160 个 EMA 项；配套参数及 SHA-256 已核对。
- 三组正式评测数值取自具有明确完成标记的原始 stdout；完整 stderr、环境测试及历史失败记录保留。

可再次执行的检查：

```bash
python3 reproduction/scripts/hpc/test_runtime.py
python3 reproduction/scripts/hpc/prepare_dependencies.py --check
python3 reproduction/scripts/hpc/download_release.py --check
bash -n reproduction/scripts/hpc/dip_eval_repro_300k.slurm
bash -n reproduction/scripts/hpc/dip_eval_repro_600k.slurm
bash reproduction/scripts/hpc/dip_eval_recipe_300k.slurm --check
```

限制：未从零重建容器/Conda 环境，未在 GPU 上重新验证迁移后的训练和评测执行。参数与路径检查、已有模型的 CPU 加载，不等同于新一次完整复现实验。Release 的远端附件与下载校验状态以发布完成记录为准。

## 2026-09-27 目录调整与 DIMP_FINAL smoke

- 删除 5115268 的失败训练输出；原始调度日志保留。
- `runs`、`scripts` 移至 `reproduction`，迁移时核对文件/链接的 inode 和大小；记录见 `experiments/layout-update-20260927.json`。
- 更新路径推导、提交入口、环境设置、下载恢复路径、Git 忽略规则及文档；原始实验参数和日志不改写。
- 正式脚本改为 DIMP_FINAL；新增短时分区的完整流程 smoke，初版复用原训练器的 integration-test 退出机制（现已由下述统一停止入口替代）。
- 本次没有提交新作业；GPU 实际结果需以用户提交后的 smoke 日志为准。
- 调整后 12 个旧入口的检查模式、4 个路径/续训回归测试、Shell/Python 语法及权重校验均通过。
- 在现有容器内解析两份独立 Slurm 的完整参数，均为 DIMP_FINAL；CPU 虚拟输入经过实际评测构造函数并到达采样边界，没有原来的目标配置断言。
- 用替身计算调用实际训练循环，确认 step 0、10 按保存→评测→生成顺序执行，并在 step 10 正常返回。该检查没有执行模型采样或 GPU 运算。

## 2026-09-27 指定 checkpoint 完整周期后正常停止

- 再查调度记录及本地日志，只发现 5115268 的 DIMP_FULL 失败；用户确认指的是该任务，同意保留 DIMP_FINAL。
- 两份独立 Slurm 共用 `scripts/hpc/train_until_checkpoint.py`，目标分别为 300000 和 10。删除正式版的文件非空轮询及 kill；smoke 不再使用 DIFFUSION_TRAINING_TEST。
- 包装类调用原训练循环，在目标 step 的保存、评测及全部生成返回后结束循环，原 main 随后关闭训练记录平台。正常提前结束但未完成目标也会报错。原项目 `closd/` 源码未修改。
- 容器内 `python reproduction/scripts/hpc/test_checkpoint_stop.py` 的 5 项 CPU 测试通过：实际训练循环的 smoke 顺序、300k 边界、保存/评测/生成异常传播、未完成目标不得成功、非法停止参数拒绝。模型计算及保存/评测/生成以替身执行，不代表实际 GPU 流程通过。
- 两份 Slurm 的命令参数经过真实 train_args 解析，确认 DIMP_FINAL、评测及生成开启、停止点与保存间隔一致；Shell/Python 语法及 Git 空白检查通过。
- 本次未提交作业；完整流程仍需 GPU smoke 验证。

## 2026-09-27 smoke 5121606 单样本可视化错误

- 用户提交的 5121606 运行 1 分 25 秒，FAILED / 1:0。step 0 训练、checkpoint 保存和 HumanML 评测完成；动作结果已写入 results.npy，随后在视频绘制标题处失败。
- stderr 为 `plot_script.py:146` 的 `IndexError: invalid index to scalar variable.`。`generate.py` 的 heading 数组原为 `[batch, 1, frames]`，无参数 squeeze 在 batch=1 时删除样本维；随后 heading_all[sample_i] 变成标量，而绘图仍按帧索引。
- 将 smoke 的 gen_num_samples 从 1 改为 2；正式版保持 6，DIMP_FINAL 和训练/评测设置保持不变，原项目源码不改动。失败日志和输出保留。
- 容器内 CPU 检查使用该作业 results.npy 的前三帧，实际调用原 heading 恢复和绘图函数，重现单样本相同异常；复制为两个样本后，逐帧绘制及原 save_multiple_samples 的 MP4 编码成功。此检查不包含重新生成动作或完整 GPU 训练。
- 修改后的 Shell 语法及 Git 空白检查通过；未代用户提交新作业。

## 2026-09-29 新训练 300k 评测入口

- 用户提交的 smoke 5121612 日志含完整 step 10 周期结束及 SMOKE PASSED；训练 5121622 日志含完整 step 300000 周期结束及 TARGET 300k RUN FINISHED。
- 新增 `dip_eval_recipe_300k` 注册项及 Slurm，模型指向 `reproduction/runs/DiP_multi-target_official_recipe_seed10/model000300000.pt`，保留相邻原始 args.json。
- 与 dip_eval_official、dip_eval_repro_300k 逐项对比：Python 评测参数完全一致，Slurm 资源一致，仅被测模型及作业名称不同。新 Slurm 明确仓库根目录和日志路径，可从 Zhengying 直接 sbatch。
- 新入口检查模式、Shell 语法及 Git 空白检查通过；核对 checkpoint 存在、相邻参数为 DIMP_FINAL / use_ema。此检查没有运行 GPU 仿真评测，未代用户提交作业。

## 2026-09-29 恢复完整旧 Slurm

- 按用户要求，从提交 `bfe977738be9156f26653b3548623bc535ee5aed` 的 `reproduction/history/scripts/*.slurm` 恢复 12 份旧脚本到当前 `reproduction/scripts/hpc`。逐项对比原文，差异仅为日志、产物和专属缓存路径适配；旧训练 smoke 输出目录增加作业号后缀，保留原来的清空本次输出目录逻辑，但不触碰已有历史 smoke 目录。
- 原始训练及评测参数、资源申请、容器启动和脚本内部环境设置恢复；每个旧文件直接写出完整 Python 命令，不调用 run_job.py。之前整理工具暂留，但文档与工作笔记改为直接 sbatch。
- 三份新建文件（300k 正式训练、训练 smoke、新 300k 评测）保留当前已修复的完整脚本。正式及 smoke 训练继续调用获授权的正常停止入口。
- 全部 15 份 Slurm 通过 bash -n，确认没有 run_job.py 调用；新评测检查模式通过。4 项已有辅助工具 CPU 测试通过，其中 spool 路径测试已改用支持检查模式的新完整评测文件，不执行旧 GPU 脚本。
- 未提交、取消作业，未改动权重或历史日志；旧完整脚本本轮未重新在 GPU 上运行。

## 2026-10-01 30 分钟评测副本

- `sacct -X` 确认官方 300k 评测 5100756 完成于 6 分 32 秒，旧复现 300k 评测 5114966 完成于 11 分 34 秒；各自 stdout 都有完整评测结束标记。当前新评测 5133827 为 PENDING、4 小时时限。
- 从新训练 300k 的完整评测 Slurm 复制 30 分钟副本。逐字对比确认只修改 `--time` 和作业名称；Shell 语法及 `--check` 通过。未提交新作业、未取消已有作业。
