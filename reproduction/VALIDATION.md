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
python3 scripts/hpc/test_runtime.py
python3 scripts/hpc/prepare_dependencies.py --check
python3 scripts/hpc/download_release.py --check
bash scripts/hpc/submit.sh dip_eval_repro_300k --check
bash scripts/hpc/submit.sh dip_eval_repro_600k --check
```

限制：未从零重建容器/Conda 环境，未在 GPU 上重新验证迁移后的训练和评测执行。参数与路径检查、已有模型的 CPU 加载，不等同于新一次完整复现实验。Release 的远端附件与下载校验状态以发布完成记录为准。
