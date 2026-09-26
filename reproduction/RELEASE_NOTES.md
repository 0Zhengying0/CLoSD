# DiP multi-target seed-10 reproduction: 300k / 600k

本次发布 DiP multi-target 从零训练的两个评测 checkpoint，以及完整训练/评测材料。运动控制器使用官方 CLoSD_multitask_finetune；没有重新训练整套 CLoSD 控制器。

- Training commit: `4af01718653dbabf1de1dd9a2600aa54c91acc38`
- Training job: `5099223`, seed 10, completed 600000 steps.
- Evaluation jobs: official baseline `5100756`, reproduced 300k `5114966`, reproduced 600k `5114967`.
- Evaluation: 4096 environments, episode length 500, CFG 7.5, 1000 samples per task.
- Assets: `model000300000.pt`, `model000600000.pt`, `args.json`, `SHA256SUMS`.

权重是原始训练文件，包含模型和 EMA；`args.json` 保留原始训练路径。将参数和权重放在同一目录。优化器及其它中间 checkpoint 只在本地保存，本 Release 用于评测，不是完整续训备份。

复现材料见 `reproduction/README.md`，单次结果和日志来源见 `reproduction/experiments/results.md`。下载并验证：

```bash
python3 scripts/hpc/download_release.py
```

整理后的代码未重跑完整训练或正式评测；验证范围见 `reproduction/VALIDATION.md`。
