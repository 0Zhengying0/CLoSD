# 实际环境与恢复说明

原始训练日志记录 Python 3.8.19、PyTorch 2.2.2+cu121、CUDA build 12.1、NVIDIA A100-SXM4-40GB。依赖清单在整理时从现有容器环境导出；它是当前安装快照，不冒充训练当天逐包锁定的历史记录。

- `pip-freeze.txt`：原始 `pip freeze`，包括本机 Isaac Gym editable 路径。
- `requirements-runtime.txt`：从 freeze 去掉本机 editable 路径、加上 PyTorch cu121 索引，Isaac Gym 单独安装。
- `conda-packages.json`、`conda-history.txt`：已安装 Conda 包、构建版本和安装历史。
- `closd_requirements_base.txt`、`pip_before_closd.txt`：原先保留的安装过程材料，并非最终环境。
- `containers.json`：两份本地容器的 SHA-256、大小、Apptainer 元数据；训练使用 CUDA 12.1 devel 版本。
- `official-resources.json`：官方 CLoSD 与 DistilBERT 的实际缓存 revision。

容器标签表明正式环境来自 `docker://nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04`，基础镜像 digest 记录在 `containers.json`。原始构建配方和额外系统包安装全过程没有完整存档，因此**新机器从零重建容器尚未验证**；不能声称仅拉取该镜像即可完全复现现有环境。建议先使用已校验的本地 SIF。

现有 `../envs/closd` 保留安装前缀；不要直接移动其目录。新建环境时在 Ubuntu 20.04 容器内创建 Python 3.8.19 环境，安装 `requirements-runtime.txt`，再从已获取的 Isaac Gym 安装 `pip install -e <isaacgym>/python`。该重建流程尚未在全新环境执行；若遇到依赖约束冲突，以安装快照及原始安装记录排查，不默默更换版本。

## 官方资源固定与恢复

官方资源来源为 https://huggingface.co/guytevet/CLoSD ，版本 `de7106b947b6f70700b5320d1cd61fef4a9ebc9b`；DistilBERT 版本在 JSON 中。官方数据/模型不重复发布，沿用上游下载入口和使用条款。

当前专属 HF 缓存在 `.local/cache/huggingface`。DistilBERT 缓存通过相对软链接复用仓库外通用缓存。运行入口默认离线，避免官方 `main` 更新影响复现；新机器须先准备缓存。

在已有 Python 环境的容器 shell 内，从仓库根目录执行：

```bash
source scripts/hpc/closd_env.sh
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/hpc/prepare_dependencies.py
python scripts/hpc/prepare_dependencies.py --check
```

准备脚本下载记录的 revision、固定本地 `refs/main`，并建立官方检查点和评测资源链接。它拒绝把已有官方路径悄悄替换为别的模型。新机器恢复需要联网；已有缓存检查无需联网。

原训练是在官方 checkpoint 之外独立训练 DiP；推理控制器仍为官方 `CLoSD_multitask_finetune`。所有结果比较均使用该控制器。
