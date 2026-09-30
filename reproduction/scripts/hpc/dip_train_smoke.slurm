#!/bin/bash
#SBATCH --job-name=dip_train_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
SAVE_DIR=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_smoke_${SLURM_JOB_ID}

# 每次 smoke 都从干净目录开始
rm -rf "$SAVE_DIR"
mkdir -p \
    "$SAVE_DIR" \
    "$ROOT/CLoSD/reproduction/logs" \
    "$ROOT/CLoSD/reproduction/runs/wandb" \
    "$ROOT/CLoSD/.local/cache/wandb" \
    "$ROOT/CLoSD/.local/cache/wandb_config"

echo "=================================================="
echo "DiP TRAINING SMOKE TEST"
echo "Host:     $(hostname)"
echo "Start:    $(date)"
echo "Save dir: $SAVE_DIR"
echo "Job ID:   ${SLURM_JOB_ID:-unknown}"
echo "=================================================="

nvidia-smi

apptainer exec \
    --nv \
    --bind $ROOT:$ROOT \
    $CONTAINER \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        SAVE_DIR=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_smoke_${SLURM_JOB_ID}

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/CLoSD/.local/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/CLoSD/.local/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip

        export TMPDIR=${SLURM_TMPDIR:-$ROOT/tmp}
        export TORCH_EXTENSIONS_DIR=$ROOT/CLoSD/.local/cache/torch_extensions
        export PYTHONUNBUFFERED=1

        export WANDB_MODE=offline
        export WANDB_DIR=$ROOT/CLoSD/reproduction/runs/wandb
        export WANDB_CACHE_DIR=$ROOT/CLoSD/.local/cache/wandb
        export WANDB_CONFIG_DIR=$ROOT/CLoSD/.local/cache/wandb_config

        # CLoSD 训练代码自带的 integration-test 退出机制：
        # 在第一次非零 step 的 checkpoint 保存后退出。
        export DIFFUSION_TRAINING_TEST=1

        cd $ROOT/CLoSD

        echo "=== PYTORCH / GPU ==="
        python - << "PY"
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

        echo "=== TRAINING SMOKE START ==="

        python -m closd.diffusion_planner.train.train_mdm \
            --save_dir "$SAVE_DIR" \
            --overwrite \
            --seed 10 \
            --batch_size 64 \
            --dataset humanml \
            --arch trans_dec \
            --text_encoder_type bert \
            --layers 8 \
            --latent_dim 512 \
            --cond_mask_prob 0.1 \
            --emb_policy concat \
            --diffusion_steps 10 \
            --noise_schedule cosine \
            --context_len 20 \
            --pred_len 40 \
            --mask_frames \
            --lambda_target_loc 1.0 \
            --multi_target_cond \
            --multi_encoder_type single \
            --target_enc_layers 1 \
            --target_joint_names DIMP_FULL \
            --use_ema \
            --avg_model_beta 0.9999 \
            --autoregressive \
            --lr 0.0001 \
            --weight_decay 0.0 \
            --adam_beta2 0.999 \
            --num_steps 20 \
            --save_interval 10 \
            --log_interval 1 \
            --gen_guidance_param 2.5 \
            --train_platform_type WandBPlatform

        echo "=== TRAINING SMOKE PASSED ==="

        echo "=== OUTPUT FILES ==="
        ls -lh "$SAVE_DIR"
    '
