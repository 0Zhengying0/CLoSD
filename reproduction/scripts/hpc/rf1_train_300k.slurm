#!/bin/bash
#SBATCH --job-name=rf1_300k
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-23:30:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k

mkdir -p "$CLOSD_ROOT/logs"
mkdir -p "$SAVE_DIR"

echo "===== RF-1 formal training ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Save dir: $SAVE_DIR"
echo "================================"

apptainer exec \
    --nv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        CLOSD_ROOT=$ROOT/CLoSD
        SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:${PYTHONPATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export PYTHONUNBUFFERED=1
        unset DIFFUSION_TRAINING_TEST

        export WANDB_MODE=offline
        export WANDB_DIR=$CLOSD_ROOT/reproduction/runs/wandb
        export WANDB_CACHE_DIR=$CLOSD_ROOT/.local/cache/wandb
        export WANDB_CONFIG_DIR=$CLOSD_ROOT/.local/cache/wandb_config
        mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

        cd "$CLOSD_ROOT"

        python -m closd.diffusion_planner.train.train_rectified_flow \
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
            --context_len 20 \
            --pred_len 40 \
            --mask_frames \
            --lambda_target_loc 1.0 \
            --multi_target_cond \
            --multi_encoder_type single \
            --target_enc_layers 1 \
            --target_joint_names DIMP_FINAL \
            --use_ema \
            --avg_model_beta 0.9999 \
            --autoregressive \
            --lr 0.0001 \
            --weight_decay 0.0 \
            --adam_beta2 0.999 \
            --num_steps 300000 \
            --save_interval 10000 \
            --log_interval 1000 \
            --train_platform_type WandBPlatform

        echo "===== RF-1 training finished ====="
    '
