#!/bin/bash
#SBATCH --job-name=rf2_300k
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

RF1_MODEL=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k/model000300000.pt

PAIR_GLOB="$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_heun10_seed*.pt"

SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow2_DIMP_FINAL_300k

mkdir -p "$CLOSD_ROOT/logs"
mkdir -p "$SAVE_DIR"

echo "===== RF-2 formal training ====="
echo "RF-1 init  = $RF1_MODEL"
echo "Pairs      = $PAIR_GLOB"
echo "Save dir   = $SAVE_DIR"
echo "Steps      = 300000"
echo "================================"

apptainer exec \
    --nv \
    --cleanenv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        CLOSD_ROOT=$ROOT/CLoSD

        RF1_MODEL=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k/model000300000.pt
        PAIR_GLOB="$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_heun10_seed*.pt"
        SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow2_DIMP_FINAL_300k

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions
        export PYTHONUNBUFFERED=1

        cd "$CLOSD_ROOT"

        python -m closd.diffusion_planner.train.train_rf2 \
            --reflow_pairs "$PAIR_GLOB" \
            --rf1_init_model "$RF1_MODEL" \
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
            --train_platform_type NoPlatform
    '
