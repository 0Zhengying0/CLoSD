#!/bin/bash
#SBATCH --job-name=rf2_train_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

RF1_MODEL=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k/model000300000.pt
PAIR_FILE=$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_smoke128.pt
SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow2_smoke

mkdir -p "$CLOSD_ROOT/logs"

# Smoke only. Avoid stale checkpoint auto-resume.
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

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
        PAIR_FILE=$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_smoke128.pt
        SAVE_DIR=$CLOSD_ROOT/reproduction/runs/RectifiedFlow2_smoke

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export PYTHONUNBUFFERED=1

        # Makes TrainLoop exit after it has completed and saved
        # the second optimizer step.
        export DIFFUSION_TRAINING_TEST=1

        cd "$CLOSD_ROOT"

        echo "===== RF-2 training smoke ====="
        echo "RF-1 init : $RF1_MODEL"
        echo "Pairs     : $PAIR_FILE"
        echo "Save dir  : $SAVE_DIR"
        echo "==============================="

        python -m closd.diffusion_planner.train.train_rf2 \
            --reflow_pairs "$PAIR_FILE" \
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
            --num_steps 100 \
            --save_interval 1 \
            --log_interval 1 \
            --train_platform_type NoPlatform

        echo "===== RF-2 training smoke finished ====="
    '
