#!/bin/bash
#SBATCH --job-name=flow_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
SAVE_DIR=$ROOT/CLoSD/reproduction/runs/FlowMatching_smoke

rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"
mkdir -p "$ROOT/logs"

apptainer exec \
    --nv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export DIFFUSION_TRAINING_TEST=1
        export PYTHONUNBUFFERED=1

        cd $ROOT/CLoSD

        python -m closd.diffusion_planner.train.train_flow \
            --save_dir "$ROOT/CLoSD/reproduction/runs/FlowMatching_smoke" \
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
            --num_steps 600000 \
            --save_interval 1 \
            --log_interval 1 \
            --train_platform_type NoPlatform
    '