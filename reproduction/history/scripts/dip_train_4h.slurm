#!/bin/bash
#SBATCH --job-name=dip_mt_4h
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
SAVE_DIR=$ROOT/runs/DiP_multi-target_repro_seed10_short

mkdir -p \
    "$ROOT/logs" \
    "$ROOT/runs" \
    "$ROOT/runs/wandb" \
    "$ROOT/tmp"

echo "=================================================="
echo "DiP multi-target training"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Save dir:  $SAVE_DIR"
echo "Job ID:    ${SLURM_JOB_ID:-unknown}"
echo "=================================================="

nvidia-smi

apptainer exec \
    --nv \
    --bind $ROOT:$ROOT \
    $CONTAINER \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        SAVE_DIR=$ROOT/runs/DiP_multi-target_repro_seed10_short

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip

        export TMPDIR=${SLURM_TMPDIR:-$ROOT/tmp}
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export PYTHONUNBUFFERED=1

        # W&B offline logging
        export WANDB_MODE=offline
        export WANDB_DIR=$ROOT/runs/wandb
        export WANDB_CACHE_DIR=$ROOT/cache/wandb
        export WANDB_CONFIG_DIR=$ROOT/cache/wandb_config

        mkdir -p \
            "$WANDB_DIR" \
            "$WANDB_CACHE_DIR" \
            "$WANDB_CONFIG_DIR"

        cd $ROOT/CLoSD

        echo "=== REPRODUCIBILITY INFO ==="

        python --version

        echo "Git commit:"
        git rev-parse HEAD || true

        echo "Local modifications:"
        git status --short || true

        python - << "PY"
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

        echo "=== W&B ==="
        echo "WANDB_MODE=$WANDB_MODE"
        echo "WANDB_DIR=$WANDB_DIR"

        echo "=== TRAINING START ==="

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
            --num_steps 600000 \
            --save_interval 10000 \
            --log_interval 1000 \
            --gen_guidance_param 2.5 \
            --train_platform_type WandBPlatform

        echo "=== TRAINING FINISHED ==="
        date
    '
