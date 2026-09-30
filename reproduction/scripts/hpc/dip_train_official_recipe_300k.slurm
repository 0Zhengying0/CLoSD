#!/bin/bash
#SBATCH --job-name=dip_exact300
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-08:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

# DIMP_FINAL 适配当前训练中评测；与官方历史 DIMP_FULL checkpoint 配置不同。
# 全新的目录，绝对不要和之前训练混在一起
SAVE_DIR=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_official_recipe_seed10

mkdir -p "$ROOT/CLoSD/reproduction/logs"
mkdir -p "$SAVE_DIR"

# 这个实验必须从 step 0 开始。
# 如果这个目录里已经有 checkpoint，就停止，防止误续训。
if compgen -G "$SAVE_DIR/model*.pt" > /dev/null; then
    echo "ERROR: checkpoint already exists in:"
    echo "$SAVE_DIR"
    echo "This control experiment must start from step 0."
    exit 2
fi

echo "=================================================="
echo "DiP OFFICIAL-RECIPE 300k CONTROL RUN"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-unknown}"
echo "Save dir:   $SAVE_DIR"
echo "Target:     model000300000.pt"
echo "Start:      $(date)"
echo "=================================================="

nvidia-smi

apptainer exec \
    --nv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        SAVE_DIR=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_official_recipe_seed10

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/CLoSD/.local/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/CLoSD/.local/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/CLoSD/.local/cache/torch_extensions
        export APPTAINER_CACHEDIR=$ROOT/cache/apptainer

        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}

        # 使用计算节点本地临时目录，避免之前的 .nfs 清理报错
        export TMPDIR=/tmp/closd_${SLURM_JOB_ID}
        mkdir -p "$TMPDIR"

        export PYTHONUNBUFFERED=1
        unset DIFFUSION_TRAINING_TEST

        # 共享 HPC 账号，因此 W&B 只离线记录
        export WANDB_MODE=offline
        export WANDB_DIR=$ROOT/CLoSD/reproduction/runs/wandb
        export WANDB_CACHE_DIR=$ROOT/CLoSD/.local/cache/wandb
        export WANDB_CONFIG_DIR=$ROOT/CLoSD/.local/cache/wandb_config

        cd $ROOT/CLoSD

        echo "=== REPRODUCIBILITY INFO ==="
        python --version
        echo "Git commit:"
        git rev-parse HEAD
        echo "Local modifications:"
        git status --short

        python - << "PY"
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

        echo "=== OFFICIAL-RECIPE TRAINING START ==="

        python reproduction/scripts/hpc/train_until_checkpoint.py \
            --stop-after-checkpoint 300000 \
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
            --target_joint_names DIMP_FINAL \
            --use_ema \
            --avg_model_beta 0.9999 \
            --autoregressive \
            --lr 0.0001 \
            --weight_decay 0.0 \
            --adam_beta2 0.999 \
            --num_steps 600000 \
            --save_interval 50000 \
            --log_interval 1000 \
            --eval_during_training \
            --eval_num_samples 1000 \
            --eval_rep_times 3 \
            --eval_split test \
            --gen_during_training \
            --gen_guidance_param 2.5 \
            --gen_num_samples 6 \
            --gen_num_repetitions 1 \
            --train_platform_type WandBPlatform

        echo "=== TARGET 300k RUN FINISHED ==="
    '
