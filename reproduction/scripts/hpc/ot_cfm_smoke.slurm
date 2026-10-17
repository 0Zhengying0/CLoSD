#!/bin/bash
#SBATCH --job-name=ot_cfm_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
SAVE_DIR=$CLOSD_ROOT/reproduction/runs/OTCFM_smoke_${SLURM_JOB_ID:?submit with sbatch}

echo "===== OT-CFM smoke test ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CLoSD root: $CLOSD_ROOT"
echo "Save dir: $SAVE_DIR"
echo "===================================="

apptainer exec \
    --nv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        CLOSD_ROOT=$ROOT/CLoSD
        SAVE_DIR=$CLOSD_ROOT/reproduction/runs/OTCFM_smoke_${SLURM_JOB_ID}

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:${PYTHONPATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export DIFFUSION_TRAINING_TEST=1
        export PYTHONUNBUFFERED=1

        cd "$CLOSD_ROOT"

        python -B reproduction/scripts/hpc/test_ot_cfm.py

        echo "===== Container check ====="
        echo "Python: $(which python)"
        python --version

        python - << "PY"
import torch
from closd.diffusion_planner.diffusion.ot_cfm import OptimalTransportConditionalFlowMatching

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

fm = OptimalTransportConditionalFlowMatching(
    num_timesteps=1000,
    lambda_target_loc=1.0,
)

print("OptimalTransportConditionalFlowMatching import OK")
print("num_timesteps =", fm.num_timesteps)
print("lambda_target_loc =", fm.lambda_target_loc)
PY

        echo "===== Start training smoke test ====="

        python -m closd.diffusion_planner.train.train_ot_cfm \
            --save_dir "$SAVE_DIR" \
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
            --num_steps 10 \
            --save_interval 10 \
            --log_interval 1 \
            --train_platform_type NoPlatform

        CHECKPOINT=$(find "$SAVE_DIR" -maxdepth 1 -name "model*.pt" | sort | tail -1)
        test -s "$CHECKPOINT"
        python -B reproduction/scripts/hpc/ot_cfm_sample_check.py \
            --checkpoint "$CHECKPOINT" \
            --cases "$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/conditions/manifest.json" \
            --output "$SAVE_DIR/sample_check"
        echo "OT_CFM_SMOKE_PASS checkpoint=$CHECKPOINT"
    '
