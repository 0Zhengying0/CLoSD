#!/bin/bash
#SBATCH --job-name=rf2_pairs
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

SEED=${SEED:?SEED not set}

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

TEACHER=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k/model000300000.pt
OUTPUT=$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_heun10_seed${SEED}.pt

mkdir -p "$CLOSD_ROOT/logs"
mkdir -p "$CLOSD_ROOT/reproduction/reflow_pairs"

echo "===== Formal RF-1 -> RF-2 pair generation ====="
echo "seed    = $SEED"
echo "teacher = $TEACHER"
echo "output  = $OUTPUT"
echo "solver  = Heun"
echo "steps   = 10"
echo "NFE     = 20"
echo "==============================================="

apptainer exec \
    --nv \
    --cleanenv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc "
        set -euo pipefail

        ROOT=$ROOT
        CLOSD_ROOT=\$ROOT/CLoSD

        export PATH=\$ROOT/envs/closd/bin:\$PATH
        export PYTHONPATH=\$CLOSD_ROOT/closd:\$CLOSD_ROOT

        export HF_HOME=\$ROOT/cache/huggingface
        export TORCH_HOME=\$ROOT/cache/torch
        export XDG_CACHE_HOME=\$ROOT/cache/xdg
        export TORCH_EXTENSIONS_DIR=\$ROOT/cache/torch_extensions
        export PYTHONUNBUFFERED=1

        cd \$CLOSD_ROOT

        python -m closd.diffusion_planner.train.generate_reflow_pairs \
            --teacher_model '$TEACHER' \
            --output '$OUTPUT' \
            --num_pairs -1 \
            --batch_size 64 \
            --teacher_steps 10 \
            --teacher_solver heun \
            --seed '$SEED'
    "
