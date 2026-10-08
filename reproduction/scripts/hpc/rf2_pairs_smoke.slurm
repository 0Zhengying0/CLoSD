#!/bin/bash
#SBATCH --job-name=rf2_pairs_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

mkdir -p "$CLOSD_ROOT/logs"
mkdir -p "$CLOSD_ROOT/reproduction/reflow_pairs"

apptainer exec \
    --nv \
    --cleanenv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        CLOSD_ROOT=$ROOT/CLoSD

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        cd "$CLOSD_ROOT"

        python -m closd.diffusion_planner.train.generate_reflow_pairs \
            --teacher_model "$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_DIMP_FINAL_seed10_300k/model000300000.pt" \
            --output "$CLOSD_ROOT/reproduction/reflow_pairs/rf1_300k_smoke128.pt" \
            --num_pairs 128 \
            --batch_size 64 \
            --teacher_steps 4 \
            --teacher_solver euler \
            --seed 10
    '
