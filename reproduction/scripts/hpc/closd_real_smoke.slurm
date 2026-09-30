#!/bin/bash
#SBATCH --job-name=closd_real_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

apptainer exec \
    --nv \
    --bind $ROOT:$ROOT \
    $CONTAINER \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/CLoSD/.local/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/CLoSD/.local/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip

        export TMPDIR=$ROOT/tmp
        export TORCH_EXTENSIONS_DIR=$ROOT/CLoSD/.local/cache/torch_extensions

        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}

        export MAX_JOBS=4

        cd $ROOT/CLoSD

        echo "=== REAL CLoSD SMOKE START ==="

        python closd/run.py \
            learning=im_big \
            robot=smpl_humanoid \
            env=closd_multitask \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            test=True \
            no_virtual_display=True \
            headless=True \
            no_log=True \
            env.num_envs=4 \
            env.episode_length=100 \
            learning.params.config.player.games_num=1

        echo "=== REAL CLoSD SMOKE PASSED ==="
    '
