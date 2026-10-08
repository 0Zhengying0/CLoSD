#!/bin/bash
#SBATCH --job-name=closd_rf_smoke
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

MODEL=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_smoke/model000000001.pt

mkdir -p "$CLOSD_ROOT/logs"

echo "===== CLoSD + RF ODE smoke ====="
echo "Host: $(hostname)"
echo "Model: $MODEL"
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
        MODEL=$CLOSD_ROOT/reproduction/runs/RectifiedFlow1_smoke/model000000001.pt

        export PATH=$ROOT/envs/closd/bin:$PATH

        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:$ROOT/isaacgym/python

        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        # New ODE planner settings
        export CLOSD_GENERATOR_TYPE=rectified_flow
        export CLOSD_ODE_STEPS=4
        export CLOSD_ODE_SOLVER=euler

        cd "$CLOSD_ROOT"

        echo "===== Runtime settings ====="
        echo "generator = $CLOSD_GENERATOR_TYPE"
        echo "ODE steps = $CLOSD_ODE_STEPS"
        echo "ODE solver = $CLOSD_ODE_SOLVER"
        echo "model = $MODEL"
        echo "============================"

        # We only need to prove that CLoSD can enter the environment
        # and repeatedly call the ODE planner.
        #
        # timeout=5 min prevents a smoke test from running the full
        # 1000-per-task evaluation.

        set +e

        timeout 300s python closd/run.py \
            learning=im_big \
            env=closd_multitask \
            robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            env.episode_length=500 \
            env.dip.cfg_param=7.5 \
            env.dip.model_path="$MODEL" \
            env.num_envs=32 \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True

        RC=$?

        set -e

        if [ "$RC" -eq 124 ]; then
            echo "===== Smoke test reached 5-minute timeout after successful launch ====="
            exit 0
        fi

        exit "$RC"
    '
