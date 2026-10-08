#!/bin/bash
#SBATCH --job-name=closd_vel_eval
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

# Supply GENERATOR and MODEL with sbatch --export; ODE_STEPS is required for velocity models.
GENERATOR=${GENERATOR:?GENERATOR must be diffusion, flow, or rectified_flow}
MODEL=${MODEL:?MODEL must be an absolute checkpoint path}
ODE_STEPS=${ODE_STEPS:-0}
ODE_SOLVER=${ODE_SOLVER:-euler}

case "$GENERATOR" in
    diffusion)
        ODE_STEPS=0
        ODE_SOLVER=diffusion
        ;;
    flow|rectified_flow)
        case "$ODE_SOLVER" in
            euler|heun) ;;
            *) echo "Unsupported ODE_SOLVER: $ODE_SOLVER" >&2; exit 2 ;;
        esac
        if [[ ! "$ODE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
            echo "ODE_STEPS must be a positive integer: $ODE_STEPS" >&2
            exit 2
        fi
        ;;
    *) echo "Unsupported GENERATOR: $GENERATOR" >&2; exit 2 ;;
esac
if [[ ! -s "$MODEL" || ! -s "$(dirname "$MODEL")/args.json" ]]; then
    echo "Missing or empty checkpoint / adjacent args.json: $MODEL" >&2
    exit 2
fi
MODEL=$(realpath "$MODEL")
for resource in "$CLOSD_ROOT/closd/run.py" "$ROOT/envs/closd/bin/python" \
                "$ROOT/isaacgym/python/isaacgym" "$CONTAINER"; do
    if [[ ! -e "$resource" ]]; then
        echo "Missing resource: $resource" >&2
        exit 2
    fi
done

echo "===== Formal CLoSD evaluation ====="
echo "Generator: $GENERATOR"
echo "Model:     $MODEL"
echo "ODE steps: $ODE_STEPS"
echo "Solver:    $ODE_SOLVER"
echo "Batch:     4096 environments"
echo "============================================"

if [[ "${1:-}" == "--check" ]]; then
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: sbatch --export=ALL,GENERATOR=...,MODEL=...[,ODE_STEPS=...] $0" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch; use --check to validate inputs on the login node." >&2
    exit 2
fi

apptainer exec --nv --cleanenv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail
        GENERATOR=$1
        ODE_STEPS=$2
        ODE_SOLVER=$3
        MODEL=$4
        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        CLOSD_ROOT=$ROOT/CLoSD

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:$ROOT/isaacgym/python
        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions
        export WANDB_DIR=$CLOSD_ROOT/reproduction/runs/wandb
        export WANDB_CACHE_DIR=$CLOSD_ROOT/.local/cache/wandb
        export WANDB_CONFIG_DIR=$CLOSD_ROOT/.local/cache/wandb_config
        export CLOSD_GENERATOR_TYPE=$GENERATOR
        export CLOSD_ODE_STEPS=$ODE_STEPS
        export CLOSD_ODE_SOLVER=$ODE_SOLVER
        export CLOSD_PROFILE_PLANNER=1
        export PYTHONUNBUFFERED=1
        export WANDB_MODE=offline
        unset DIFFUSION_TRAINING_TEST
        mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

        cd "$CLOSD_ROOT"
        exec python closd/run.py \
            learning=im_big \
            env=closd_multitask \
            robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            env.episode_length=500 \
            env.dip.cfg_param=7.5 \
            "env.dip.model_path=$MODEL" \
            env.num_envs=4096 \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True \
            no_log=True
    ' closd-velocity-eval "$GENERATOR" "$ODE_STEPS" "$ODE_SOLVER" "$MODEL"
