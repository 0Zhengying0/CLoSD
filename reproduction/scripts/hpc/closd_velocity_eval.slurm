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
EXPERIMENT_ID=${EXPERIMENT_ID:-}
SWEEP_RUN_ROOT=${SWEEP_RUN_ROOT:-}

case "$GENERATOR" in
    diffusion)
        ODE_STEPS=0
        ODE_SOLVER=diffusion
        ;;
    flow|rectified_flow)
        case "$ODE_SOLVER" in
            euler|heun|rk4) ;;
            *) echo "Unsupported ODE_SOLVER: $ODE_SOLVER" >&2; exit 2 ;;
        esac
        if [[ ! "$ODE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
            echo "ODE_STEPS must be a positive integer: $ODE_STEPS" >&2
            exit 2
        fi
        ;;
    *) echo "Unsupported GENERATOR: $GENERATOR" >&2; exit 2 ;;
esac

if [[ -n "$EXPERIMENT_ID" || -n "$SWEEP_RUN_ROOT" ]]; then
    [[ -n "$EXPERIMENT_ID" && -n "$SWEEP_RUN_ROOT" ]] || {
        echo "EXPERIMENT_ID and SWEEP_RUN_ROOT must be supplied together" >&2; exit 2;
    }
    [[ "$SWEEP_RUN_ROOT" == /* ]] || {
        echo "SWEEP_RUN_ROOT must be an absolute path" >&2; exit 2;
    }
    [[ "$GENERATOR" == flow ]] || {
        echo "Solver sweep requires GENERATOR=flow" >&2; exit 2;
    }
    case "$EXPERIMENT_ID" in
        FM-E12|FM-E20|FM-E40) expected_solver=euler; nfe_per_step=1 ;;
        FM-H12|FM-H20|FM-H40) expected_solver=heun; nfe_per_step=2 ;;
        FM-R12|FM-R20|FM-R40) expected_solver=rk4; nfe_per_step=4 ;;
        *) echo "Unknown solver sweep experiment: $EXPERIMENT_ID" >&2; exit 2 ;;
    esac
    expected_nfe=${EXPERIMENT_ID: -2}
    if [[ "$ODE_SOLVER" != "$expected_solver" || $((ODE_STEPS * nfe_per_step)) -ne $expected_nfe ]]; then
        echo "Experiment $EXPERIMENT_ID requires $expected_solver and NFE=$expected_nfe" >&2
        exit 2
    fi
fi

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
if [[ -n "$EXPERIMENT_ID" && ! -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" ]]; then
    echo "Missing pretrained CLoSD controller checkpoint" >&2
    exit 2
fi

echo "===== Formal CLoSD evaluation ====="
echo "Generator: $GENERATOR"
echo "Model:     $MODEL"
echo "ODE steps: $ODE_STEPS"
echo "Solver:    $ODE_SOLVER"
if [[ "$GENERATOR" != diffusion ]]; then
    case "$ODE_SOLVER" in
        euler) nfe_per_step=1 ;;
        heun) nfe_per_step=2 ;;
        rk4) nfe_per_step=4 ;;
    esac
    echo "NFE:       $((ODE_STEPS * nfe_per_step))"
fi
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
if [[ -n "$EXPERIMENT_ID" ]]; then
    trap 'status=$?; echo "SWEEP_JOB_EXIT=$status"' EXIT
fi

SWEEP_WORK_DIR=
if [[ -n "$EXPERIMENT_ID" ]]; then
    SWEEP_WORK_DIR="$SWEEP_RUN_ROOT/work/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
    mkdir "$SWEEP_WORK_DIR"
    mkdir -p "$SWEEP_WORK_DIR/controller" "$SWEEP_WORK_DIR/hydra" \
        "$SWEEP_WORK_DIR/tmp" "$SWEEP_WORK_DIR/cache/torch_extensions" \
        "$SWEEP_WORK_DIR/cache/torch" "$SWEEP_WORK_DIR/cache/xdg" \
        "$SWEEP_WORK_DIR/cache/wandb" "$SWEEP_WORK_DIR/cache/isaacgym/vhacd"
    ln -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" \
        "$SWEEP_WORK_DIR/controller/Humanoid.pth"
    # Isaac Gym otherwise writes VHACD meshes under the shared user home.
    # Seed an isolated cache from the existing, small read-only cache.
    if [[ -d /home/it4i-dianyu/.isaacgym/vhacd ]]; then
        shopt -s nullglob
        existing_vhacd=(/home/it4i-dianyu/.isaacgym/vhacd/*.vhacd)
        shopt -u nullglob
        if (( ${#existing_vhacd[@]} )); then
            cp -n "${existing_vhacd[@]}" "$SWEEP_WORK_DIR/cache/isaacgym/vhacd/"
        fi
    fi
    echo "SWEEP_EXPERIMENT_ID=$EXPERIMENT_ID"
    echo "SWEEP_JOB_ID=$SLURM_JOB_ID"
    echo "SWEEP_CHECKPOINT=$MODEL"
    echo "SWEEP_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$CLOSD_ROOT" status --porcelain --untracked-files=all -- . ':(exclude)reproduction/experiments/fm_solver_sweep/runs')" ]]; then
        echo "SWEEP_GIT_DIRTY=1"
    else
        echo "SWEEP_GIT_DIRTY=0"
    fi
    echo "SWEEP_SOURCE_SHA256=$(sha256sum \
        "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" \
        "$CLOSD_ROOT/closd/env/tasks/closd.py" \
        "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/submit_fm_solver_sweep.sh" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/summarize_fm_solver_sweep.py" | sha256sum | cut -d ' ' -f1)"
    echo "SWEEP_SEED=0"
    echo "SWEEP_CONFIG=learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False"
fi

container_binds=(--bind "$ROOT:$ROOT")
if [[ -n "$EXPERIMENT_ID" ]]; then
    container_binds+=(--bind "$SWEEP_WORK_DIR/cache/isaacgym:/home/it4i-dianyu/.isaacgym")
fi
apptainer exec --nv --cleanenv \
    "${container_binds[@]}" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail
        GENERATOR=$1
        ODE_STEPS=$2
        ODE_SOLVER=$3
        MODEL=$4
        EXPERIMENT_ID=$5
        SWEEP_WORK_DIR=$6
        if [[ -n "$EXPERIMENT_ID" ]]; then
            trap '\''status=$?; echo "SWEEP_PROCESS_EXIT=$status"'\'' EXIT
        fi
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
        if [[ -n "$EXPERIMENT_ID" ]]; then
            export CLOSD_SOLVER_SWEEP=1
            # multiprocessing.Manager() creates an AF_UNIX socket below TMPDIR.
            # The full sweep work path exceeds the 107-byte AF_UNIX path limit.
            # A unique, node-local /tmp directory keeps the socket path short.
            export TMPDIR=$(mktemp -d /tmp/cfm.XXXXXXXX)
            echo "SWEEP_TMPDIR=$TMPDIR"
            python -c "import os, socket; p=os.path.join(os.environ[\"TMPDIR\"], \"socket_test\"); s=socket.socket(socket.AF_UNIX); s.bind(p); s.close(); os.unlink(p)"
            export TORCH_EXTENSIONS_DIR=$SWEEP_WORK_DIR/cache/torch_extensions
            export TORCH_HOME=$SWEEP_WORK_DIR/cache/torch
            export XDG_CACHE_HOME=$SWEEP_WORK_DIR/cache/xdg
            export WANDB_DIR=$SWEEP_WORK_DIR/cache/wandb
            export WANDB_CACHE_DIR=$SWEEP_WORK_DIR/cache/wandb
            export WANDB_CONFIG_DIR=$SWEEP_WORK_DIR/cache/wandb
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
        fi
        mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

        cd "$CLOSD_ROOT"
        runtime_overrides=()
        if [[ -n "$EXPERIMENT_ID" ]]; then
            runtime_overrides+=("output_path=$SWEEP_WORK_DIR/controller")
            runtime_overrides+=("hydra.run.dir=$SWEEP_WORK_DIR/hydra")
            runtime_overrides+=("hydra.job.chdir=False")
        fi
        python closd/run.py \
            learning=im_big \
            env=closd_multitask \
            robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            seed=0 \
            torch_deterministic=False \
            env.episode_length=500 \
            env.dip.cfg_param=7.5 \
            "env.dip.model_path=$MODEL" \
            env.num_envs=4096 \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True \
            no_log=True \
            "${runtime_overrides[@]}"
    ' closd-velocity-eval "$GENERATOR" "$ODE_STEPS" "$ODE_SOLVER" "$MODEL" "$EXPERIMENT_ID" "$SWEEP_WORK_DIR"
