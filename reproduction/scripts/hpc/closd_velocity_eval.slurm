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
DIAG_LABEL=${DIAG_LABEL:-}
DIAG_RUN_ROOT=${DIAG_RUN_ROOT:-}
CFG_SWEEP_ID=${CFG_SWEEP_ID:-}
CFG_SWEEP_RUN_ROOT=${CFG_SWEEP_RUN_ROOT:-}
CFG_SCALE=${CFG_SCALE:-7.5}
DOPRI5_ID=${DOPRI5_ID:-}
DOPRI5_RUN_ROOT=${DOPRI5_RUN_ROOT:-}
DOPRI5_RTOL=${DOPRI5_RTOL:-0.001}
DOPRI5_ATOL=${DOPRI5_ATOL:-0.0001}
DOPRI5_MAX_NUM_STEPS=${DOPRI5_MAX_NUM_STEPS:-128}
DOPRI5_MAX_NFE=${DOPRI5_MAX_NFE:-256}
MATCHED_ID=${MATCHED_ID:-}
MATCHED_RUN_ROOT=${MATCHED_RUN_ROOT:-}
MATCHED_NFE=${MATCHED_NFE:-}

case "$GENERATOR" in
    diffusion)
        ODE_STEPS=0
        ODE_SOLVER=diffusion
        ;;
    flow|rectified_flow)
        case "$ODE_SOLVER" in
            euler|heun|rk4|dopri5) ;;
            *) echo "Unsupported ODE_SOLVER: $ODE_SOLVER" >&2; exit 2 ;;
        esac
        if [[ "$ODE_SOLVER" == dopri5 ]]; then
            [[ "$ODE_STEPS" == 0 ]] || { echo "DOPRI5 uses adaptive steps; set ODE_STEPS=0" >&2; exit 2; }
        elif [[ ! "$ODE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
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

if [[ -n "$DOPRI5_ID$DOPRI5_RUN_ROOT" ]]; then
    [[ -z "$EXPERIMENT_ID$SWEEP_RUN_ROOT$DIAG_LABEL$DIAG_RUN_ROOT$CFG_SWEEP_ID$CFG_SWEEP_RUN_ROOT" ]] || {
        echo "DOPRI5 mode cannot be combined with another evaluation mode" >&2; exit 2;
    }
    [[ "$DOPRI5_ID" =~ ^FM-DOPRI5-(12|20|40)$ && "$DOPRI5_RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/fm_dopri5/runs/"* ]] || {
        echo "Invalid DOPRI5 experiment ID or isolated run root" >&2; exit 2;
    }
    [[ "$GENERATOR" == flow && "$ODE_SOLVER" == dopri5 && "$ODE_STEPS" == 0 && "$CFG_SCALE" == 7.5 ]] || {
        echo "DOPRI5 formal evaluation requires FM, adaptive steps, CFG 7.5" >&2; exit 2;
    }
    "$ROOT/envs/closd/bin/python" - "$DOPRI5_RTOL" "$DOPRI5_ATOL" "$DOPRI5_MAX_NUM_STEPS" "$DOPRI5_MAX_NFE" <<'PY'
import math
import sys
rtol, atol = map(float, sys.argv[1:3])
steps, nfe = map(int, sys.argv[3:5])
if not all(math.isfinite(x) and x > 0 for x in (rtol, atol)) or steps < 1 or nfe < 1:
    raise SystemExit("Invalid DOPRI5 tolerances or limits")
PY
    "$ROOT/envs/closd/bin/python" -c 'import torchdiffeq; assert torchdiffeq.__version__ == "0.2.5"'
fi

if [[ -n "$MATCHED_ID$MATCHED_RUN_ROOT$MATCHED_NFE" ]]; then
    [[ -z "$EXPERIMENT_ID$SWEEP_RUN_ROOT$DIAG_LABEL$DIAG_RUN_ROOT$CFG_SWEEP_ID$CFG_SWEEP_RUN_ROOT$DOPRI5_ID$DOPRI5_RUN_ROOT" ]] || {
        echo "Matched NFE mode cannot be combined with another mode" >&2; exit 2;
    }
    [[ "$MATCHED_RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/fm_dopri5/runs/"* ]] || {
        echo "Invalid matched run root" >&2; exit 2;
    }
    [[ "$MATCHED_NFE" =~ ^[1-9][0-9]*$ ]] && (( MATCHED_NFE % 4 == 0 )) || {
        echo "MATCHED_NFE must be a positive multiple of four" >&2; exit 2;
    }
    [[ "$GENERATOR" == flow && "$CFG_SCALE" == 7.5 ]] || {
        echo "Matched comparison requires FM and CFG 7.5" >&2; exit 2;
    }
    case "$MATCHED_ID" in
        FM-ME) expected_solver=euler; factor=1 ;;
        FM-MH) expected_solver=heun; factor=2 ;;
        FM-MR) expected_solver=rk4; factor=4 ;;
        FM-MD) expected_solver=dopri5; factor=0 ;;
        *) echo "Unknown MATCHED_ID: $MATCHED_ID" >&2; exit 2 ;;
    esac
    if [[ "$ODE_SOLVER" != "$expected_solver" ]]; then
        echo "$MATCHED_ID requires $expected_solver" >&2; exit 2;
    fi
    if [[ "$factor" == 0 ]]; then
        [[ "$ODE_STEPS" == 0 ]] || { echo "Adaptive DOPRI5 requires ODE_STEPS=0" >&2; exit 2; }
    else
        (( ODE_STEPS * factor == MATCHED_NFE )) || {
            echo "$MATCHED_ID requires NFE=$MATCHED_NFE" >&2; exit 2;
        }
    fi
fi

if [[ -n "$DIAG_LABEL" || -n "$DIAG_RUN_ROOT" ]]; then
    [[ -z "$EXPERIMENT_ID" && -z "$SWEEP_RUN_ROOT" ]] || {
        echo "Sweep and episode diagnostics modes cannot be combined" >&2; exit 2;
    }
    [[ -n "$DIAG_LABEL" && "$DIAG_RUN_ROOT" == /* ]] || {
        echo "DIAG_LABEL and absolute DIAG_RUN_ROOT are required together" >&2; exit 2;
    }
    case "$DIAG_LABEL" in
        FM-RK4-3)
            [[ "$GENERATOR" == flow && "$ODE_SOLVER" == rk4 && "$ODE_STEPS" == 3 ]] || {
                echo "FM-RK4-3 requires flow, rk4, three steps" >&2; exit 2;
            }
            ;;
        DIP-ORIGINAL)
            [[ "$GENERATOR" == diffusion ]] || {
                echo "DIP-ORIGINAL requires the original diffusion sampler" >&2; exit 2;
            }
            ;;
        *) echo "Unsupported DIAG_LABEL: $DIAG_LABEL" >&2; exit 2 ;;
    esac
fi

if [[ -n "$CFG_SWEEP_ID" || -n "$CFG_SWEEP_RUN_ROOT" ]]; then
    [[ -z "$EXPERIMENT_ID$SWEEP_RUN_ROOT$DIAG_LABEL$DIAG_RUN_ROOT" ]] || {
        echo "CFG sweep cannot be combined with other modes" >&2; exit 2;
    }
    [[ -n "$CFG_SWEEP_ID" && "$CFG_SWEEP_RUN_ROOT" == /* ]] || {
        echo "CFG_SWEEP_ID and absolute CFG_SWEEP_RUN_ROOT are required together" >&2; exit 2;
    }
    [[ "$GENERATOR" == flow && "$ODE_SOLVER" == rk4 && "$ODE_STEPS" == 3 ]] || {
        echo "CFG sweep requires flow, RK4, three steps" >&2; exit 2;
    }
    case "$CFG_SWEEP_ID" in
        FM-CFG1) expected_cfg=1 ;; FM-CFG2p5) expected_cfg=2.5 ;;
        FM-CFG5) expected_cfg=5 ;; FM-CFG7p5) expected_cfg=7.5 ;;
        FM-CFG10) expected_cfg=10 ;; FM-CFG15) expected_cfg=15 ;;
        *) echo "Unsupported CFG sweep experiment: $CFG_SWEEP_ID" >&2; exit 2 ;;
    esac
    [[ "$CFG_SCALE" == "$expected_cfg" ]] || {
        echo "$CFG_SWEEP_ID requires CFG_SCALE=$expected_cfg" >&2; exit 2;
    }
fi

if [[ ! -s "$MODEL" || ! -s "$(dirname "$MODEL")/args.json" ]]; then
    echo "Missing or empty checkpoint / adjacent args.json: $MODEL" >&2
    exit 2
fi
# Preserve the checkpoint symlink name: args.json is adjacent to that name,
# whereas a Hugging Face blob resolved by realpath has no adjacent args.json.
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
for resource in "$CLOSD_ROOT/closd/run.py" "$ROOT/envs/closd/bin/python" \
                "$ROOT/isaacgym/python/isaacgym" "$CONTAINER"; do
    if [[ ! -e "$resource" ]]; then
        echo "Missing resource: $resource" >&2
        exit 2
    fi
done
if [[ -n "$EXPERIMENT_ID$DIAG_LABEL$CFG_SWEEP_ID$DOPRI5_ID$MATCHED_ID" && ! -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" ]]; then
    echo "Missing pretrained CLoSD controller checkpoint" >&2
    exit 2
fi

echo "===== Formal CLoSD evaluation ====="
echo "Generator: $GENERATOR"
echo "Model:     $MODEL"
echo "ODE steps: $ODE_STEPS"
echo "Solver:    $ODE_SOLVER"
echo "CFG:       $CFG_SCALE"
if [[ "$GENERATOR" != diffusion ]]; then
    case "$ODE_SOLVER" in
        euler) nfe_per_step=1 ;;
        heun) nfe_per_step=2 ;;
        rk4) nfe_per_step=4 ;;
        dopri5) nfe_per_step=adaptive ;;
    esac
    if [[ "$ODE_SOLVER" == dopri5 ]]; then
        echo "NFE:       adaptive (measured for every planner call)"
        echo "Tolerance: rtol=$DOPRI5_RTOL atol=$DOPRI5_ATOL max_num_steps=$DOPRI5_MAX_NUM_STEPS max_nfe=$DOPRI5_MAX_NFE"
    else
        echo "NFE:       $((ODE_STEPS * nfe_per_step))"
    fi
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
elif [[ -n "$DIAG_LABEL" ]]; then
    trap 'status=$?; echo "DIAG_JOB_EXIT=$status"' EXIT
elif [[ -n "$CFG_SWEEP_ID" ]]; then
    trap 'status=$?; echo "CFG_JOB_EXIT=$status"' EXIT
elif [[ -n "$DOPRI5_ID" ]]; then
    trap 'status=$?; echo "DOPRI5_JOB_EXIT=$status"' EXIT
elif [[ -n "$MATCHED_ID" ]]; then
    trap 'status=$?; echo "MATCHED_JOB_EXIT=$status"' EXIT
fi

SWEEP_WORK_DIR=
ISOLATED_ID=${EXPERIMENT_ID:-${DIAG_LABEL:-${CFG_SWEEP_ID:-${DOPRI5_ID:-$MATCHED_ID}}}}
ISOLATED_RUN_ROOT=${SWEEP_RUN_ROOT:-${DIAG_RUN_ROOT:-${CFG_SWEEP_RUN_ROOT:-${DOPRI5_RUN_ROOT:-$MATCHED_RUN_ROOT}}}}
if [[ -n "$ISOLATED_ID" ]]; then
    SWEEP_WORK_DIR="$ISOLATED_RUN_ROOT/work/${ISOLATED_ID}_${SLURM_JOB_ID}"
    mkdir -p "$ISOLATED_RUN_ROOT/work"
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
fi
if [[ -n "$EXPERIMENT_ID" ]]; then
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
elif [[ -n "$DIAG_LABEL" ]]; then
    echo "DIAG_LABEL=$DIAG_LABEL"
    echo "DIAG_JOB_ID=$SLURM_JOB_ID"
    echo "DIAG_CHECKPOINT=$MODEL"
    echo "DIAG_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
    echo "DIAG_SOURCE_SHA256=$(sha256sum \
        "$CLOSD_ROOT/closd/learning/episode_diagnostics.py" \
        "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
        "$CLOSD_ROOT/closd/env/tasks/humanoid.py" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm" | sha256sum | cut -d ' ' -f1)"
    echo "DIAG_CONFIG=seed:0,num_envs:4096,episode_length:500,cfg:7.5,samples_per_task:1000"
elif [[ -n "$CFG_SWEEP_ID" ]]; then
    echo "CFG_EXPERIMENT_ID=$CFG_SWEEP_ID"
    echo "CFG_JOB_ID=$SLURM_JOB_ID"
    echo "CFG_CHECKPOINT=$MODEL"
    echo "CFG_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
    echo "CFG_GIT_DIRTY=$(if [[ -n "$(git -C "$CLOSD_ROOT" status --porcelain --untracked-files=all -- . ':(exclude)reproduction/experiments/fm_cfg_sweep/runs')" ]]; then echo 1; else echo 0; fi)"
    echo "CFG_SOURCE_SHA256=$(sha256sum \
        "$CLOSD_ROOT/closd/env/tasks/closd.py" \
        "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
        "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" \
        "$CLOSD_ROOT/closd/diffusion_planner/utils/sampler_util.py" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/submit_fm_cfg_sweep.sh" \
        "$CLOSD_ROOT/reproduction/scripts/hpc/summarize_fm_cfg_sweep.py" | sha256sum | cut -d ' ' -f1)"
    echo "CFG_SCALE=$CFG_SCALE"
    echo "CFG_GUIDANCE_TYPE=text"
    echo "CFG_SEED=0"
    echo "CFG_CONFIG=learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,solver:rk4,steps:3,nfe:12,cfg:$CFG_SCALE,guidance:text,samples_per_task:1000,torch_deterministic:False"
elif [[ -n "$DOPRI5_ID" ]]; then
    echo "DOPRI5_EXPERIMENT_ID=$DOPRI5_ID"
    echo "DOPRI5_JOB_ID=$SLURM_JOB_ID"
    echo "DOPRI5_CHECKPOINT=$MODEL"
    echo "DOPRI5_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
    echo "DOPRI5_SOURCE_SHA256=$(sha256sum "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" "$CLOSD_ROOT/closd/env/tasks/closd.py" "$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm" | sha256sum | cut -d ' ' -f1)"
    echo "DOPRI5_SEED=0"
    echo "DOPRI5_TARGET_NFE=${DOPRI5_ID##*-}"
    echo "DOPRI5_CONFIG=learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False,rtol:$DOPRI5_RTOL,atol:$DOPRI5_ATOL,max_num_steps:$DOPRI5_MAX_NUM_STEPS,max_nfe:$DOPRI5_MAX_NFE"
elif [[ -n "$MATCHED_ID" ]]; then
    echo "MATCHED_ID=$MATCHED_ID"
    echo "MATCHED_TARGET_NFE=$MATCHED_NFE"
    echo "MATCHED_JOB_ID=$SLURM_JOB_ID"
    echo "MATCHED_CHECKPOINT=$MODEL"
    echo "MATCHED_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
    echo "MATCHED_SOURCE_SHA256=$(sha256sum "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" "$CLOSD_ROOT/closd/env/tasks/closd.py" "$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm" | sha256sum | cut -d ' ' -f1)"
    echo "MATCHED_SEED=0"
    echo "MATCHED_CONFIG=learning:im_big,env:closd_multitask,robot:smpl_humanoid,controller:CLoSD_multitask_finetune,episode_length:500,num_envs:4096,cfg:7.5,samples_per_task:1000,torch_deterministic:False,solver:$ODE_SOLVER,steps:$ODE_STEPS,target_nfe:$MATCHED_NFE,rtol:$DOPRI5_RTOL,atol:$DOPRI5_ATOL"
fi

container_binds=(--bind "$ROOT:$ROOT")
if [[ -n "$ISOLATED_ID" ]]; then
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
        DIAG_LABEL=$7
        DIAG_RUN_ROOT=$8
        DIAG_JOB_ID=$9
        CFG_SWEEP_ID=${10}
        CFG_SCALE=${11}
        DOPRI5_ID=${12}
        DOPRI5_RTOL=${13}
        DOPRI5_ATOL=${14}
        DOPRI5_MAX_NUM_STEPS=${15}
        DOPRI5_MAX_NFE=${16}
        MATCHED_ID=${17}
        if [[ -n "$EXPERIMENT_ID" ]]; then
            trap '\''status=$?; echo "SWEEP_PROCESS_EXIT=$status"'\'' EXIT
        elif [[ -n "$DIAG_LABEL" ]]; then
            trap '\''status=$?; echo "DIAG_PROCESS_EXIT=$status"'\'' EXIT
        elif [[ -n "$CFG_SWEEP_ID" ]]; then
            trap '\''status=$?; echo "CFG_PROCESS_EXIT=$status"'\'' EXIT
        elif [[ -n "$DOPRI5_ID" ]]; then
            trap '\''status=$?; echo "DOPRI5_PROCESS_EXIT=$status"'\'' EXIT
        elif [[ -n "$MATCHED_ID" ]]; then
            trap '\''status=$?; echo "MATCHED_PROCESS_EXIT=$status"'\'' EXIT
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
        export CLOSD_ODE_RTOL=$DOPRI5_RTOL
        export CLOSD_ODE_ATOL=$DOPRI5_ATOL
        export CLOSD_ODE_MAX_NUM_STEPS=$DOPRI5_MAX_NUM_STEPS
        export CLOSD_ODE_MAX_NFE=$DOPRI5_MAX_NFE
        export CLOSD_PROFILE_PLANNER=1
        export PYTHONUNBUFFERED=1
        export WANDB_MODE=offline
        unset DIFFUSION_TRAINING_TEST
        if [[ -n "$EXPERIMENT_ID$DIAG_LABEL$CFG_SWEEP_ID$DOPRI5_ID$MATCHED_ID" ]]; then
            if [[ -n "$EXPERIMENT_ID$CFG_SWEEP_ID$DOPRI5_ID$MATCHED_ID" ]]; then
                export CLOSD_SOLVER_SWEEP=1
                if [[ -n "$CFG_SWEEP_ID" ]]; then
                    export CLOSD_CFG_SWEEP=1
                fi
            else
                export CLOSD_EPISODE_DIAG_DIR=$DIAG_RUN_ROOT/diagnostics
                export CLOSD_DIAG_JOB_ID=$DIAG_JOB_ID
            fi
            # multiprocessing.Manager() creates an AF_UNIX socket below TMPDIR.
            # The full sweep work path exceeds the 107-byte AF_UNIX path limit.
            # A unique, node-local /tmp directory keeps the socket path short.
            export TMPDIR=$(mktemp -d /tmp/cfm.XXXXXXXX)
            echo "JOB_TMPDIR=$TMPDIR"
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
        if [[ -n "$EXPERIMENT_ID$DIAG_LABEL$CFG_SWEEP_ID$DOPRI5_ID$MATCHED_ID" ]]; then
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
            "env.dip.cfg_param=$CFG_SCALE" \
            "env.dip.model_path=$MODEL" \
            env.num_envs=4096 \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True \
            no_log=True \
            "${runtime_overrides[@]}"
    ' closd-velocity-eval "$GENERATOR" "$ODE_STEPS" "$ODE_SOLVER" "$MODEL" "$EXPERIMENT_ID" "$SWEEP_WORK_DIR" "$DIAG_LABEL" "$DIAG_RUN_ROOT" "$SLURM_JOB_ID" "$CFG_SWEEP_ID" "$CFG_SCALE" "$DOPRI5_ID" "$DOPRI5_RTOL" "$DOPRI5_ATOL" "$DOPRI5_MAX_NUM_STEPS" "$DOPRI5_MAX_NFE" "$MATCHED_ID"
