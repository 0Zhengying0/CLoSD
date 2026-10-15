#!/usr/bin/env bash
#SBATCH --job-name=quality_execution
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
RUN_ROOT=${QUALITY_RUN_ROOT:?QUALITY_RUN_ROOT required}
MODEL_KIND=${QUALITY_MODEL_KIND:?QUALITY_MODEL_KIND required}
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/"* ]] || {
    echo "Invalid quality run root" >&2; exit 2;
}
case "$MODEL_KIND" in
    dip)
        MODEL=$CLOSD_ROOT/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt
        GENERATOR=diffusion; ODE_SOLVER=euler; ODE_STEPS=10 ;;
    fm)
        MODEL=$CLOSD_ROOT/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt
        GENERATOR=flow; ODE_SOLVER=rk4; ODE_STEPS=3 ;;
    *) echo "QUALITY_MODEL_KIND must be dip or fm" >&2; exit 2 ;;
esac
[[ -s "$MODEL" && -s "$(dirname "$MODEL")/args.json" && -s "$CONTAINER" &&
   -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" &&
   -x "$ROOT/envs/closd/bin/python" && -d "$ROOT/isaacgym/python/isaacgym" ]] || {
    echo "Missing model, args, controller, container or runtime" >&2; exit 2;
}
OUT="$RUN_ROOT/execution/$MODEL_KIND"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
if [[ "${1:-}" == --check ]]; then
    sbatch --test-only "$0"
    echo "QUALITY_EXECUTION_SCHEDULER_CHECK=PASS; no job submitted"
    exit 0
fi
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch or --check" >&2; exit 2; }
trap 'status=$?; echo "QUALITY_EXECUTION_EXIT=$status MODEL=$MODEL_KIND"' EXIT
WORK="$RUN_ROOT/work/execution_${MODEL_KIND}_$SLURM_JOB_ID"
mkdir -p "$WORK/controller" "$WORK/hydra" "$WORK/cache/torch_extensions" \
    "$WORK/cache/torch" "$WORK/cache/xdg" "$WORK/cache/wandb" "$WORK/cache/isaacgym/vhacd"
ln -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" "$WORK/controller/Humanoid.pth"
if [[ -d /home/it4i-dianyu/.isaacgym/vhacd ]]; then
    shopt -s nullglob
    old_cache=(/home/it4i-dianyu/.isaacgym/vhacd/*.vhacd)
    shopt -u nullglob
    if (( ${#old_cache[@]} )); then cp -n "${old_cache[@]}" "$WORK/cache/isaacgym/vhacd/"; fi
fi
echo "QUALITY_EXECUTION_JOB_ID=$SLURM_JOB_ID MODEL=$MODEL_KIND CHECKPOINT=$MODEL GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
apptainer exec --nv --cleanenv --bind "$ROOT:$ROOT" \
    --bind "$WORK/cache/isaacgym:/home/it4i-dianyu/.isaacgym" \
    "$CONTAINER" bash -lc '
        set -euo pipefail
        ROOT=$1; CLOSD_ROOT=$2; WORK=$3; RUN_ROOT=$4; MODEL=$5
        GENERATOR=$6; ODE_SOLVER=$7; ODE_STEPS=$8; MODEL_KIND=$9; export SLURM_JOB_ID=${10}
        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:$ROOT/isaacgym/python
        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$WORK/cache/torch
        export XDG_CACHE_HOME=$WORK/cache/xdg
        export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
        export WANDB_DIR=$WORK/cache/wandb WANDB_CACHE_DIR=$WORK/cache/wandb WANDB_CONFIG_DIR=$WORK/cache/wandb
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
        export CLOSD_GENERATOR_TYPE=$GENERATOR CLOSD_ODE_SOLVER=$ODE_SOLVER CLOSD_ODE_STEPS=$ODE_STEPS
        export CLOSD_QUALITY_EXEC_RECORD_DIR=$RUN_ROOT/execution/$MODEL_KIND
        export CLOSD_QUALITY_EXEC_RECORD_STEPS=500
        export TMPDIR=$(mktemp -d /tmp/cqe.XXXXXXXX)
        cd "$CLOSD_ROOT"
        python closd/run.py learning=im_big env=closd_multitask robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune epoch=-1 seed=0 torch_deterministic=False \
            env.episode_length=500 env.num_envs=128 env.dip.cfg_param=7.5 \
            "env.dip.model_path=$MODEL" test=True no_virtual_display=True headless=True \
            closd_eval=True no_log=True "output_path=$WORK/controller" \
            "hydra.run.dir=$WORK/hydra" hydra.job.chdir=False
    ' quality-execution "$ROOT" "$CLOSD_ROOT" "$WORK" "$RUN_ROOT" "$MODEL" \
    "$GENERATOR" "$ODE_SOLVER" "$ODE_STEPS" "$MODEL_KIND" "$SLURM_JOB_ID"
[[ -s "$OUT/episodes.pkl" && -s "$OUT/metadata.json" ]] || {
    echo "Execution recording missing for $MODEL_KIND" >&2; exit 1;
}
