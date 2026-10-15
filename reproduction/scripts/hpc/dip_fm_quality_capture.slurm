#!/usr/bin/env bash
#SBATCH --job-name=dip_fm_capture
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
MODEL=$CLOSD_ROOT/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt
RUN_ROOT=${QUALITY_RUN_ROOT:?QUALITY_RUN_ROOT required}
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/"* ]] || {
    echo "Invalid isolated quality run root" >&2; exit 2;
}
[[ -s "$MODEL" && -s "$(dirname "$MODEL")/args.json" && -s "$CONTAINER" &&
   -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" &&
   -x "$ROOT/envs/closd/bin/python" && -d "$ROOT/isaacgym/python/isaacgym" ]] || {
    echo "Missing DiP checkpoint, args.json, controller, container or runtime" >&2; exit 2;
}
[[ ! -e "$RUN_ROOT/conditions" ]] || { echo "Conditions already exist" >&2; exit 2; }
if [[ "${1:-}" == --check ]]; then
    sbatch --test-only "$0"
    echo "QUALITY_CAPTURE_SCHEDULER_CHECK=PASS; no job submitted"
    exit 0
fi
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch or --check" >&2; exit 2; }
trap 'status=$?; echo "QUALITY_CAPTURE_JOB_EXIT=$status"' EXIT
WORK="$RUN_ROOT/work/capture_$SLURM_JOB_ID"
mkdir "$WORK"
mkdir -p "$WORK/controller" "$WORK/hydra" "$WORK/cache/torch_extensions" \
    "$WORK/cache/torch" "$WORK/cache/xdg" "$WORK/cache/wandb" "$WORK/cache/isaacgym/vhacd"
ln -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" "$WORK/controller/Humanoid.pth"
if [[ -d /home/it4i-dianyu/.isaacgym/vhacd ]]; then
    shopt -s nullglob
    old_cache=(/home/it4i-dianyu/.isaacgym/vhacd/*.vhacd)
    shopt -u nullglob
    if (( ${#old_cache[@]} )); then cp -n "${old_cache[@]}" "$WORK/cache/isaacgym/vhacd/"; fi
fi
echo "QUALITY_CAPTURE_JOB_ID=$SLURM_JOB_ID"
echo "QUALITY_CAPTURE_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
echo "QUALITY_CAPTURE_MODEL=$MODEL"
echo "QUALITY_CAPTURE_CONFIG=seed:0,num_envs:128,episode_length:500,cfg:7.5,candidates_per_task:16,selected_per_task:4"
apptainer exec --nv --cleanenv --bind "$ROOT:$ROOT" \
    --bind "$WORK/cache/isaacgym:/home/it4i-dianyu/.isaacgym" \
    "$CONTAINER" bash -lc '
        set -euo pipefail
        ROOT=$1; CLOSD_ROOT=$2; WORK=$3; RUN_ROOT=$4; MODEL=$5; export SLURM_JOB_ID=$6
        trap '\''status=$?; echo "QUALITY_CAPTURE_PROCESS_EXIT=$status"'\'' EXIT
        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:$ROOT/isaacgym/python
        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$WORK/cache/torch
        export XDG_CACHE_HOME=$WORK/cache/xdg
        export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
        export WANDB_DIR=$WORK/cache/wandb
        export WANDB_CACHE_DIR=$WORK/cache/wandb
        export WANDB_CONFIG_DIR=$WORK/cache/wandb
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
        export CLOSD_GENERATOR_TYPE=diffusion
        export CLOSD_QUALITY_CAPTURE_DIR=$RUN_ROOT/conditions
        export CLOSD_QUALITY_CAPTURE_PER_TASK=16 CLOSD_QUALITY_SELECTED_PER_TASK=4
        export CLOSD_QUALITY_CAPTURE_MAX_CALLS=400
        export TMPDIR=$(mktemp -d /tmp/cfq.XXXXXXXX)
        mkdir -p "$WANDB_DIR"
        cd "$CLOSD_ROOT"
        python closd/run.py learning=im_big env=closd_multitask robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune epoch=-1 seed=0 torch_deterministic=False \
            env.episode_length=500 env.num_envs=128 env.dip.cfg_param=7.5 \
            "env.dip.model_path=$MODEL" test=True no_virtual_display=True headless=True \
            closd_eval=True no_log=True "output_path=$WORK/controller" \
            "hydra.run.dir=$WORK/hydra" hydra.job.chdir=False
    ' quality-capture "$ROOT" "$CLOSD_ROOT" "$WORK" "$RUN_ROOT" "$MODEL" "$SLURM_JOB_ID"
[[ -s "$RUN_ROOT/conditions/manifest.json" ]] || {
    echo "Capture stopped without complete condition manifest" >&2; exit 1;
}
