#!/usr/bin/env bash
#SBATCH --job-name=dip_eval_recipe30m
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

# Complete CLoSD task evaluation recipe; paths may be overridden in the environment.
export CLOSD_ROOT="${CLOSD_ROOT:-/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD}"
export CLOSD_SHARED_ROOT="${CLOSD_SHARED_ROOT:-$(dirname "$CLOSD_ROOT")}"
export CLOSD_ENV="${CLOSD_ENV:-$CLOSD_SHARED_ROOT/envs/closd}"
export CLOSD_ISAACGYM="${CLOSD_ISAACGYM:-$CLOSD_SHARED_ROOT/isaacgym}"
export CLOSD_CONTAINER="${CLOSD_CONTAINER:-$CLOSD_SHARED_ROOT/containers/closd_cuda121_ubuntu20_devel.sif}"
export CLOSD_CACHE="${CLOSD_CACHE:-$CLOSD_ROOT/.local/cache}"
export CLOSD_LOGS="${CLOSD_LOGS:-$CLOSD_ROOT/reproduction/logs}"
MODEL="$CLOSD_ROOT/reproduction/runs/DiP_multi-target_official_recipe_seed10/model000300000.pt"
CHECK_ONLY=0
while (( $# )); do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --model)
            if (( $# < 2 )); then echo "--model requires a checkpoint path" >&2; exit 2; fi
            MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1 (supported: --check, --model PATH)" >&2; exit 2 ;;
    esac
done

if [[ ! -s "$MODEL" || ! -s "$(dirname "$MODEL")/args.json" ]]; then
    echo "Missing or empty model / adjacent args.json: $MODEL" >&2
    exit 2
fi
MODEL="$(realpath "$MODEL")"
for resource in "$CLOSD_ROOT/closd/run.py" "$CLOSD_ENV/bin/python" "$CLOSD_ISAACGYM/python/isaacgym" "$CLOSD_CONTAINER"; do
    if [[ ! -e "$resource" ]]; then echo "Missing resource: $resource" >&2; exit 2; fi
done

# Official pretrained controller; replace only the DiP checkpoint.
# These settings match the earlier official and reproduction evaluations.
EVAL_COMMAND=(
    python closd/run.py
    learning=im_big
    env=closd_multitask
    robot=smpl_humanoid
    exp_name=CLoSD_multitask_finetune
    epoch=-1
    env.episode_length=500
    env.dip.cfg_param=7.5
    "env.dip.model_path=$MODEL"
    env.num_envs=4096
    test=True
    no_virtual_display=True
    headless=True
    closd_eval=True
    no_log=True
)

echo "=== dip_eval_recipe30m ==="
echo "Repository: $CLOSD_ROOT"
echo "Container:  $CLOSD_CONTAINER"
echo "Conda env:  $CLOSD_ENV"
echo "Isaac Gym:  $CLOSD_ISAACGYM"
echo "Model:      $MODEL"
printf 'Evaluation command: '; printf '%q ' "${EVAL_COMMAND[@]}"; printf '\n'
if (( CHECK_ONLY )); then exit 0; fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this file with sbatch; use --check on the login node." >&2
    exit 2
fi

mkdir -p "$CLOSD_LOGS" "$CLOSD_CACHE"
nvidia-smi

apptainer exec --nv \
    --bind "$CLOSD_SHARED_ROOT:$CLOSD_SHARED_ROOT" \
    --bind "$CLOSD_ROOT:$CLOSD_ROOT" \
    --bind "$CLOSD_ENV:$CLOSD_ENV" \
    --bind "$CLOSD_ISAACGYM:$CLOSD_ISAACGYM" \
    --bind "$CLOSD_CACHE:$CLOSD_CACHE" \
    --bind "$(dirname "$MODEL"):$(dirname "$MODEL")" \
    "$CLOSD_CONTAINER" bash -c '
        set -euo pipefail
        export PATH="$CLOSD_ENV/bin:$PATH"
        export PYTHONPATH="$CLOSD_ROOT/closd:$CLOSD_ROOT:${PYTHONPATH:-}"
        export ISAACGYM_BINDINGS="$CLOSD_ISAACGYM/python/isaacgym/_bindings/linux-x86_64"
        export LD_LIBRARY_PATH="$ISAACGYM_BINDINGS:$CLOSD_ENV/lib:${LD_LIBRARY_PATH:-}"
        export HF_HOME="$CLOSD_CACHE/huggingface"
        export HF_HUB_CACHE="$HF_HOME/hub"
        export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        export TORCH_HOME="$CLOSD_SHARED_ROOT/cache/torch"
        export PIP_CACHE_DIR="$CLOSD_SHARED_ROOT/cache/pip"
        export APPTAINER_CACHEDIR="$CLOSD_SHARED_ROOT/cache/apptainer"
        export XDG_CACHE_HOME="$CLOSD_CACHE/xdg"
        export TORCH_EXTENSIONS_DIR="$CLOSD_CACHE/torch_extensions"
        export TMPDIR="/tmp/closd_${SLURM_JOB_ID}"
        export APPTAINER_TMPDIR="$TMPDIR"
        export WANDB_MODE=offline
        export WANDB_DIR="$CLOSD_ROOT/reproduction/runs/wandb"
        export WANDB_CACHE_DIR="$CLOSD_CACHE/wandb"
        export WANDB_CONFIG_DIR="$CLOSD_CACHE/wandb_config"
        export MAX_JOBS="${MAX_JOBS:-4}"
        export PYTHONUNBUFFERED=1
        unset DIFFUSION_TRAINING_TEST
        mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" \
            "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
        cd "$CLOSD_ROOT"
        echo "=== ENVIRONMENT ==="
        python --version
        git rev-parse HEAD
        git status --short
        python -c "import torch; print(\"PyTorch\", torch.__version__, \"CUDA\", torch.version.cuda); print(\"GPU\", torch.cuda.get_device_name(0))"
        echo "=== CLoSD TASK EVALUATION START ==="
        exec "$@"
    ' closd-eval "${EVAL_COMMAND[@]}"

echo "=== EVALUATION FINISHED ==="
