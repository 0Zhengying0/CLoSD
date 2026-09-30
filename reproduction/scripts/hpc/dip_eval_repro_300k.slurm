#!/bin/bash
#SBATCH --job-name=dip_eval_repro300
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
MODEL=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_repro_seed10/model000300000.pt

mkdir -p "$ROOT/CLoSD/reproduction/logs"

echo "=================================================="
echo "Official DiP multi-target 300k - FORMAL EVALUATION"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-unknown}"
echo "Model:      $MODEL"
echo "num_envs:   4096"
echo "episode:    500"
echo "CFG:        7.5"
echo "Start:      $(date)"
echo "=================================================="

test -f "$MODEL"

nvidia-smi

apptainer exec \
    --nv \
    --bind "$ROOT:$ROOT" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        MODEL=$ROOT/CLoSD/reproduction/runs/DiP_multi-target_repro_seed10/model000300000.pt

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/CLoSD/.local/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/CLoSD/.local/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/CLoSD/.local/cache/torch_extensions

        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}

        export TMPDIR=/tmp
        export PYTHONUNBUFFERED=1

        cd $ROOT/CLoSD

        echo "=== GPU ==="
        python - << "PY"
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

        echo "=== FORMAL EVALUATION START ==="

        python closd/run.py \
            learning=im_big \
            env=closd_multitask \
            robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            env.episode_length=500 \
            env.dip.cfg_param=7.5 \
            env.dip.model_path="$MODEL" \
            env.num_envs=4096 \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True \
            no_log=True

        echo "=== FORMAL EVALUATION FINISHED ==="
        date
    '
