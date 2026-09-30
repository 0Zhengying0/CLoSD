#!/bin/bash
#SBATCH --job-name=dip_envtest
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/reproduction/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
MODEL=$ROOT/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt

NUM_ENVS=${NUM_ENVS:-128}

mkdir -p "$ROOT/CLoSD/reproduction/logs"

echo "=================================================="
echo "CLoSD evaluation env-count test"
echo "Job ID:       ${SLURM_JOB_ID:-unknown}"
echo "Host:         $(hostname)"
echo "NUM_ENVS:     $NUM_ENVS"
echo "episode_len:  500"
echo "Model:        $MODEL"
echo "Start:        $(date)"
echo "=================================================="

test -f "$MODEL"

GPU_LOG=$ROOT/CLoSD/reproduction/logs/gpu_${SLURM_JOB_ID}_${NUM_ENVS}env.csv

# 每两秒记录一次显存占用
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Monitoring GPU=$GPU_ID"

stdbuf -oL nvidia-smi \
    -i "$GPU_ID" \
    --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    -l 1 > "$GPU_LOG" &

MON_PID=$!

cleanup() {
    kill "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

set +e

apptainer exec \
    --nv \
    --bind $ROOT:$ROOT \
    $CONTAINER \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
        MODEL=$ROOT/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/CLoSD/.local/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/CLoSD/.local/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip
        export TORCH_EXTENSIONS_DIR=$ROOT/CLoSD/.local/cache/torch_extensions
        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
        export TMPDIR=${SLURM_TMPDIR:-$ROOT/tmp}
        export PYTHONUNBUFFERED=1

        cd $ROOT/CLoSD

        echo "=== GPU ==="
        python - << "PY"
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Total VRAM GiB:",
      torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

        echo "=== EVAL ENV TEST START ==="
        echo "NUM_ENVS='"$NUM_ENVS"'"

        python closd/run.py \
            learning=im_big \
            env=closd_multitask \
            robot=smpl_humanoid \
            exp_name=CLoSD_multitask_finetune \
            epoch=-1 \
            env.episode_length=500 \
            env.dip.cfg_param=7.5 \
            env.dip.model_path="$MODEL" \
            env.num_envs='"$NUM_ENVS"' \
            test=True \
            no_virtual_display=True \
            headless=True \
            closd_eval=True \
            no_log=True \
            learning.params.config.player.games_num=1
    '

RC=$?

set -e

cleanup
trap - EXIT

echo
echo "=================================================="
echo "Process exit code: $RC"

awk -F',' '
{
    gsub(/ /, "", $2)
    if ($2 + 0 > max) max=$2 + 0
}
END {
    print "Peak GPU memory:", max, "MiB"
}
' "$GPU_LOG"

if [ "$RC" -eq 0 ]; then
    echo "=== EVAL ENV TEST PASSED ==="
else
    echo "=== EVAL ENV TEST FAILED ==="
fi

echo "End: $(date)"
echo "=================================================="

exit "$RC"
