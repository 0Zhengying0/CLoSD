#!/bin/bash
#SBATCH --job-name=closd_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/logs/%x_%j.err

set -euo pipefail

ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif

echo "=== HOST ==="
hostname
date

echo "=== GPU ==="
nvidia-smi

echo "=== CONTAINER TEST ==="

apptainer exec \
    --nv \
    --bind $ROOT:$ROOT \
    $CONTAINER \
    bash -lc '
        set -euo pipefail

        ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying

        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$ROOT/CLoSD/closd:$ROOT/CLoSD:${PYTHONPATH:-}

        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export PIP_CACHE_DIR=$ROOT/cache/pip

        export TMPDIR=$ROOT/tmp
        export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions

        export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
        export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}

        export MAX_JOBS=4

        cd $ROOT/CLoSD

        echo "=== PYTHON / TORCH ==="
        python - << "PY"
import torch

print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available")
PY

        echo "=== NINJA ==="
        command -v ninja
        ninja --version

        echo "=== ISAAC GYM / GYMTORCH ==="
        python - << "PY"
from isaacgym import gymapi
from isaacgym import gymtorch
print("Isaac Gym + gymtorch import OK")
PY

        echo "=== CLoSD IMPORT ==="
        python - << "PY"
import closd.run
print("CLoSD run.py import OK")
PY

        echo "=== SMOKE TEST PASSED ==="
    '
