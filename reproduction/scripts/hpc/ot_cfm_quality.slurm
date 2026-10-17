#!/bin/bash
#SBATCH --job-name=ot_cfm_quality
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
RUN_ROOT=${OT_RUN_ROOT:?}
MODEL=$CLOSD_ROOT/reproduction/runs/OTCFM_DIMP_FINAL_seed10_300k/model000300000.pt
CASES=$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/conditions/manifest.json
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/ot_cfm/runs/"* ]] || exit 2
[[ -s "$MODEL" && -s "$CASES" && ! -e "$RUN_ROOT/quality" ]] || exit 2
apptainer exec --nv --cleanenv --bind "$ROOT:$ROOT" \
    "$ROOT/containers/closd_cuda121_ubuntu20_devel.sif" bash -lc '
        set -euo pipefail
        ROOT=$1 CLOSD_ROOT=$2 RUN_ROOT=$3 MODEL=$4 CASES=$5
        export PATH=$ROOT/envs/closd/bin:$PATH
        export PYTHONPATH=$CLOSD_ROOT/closd:$CLOSD_ROOT:$ROOT/isaacgym/python
        export HF_HOME=$ROOT/cache/huggingface
        export TORCH_HOME=$ROOT/cache/torch
        export XDG_CACHE_HOME=$ROOT/cache/xdg
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
        cd "$CLOSD_ROOT"
        python -B reproduction/scripts/hpc/ot_cfm_quality.py \
            --checkpoint "$MODEL" --cases "$CASES" --output "$RUN_ROOT/quality"
    ' ot-quality "$ROOT" "$CLOSD_ROOT" "$RUN_ROOT" "$MODEL" "$CASES"
