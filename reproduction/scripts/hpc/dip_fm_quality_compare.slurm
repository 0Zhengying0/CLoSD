#!/usr/bin/env bash
#SBATCH --job-name=dip_fm_quality
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
RUN_ROOT=${QUALITY_RUN_ROOT:?QUALITY_RUN_ROOT required}
MODE=${QUALITY_MODE:?QUALITY_MODE must be smoke or formal}
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/"* ]] || {
    echo "Invalid isolated quality run root" >&2; exit 2;
}
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo "Invalid QUALITY_MODE" >&2; exit 2; }
[[ -s "$RUN_ROOT/conditions/manifest.json" && -s "$CONTAINER" ]] || {
    echo "Missing complete condition manifest or container" >&2; exit 2;
}
if [[ "$MODE" == formal && -e "$RUN_ROOT/results" ]]; then
    echo "Historical formal results path exists" >&2; exit 2;
fi
if [[ "$MODE" == formal ]]; then
    smoke_ready=0
    shopt -s nullglob
    for smoke_dir in "$RUN_ROOT"/smoke_*/; do
        if [[ -s "$smoke_dir/summary.md" && -s "$smoke_dir/results.csv" &&
              -s "$smoke_dir/group_summary.csv" ]]; then
            smoke_ready=1
            break
        fi
    done
    shopt -u nullglob
    (( smoke_ready )) || { echo "Completed GPU smoke output required" >&2; exit 2; }
fi
if [[ "${1:-}" == --check ]]; then
    sbatch --test-only "$0"
    echo "QUALITY_COMPARE_SCHEDULER_CHECK=PASS; no job submitted"
    exit 0
fi
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch or --check" >&2; exit 2; }
trap 'status=$?; echo "QUALITY_COMPARE_JOB_EXIT=$status"' EXIT
WORK="$RUN_ROOT/work/compare_${MODE}_$SLURM_JOB_ID"
mkdir "$WORK"
mkdir -p "$WORK/cache/torch_extensions" "$WORK/cache/torch" "$WORK/cache/xdg" \
    "$WORK/cache/wandb" "$WORK/mpl"
if [[ "$MODE" == smoke ]]; then
    OUTPUT="$RUN_ROOT/smoke_$SLURM_JOB_ID"
else
    OUTPUT="$RUN_ROOT/results"
fi
echo "QUALITY_COMPARE_JOB_ID=$SLURM_JOB_ID"
echo "QUALITY_COMPARE_MODE=$MODE"
echo "QUALITY_COMPARE_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
apptainer exec --nv --cleanenv --bind "$ROOT:$ROOT" "$CONTAINER" bash -lc '
    set -euo pipefail
    ROOT=$1; CLOSD_ROOT=$2; WORK=$3; RUN_ROOT=$4; OUTPUT=$5; MODE=$6; export SLURM_JOB_ID=$7
    trap '\''status=$?; echo "QUALITY_COMPARE_PROCESS_EXIT=$status"'\'' EXIT
    export PATH=$ROOT/envs/closd/bin:$PATH
    export PYTHONPATH=$CLOSD_ROOT:$CLOSD_ROOT/closd:$ROOT/isaacgym/python
    export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
    export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
    export HF_HOME=$ROOT/cache/huggingface
    export TORCH_HOME=$WORK/cache/torch
    export XDG_CACHE_HOME=$WORK/cache/xdg
    export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
    export MPLCONFIGDIR=$WORK/mpl
    export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
    export TMPDIR=$(mktemp -d /tmp/cfq.XXXXXXXX)
    cd "$CLOSD_ROOT"
    if [[ "$MODE" == smoke ]]; then
        python reproduction/scripts/hpc/run_dip_fm_quality.py \
            --cases "$RUN_ROOT/conditions/manifest.json" --output "$OUTPUT" --max-cases 1
    else
        python reproduction/scripts/hpc/run_dip_fm_quality.py \
            --cases "$RUN_ROOT/conditions/manifest.json" --output "$OUTPUT"
    fi
    ' quality-compare "$ROOT" "$CLOSD_ROOT" "$WORK" "$RUN_ROOT" "$OUTPUT" "$MODE" "$SLURM_JOB_ID"
[[ -s "$OUTPUT/summary.md" && -s "$OUTPUT/results.csv" && -s "$OUTPUT/group_summary.csv" ]] || {
    echo "Comparison ended without complete output" >&2; exit 1;
}
