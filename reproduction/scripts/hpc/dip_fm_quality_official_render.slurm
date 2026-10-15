#!/usr/bin/env bash
#SBATCH --job-name=quality_official_render
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
RUN_ROOT=${QUALITY_RUN_ROOT:?QUALITY_RUN_ROOT required}
KIND=${QUALITY_RENDER_KIND:?QUALITY_RENDER_KIND required}
MODE=${QUALITY_RENDER_MODE:-formal}
PYTHON=$ROOT/envs/closd/bin/python
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/"* && -x "$PYTHON" ]] || {
    echo "Invalid run root or Python runtime" >&2; exit 2;
}
case "$KIND" in
    skeleton)
        SOURCE=${QUALITY_SOURCE_DIR:?QUALITY_SOURCE_DIR required for skeleton}
        [[ "$SOURCE" == "$RUN_ROOT/"* && -s "$SOURCE/provenance.json" &&
           -s "$RUN_ROOT/conditions/manifest.json" ]] || {
            echo "Completed planner trajectory source required" >&2; exit 2;
        } ;;
    execution)
        [[ -s "$RUN_ROOT/execution/dip/episodes.pkl" && -s "$RUN_ROOT/execution/fm/episodes.pkl" ]] || {
            echo "Both real simulator records required" >&2; exit 2;
        } ;;
    *) echo "QUALITY_RENDER_KIND must be skeleton or execution" >&2; exit 2 ;;
esac
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo "Invalid render mode" >&2; exit 2; }
if [[ "${1:-}" == --check ]]; then
    # Verify the actual Slurm account/partition/GRES combination without
    # submitting a job; the previous file-only check missed this failure.
    sbatch --test-only "$0"
    echo "QUALITY_RENDER_SCHEDULER_CHECK=PASS; no job submitted"
    exit 0
fi
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch or --check" >&2; exit 2; }
OUT=$RUN_ROOT/official_${KIND}_${MODE}_$SLURM_JOB_ID
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
export MPLBACKEND=Agg MPLCONFIGDIR=$RUN_ROOT/work/mpl_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"
cd "$CLOSD_ROOT"
if [[ "$KIND" == skeleton ]]; then
    if [[ "$MODE" == smoke ]]; then
        "$PYTHON" reproduction/scripts/hpc/render_quality_official_skeleton.py \
            --cases "$RUN_ROOT/conditions/manifest.json" --source "$SOURCE" \
            --output "$OUT" --frames 3
    else
        "$PYTHON" reproduction/scripts/hpc/render_quality_official_skeleton.py \
            --cases "$RUN_ROOT/conditions/manifest.json" --source "$SOURCE" \
            --output "$OUT"
    fi
else
    module load Blender/4.0.2-GCCcore-12.2.0
    if [[ "$MODE" == smoke ]]; then
        "$PYTHON" reproduction/scripts/hpc/render_quality_execution.py \
            --run-dir "$RUN_ROOT" --output "$OUT" --tasks REACH --preview-frames 3 \
            --blender "$(command -v blender)"
    else
        "$PYTHON" reproduction/scripts/hpc/render_quality_execution.py \
            --run-dir "$RUN_ROOT" --output "$OUT" \
            --blender "$(command -v blender)"
    fi
fi
[[ -s "$OUT/index.csv" ]] || { echo "Renderer ended without index.csv" >&2; exit 1; }
echo "QUALITY_OFFICIAL_RENDER_COMPLETE=1 KIND=$KIND MODE=$MODE OUTPUT=$OUT"
