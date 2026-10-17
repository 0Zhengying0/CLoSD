#!/usr/bin/env bash
#SBATCH --job-name=fm_dopri5_calibrate
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --mail-user=zy.xu@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.out
#SBATCH --error=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/CLoSD/logs/%x_%j.err

set -euo pipefail
ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying
CLOSD_ROOT=$ROOT/CLoSD
RUN_ROOT=${DOPRI5_RUN_ROOT:?DOPRI5_RUN_ROOT required}
CASES=$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/conditions/manifest.json
CONTAINER=$ROOT/containers/closd_cuda121_ubuntu20_devel.sif
[[ "$RUN_ROOT" == "$CLOSD_ROOT/reproduction/experiments/fm_dopri5/runs/"* && -d "$RUN_ROOT" ]] || {
    echo "Invalid run root" >&2; exit 2;
}
[[ -s "$CASES" && -s "$CONTAINER" ]] || { echo "Missing real cases or container" >&2; exit 2; }
"$ROOT/envs/closd/bin/python" -c 'import torchdiffeq; assert torchdiffeq.__version__ == "0.2.5"'
if [[ "${1:-}" == --check ]]; then
    sbatch --test-only "$0"
    echo "DOPRI5_CALIBRATION_CHECK=PASS; no job submitted"
    exit 0
fi
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch or --check" >&2; exit 2; }
trap 'status=$?; echo "DOPRI5_CALIBRATION_JOB_EXIT=$status"' EXIT
OUTPUT=$RUN_ROOT/calibration_$SLURM_JOB_ID
[[ ! -e "$OUTPUT" ]] || { echo "Calibration output already exists" >&2; exit 2; }
echo "DOPRI5_CALIBRATION_JOB_ID=$SLURM_JOB_ID"
echo "DOPRI5_CALIBRATION_GIT_COMMIT=$(git -C "$CLOSD_ROOT" rev-parse HEAD)"
apptainer exec --nv --cleanenv --bind "$ROOT:$ROOT" "$CONTAINER" bash -lc '
    set -euo pipefail
    ROOT=$1; CLOSD_ROOT=$2; CASES=$3; OUTPUT=$4; MAX_STEPS=$5; MAX_NFE=$6
    trap '\''status=$?; echo "DOPRI5_CALIBRATION_PROCESS_EXIT=$status"'\'' EXIT
    export PATH=$ROOT/envs/closd/bin:$PATH
    export PYTHONPATH=$CLOSD_ROOT:$CLOSD_ROOT/closd:$ROOT/isaacgym/python
    export ISAACGYM_BINDINGS=$ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
    export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}
    export HF_HOME=$ROOT/cache/huggingface
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
    export TMPDIR=$(mktemp -d /tmp/cfdc.XXXXXXXX)
    cd "$CLOSD_ROOT"
    python reproduction/scripts/hpc/calibrate_fm_dopri5.py --cases "$CASES" --output "$OUTPUT" \
        --max-num-steps "$MAX_STEPS" --max-nfe "$MAX_NFE"
    ' fm-dopri5-calibrate "$ROOT" "$CLOSD_ROOT" "$CASES" "$OUTPUT" \
    "${DOPRI5_MAX_NUM_STEPS:-128}" "${DOPRI5_MAX_NFE:-256}"
[[ -s "$OUTPUT/calibration.json" ]] || { echo "Missing calibration result" >&2; exit 1; }
