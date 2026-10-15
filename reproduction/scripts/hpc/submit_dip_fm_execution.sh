#!/usr/bin/env bash
# Submit separate, real-controller execution recordings for official Blender visualization.
set -euo pipefail
CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=$(dirname "$CLOSD_ROOT")
BASE=$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs
SLURM=$CLOSD_ROOT/reproduction/scripts/hpc/dip_fm_quality_execution.slurm
PYTHON=$ROOT/envs/closd/bin/python
MODE=${1:-}
[[ "$MODE" == --check || "$MODE" == --submit ]] || {
    echo "Usage: $0 --check|--submit --run-tag TAG" >&2; exit 2;
}
[[ "${2:-}" == --run-tag && -n "${3:-}" && $# -eq 3 ]] || {
    echo "Usage: $0 --check|--submit --run-tag TAG" >&2; exit 2;
}
TAG=$3
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || { echo "Invalid tag" >&2; exit 2; }
RUN_ROOT=$BASE/$TAG
[[ -d "$RUN_ROOT" && -s "$RUN_ROOT/conditions/manifest.json" && -x "$PYTHON" ]] || {
    echo "Run root, captured conditions or runtime missing" >&2; exit 2;
}
command -v sbatch >/dev/null || { echo "sbatch unavailable" >&2; exit 2; }
bash -n "$SLURM"
"$PYTHON" -B -c 'import sys; sys.path.insert(0, sys.argv[1]); from run_dip_fm_quality import validate_models; validate_models(check_ema=True); print("CHECKPOINTS=PASS")' \
    "$CLOSD_ROOT/reproduction/scripts/hpc"
for model in dip fm; do
    QUALITY_RUN_ROOT=$RUN_ROOT QUALITY_MODEL_KIND=$model bash "$SLURM" --check
    [[ ! -e "$RUN_ROOT/execution/$model" ]] || {
        echo "Existing execution result: $RUN_ROOT/execution/$model" >&2; exit 2;
    }
done
for model in dip fm; do
    cmd=(sbatch --parsable "--chdir=$CLOSD_ROOT"
         "--job-name=quality_exec_${model}_$TAG"
         "--output=$RUN_ROOT/logs/execution_${model}_%j.out"
         "--error=$RUN_ROOT/logs/execution_${model}_%j.err"
         "--export=ALL,QUALITY_RUN_ROOT=$RUN_ROOT,QUALITY_MODEL_KIND=$model" "$SLURM")
    printf '%s command:' "$model"; printf ' %q' "${cmd[@]}"; printf '\n'
    if [[ "$MODE" == --submit ]]; then
        if response=$("${cmd[@]}" 2>&1); then
            job_id=${response%%;*}
            [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Job ID: $response" >&2; exit 1; }
            if [[ ! -e "$RUN_ROOT/execution_jobs.tsv" ]]; then
                printf 'model\tjob_id\tjob_name\n' > "$RUN_ROOT/execution_jobs.tsv"
            fi
            printf '%s\t%s\t%s\n' "$model" "$job_id" "quality_exec_${model}_$TAG" >> "$RUN_ROOT/execution_jobs.tsv"
            echo "SUBMITTED_MODEL=$model SLURM_JOB_ID=$job_id"
        else
            echo "SUBMISSION_FAILED_MODEL=$model RESPONSE=$response" >&2
            exit 1
        fi
    fi
done
[[ "$MODE" == --submit ]] || echo "QUALITY_EXECUTION_PRECHECK=PASS; no GPU job submitted"
