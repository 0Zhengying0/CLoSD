#!/usr/bin/env bash
# Staged, isolated Karolina jobs. Only --capture/--smoke/--compare call sbatch.
set -euo pipefail

CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHARED_ROOT=$(dirname "$CLOSD_ROOT")
BASE="$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs"
PYTHON="$SHARED_ROOT/envs/closd/bin/python"
CAPTURE="$CLOSD_ROOT/reproduction/scripts/hpc/dip_fm_quality_capture.slurm"
COMPARE="$CLOSD_ROOT/reproduction/scripts/hpc/dip_fm_quality_compare.slurm"
RUNNER="$CLOSD_ROOT/reproduction/scripts/hpc/run_dip_fm_quality.py"
TEST="$CLOSD_ROOT/reproduction/scripts/hpc/test_dip_fm_quality.py"
usage() { echo "Usage: $0 --check|--capture|--smoke|--compare --run-tag TAG" >&2; exit 2; }
MODE=
TAG=
while (( $# )); do
    case "$1" in
        --check|--capture|--smoke|--compare) [[ -z "$MODE" ]] || usage; MODE=$1; shift ;;
        --run-tag) (( $# >= 2 )) || usage; TAG=$2; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$MODE" && "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || usage
RUN_ROOT="$BASE/$TAG"
[[ -x "$PYTHON" && -f "$CAPTURE" && -f "$COMPARE" && -f "$RUNNER" && -f "$TEST" ]] || {
    echo "Required scripts or Python runtime missing" >&2; exit 2;
}
command -v sbatch >/dev/null || { echo "sbatch unavailable" >&2; exit 2; }
bash -n "$0"
bash -n "$CAPTURE"
bash -n "$COMPARE"
bash -n "$CLOSD_ROOT/reproduction/scripts/hpc/dip_fm_quality_execution.slurm"
bash -n "$CLOSD_ROOT/reproduction/scripts/hpc/submit_dip_fm_execution.sh"
"$PYTHON" -B -c 'import ast,pathlib,sys; [ast.parse(pathlib.Path(x).read_text(), filename=x) for x in sys.argv[1:]]' \
    "$CLOSD_ROOT/closd/learning/quality_snapshot.py" \
    "$CLOSD_ROOT/closd/env/tasks/closd.py" \
    "$CLOSD_ROOT/closd/learning/quality_execution_record.py" \
    "$CLOSD_ROOT/closd/blender/blender_utils.py" \
    "$CLOSD_ROOT/closd/blender/record2anim.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/quality_metrics.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/quality_render.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/prepare_official_sim_record.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/render_quality_official_skeleton.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/render_quality_execution.py" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/summarize_dip_fm_quality.py" \
    "$RUNNER" "$TEST"
if test_output=$("$PYTHON" -B "$TEST" 2>&1); then
    echo "CPU_TESTS=PASS (11 tests; fixtures only)"
else
    printf '%s\n' "$test_output" >&2
    echo "CPU_TESTS=FAIL" >&2
    exit 1
fi
"$PYTHON" -B "$RUNNER" --help >/dev/null
"$PYTHON" -B -c 'import sys; sys.path.insert(0, sys.argv[1]); from run_dip_fm_quality import validate_models; r=validate_models(check_ema=True); print("CHECKPOINTS=PASS DiP_steps={} FM_steps={} separate_args=True ema=True".format(r["dip"]["args"]["diffusion_steps"], r["fm"]["args"]["diffusion_steps"]))' \
    "$CLOSD_ROOT/reproduction/scripts/hpc"

capture_cmd=(sbatch --parsable "--chdir=$CLOSD_ROOT"
    "--job-name=quality_capture_$TAG"
    "--output=$RUN_ROOT/logs/capture_%j.out"
    "--error=$RUN_ROOT/logs/capture_%j.err"
    "--export=ALL,QUALITY_RUN_ROOT=$RUN_ROOT" "$CAPTURE")
smoke_cmd=(sbatch --parsable "--chdir=$CLOSD_ROOT"
    "--job-name=quality_smoke_$TAG"
    "--output=$RUN_ROOT/logs/smoke_%j.out"
    "--error=$RUN_ROOT/logs/smoke_%j.err"
    "--export=ALL,QUALITY_RUN_ROOT=$RUN_ROOT,QUALITY_MODE=smoke" "$COMPARE")
formal_cmd=(sbatch --parsable "--chdir=$CLOSD_ROOT"
    "--job-name=quality_compare_$TAG"
    "--output=$RUN_ROOT/logs/compare_%j.out"
    "--error=$RUN_ROOT/logs/compare_%j.err"
    "--export=ALL,QUALITY_RUN_ROOT=$RUN_ROOT,QUALITY_MODE=formal" "$COMPARE")
echo "RUN_ROOT=$RUN_ROOT"
if [[ "$MODE" == --check ]]; then
    for stage in capture smoke compare; do
        case "$stage" in capture) cmd=("${capture_cmd[@]}") ;;
            smoke) cmd=("${smoke_cmd[@]}") ;;
            compare) cmd=("${formal_cmd[@]}") ;;
        esac
        printf '%s command:' "$stage"
        printf ' %q' "${cmd[@]}"
        printf '\n'
    done
    if [[ -e "$RUN_ROOT/conditions" && ! -s "$RUN_ROOT/conditions/manifest.json" ]]; then
        echo "Incomplete condition capture exists: $RUN_ROOT/conditions" >&2; exit 2;
    elif [[ ! -e "$RUN_ROOT/conditions" ]]; then
        QUALITY_RUN_ROOT="$RUN_ROOT" bash "$CAPTURE" --check
    fi
    if [[ -s "$RUN_ROOT/conditions/manifest.json" ]]; then
        "$PYTHON" -B "$RUNNER" --cases "$RUN_ROOT/conditions/manifest.json" \
            --output "$RUN_ROOT/results" --validate-only
    fi
    echo "QUALITY_PRECHECK=PASS; no GPU job submitted"
    exit 0
fi
if [[ "$MODE" == --capture ]]; then
    [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
        echo "Run root exists; refusing to overwrite historical data: $RUN_ROOT" >&2; exit 2;
    }
    QUALITY_RUN_ROOT="$RUN_ROOT" bash "$CAPTURE" --check
    mkdir -p "$BASE"
    mkdir "$RUN_ROOT"
    mkdir "$RUN_ROOT/logs" "$RUN_ROOT/work"
    cmd=("${capture_cmd[@]}")
else
    [[ -s "$RUN_ROOT/conditions/manifest.json" ]] || {
        echo "Complete captured conditions missing" >&2; exit 2;
    }
    "$PYTHON" -B "$RUNNER" --cases "$RUN_ROOT/conditions/manifest.json" \
        --output "$RUN_ROOT/results" --validate-only
    if [[ "$MODE" == --smoke ]]; then
        QUALITY_RUN_ROOT="$RUN_ROOT" QUALITY_MODE=smoke bash "$COMPARE" --check
        cmd=("${smoke_cmd[@]}")
    else
        [[ ! -e "$RUN_ROOT/results" ]] || {
            echo "Formal results path already exists" >&2; exit 2;
        }
        good_smoke=0
        shopt -s nullglob
        for smoke_dir in "$RUN_ROOT"/smoke_*/; do
            if [[ -s "$smoke_dir/summary.md" && -s "$smoke_dir/results.csv" ]]; then
                good_smoke=1
                break
            fi
        done
        shopt -u nullglob
        (( good_smoke )) || { echo "Completed one-case GPU smoke output required before formal submission" >&2; exit 2; }
        QUALITY_RUN_ROOT="$RUN_ROOT" QUALITY_MODE=formal bash "$COMPARE" --check
        cmd=("${formal_cmd[@]}")
    fi
fi
if response=$("${cmd[@]}" 2>&1); then
    job_id=${response%%;*}
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm Job ID: $response" >&2; exit 1; }
    if [[ ! -e "$RUN_ROOT/jobs.tsv" ]]; then
        printf 'stage\tjob_id\tjob_name\n' > "$RUN_ROOT/jobs.tsv"
    fi
    printf '%s\t%s\t%s\n' "${MODE#--}" "$job_id" "quality_${MODE#--}_$TAG" >> "$RUN_ROOT/jobs.tsv"
    echo "SUBMITTED_STAGE=${MODE#--}"
    echo "SLURM_JOB_ID=$job_id"
    echo "LOG_PATTERN=$RUN_ROOT/logs/${MODE#--}_${job_id}.out"
else
    echo "SUBMIT FAILED for $MODE: $response" >&2
    exit 1
fi
