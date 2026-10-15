#!/usr/bin/env bash
set -euo pipefail

CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHARED_ROOT=$(dirname "$CLOSD_ROOT")
PYTHON="$SHARED_ROOT/envs/closd/bin/python"
SLURM_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm"
TEST_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/test_episode_diagnostics.py"
FM_MODEL="$CLOSD_ROOT/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
DIP_MODEL="$CLOSD_ROOT/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt"
BASE="$CLOSD_ROOT/reproduction/experiments/episode_diagnostics/runs"

usage() {
    echo "Usage: $0 --check|--submit --run-tag NEW_TAG" >&2
    exit 2
}

MODE=
RUN_TAG=
while (( $# )); do
    case "$1" in
        --check|--submit)
            [[ -z "$MODE" ]] || usage
            MODE=$1
            shift
            ;;
        --run-tag)
            (( $# >= 2 )) || usage
            RUN_TAG=$2
            shift 2
            ;;
        *) usage ;;
    esac
done
[[ -n "$MODE" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || usage
RUN_ROOT="$BASE/$RUN_TAG"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
    echo "Existing diagnostic artifacts would be reused: $RUN_ROOT" >&2
    exit 2
}

for resource in "$PYTHON" "$SLURM_SCRIPT" "$TEST_SCRIPT" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/analyze_episode_diagnostics.py" \
    "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth"; do
    [[ -s "$resource" ]] || { echo "Missing resource: $resource" >&2; exit 2; }
done
for checkpoint in "$FM_MODEL" "$DIP_MODEL"; do
    [[ -s "$checkpoint" && -s "$(dirname "$checkpoint")/args.json" ]] || {
        echo "Missing checkpoint or adjacent args.json: $checkpoint" >&2; exit 2;
    }
done
command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 2; }

bash -n "$SLURM_SCRIPT"
bash -n "$0"
"$PYTHON" -B -c '
import ast, json, pathlib, sys
for name in sys.argv[1:6]:
    ast.parse(pathlib.Path(name).read_text(), filename=name)
for filename, expected_steps in ((sys.argv[6], 1000), (sys.argv[7], 10)):
    args = json.loads(pathlib.Path(filename).read_text())
    for key, value in (("diffusion_steps", expected_steps), ("context_len", 20),
                       ("pred_len", 40), ("multi_target_cond", True), ("use_ema", True)):
        if args.get(key) != value:
            raise SystemExit("checkpoint configuration mismatch: {} {}".format(filename, key))
' \
    "$CLOSD_ROOT/closd/learning/episode_diagnostics.py" \
    "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
    "$CLOSD_ROOT/closd/env/tasks/humanoid.py" \
    "$TEST_SCRIPT" \
    "$CLOSD_ROOT/reproduction/scripts/hpc/analyze_episode_diagnostics.py" \
    "$(dirname "$FM_MODEL")/args.json" "$(dirname "$DIP_MODEL")/args.json"
"$PYTHON" -B "$TEST_SCRIPT"

LABELS=(FM-RK4-3 DIP-ORIGINAL)
SLUGS=(fm_rk4_3 dip_original)
GENERATORS=(flow diffusion)
SOLVERS=(rk4 diffusion)
STEPS=(3 0)
MODELS=("$FM_MODEL" "$DIP_MODEL")

echo "RUN_ROOT=$RUN_ROOT"
echo "EVAL_SEED=0 ENVIRONMENTS=4096 EPISODE_LENGTH=500 CFG=7.5 SAMPLES_PER_TASK=1000"
for index in "${!LABELS[@]}"; do
    label=${LABELS[$index]}
    slug=${SLUGS[$index]}
    model=${MODELS[$index]}
    run_dir="$RUN_ROOT/$slug"
    checked=$(GENERATOR=${GENERATORS[$index]} MODEL="$model" \
        ODE_STEPS=${STEPS[$index]} ODE_SOLVER=${SOLVERS[$index]} \
        DIAG_LABEL="$label" DIAG_RUN_ROOT="$run_dir" \
        bash "$SLURM_SCRIPT" --check 2>&1) || {
        echo "Invalid $label: $checked" >&2
        exit 2
    }
    job_name="diag_${slug}_${RUN_TAG}"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=$job_name"
        "--output=$run_dir/logs/${slug}_%j.out"
        "--error=$run_dir/logs/${slug}_%j.err"
        "--export=ALL,GENERATOR=${GENERATORS[$index]},MODEL=$model,ODE_STEPS=${STEPS[$index]},ODE_SOLVER=${SOLVERS[$index]},DIAG_LABEL=$label,DIAG_RUN_ROOT=$run_dir"
        "$SLURM_SCRIPT")
    printf '%s command:' "$label"
    printf ' %q' "${command[@]}"
    printf '\n'
done

if [[ "$MODE" == --check ]]; then
    echo "PRECHECK_STATUS=PASS; no jobs submitted and no run directory created"
    exit 0
fi

mkdir -p "$BASE"
mkdir "$RUN_ROOT"
for slug in "${SLUGS[@]}"; do
    mkdir -p "$RUN_ROOT/$slug/logs" "$RUN_ROOT/$slug/work"
done
manifest="$RUN_ROOT/submissions.tsv"
printf 'label\tgenerator\tsteps\tmodel\tjob_id\tsubmit_status\tstdout\tstderr\n' > "$manifest"
failed=0
for index in "${!LABELS[@]}"; do
    label=${LABELS[$index]}
    slug=${SLUGS[$index]}
    model=${MODELS[$index]}
    run_dir="$RUN_ROOT/$slug"
    stdout="$run_dir/logs/${slug}_%j.out"
    stderr="$run_dir/logs/${slug}_%j.err"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT"
        "--job-name=diag_${slug}_${RUN_TAG}"
        "--output=$stdout" "--error=$stderr"
        "--export=ALL,GENERATOR=${GENERATORS[$index]},MODEL=$model,ODE_STEPS=${STEPS[$index]},ODE_SOLVER=${SOLVERS[$index]},DIAG_LABEL=$label,DIAG_RUN_ROOT=$run_dir"
        "$SLURM_SCRIPT")
    if response=$("${command[@]}" 2>&1); then
        job_id=${response%%;*}
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            status=submitted
            echo "$label Slurm_Job_ID=$job_id"
        else
            job_id=
            status=submit_failed
            failed=1
            echo "SUBMIT FAILED: $label: $response" >&2
        fi
    else
        job_id=
        status=submit_failed
        failed=1
        echo "SUBMIT FAILED: $label: $response" >&2
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "${GENERATORS[$index]}" "${STEPS[$index]}" "$model" \
        "$job_id" "$status" "$stdout" "$stderr" >> "$manifest"
done
echo "SUBMISSION_MANIFEST=$manifest"
if (( failed )); then
    exit 1
fi
echo "SUBMISSION_STATUS=BOTH_SUBMITTED"
