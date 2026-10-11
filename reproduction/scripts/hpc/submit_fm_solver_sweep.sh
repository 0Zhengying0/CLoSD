#!/usr/bin/env bash
set -euo pipefail

CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHARED_ROOT=$(dirname "$CLOSD_ROOT")
PYTHON="$SHARED_ROOT/envs/closd/bin/python"
SLURM_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm"
TEST_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/test_fm_solver_sweep.py"
SUMMARY_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/summarize_fm_solver_sweep.py"
MODEL="$CLOSD_ROOT/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
SWEEP_BASE="$CLOSD_ROOT/reproduction/experiments/fm_solver_sweep/runs"

usage() {
    echo "Usage: $0 --check|--submit [--run-tag TAG]" >&2
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
[[ -n "$MODE" ]] || usage
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || {
    echo "Run tag must be 1-32 letters, numbers, underscores or dashes" >&2
    exit 2
}
RUN_ROOT="$SWEEP_BASE/$RUN_TAG"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
    echo "Existing sweep artifacts would be reused: $RUN_ROOT" >&2
    exit 2
}

[[ -s "$MODEL" && -s "$(dirname "$MODEL")/args.json" ]] || {
    echo "Missing FM 300k checkpoint or adjacent args.json: $MODEL" >&2
    exit 2
}
[[ -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" ]] || {
    echo "Missing pretrained CLoSD controller checkpoint" >&2
    exit 2
}
[[ -x "$PYTHON" && -f "$SLURM_SCRIPT" && -f "$TEST_SCRIPT" && -f "$SUMMARY_SCRIPT" ]] || {
    echo "Missing runtime Python, Slurm script or sweep helper" >&2
    exit 2
}
command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 2; }

# Static syntax checks have no filesystem side effects. The CPU tests are a
# gate for both --check and --submit; a failure prevents every submission.
while IFS= read -r -d '' shell_script; do
    bash -n "$shell_script"
done < <(find "$CLOSD_ROOT/reproduction/scripts/hpc" -maxdepth 1 -type f \
    \( -name '*.sh' -o -name '*.slurm' \) -print0)
"$PYTHON" -B -c '
import ast
import pathlib
import sys
for name in sys.argv[1:]:
    ast.parse(pathlib.Path(name).read_text(), filename=name)
' \
    "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" \
    "$CLOSD_ROOT/closd/env/tasks/closd.py" \
    "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
    "$TEST_SCRIPT" "$SUMMARY_SCRIPT"
"$PYTHON" -B -c '
import json
import sys
with open(sys.argv[1]) as handle:
    args = json.load(handle)
expected = {"diffusion_steps": 1000, "pred_len": 40, "context_len": 20,
            "multi_target_cond": True, "use_ema": True}
for key, value in expected.items():
    if args.get(key) != value:
        raise SystemExit("checkpoint args.json mismatch: {}={!r}, expected {!r}".format(
            key, args.get(key), value))
' "$(dirname "$MODEL")/args.json"
"$PYTHON" -B -c '
import pathlib
import sys
repo = pathlib.Path(sys.argv[1])
shared = pathlib.Path(sys.argv[2])
snapshots = list((shared / "cache/huggingface/hub/models--guytevet--CLoSD/snapshots").glob("*"))
if len(snapshots) != 1:
    raise SystemExit("expected one locally cached CLoSD dependency snapshot")
snapshot = snapshots[0]
for source, destination in (("checkpoints/dip", "closd/diffusion_planner/save"),
                            ("evaluation", "closd/diffusion_planner/saved_motions"),
                            ("checkpoints/closd", "output/CLoSD")):
    for item in (snapshot / source).iterdir():
        if item.is_dir() and not (repo / destination / item.name).exists():
            raise SystemExit("shared dependency link is missing: " + destination + "/" + item.name)
' "$CLOSD_ROOT" "$SHARED_ROOT"
"$PYTHON" -B "$TEST_SCRIPT"

# Keep this table explicit so a typo cannot silently change the NFE budget.
IDS=(FM-E12 FM-H12 FM-R12 FM-E20 FM-H20 FM-R20 FM-E40 FM-H40 FM-R40)
SOLVERS=(euler heun rk4 euler heun rk4 euler heun rk4)
STEPS=(12 6 3 20 10 5 40 20 10)
NFES=(12 12 12 20 20 20 40 40 40)

echo "CHECKPOINT=$MODEL"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "EVAL_SEED=0 CHECKPOINT_TRAIN_SEED=10 TORCH_DETERMINISTIC=False ENVIRONMENTS=4096 EPISODE_LENGTH=500 CFG=7.5 SAMPLES_PER_TASK=1000"
echo "Commands for all nine experiments:"

for index in "${!IDS[@]}"; do
    experiment_id=${IDS[$index]}
    solver=${SOLVERS[$index]}
    steps=${STEPS[$index]}
    nfe=${NFES[$index]}
    case "$solver" in euler) factor=1 ;; heun) factor=2 ;; rk4) factor=4 ;; esac
    [[ $((steps * factor)) -eq $nfe ]] || {
        echo "Bad NFE matrix entry: $experiment_id" >&2; exit 2;
    }
    validation_output=$(GENERATOR=flow MODEL="$MODEL" ODE_STEPS="$steps" \
        ODE_SOLVER="$solver" EXPERIMENT_ID="$experiment_id" \
        SWEEP_RUN_ROOT="$RUN_ROOT" bash "$SLURM_SCRIPT" --check 2>&1) || {
        echo "Invalid task $experiment_id: $validation_output" >&2
        exit 2
    }

    job_name="fm_${experiment_id#FM-}_${RUN_TAG}"
    job_name=${job_name,,}
    stdout="$RUN_ROOT/logs/${experiment_id}_%j.out"
    stderr="$RUN_ROOT/logs/${experiment_id}_%j.err"
    export_spec="ALL,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=$steps,ODE_SOLVER=$solver,EXPERIMENT_ID=$experiment_id,SWEEP_RUN_ROOT=$RUN_ROOT"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=$job_name"
        "--output=$stdout" "--error=$stderr" "--export=$export_spec" "$SLURM_SCRIPT")
    printf '%s solver=%s steps=%s NFE=%s command:' "$experiment_id" "$solver" "$steps" "$nfe"
    printf ' %q' "${command[@]}"
    printf '\n'
done

if [[ "$MODE" == --check ]]; then
    echo "PRECHECK_STATUS=PASS; no jobs submitted and no sweep directory created"
    exit 0
fi

# mkdir without -p makes run-tag reservation atomic and refuses old artifacts.
mkdir -p "$SWEEP_BASE"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/work"
manifest="$RUN_ROOT/submissions.tsv"
printf 'experiment_id\tsolver\tsteps\tnfe\tjob_name\tjob_id\tstdout\tstderr\tsubmit_status\tcheckpoint\n' > "$manifest"

failed=0
for index in "${!IDS[@]}"; do
    experiment_id=${IDS[$index]}
    solver=${SOLVERS[$index]}
    steps=${STEPS[$index]}
    nfe=${NFES[$index]}
    job_name="fm_${experiment_id#FM-}_${RUN_TAG}"
    job_name=${job_name,,}
    stdout="$RUN_ROOT/logs/${experiment_id}_%j.out"
    stderr="$RUN_ROOT/logs/${experiment_id}_%j.err"
    export_spec="ALL,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=$steps,ODE_SOLVER=$solver,EXPERIMENT_ID=$experiment_id,SWEEP_RUN_ROOT=$RUN_ROOT"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=$job_name"
        "--output=$stdout" "--error=$stderr" "--export=$export_spec" "$SLURM_SCRIPT")
    if response=$("${command[@]}" 2>&1); then
        job_id=${response%%;*}
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            status=submitted
            echo "$experiment_id solver=$solver steps=$steps NFE=$nfe Slurm_Job_ID=$job_id"
        else
            status=submit_failed
            job_id=
            failed=1
            echo "SUBMIT FAILED: $experiment_id returned an invalid Job ID: $response" >&2
        fi
    else
        status=submit_failed
        job_id=
        failed=1
        echo "SUBMIT FAILED: $experiment_id: $response" >&2
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$experiment_id" "$solver" "$steps" "$nfe" "$job_name" "$job_id" \
        "$stdout" "$stderr" "$status" "$MODEL" >> "$manifest"
done
echo "SUBMISSION_MANIFEST=$manifest"
if (( failed )); then
    echo "Some submissions failed; see the manifest. No running jobs were cancelled." >&2
    exit 1
fi
echo "SUBMISSION_STATUS=ALL_NINE_SUBMITTED"
