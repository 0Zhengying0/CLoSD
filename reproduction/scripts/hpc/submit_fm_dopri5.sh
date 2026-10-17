#!/usr/bin/env bash
# User-driven DOPRI5 smoke, 20-case calibration, and four-way matched-NFE sweep.
set -euo pipefail

CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=$(dirname "$CLOSD_ROOT")
PYTHON=$ROOT/envs/closd/bin/python
MODEL=$CLOSD_ROOT/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt
BASE=$CLOSD_ROOT/reproduction/experiments/fm_dopri5/runs
CASES=$CLOSD_ROOT/reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/conditions/manifest.json
SMOKE=$CLOSD_ROOT/reproduction/scripts/hpc/fm_dopri5_smoke.slurm
CALIBRATE=$CLOSD_ROOT/reproduction/scripts/hpc/fm_dopri5_calibrate.slurm
FORMAL=$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm
TEST=$CLOSD_ROOT/reproduction/scripts/hpc/test_fm_dopri5.py
RTOL=0.001
ATOL=0.0001
MAX_STEPS=128
MAX_NFE=256

usage() { echo "Usage: $0 --check|--smoke|--calibrate|--submit --run-tag TAG [--retry-tag RETRY]" >&2; exit 2; }
[[ ( $# -eq 3 || $# -eq 5 ) && "$2" == --run-tag ]] || usage
MODE=$1
TAG=$3
RETRY_TAG=
if (( $# == 5 )); then
    [[ "$4" == --retry-tag ]] || usage
    RETRY_TAG=$5
    [[ "$MODE" == --check || "$MODE" == --submit ]] || usage
    [[ "$RETRY_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,11}$ ]] || usage
fi
[[ "$MODE" == --check || "$MODE" == --smoke || "$MODE" == --calibrate || "$MODE" == --submit ]] || usage
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || usage
RUN_ROOT=$BASE/$TAG

[[ -x "$PYTHON" && -s "$MODEL" && -s "$(dirname "$MODEL")/args.json" && -s "$CASES" ]] || {
    echo "Missing Python, FM checkpoint, args.json or real-case manifest" >&2; exit 2;
}
command -v sbatch >/dev/null || { echo "sbatch unavailable" >&2; exit 2; }
for script in "$SMOKE" "$CALIBRATE" "$FORMAL" "$0"; do bash -n "$script"; done
"$PYTHON" -B - "$MODEL" "$CASES" <<'PY'
import json
import pathlib
import sys
import torchdiffeq
assert torchdiffeq.__version__ == "0.2.5"
args = json.loads((pathlib.Path(sys.argv[1]).parent / "args.json").read_text())
for key, value in {"diffusion_steps": 1000, "pred_len": 40, "context_len": 20,
                   "multi_target_cond": True, "use_ema": True}.items():
    assert args.get(key) == value, (key, args.get(key))
cases = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert cases["source"] == "real CLoSD planner inputs before sampling"
assert len(cases["cases"]) == 20
assert {"REACH_00_normal", "GET_UP_00_normal"}.issubset(
    {case["case_id"] for case in cases["cases"]})
PY
"$PYTHON" -B "$TEST"
DOPRI5_RUN_ROOT="$RUN_ROOT" bash "$SMOKE" --check
DOPRI5_RUN_ROOT="$RUN_ROOT" bash "$CALIBRATE" --check

SMOKE_EXPORT="ALL,DOPRI5_RUN_ROOT=$RUN_ROOT,DOPRI5_CASES=$CASES,DOPRI5_RTOL=$RTOL,DOPRI5_ATOL=$ATOL,DOPRI5_MAX_NUM_STEPS=$MAX_STEPS,DOPRI5_MAX_NFE=$MAX_NFE"
CALIBRATE_EXPORT="ALL,DOPRI5_RUN_ROOT=$RUN_ROOT,DOPRI5_MAX_NUM_STEPS=$MAX_STEPS,DOPRI5_MAX_NFE=$MAX_NFE"
SMOKE_CMD=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=dopri5_smoke_$TAG"
    "--output=$RUN_ROOT/logs/smoke_%j.out" "--error=$RUN_ROOT/logs/smoke_%j.err"
    "--export=$SMOKE_EXPORT" "$SMOKE")
CALIBRATE_CMD=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=dopri5_calibrate_$TAG"
    "--output=$RUN_ROOT/logs/calibration_%j.out" "--error=$RUN_ROOT/logs/calibration_%j.err"
    "--export=$CALIBRATE_EXPORT" "$CALIBRATE")

if [[ "$MODE" == --smoke ]]; then
    [[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 2; }
    mkdir -p "$BASE"
    mkdir "$RUN_ROOT"
    mkdir "$RUN_ROOT/logs" "$RUN_ROOT/work"
    "$PYTHON" -B - "$RUN_ROOT/config.json" "$MODEL" "$CASES" <<'PY'
import json
import pathlib
import sys
path, model, cases = sys.argv[1:4]
pathlib.Path(path).write_text(json.dumps({"checkpoint": model, "case_manifest": cases,
    "rtol": 0.001, "atol": 0.0001, "max_num_steps": 128, "max_nfe": 256,
    "cfg": 7.5, "seed": 0, "num_envs": 4096, "episode_length": 500,
    "samples_per_task": 1000, "solver": "dopri5", "time_encoding": "round(t * 999)"}, indent=2) + "\n")
PY
    if response=$("${SMOKE_CMD[@]}" 2>&1); then
        job_id=${response%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid smoke Job ID: $response" >&2; exit 1; }
        printf '%s\n' "$job_id" > "$RUN_ROOT/smoke_job_id.txt"
        echo "DOPRI5_SMOKE_JOB_ID=$job_id"
    else
        echo "DOPRI5_SMOKE_SUBMIT_FAILED=$response" >&2; exit 1
    fi
    exit 0
fi

if [[ "$MODE" == --calibrate ]]; then
    [[ -s "$RUN_ROOT/config.json" && -s "$RUN_ROOT/smoke_job_id.txt" ]] || {
        echo "Successful smoke job required before calibration" >&2; exit 2;
    }
    [[ ! -e "$RUN_ROOT/calibration_job_id.txt" ]] || {
        echo "Calibration already submitted" >&2; exit 2;
    }
    smoke_job=$(cat "$RUN_ROOT/smoke_job_id.txt")
    [[ "$smoke_job" =~ ^[0-9]+$ && -s "$RUN_ROOT/smoke_$smoke_job/smoke.json" ]] || {
        echo "Smoke output missing" >&2; exit 2;
    }
    grep -q '^DOPRI5_SMOKE_JOB_EXIT=0$' "$RUN_ROOT/logs/smoke_${smoke_job}.out" || {
        echo "Smoke job did not complete successfully" >&2; exit 2;
    }
    if response=$("${CALIBRATE_CMD[@]}" 2>&1); then
        job_id=${response%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid calibration Job ID: $response" >&2; exit 1; }
        printf '%s\n' "$job_id" > "$RUN_ROOT/calibration_job_id.txt"
        echo "DOPRI5_CALIBRATION_JOB_ID=$job_id"
    else
        echo "DOPRI5_CALIBRATION_SUBMIT_FAILED=$response" >&2; exit 1
    fi
    exit 0
fi

COMMON_NFE=
PILOT_NFE=
if [[ -s "$RUN_ROOT/calibration_job_id.txt" ]]; then
    calibration_job=$(cat "$RUN_ROOT/calibration_job_id.txt")
    [[ "$calibration_job" =~ ^[0-9]+$ ]] || { echo "Invalid calibration Job ID" >&2; exit 2; }
    calibration_log=$RUN_ROOT/logs/calibration_${calibration_job}.out
    if [[ -s "$calibration_log" ]] && \
        grep -q '^DOPRI5_CALIBRATION_COMPLETE=1 ' "$calibration_log" && \
        grep -q '^DOPRI5_CALIBRATION_PROCESS_EXIT=0$' "$calibration_log" && \
        grep -q '^DOPRI5_CALIBRATION_JOB_EXIT=0$' "$calibration_log"; then
        calibration_result=$("$PYTHON" -B - "$RUN_ROOT/calibration_${calibration_job}/calibration.json" "$MODEL" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
pilot = data["default_dopri5"]
common = data["recommended_common_nfe"]
if (data["checkpoint"] != sys.argv[2] or data["case_count"] != 20 or
        data["max_num_steps"] != 128 or data["max_nfe"] != 256 or
        pilot["status"] != "COMPLETE" or pilot["rtol"] != 0.001 or
        pilot["atol"] != 0.0001 or not isinstance(common, int) or
        common < 4 or common % 4 != 0 or abs(common - pilot["nfe"]) > 2):
    raise SystemExit("Calibration did not produce a valid common NFE")
print(common, pilot["nfe"])
PY
        ) || exit 2
        read -r COMMON_NFE PILOT_NFE <<< "$calibration_result"
    fi
fi

if [[ -z "$COMMON_NFE" ]]; then
    [[ "$MODE" == --check ]] || { echo "20-case NFE calibration is pending or failed" >&2; exit 2; }
    printf 'SMOKE_COMMAND:'; printf ' %q' "${SMOKE_CMD[@]}"; printf '\n'
    printf 'CALIBRATION_COMMAND:'; printf ' %q' "${CALIBRATE_CMD[@]}"; printf '\n'
    echo "NFE_CALIBRATION=PENDING; no formal commands or jobs submitted"
    exit 0
fi

IDS=(FM-ME FM-MH FM-MR FM-MD)
SOLVERS=(euler heun rk4 dopri5)
STEPS=("$COMMON_NFE" "$((COMMON_NFE / 2))" "$((COMMON_NFE / 4))" 0)
MANIFEST=$RUN_ROOT/matched_submissions.tsv
JOB_SUFFIX=
LOG_SUFFIX=
if [[ -n "$RETRY_TAG" ]]; then
    [[ -s "$RUN_ROOT/matched_submissions.tsv" ]] || {
        echo "Original matched submission manifest missing; retry requires a prior submission" >&2; exit 2;
    }
    MANIFEST=$RUN_ROOT/matched_submissions_${RETRY_TAG}.tsv
    JOB_SUFFIX=_${RETRY_TAG}
    LOG_SUFFIX=_${RETRY_TAG}
fi
[[ ! -e "$MANIFEST" ]] || {
    echo "Submission manifest already exists; choose a fresh retry tag: $MANIFEST" >&2; exit 2;
}
CLEAR_MODES="EXPERIMENT_ID=,SWEEP_RUN_ROOT=,DIAG_LABEL=,DIAG_RUN_ROOT=,CFG_SWEEP_ID=,CFG_SWEEP_RUN_ROOT=,DOPRI5_ID=,DOPRI5_RUN_ROOT="
echo "COMMON_NFE=$COMMON_NFE DOPRI5_PILOT_NFE=$PILOT_NFE CFG=7.5 SEED=0 ENVIRONMENTS=4096 EPISODE_LENGTH=500"
for i in "${!IDS[@]}"; do
    experiment_id=${IDS[$i]}
    solver=${SOLVERS[$i]}
    steps=${STEPS[$i]}
    EXPERIMENT_ID= SWEEP_RUN_ROOT= DIAG_LABEL= DIAG_RUN_ROOT= \
        CFG_SWEEP_ID= CFG_SWEEP_RUN_ROOT= DOPRI5_ID= DOPRI5_RUN_ROOT= \
        GENERATOR=flow MODEL="$MODEL" ODE_STEPS="$steps" ODE_SOLVER="$solver" CFG_SCALE=7.5 \
        MATCHED_ID="$experiment_id" MATCHED_RUN_ROOT="$RUN_ROOT" MATCHED_NFE="$COMMON_NFE" \
        DOPRI5_RTOL="$RTOL" DOPRI5_ATOL="$ATOL" \
        DOPRI5_MAX_NUM_STEPS="$MAX_STEPS" DOPRI5_MAX_NFE="$MAX_NFE" \
        bash "$FORMAL" --check >/dev/null
    export_spec="ALL,$CLEAR_MODES,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=$steps,ODE_SOLVER=$solver,CFG_SCALE=7.5,MATCHED_ID=$experiment_id,MATCHED_RUN_ROOT=$RUN_ROOT,MATCHED_NFE=$COMMON_NFE,DOPRI5_RTOL=$RTOL,DOPRI5_ATOL=$ATOL,DOPRI5_MAX_NUM_STEPS=$MAX_STEPS,DOPRI5_MAX_NFE=$MAX_NFE"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=${experiment_id,,}_${TAG}${JOB_SUFFIX}"
        "--output=$RUN_ROOT/logs/${experiment_id}${LOG_SUFFIX}_%j.out"
        "--error=$RUN_ROOT/logs/${experiment_id}${LOG_SUFFIX}_%j.err"
        "--export=$export_spec" "$FORMAL")
    sbatch --test-only "${command[@]:2}" >/dev/null
    printf '%s solver=%s steps=%s NFE=%s command:' "$experiment_id" "$solver" "$steps" \
        "$([[ "$solver" == dopri5 ]] && echo adaptive || echo "$COMMON_NFE")"
    printf ' %q' "${command[@]}"; printf '\n'
done
if [[ "$MODE" == --check ]]; then
    echo "MATCHED_PRECHECK=PASS; no jobs submitted"
    exit 0
fi

mkdir -p "$RUN_ROOT/work"
set -C  # Refuse to replace a manifest created by another concurrent submission.
printf 'experiment_id\tsolver\tsteps\ttarget_nfe\tpilot_nfe\tjob_id\tstdout\tstderr\tsubmit_status\tcheckpoint\n' > "$MANIFEST"
set +C
failed=0
for i in "${!IDS[@]}"; do
    experiment_id=${IDS[$i]}; solver=${SOLVERS[$i]}; steps=${STEPS[$i]}
    stdout=$RUN_ROOT/logs/${experiment_id}${LOG_SUFFIX}_%j.out
    stderr=$RUN_ROOT/logs/${experiment_id}${LOG_SUFFIX}_%j.err
    export_spec="ALL,$CLEAR_MODES,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=$steps,ODE_SOLVER=$solver,CFG_SCALE=7.5,MATCHED_ID=$experiment_id,MATCHED_RUN_ROOT=$RUN_ROOT,MATCHED_NFE=$COMMON_NFE,DOPRI5_RTOL=$RTOL,DOPRI5_ATOL=$ATOL,DOPRI5_MAX_NUM_STEPS=$MAX_STEPS,DOPRI5_MAX_NFE=$MAX_NFE"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=${experiment_id,,}_${TAG}${JOB_SUFFIX}"
        "--output=$stdout" "--error=$stderr" "--export=$export_spec" "$FORMAL")
    if response=$("${command[@]}" 2>&1); then
        job_id=${response%%;*}
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            status=submitted
            echo "$experiment_id solver=$solver steps=$steps target_NFE=$COMMON_NFE Job_ID=$job_id"
        else
            status=submit_failed; job_id=; failed=1
            echo "SUBMIT_FAILED $experiment_id: invalid Job ID: $response" >&2
        fi
    else
        status=submit_failed; job_id=; failed=1
        echo "SUBMIT_FAILED $experiment_id: $response" >&2
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$experiment_id" "$solver" "$steps" "$COMMON_NFE" "$PILOT_NFE" "$job_id" \
        "$stdout" "$stderr" "$status" "$MODEL" >> "$MANIFEST"
done
echo "MATCHED_SUBMISSION_MANIFEST=$MANIFEST"
(( failed == 0 )) || exit 1
echo "MATCHED_SUBMISSION_STATUS=FOUR_SUBMITTED"
