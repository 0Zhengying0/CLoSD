#!/usr/bin/env bash
# Six independent formal CLoSD evaluations; --check never submits or creates a run.
set -euo pipefail

CLOSD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHARED_ROOT=$(dirname "$CLOSD_ROOT")
PYTHON="$SHARED_ROOT/envs/closd/bin/python"
SLURM_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/closd_velocity_eval.slurm"
TEST_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/test_fm_cfg_sweep.py"
SUMMARY_SCRIPT="$CLOSD_ROOT/reproduction/scripts/hpc/summarize_fm_cfg_sweep.py"
MODEL="$CLOSD_ROOT/reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
RUN_BASE="$CLOSD_ROOT/reproduction/experiments/fm_cfg_sweep/runs"

usage() { echo "Usage: $0 --check|--submit [--run-tag TAG]" >&2; exit 2; }
MODE=
RUN_TAG=
while (( $# )); do
    case "$1" in
        --check|--submit) [[ -z "$MODE" ]] || usage; MODE=$1; shift ;;
        --run-tag) (( $# >= 2 )) || usage; RUN_TAG=$2; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$MODE" ]] || usage
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] || {
    echo "Run tag must have 1-32 letters, digits, underscores or dashes" >&2; exit 2;
}
RUN_ROOT="$RUN_BASE/$RUN_TAG"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
    echo "Historical run path exists; choose a new --run-tag: $RUN_ROOT" >&2; exit 2;
}
[[ -s "$MODEL" && -s "$(dirname "$MODEL")/args.json" ]] || {
    echo "Missing FM 300k EMA checkpoint or adjacent args.json: $MODEL" >&2; exit 2;
}
[[ -s "$CLOSD_ROOT/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth" ]] || {
    echo "Missing pretrained CLoSD controller checkpoint" >&2; exit 2;
}
for path in "$PYTHON" "$SLURM_SCRIPT" "$TEST_SCRIPT" "$SUMMARY_SCRIPT" \
            "$SHARED_ROOT/containers/closd_cuda121_ubuntu20_devel.sif" \
            "$SHARED_ROOT/isaacgym/python/isaacgym"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 2; }

bash -n "$0"
bash -n "$SLURM_SCRIPT"
"$PYTHON" -B -c 'import ast,pathlib,sys; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in sys.argv[1:]]' \
    "$CLOSD_ROOT/closd/env/tasks/closd.py" \
    "$CLOSD_ROOT/closd/learning/im_amp_players.py" \
    "$CLOSD_ROOT/closd/diffusion_planner/diffusion/ode_sampler.py" \
    "$CLOSD_ROOT/closd/diffusion_planner/utils/sampler_util.py" \
    "$TEST_SCRIPT" "$SUMMARY_SCRIPT"
"$PYTHON" -B -c '
import json,sys
with open(sys.argv[1]) as handle: args=json.load(handle)
for key,value in {"diffusion_steps":1000,"pred_len":40,"context_len":20,"multi_target_cond":True,"use_ema":True}.items():
    if args.get(key) != value: raise SystemExit("checkpoint args.json mismatch: {}={!r}, expected {!r}".format(key,args.get(key),value))
' "$(dirname "$MODEL")/args.json"
"$PYTHON" -B "$TEST_SCRIPT"

IDS=(FM-CFG1 FM-CFG2p5 FM-CFG5 FM-CFG7p5 FM-CFG10 FM-CFG15)
SCALES=(1 2.5 5 7.5 10 15)
echo "CHECKPOINT=$MODEL"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "EVAL_SEED=0 ENVIRONMENTS=4096 EPISODE_LENGTH=500 SAMPLES_PER_TASK=1000 SOLVER=rk4 STEPS=3 NFE=12 GUIDANCE=text"
echo "Commands for all six experiments:"
for index in "${!IDS[@]}"; do
    id=${IDS[$index]}
    scale=${SCALES[$index]}
    validation=$(GENERATOR=flow MODEL="$MODEL" ODE_STEPS=3 ODE_SOLVER=rk4 \
        CFG_SWEEP_ID="$id" CFG_SWEEP_RUN_ROOT="$RUN_ROOT" CFG_SCALE="$scale" \
        bash "$SLURM_SCRIPT" --check 2>&1) || {
        echo "Invalid task $id: $validation" >&2; exit 2;
    }
    job_name="cfg_${id#FM-CFG}_${RUN_TAG}"
    stdout="$RUN_ROOT/logs/${id}_%j.out"
    stderr="$RUN_ROOT/logs/${id}_%j.err"
    export_spec="ALL,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=3,ODE_SOLVER=rk4,CFG_SWEEP_ID=$id,CFG_SWEEP_RUN_ROOT=$RUN_ROOT,CFG_SCALE=$scale"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=$job_name"
        "--output=$stdout" "--error=$stderr" "--export=$export_spec" "$SLURM_SCRIPT")
    printf '%s CFG=%s RK4 steps=3 NFE=12 expected_MDM_forwards=%s command:' \
        "$id" "$scale" "$([[ "$scale" == 1 ]] && echo 12 || echo 24)"
    printf ' %q' "${command[@]}"
    printf '\n'
done
if [[ "$MODE" == --check ]]; then
    echo "PRECHECK_STATUS=PASS; no jobs submitted and no run directory created"
    exit 0
fi

mkdir -p "$RUN_BASE"
mkdir "$RUN_ROOT"  # Atomic reservation; refuses to reuse an existing run.
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/work"
manifest="$RUN_ROOT/submissions.tsv"
printf 'experiment_id\tcfg\tsolver\tsteps\tnfe\tjob_name\tjob_id\tstdout\tstderr\tsubmit_status\tcheckpoint\n' > "$manifest"
failed=0
for index in "${!IDS[@]}"; do
    id=${IDS[$index]}
    scale=${SCALES[$index]}
    job_name="cfg_${id#FM-CFG}_${RUN_TAG}"
    stdout="$RUN_ROOT/logs/${id}_%j.out"
    stderr="$RUN_ROOT/logs/${id}_%j.err"
    export_spec="ALL,GENERATOR=flow,MODEL=$MODEL,ODE_STEPS=3,ODE_SOLVER=rk4,CFG_SWEEP_ID=$id,CFG_SWEEP_RUN_ROOT=$RUN_ROOT,CFG_SCALE=$scale"
    command=(sbatch --parsable "--chdir=$CLOSD_ROOT" "--job-name=$job_name"
        "--output=$stdout" "--error=$stderr" "--export=$export_spec" "$SLURM_SCRIPT")
    if response=$("${command[@]}" 2>&1); then
        job_id=${response%%;*}
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            status=submitted
            echo "$id CFG=$scale RK4 steps=3 NFE=12 Slurm_Job_ID=$job_id"
        else
            status=submit_failed; job_id=; failed=1
            echo "SUBMIT FAILED: $id returned invalid Job ID: $response" >&2
        fi
    else
        status=submit_failed; job_id=; failed=1
        echo "SUBMIT FAILED: $id: $response" >&2
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$scale" rk4 3 12 "$job_name" "$job_id" "$stdout" "$stderr" "$status" "$MODEL" >> "$manifest"
done
echo "SUBMISSION_MANIFEST=$manifest"
if (( failed )); then
    echo "Some submissions failed; see the manifest. Submitted jobs were not cancelled." >&2
    exit 1
fi
echo "SUBMISSION_STATUS=ALL_SIX_SUBMITTED"
