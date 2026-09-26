#!/usr/bin/env bash
#SBATCH --job-name=closd_smoke
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu_exp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
set -euo pipefail
# Slurm copies this script to a spool directory. submit.sh supplies CLOSD_ROOT.
export CLOSD_ROOT="${CLOSD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
exec /usr/bin/python3 "$CLOSD_ROOT/scripts/hpc/run_job.py" closd_smoke "$@"
