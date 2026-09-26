#!/usr/bin/env bash
#SBATCH --job-name=dip_mt_2h
#SBATCH --account=EU-26-37
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -euo pipefail
# Slurm copies this script to a spool directory. submit.sh supplies CLOSD_ROOT.
export CLOSD_ROOT="${CLOSD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
exec /usr/bin/python3 "$CLOSD_ROOT/scripts/hpc/run_job.py" dip_train_2h "$@"
