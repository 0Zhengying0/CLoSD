#!/usr/bin/env bash
set -euo pipefail
source "${CLOSD_ROOT:?CLOSD_ROOT must be exported by the submitter}/scripts/hpc/closd_env.sh"
mkdir -p "$CLOSD_LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
cd "$CLOSD_ROOT"
exec python "$CLOSD_ROOT/scripts/hpc/run_job.py" "$@" --inside
