#!/usr/bin/env bash
set -euo pipefail
export CLOSD_ROOT="${CLOSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
exec /usr/bin/python3 "$CLOSD_ROOT/reproduction/scripts/hpc/run_job.py" "$@" --submit
