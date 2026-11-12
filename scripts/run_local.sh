#!/usr/bin/env bash
# Run commands from stdin in parallel on this machine (no scheduler).
# Usage: scripts/cmds.sh train | scripts/run_local.sh 30
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${1:-30}
xargs -P "$JOBS" -I{} bash -c "{}"
echo "run_local: all jobs finished at $(date)"
