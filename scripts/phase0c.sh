#!/usr/bin/env bash
# Unattended Phase 0c follow-up for one machine group (see phase0c.py).
#   GROUP=a JOBS=30 scripts/phase0c.sh      # run inside tmux, env marl0 active
# Safe to re-run: finished runs and results are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
GROUP=${GROUP:?set GROUP=a|b}
JOBS=${JOBS:-30}
LOG=phase0c_$GROUP.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "start group=$GROUP jobs=$JOBS sha=$(git rev-parse --short HEAD 2>/dev/null)"
python phase0c.py train --group "$GROUP" | scripts/run_local.sh "$JOBS"
log "training done: $(ls runs_dose/*/*/DONE runs_val/*/*/DONE 2>/dev/null | wc -l) finished run dirs on this machine"
python phase0c.py post --group "$GROUP" | scripts/run_local.sh "$JOBS"
log "post-processing done: $(ls runs_dose/*/*/eval_curves.csv 2>/dev/null | wc -l) eval, $(ls runs_dose/*/*/adv/*.json 2>/dev/null | wc -l) adv, $(ls runs_val/*/*/prey_s*.json 2>/dev/null | wc -l) prey"
touch "PHASE0C_${GROUP}_DONE"
log "finished"
