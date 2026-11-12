#!/usr/bin/env bash
# Unattended Phase 0 pipeline for one machine:
#   wait for tuning -> select hparams -> train grid -> eval critic + R/G curves
#   -> learned adversaries -> PIPELINE_DONE
# Usage (workstation): RUNNER=local JOBS=24 METHODS="maddpg pr nr marta" scripts/pipeline.sh
#        (HPC, inside a SLURM job): RUNNER=slurm MAXC=30 METHODS="ernie" scripts/pipeline.sh
# Safe to re-run: every stage skips finished work.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNNER=${RUNNER:-local}
JOBS=${JOBS:-24}
MAXC=${MAXC:-30}
METHODS=${METHODS:?set METHODS}
TASKS=${TASKS:-"spread tag"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TUNE_EXPECTED=${TUNE_EXPECTED:-0}   # number of tuning runs to wait for on this machine
export OUT=${OUT:-runs}
LOG=$OUT/pipeline.log
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run() {  # run commands from stdin with the configured runner
  local name=$1 time=$2
  if [ "$RUNNER" = slurm ]; then
    WAIT=1 scripts/slurm_array.sh "$name" "$MAXC" "$time"
  else
    scripts/run_local.sh "$JOBS"
  fi
}

log "pipeline start: methods=[$METHODS] tasks=[$TASKS] runner=$RUNNER sha=$(git rev-parse --short HEAD)"
[ -f assets/tag_prey.pt ] || { log "missing assets/tag_prey.pt"; exit 1; }

if [ "$TUNE_EXPECTED" -gt 0 ]; then
  until [ "$(ls runs_tune/*/*/DONE 2>/dev/null | wc -l)" -ge "$TUNE_EXPECTED" ]; do sleep 120; done
  log "tuning finished"
fi
for m in $METHODS; do
  [ -f "$OUT/hparams_$m.txt" ] || python select_hparams.py --method "$m" --out "$OUT" >> "$LOG" 2>&1
  log "hparams $m: '$(cat "$OUT/hparams_$m.txt")'"
done

for m in $METHODS; do
  TASKS="$TASKS" SEEDS="$SEEDS" METHODS=$m EXTRA="$(cat "$OUT/hparams_$m.txt")" scripts/cmds.sh train
done | run "train" "04:00:00"
log "training done: $(ls "$OUT"/*/DONE 2>/dev/null | wc -l) runs"

scripts/cmds.sh eval | run "eval" "04:00:00"
log "eval done: $(ls "$OUT"/*/eval_curves.csv 2>/dev/null | wc -l) runs"

scripts/cmds.sh adv | run "adv" "02:00:00"
log "adversaries done: $(ls "$OUT"/*/adv/*.json 2>/dev/null | wc -l) cells"
touch "$OUT/PIPELINE_DONE"
log "pipeline finished"
