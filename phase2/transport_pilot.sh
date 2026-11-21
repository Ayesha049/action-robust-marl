#!/usr/bin/env bash
# Phase 2 Stage 3 gate pilot: can MADDPG learn VMAS Transport (BenchMARL config: 4 agents,
# package mass 50, 100 steps)? Tuning seeds 100-102 only; seeds 0-9 stay unused for the real bases.
# Grid: gamma {0.95 (Phase 0 default), 0.99} x steps 3M; everything else Phase 0 train.py defaults.
# Usage (repo root): JOBS=12 nohup bash phase2/transport_pilot.sh > phase2_pilot.log 2>&1 &
set -u
OUT=${OUT:-runs_phase2/pilot}
JOBS=${JOBS:-12}
STEPS=${STEPS:-3000000}
PY=${PY:-$HOME/miniconda3/envs/marl2/bin/python}
mkdir -p $OUT
log() { echo "[$(date '+%F %T')] $*"; }

: > $OUT/train.cmds
for g in 0.95 0.99; do
  for s in 100 101 102; do
    d=$OUT/g$g
    echo "mkdir -p $d && $PY train.py --task transport --method maddpg --seed $s --steps $STEPS --gamma $g --out-dir $d > $d/train_s$s.out 2>&1" >> $OUT/train.cmds
  done
done
log "training $(wc -l < $OUT/train.cmds) runs"
xargs -P $JOBS -I{} bash -c "{}" < $OUT/train.cmds
log "training done"
mkdir -p $OUT/eval
$PY -m phase2.eval_clean --task transport --policy random --label random --out $OUT/eval
$PY -m phase2.eval_clean --task transport --policy zero --label zero --out $OUT/eval
for g in 0.95 0.99; do
  for s in 100 101 102; do
    $PY -m phase2.eval_clean --run-dir $OUT/g$g/transport__maddpg__s$s --label g${g}_s$s --out $OUT/eval
  done
done
touch $OUT/DONE
