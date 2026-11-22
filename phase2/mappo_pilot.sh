#!/usr/bin/env bash
# Phase 2 Stage 3 gate pilot for the MAPPO Transport base. Tuning seeds 100-102 only; seeds 0-9
# stay unused for the real bases. Configs:
#   paper:    VMAS paper MAPPO settings (lr 5e-5, 45 epochs, minibatch 4096, 600 envs, lambda 0.9)
#   fast:     lr 3e-4, 10 epochs (same batch)
#   fast_ent: fast + entropy bonus 0.01 (fast collapsed its std to ~0.05 on mass 50)
# Then clean evaluation of every run with phase2/eval_clean.py, plus the scripted and random
# references. Gate (PHASE-2.md): clean return >= scripted and ever-delivered >= scripted.
# Usage (repo root): TASK=transport_m10 OUT=... nohup bash phase2/mappo_pilot.sh > log 2>&1 &
set -u
TASK=${TASK:-transport}
OUT=${OUT:-runs_phase2/mappo_pilot}
STEPS=${STEPS:-24000000}
CONFIGS=${CONFIGS:-"paper fast"}
PY=${PY:-$HOME/miniconda3/envs/marl2/bin/python}
mkdir -p $OUT
log() { echo "[$(date '+%F %T')] $*"; }
args() { case $1 in paper) echo "";; fast) echo "--lr 3e-4 --epochs 10";;
                     fast_ent) echo "--lr 3e-4 --epochs 10 --ent-coef 0.01";; esac; }
: > $OUT/train.cmds
for s in 100 101 102; do
  for c in $CONFIGS; do
    echo "mkdir -p $OUT/$c && $PY train_mappo.py --task $TASK --seed $s --steps $STEPS $(args $c) --out-dir $OUT/$c > $OUT/$c/train_s$s.out 2>&1" >> $OUT/train.cmds
  done
done
log "training $(wc -l < $OUT/train.cmds) runs"
xargs -P 12 -I{} bash -c "{}" < $OUT/train.cmds
log "training done"
mkdir -p $OUT/eval
for pol in scripted random; do
  $PY -m phase2.eval_clean --task $TASK --policy $pol --label $pol --out $OUT/eval
done
for c in $CONFIGS; do
  for s in 100 101 102; do
    $PY -m phase2.eval_clean --run-dir $OUT/$c/${TASK}__mappo__s$s --label ${c}_s$s --out $OUT/eval
  done
done
touch $OUT/DONE
