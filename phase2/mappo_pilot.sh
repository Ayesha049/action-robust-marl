#!/usr/bin/env bash
# Phase 2 Stage 3 gate pilot for the MAPPO Transport base (decision 2026-11-21: MADDPG did not
# learn Transport). Tuning seeds 100-102 only; seeds 0-9 stay unused for the real bases.
#   paper: VMAS paper MAPPO settings (lr 5e-5, 45 epochs, minibatch 4096, 600 envs, lambda 0.9)
#   fast:  lr 3e-4, 10 epochs (same batch), in case the paper settings are too slow on CPU
# 24M frames each. Then clean evaluation of every run with phase2/eval_clean.py.
# Usage (repo root): nohup bash phase2/mappo_pilot.sh > phase2_mappo_pilot.log 2>&1 &
set -u
OUT=${OUT:-runs_phase2/mappo_pilot}
STEPS=${STEPS:-24000000}
PY=${PY:-$HOME/miniconda3/envs/marl2/bin/python}
mkdir -p $OUT
log() { echo "[$(date '+%F %T')] $*"; }
: > $OUT/train.cmds
for s in 100 101 102; do
  echo "mkdir -p $OUT/paper && $PY train_mappo.py --seed $s --steps $STEPS --out-dir $OUT/paper > $OUT/paper/train_s$s.out 2>&1" >> $OUT/train.cmds
  echo "mkdir -p $OUT/fast && $PY train_mappo.py --seed $s --steps $STEPS --lr 3e-4 --epochs 10 --out-dir $OUT/fast > $OUT/fast/train_s$s.out 2>&1" >> $OUT/train.cmds
done
log "training $(wc -l < $OUT/train.cmds) runs"
xargs -P 6 -I{} bash -c "{}" < $OUT/train.cmds
log "training done"
mkdir -p $OUT/eval
for c in paper fast; do
  for s in 100 101 102; do
    $PY -m phase2.eval_clean --run-dir $OUT/$c/transport__mappo__s$s --label ${c}_s$s --out $OUT/eval
  done
done
touch $OUT/DONE
