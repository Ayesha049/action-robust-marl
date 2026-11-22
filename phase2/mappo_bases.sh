#!/usr/bin/env bash
# Phase 2 Stage 3: the MAPPO Transport bases, seeds 0-9 (0-4 development, 5-9 confirmation), with
# the configuration that passed the m10 gate pilot (fast_ent: lr 3e-4, 10 epochs, entropy 0.01),
# 24M frames on transport_m10. Then a clean evaluation of each base.
# Usage (repo root): nohup bash phase2/mappo_bases.sh > phase2_mappo_bases.log 2>&1 &
set -u
OUT=${OUT:-runs_phase2/mappo_bases}
PY=${PY:-$HOME/miniconda3/envs/marl2/bin/python}
mkdir -p $OUT/eval
log() { echo "[$(date '+%F %T')] $*"; }
log "training 10 bases"
seq 0 9 | xargs -P 10 -I{} bash -c "$PY train_mappo.py --task transport_m10 --seed {} --steps 24000000 --lr 3e-4 --epochs 10 --ent-coef 0.01 --out-dir $OUT > $OUT/train_s{}.out 2>&1"
log "training done"
for s in $(seq 0 9); do
  $PY -m phase2.eval_clean --run-dir $OUT/transport_m10__mappo__s$s --label s$s --out $OUT/eval
done
touch $OUT/DONE
