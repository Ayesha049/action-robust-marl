#!/usr/bin/env bash
# Phase 2 Stage 1 (development, seeds 0-4): train RC / RC-blind / REC / FT on each frozen
# Phase 0 MADDPG base under the persistent-fault mix (phase2/train_fault.py, untuned defaults),
# then evaluate every trained team on the Stage 0 cells (predictor none, same seeds and keys as
# Stage 0, so results pair with the base's Stage 0 rows).
# Usage (repo root): nohup bash phase2/stage1.sh > phase2_stage1.log 2>&1 &
set -u
TASKS=${TASKS:-"tag spread"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
KINDS=${KINDS:-"residual blind rec ft"}
OUT=${OUT:-runs_phase2/stage1}
JOBS=${JOBS:-56}
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
EXTRA=${EXTRA:-}  # extra train_fault.py arguments, e.g. "--track 1"
mkdir -p $OUT/logs
log() { echo "[$(date '+%F %T')] $*"; }
eps() { [ $1 = tag ] && echo 500 || echo 100; }
: > $OUT/train.cmds
for t in $TASKS; do for s in $SEEDS; do for k in $KINDS; do
  d=$OUT/$t/${k}__s$s
  echo "$PY -m phase2.train_fault --base-dir runs/${t}__maddpg__s$s --kind $k --seed $s --out $d $EXTRA > $OUT/logs/train_${t}_${k}_s$s.log 2>&1" >> $OUT/train.cmds
done; done; done
log "training $(wc -l < $OUT/train.cmds) runs"
xargs -P $JOBS -I{} bash -c "{}" < $OUT/train.cmds
log "training done: $(ls $OUT/*/*/DONE 2>/dev/null | wc -l) DONE"
: > $OUT/eval.cmds
for t in $TASKS; do for s in $SEEDS; do for k in $KINDS; do
  for kind in clean gain bias drift stuck wdir rot; do
    echo "$PY -m phase2.stage0_faults --run-dir $OUT/$t/${k}__s$s --out $OUT/eval_$t --kinds $kind --predictors none --episodes $(eps $t) --strengths 'bias=0.1,0.3,0.5,0.7,0.9' > $OUT/logs/eval_${t}_${k}_s${s}_$kind.log 2>&1" >> $OUT/eval.cmds
  done
done; done; done
log "evaluating $(wc -l < $OUT/eval.cmds) jobs"
xargs -P $JOBS -I{} bash -c "{}" < $OUT/eval.cmds
log "eval done: errors in $(grep -l -i error $OUT/logs/eval_*.log 2>/dev/null | wc -l) logs"
touch $OUT/DONE
