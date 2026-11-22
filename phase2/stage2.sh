#!/usr/bin/env bash
# Phase 2 Stage 2 confirmation (pre-registered in phase-2/PREREG-STAGE2.md of the docs repo).
# Held-out MADDPG bases, seeds 5-9, in $BASES (spread: Phase 1 bases; tag: trained 2026-11-22
# with the Phase 0 tag settings). Frozen Stage 1 run 3 setup: phase2/train_fault.py --track 1.
#   1. train RC / RC-blind / REC / FT on every base
#   2. evaluate the bases and the trained teams on every Stage 0 cell (predictor none)
#   3. correction fits (rot 90, all agents) for RC and FT
#   4. phase2/stage2_tests.py per task
# Usage (repo root): nohup bash phase2/stage2.sh > phase2_stage2.log 2>&1 &
set -u
TASKS=${TASKS:-"tag spread"}
SEEDS=${SEEDS:-"5 6 7 8 9"}
KINDS="residual blind rec ft"
BASES=${BASES:-runs_heldout}
OUT=${OUT:-runs_phase2/stage2}
JOBS=${JOBS:-56}
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
mkdir -p $OUT/logs $OUT/diag
log() { echo "[$(date '+%F %T')] $*"; }
eps() { [ $1 = tag ] && echo 500 || echo 100; }
CELLS="clean gain bias drift stuck wdir rot"
STR="--strengths 'bias=0.1,0.3,0.5,0.7,0.9'"

: > $OUT/train.cmds
for t in $TASKS; do for s in $SEEDS; do for k in $KINDS; do
  echo "$PY -m phase2.train_fault --base-dir $BASES/${t}__maddpg__s$s --kind $k --seed $s --track 1 --out $OUT/$t/${k}__s$s > $OUT/logs/train_${t}_${k}_s$s.log 2>&1" >> $OUT/train.cmds
done; done; done
# base rows run alongside training (they do not depend on it)
: > $OUT/base.cmds
for t in $TASKS; do for s in $SEEDS; do for c in $CELLS; do
  echo "$PY -m phase2.stage0_faults --run-dir $BASES/${t}__maddpg__s$s --out $OUT/eval_$t --kinds $c --predictors none --episodes $(eps $t) $STR > $OUT/logs/base_${t}_s${s}_$c.log 2>&1" >> $OUT/base.cmds
done; done; done
log "training $(wc -l < $OUT/train.cmds) runs + $(wc -l < $OUT/base.cmds) base evaluation jobs"
cat $OUT/train.cmds $OUT/base.cmds | xargs -P $JOBS -I{} bash -c "{}"
log "training done: $(ls $OUT/*/*/DONE 2>/dev/null | wc -l) DONE"
: > $OUT/eval.cmds
for t in $TASKS; do for s in $SEEDS; do for k in $KINDS; do
  for c in $CELLS; do
    echo "$PY -m phase2.stage0_faults --run-dir $OUT/$t/${k}__s$s --out $OUT/eval_$t --kinds $c --predictors none --episodes $(eps $t) $STR > $OUT/logs/eval_${t}_${k}_s${s}_$c.log 2>&1" >> $OUT/eval.cmds
  done
done; done; done
for t in $TASKS; do for s in $SEEDS; do for k in residual ft; do
  echo "$PY -m phase2.diag_correction --run-dir $OUT/$t/${k}__s$s --csv $OUT/diag/${t}_${k}_s$s.csv > /dev/null 2>&1" >> $OUT/eval.cmds
done; done; done
log "evaluating $(wc -l < $OUT/eval.cmds) jobs"
xargs -P $JOBS -I{} bash -c "{}" < $OUT/eval.cmds
log "eval done: errors in $(grep -l -i error $OUT/logs/*.log 2>/dev/null | wc -l) logs"
(head -1 $(ls $OUT/diag/*.csv | head -1); tail -q -n +2 $OUT/diag/*.csv) > $OUT/diag_all.csv
for t in $TASKS; do
  $PY -m phase2.stage2_tests --task $t --dirs $OUT/eval_$t --diag $OUT/diag_all.csv > $OUT/tests_$t.txt 2>&1
done
log "tests written to $OUT/tests_*.txt"
touch $OUT/DONE
