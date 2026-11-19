#!/usr/bin/env bash
# Phase 1 Stage 2: PR-MADDPG + residual, spread, seeds 0-4 (PHASE-1.md, Stage 2).
#  1. train RC (residual), RC-fast (rcfast), FT, REC, RES-MLP on the frozen Phase 0 PR-MADDPG
#     team of each seed, with the co-trained rate-bounded adversary (25 runs)
#  2. evaluate them and the Phase 0 MADDPG, PR-MADDPG and NR-MADDPG teams on the core random
#     cells plus Rphi (40 teams)
#  3. learned adversaries for the 25 new teams: the 9 Phase 0 core learned cells x 2 seeds
#     (MADDPG / PR-MADDPG / NR-MADDPG reuse their Phase 0 adversaries)
# Usage (repo root): nohup bash phase1/stage2.sh > phase1/stage2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs_phase1s2}
SEEDS=${SEEDS:-"0 1 2 3 4"}
KINDS=${KINDS:-"residual rcfast ft rec resmlp"}
JOBS=${JOBS:-100}
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
log() { echo "[$(date '+%F %T')] $*" >&2; }
mkdir -p $OUT/eval
log "start $(git rev-parse --short HEAD 2>/dev/null) + uncommitted phase1/"

for s in $SEEDS; do for k in $KINDS; do
  [ -f $OUT/${k}_s$s/DONE ] && continue
  echo "$PY -m phase1.train_stage2 --base-dir runs/spread__pr__s$s --kind $k --seed $s --out $OUT/${k}_s$s > $OUT/train_${k}_s$s.log 2>&1"
done; done > $OUT/train.cmds
xargs -P $JOBS -I{} bash -c "{}" < $OUT/train.cmds
log "training done: $(ls $OUT/*/DONE 2>/dev/null | wc -l) runs"

for s in $SEEDS; do
  for k in $KINDS; do
    [ -f $OUT/eval/${k}_s$s.csv ] || echo "$PY -m phase1.eval_stage2 --run-dir $OUT/${k}_s$s --label ${k}_s$s --out $OUT/eval > $OUT/eval_${k}_s$s.log 2>&1"
  done
  for m in maddpg pr nr; do
    [ -f $OUT/eval/${m}_s$s.csv ] || echo "$PY -m phase1.eval_stage2 --run-dir runs/spread__${m}__s$s --label ${m}_s$s --out $OUT/eval > $OUT/eval_${m}_s$s.log 2>&1"
  done
done > $OUT/eval.cmds
for s in $SEEDS; do for k in $KINDS; do for as in 0 1; do
  for op in nr pr; do for al in 0.1 0.3; do for sc in all one; do
    [ -f $OUT/${k}_s$s/adv/${op}_a${al}_${sc}_s$as.json ] || \
      echo "$PY -m phase1.train_adversary_any --run-dir $OUT/${k}_s$s --op $op --alpha $al --scope $sc --seed $as > /dev/null 2>> $OUT/adv_errors.log"
  done; done; done
  [ -f $OUT/${k}_s$s/adv/pr_a1.0_one_s$as.json ] || \
    echo "$PY -m phase1.train_adversary_any --run-dir $OUT/${k}_s$s --op pr --alpha 1.0 --scope one --seed $as > /dev/null 2>> $OUT/adv_errors.log"
done; done; done > $OUT/adv.cmds
cat $OUT/eval.cmds $OUT/adv.cmds | xargs -P $JOBS -I{} bash -c "{}"
log "eval: $(ls $OUT/eval/*.csv | wc -l) teams; adversaries: $(ls $OUT/*/adv/*.json 2>/dev/null | wc -l) of 450"
touch $OUT/STAGE2_DONE
