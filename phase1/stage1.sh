#!/usr/bin/env bash
# Phase 1 Stage 1: MADDPG + residual vs FT control, spread, seeds 0-2 (PHASE-1.md, Stage 1).
#  1. train residual and FT on the frozen Phase 0 MADDPG team of each seed (6 runs)
#  2. evaluate them and the Phase 0 MADDPG and NR-MADDPG teams under clean / R / Rphi / PR-R
#  3. learned adversaries (L-NR all alpha 0.3, takeover) against the 6 new teams, 2 seeds each
# Usage (repo root, conda env marl0): nohup bash phase1/stage1.sh > phase1/stage1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs_phase1}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-30}
# absolute interpreter: a fresh `bash -c` can resolve `python` to conda base instead of marl0
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
log() { echo "[$(date '+%F %T')] $*" >&2; }
mkdir -p $OUT/eval
log "start $(git rev-parse --short HEAD) + uncommitted phase1/"

for s in $SEEDS; do for k in residual ft; do
  [ -f $OUT/${k}_s$s/DONE ] && continue
  echo "$PY -m phase1.train_residual --base-dir runs/spread__maddpg__s$s --kind $k --seed $s --out $OUT/${k}_s$s > $OUT/train_${k}_s$s.log 2>&1"
done; done > $OUT/train.cmds
xargs -P $JOBS -I{} bash -c "{}" < $OUT/train.cmds
log "training done"

for s in $SEEDS; do
  for k in residual ft; do echo "$PY -m phase1.eval_stage1 --run-dir $OUT/${k}_s$s --label ${k}_s$s --out $OUT/eval"; done
  for m in maddpg nr; do echo "$PY -m phase1.eval_stage1 --run-dir runs/spread__${m}__s$s --label ${m}_s$s --out $OUT/eval"; done
done > $OUT/eval.cmds
for s in $SEEDS; do for k in residual ft; do for as in 0 1; do
  echo "$PY -m phase1.train_adversary_any --run-dir $OUT/${k}_s$s --op nr --alpha 0.3 --scope all --seed $as"
  echo "$PY -m phase1.train_adversary_any --run-dir $OUT/${k}_s$s --op pr --alpha 1.0 --scope one --seed $as"
done; done; done > $OUT/adv.cmds
cat $OUT/eval.cmds $OUT/adv.cmds | xargs -P $JOBS -I{} bash -c "{}"
log "eval and adversaries done"
touch $OUT/STAGE1_DONE
