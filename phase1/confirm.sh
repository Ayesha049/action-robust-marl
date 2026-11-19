#!/usr/bin/env bash
# Phase 1 confirmation run (PHASE-1.md, "Confirmation run"): fresh seeds 5-9.
#  1. Phase 0 bases: MADDPG, PR-MADDPG, NR-MADDPG, spread, seeds 5-9, Phase 0 settings
#     (train.py defaults, 1.5M steps); plus a reproduction check, PR-MADDPG seed 0 -> runs_check/
#  2. in parallel: Phase 0 learned adversaries for the 15 new bases (train_adversary.py, 9 cells x 2)
#     and RC (residual) + FT on the new PR-MADDPG bases (train_stage2.py, Stage 2 settings)
#  3. evaluation (eval_stage2.py) of the 25 teams; learned adversaries for RC and FT (9 cells x 2)
#  4. T3' (t3prime.py) and the summary (stage2_summary.py)
# Usage (repo root): nohup bash phase1/confirm.sh > phase1/confirm.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs_phase1c}
SEEDS=${SEEDS:-"5 6 7 8 9"}
JOBS=${JOBS:-100}
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
log() { echo "[$(date '+%F %T')] $*" >&2; }
xrun() { xargs -P $JOBS -I{} bash -c "{}"; }
CELLS=("pr 0.1 one" "pr 0.1 all" "pr 0.3 one" "pr 0.3 all" "nr 0.1 one" "nr 0.1 all" "nr 0.3 one" "nr 0.3 all" "pr 1.0 one")
mkdir -p $OUT/eval runs_check
log "start $(git rev-parse --short HEAD 2>/dev/null) + uncommitted phase1/"

{
  for m in maddpg pr nr; do for s in $SEEDS; do
    d=runs/spread__${m}__s$s
    [ -f $d/DONE ] || echo "mkdir -p $d && $PY train.py --task spread --method $m --seed $s --steps 1500000 --out-dir runs > $d/train.out 2>&1"
  done; done
  d=runs_check/spread__pr__s0
  [ -f $d/DONE ] || echo "mkdir -p $d && $PY train.py --task spread --method pr --seed 0 --steps 1500000 --out-dir runs_check > $d/train.out 2>&1"
} | xrun
log "bases done: $(ls runs/spread__*__s[5-9]/DONE 2>/dev/null | wc -l) of 15"

{
  for s in $SEEDS; do for k in residual ft; do
    [ -f $OUT/${k}_s$s/DONE ] || echo "$PY -m phase1.train_stage2 --base-dir runs/spread__pr__s$s --kind $k --seed $s --out $OUT/${k}_s$s > $OUT/train_${k}_s$s.log 2>&1"
  done; done
  for m in maddpg pr nr; do for s in $SEEDS; do for c in "${CELLS[@]}"; do set -- $c; for as in 0 1; do
    d=runs/spread__${m}__s$s
    [ -f $d/adv/${1}_a${2}_${3}_s$as.json ] || echo "$PY train_adversary.py --run-dir $d --op $1 --alpha $2 --scope $3 --seed $as > /dev/null 2>> $d/adv_errors.log"
  done; done; done; done
} | xrun
log "RC/FT: $(ls $OUT/*/DONE 2>/dev/null | wc -l) of 10; base adversaries: $(ls runs/spread__*__s[5-9]/adv/*.json 2>/dev/null | wc -l) of 270"

{
  for s in $SEEDS; do
    for k in residual ft; do
      [ -f $OUT/eval/${k}_s$s.csv ] || echo "$PY -m phase1.eval_stage2 --run-dir $OUT/${k}_s$s --label ${k}_s$s --out $OUT/eval > $OUT/eval_${k}_s$s.log 2>&1"
      for c in "${CELLS[@]}"; do set -- $c; for as in 0 1; do
        [ -f $OUT/${k}_s$s/adv/${1}_a${2}_${3}_s$as.json ] || echo "$PY -m phase1.train_adversary_any --run-dir $OUT/${k}_s$s --op $1 --alpha $2 --scope $3 --seed $as > /dev/null 2>> $OUT/adv_errors.log"
      done; done
    done
    for m in maddpg pr nr; do
      [ -f $OUT/eval/${m}_s$s.csv ] || echo "$PY -m phase1.eval_stage2 --run-dir runs/spread__${m}__s$s --label ${m}_s$s --out $OUT/eval > $OUT/eval_${m}_s$s.log 2>&1"
    done
  done
} | xrun
log "eval: $(ls $OUT/eval/*.csv | wc -l) of 25; RC/FT adversaries: $(ls $OUT/*/adv/*.json 2>/dev/null | wc -l) of 180"

$PY -m phase1.t3prime --out $OUT --seeds "$SEEDS" > $OUT/t3prime_printout.txt 2>&1
$PY -m phase1.stage2_summary --out $OUT 2>&1 | grep -v Warn > $OUT/summary_printout.txt
log "T3' and summary written"
touch $OUT/CONFIRM_DONE
