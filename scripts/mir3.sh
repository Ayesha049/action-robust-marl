#!/usr/bin/env bash
# MIR3 added to Phase 0 (0b protocol + 0c-style dose-response), one machine, unattended.
#   1. train: tuning (spread, seeds 100-101) and main grid (spread+tag, seeds 0-4) for every lam
#   2. select lam by clean return on the tuning runs (select_hparams.py, same rule as 0b)
#   3. winner -> runs_mir3/main (symlinks): full 0b post-processing (eval critic, R/G curves,
#      full learned-adversary sweep, both tasks)
#   4. other lam on spread -> R/G curves + the three 0c dose cells (learned, 2 adversary seeds)
# Usage: JOBS=60 nohup scripts/mir3.sh > mir3.log 2>&1 &     Safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-60}
LAMS=${LAMS:-"0.0005 0.01 0.1 1.0"}
STEPS=${STEPS:-1500000}
R=runs_mir3
mkdir -p $R
log() { echo "[$(date '+%F %T')] $*"; }
log "mir3 start sha=$(git rev-parse --short HEAD) lams=[$LAMS]"
git diff HEAD > $R/code_at_launch.diff

for L in $LAMS; do
  for s in 100 101; do
    d=runs_tune/mir3_lam$L/spread__mir3__s$s
    [ -f $d/DONE ] || echo "mkdir -p $d && python train.py --task spread --method mir3 --seed $s --steps $STEPS --out-dir runs_tune/mir3_lam$L --mir3-lam $L > $d/train.out 2>&1"
  done
done > $R/train.cmds
for L in $LAMS; do for t in spread tag; do for s in 0 1 2 3 4; do
  d=$R/lam$L/${t}__mir3__s$s
  [ -f $d/DONE ] || echo "mkdir -p $d && python train.py --task $t --method mir3 --seed $s --steps $STEPS --out-dir $R/lam$L --mir3-lam $L > $d/train.out 2>&1"
done; done; done >> $R/train.cmds
scripts/run_local.sh $JOBS < $R/train.cmds
log "training done: $(ls runs_tune/mir3_lam*/*/DONE $R/lam*/*/DONE 2>/dev/null | wc -l) runs"

[ -f $R/hparams_mir3.txt ] || python select_hparams.py --method mir3 --tune-root runs_tune --out $R
W=$(awk '{print $2}' $R/hparams_mir3.txt)
log "selected lam=$W ($(cat $R/hparams_mir3.txt))"
mkdir -p $R/main
for d in $R/lam$W/*__mir3__s*; do ln -sfn "../lam$W/$(basename $d)" $R/main/$(basename $d); done

{
  OUT=$R/main scripts/cmds.sh adv
  for L in $LAMS; do
    [ "$L" = "$W" ] && continue
    for d in $R/lam$L/spread__mir3__s*; do
      for cell in "pr 0.3 all" "nr 0.3 all" "pr 1.0 one"; do
        set -- $cell
        for as in 0 1; do
          [ -f $d/adv/${1}_a${2}_${3}_s$as.json ] || echo "python train_adversary.py --run-dir $d --op $1 --alpha $2 --scope $3 --seed $as > /dev/null 2>> $d/adv_errors.log"
        done
      done
    done
  done
  OUT=$R/main scripts/cmds.sh eval
  for L in $LAMS; do
    [ "$L" = "$W" ] && continue
    OUT=$R/lam$L scripts/cmds.sh eval | grep "spread__"
  done
} > $R/post.cmds
scripts/run_local.sh $JOBS < $R/post.cmds
log "post done: $(ls $R/*/*/eval_curves.csv 2>/dev/null | wc -l) eval, $(ls $R/*/*/adv/*.json 2>/dev/null | wc -l) adv"
touch $R/MIR3_DONE
log "finished"
