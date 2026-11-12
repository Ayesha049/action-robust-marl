#!/usr/bin/env bash
# Print one shell command per job (skipping finished ones) for a Phase 0 stage.
#   scripts/cmds.sh train   # env: TASKS METHODS SEEDS STEPS OUT
#   scripts/cmds.sh eval    # eval critic + R/G curves for every finished run in OUT
#   scripts/cmds.sh adv     # learned adversaries (full sweep: both tasks, PR/NR a0.1/0.3, one/all, + PR takeover)
# Pipe the output to scripts/run_local.sh or scripts/slurm_array.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=$1
OUT=${OUT:-runs}
TASKS=${TASKS:-"spread tag"}
METHODS=${METHODS:-"maddpg m3ddpg ernie pr nr marta"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
STEPS=${STEPS:-1500000}
ADV_SEEDS=${ADV_SEEDS:-"0 1"}
ADV_TASKS=${ADV_TASKS:-"spread tag"}
EXTRA=${EXTRA:-}

case $STAGE in
train)
  for t in $TASKS; do for m in $METHODS; do for s in $SEEDS; do
    d="$OUT/${t}__${m}__s${s}"
    [ -f "$d/DONE" ] && continue
    echo "mkdir -p $d && python train.py --task $t --method $m --seed $s --steps $STEPS --out-dir $OUT $EXTRA > $d/train.out 2>&1"
  done; done; done ;;
eval)
  for done_f in "$OUT"/*/DONE; do
    d=$(dirname "$done_f")
    [ -f "$d/eval_curves.csv" ] && continue
    echo "( [ -f $d/eval_critic.pt ] || python eval_critic.py --run-dir $d ) > $d/eval.out 2>&1 && python eval_curves.py --run-dir $d >> $d/eval.out 2>&1"
  done ;;
adv)
  for done_f in "$OUT"/*/DONE; do
    d=$(dirname "$done_f")
    t=$(basename "$d" | cut -d_ -f1)
    case " $ADV_TASKS " in *" $t "*) ;; *) continue ;; esac
    for cell in "pr 0.1 one" "pr 0.1 all" "pr 0.3 one" "pr 0.3 all" \
                "nr 0.1 one" "nr 0.1 all" "nr 0.3 one" "nr 0.3 all" "pr 1.0 one"; do
      set -- $cell
      for as in $ADV_SEEDS; do
        [ -f "$d/adv/${1}_a${2}_${3}_s${as}.json" ] && continue
        echo "python train_adversary.py --run-dir $d --op $1 --alpha $2 --scope $3 --seed $as > /dev/null 2>> $d/adv_errors.log"
      done
    done
  done ;;
*) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
