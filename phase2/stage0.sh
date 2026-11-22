#!/usr/bin/env bash
# Phase 2 Stage 0: persistent-fault headroom on the Phase 0 checkpoints of TASK (spread|tag), seeds 0-4.
# Seeds 5-9 are held out for the Stage 2 confirmation and are not touched here.
#   MADDPG and PR-MADDPG: every oracle predictor, one job per (team, fault kind)
#   other Phase 0 methods: no correction only, one job per team
# Usage (repo root): JOBS=60 nohup bash phase2/stage0.sh > phase2/stage0.log 2>&1 &
set -u
TASK=${TASK:-spread}
STRENGTHS=${STRENGTHS:-}  # e.g. "bias=0.1,0.3,0.5,0.7,0.9", passed to stage0_faults --strengths
EPISODES=${EPISODES:-100}
RUNS=${RUNS:-runs}  # directory holding <task>__<method>__s<seed> checkpoints
KINDS=${KINDS:-"clean gain bias drift stuck wdir"}
RULE=${RULE:-gain,bias}
ORACLE_METHODS=${ORACLE_METHODS:-"maddpg pr"}  # methods evaluated with every oracle predictor
BASE_METHOD=${BASE_METHOD:-maddpg}            # the method the stop rule reads
OUT=${OUT:-runs_phase2/stage0}
JOBS=${JOBS:-32}
SEEDS=${SEEDS:-"0 1 2 3 4"}
METHODS=${METHODS:-"maddpg pr nr m3ddpg ernie marta mir3"}
# absolute interpreter: a fresh `bash -c` can resolve `python` to conda base instead of marl0
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
mkdir -p $OUT/logs
log() { echo "[$(date '+%F %T')] $*"; }

: > $OUT/jobs.cmds
for s in $SEEDS; do
  for m in $METHODS; do
    d=$RUNS/${TASK}__${m}__s$s
    if [ ! -f $d/actor_0.pt ]; then log "missing $d, skipped"; continue; fi
    if [[ " $ORACLE_METHODS " == *" $m "* ]]; then
      for k in $KINDS; do
        echo "$PY -m phase2.stage0_faults --run-dir $d --out $OUT --episodes $EPISODES --kinds $k ${STRENGTHS:+--strengths \"$STRENGTHS\"} > $OUT/logs/${TASK}_${m}_s${s}_$k.log 2>&1" >> $OUT/jobs.cmds
      done
    else
      echo "$PY -m phase2.stage0_faults --run-dir $d --out $OUT --episodes $EPISODES --predictors none --kinds $(echo $KINDS | tr " " ,) ${STRENGTHS:+--strengths \"$STRENGTHS\"} > $OUT/logs/${TASK}_${m}_s$s.log 2>&1" >> $OUT/jobs.cmds
    fi
  done
done
log "launching $(wc -l < $OUT/jobs.cmds) jobs on $JOBS workers"
xargs -P $JOBS -I{} bash -c "{}" < $OUT/jobs.cmds
log "jobs done: $(ls $OUT/*.csv 2>/dev/null | wc -l) CSVs; errors in: $(grep -l -i error $OUT/logs/*.log 2>/dev/null | wc -l) logs"
$PY -m phase2.stage0_summary --dir $OUT --rule-kinds $RULE --base-method $BASE_METHOD --phase1-stage0 ${P1S0:-runs_phase1/stage0} > $OUT/summary_printout.txt 2>&1
log "summary written to $OUT/summary_printout.txt"
touch $OUT/DONE
