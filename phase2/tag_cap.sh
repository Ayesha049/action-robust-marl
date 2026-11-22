#!/usr/bin/env bash
# Phase 2 reserved-authority bases on tag (decision 2026-11-21): MADDPG trained with team actions
# scaled by act_cap c before the actuator (which still accepts [-1,1]), Phase 0 tag settings
# otherwise (1.5M steps, frozen Phase 0 prey). Development seeds 0-4; 5-9 stay held out.
# Then a clean eval (vs the uncapped Phase 0 bases) and the tag Stage 0 on each cap.
# Usage (repo root): nohup bash phase2/tag_cap.sh > phase2_tag_cap.log 2>&1 &
set -u
CAPS=${CAPS:-"0.5 0.7"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PY=${PY:-$HOME/miniconda3/envs/marl0/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
: > runs_phase2/tag_cap.cmds
for c in $CAPS; do
  mkdir -p runs_phase2/tag_cap$c
  for s in $SEEDS; do
    echo "$PY train.py --task tag --method maddpg --seed $s --act-cap $c --out-dir runs_phase2/tag_cap$c > runs_phase2/tag_cap$c/train_s$s.out 2>&1" >> runs_phase2/tag_cap.cmds
  done
done
log "training $(wc -l < runs_phase2/tag_cap.cmds) runs"
xargs -P 20 -I{} bash -c "{}" < runs_phase2/tag_cap.cmds
log "training done"
for c in $CAPS; do
  TASK=tag RUNS=runs_phase2/tag_cap$c EPISODES=500 STRENGTHS="bias=0.1,0.3,0.5,0.7,0.9" METHODS=maddpg \
    OUT=runs_phase2/tag_cap$c/stage0 JOBS=30 PY=$PY bash phase2/stage0.sh > runs_phase2/tag_cap$c/stage0.log 2>&1 &
done
wait
log "stage 0 done"
touch runs_phase2/tag_cap.DONE
