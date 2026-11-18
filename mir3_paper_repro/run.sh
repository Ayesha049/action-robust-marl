#!/usr/bin/env bash
# Reproduce MIR3's continuous-action claim (Li et al., TNNLS; arXiv 2310.09833, Sec. 4.2 rendezvous)
# with the authors' released code (github.com/DIG-Beihang/MIR3 @ e36d99b) plus released_code.patch.
#
# Claim tested: under a learned worst-case adversary that takes over one agent, a MIR3 team keeps
# more reward than a MADDPG team on 10-robot rendezvous.
# Protocol (paper Sec. 4.1, Tables 6-7): 10M steps, 5 seeds, MIR3 lambda 5e-5; the defender is frozen
# and one adversary is trained per victim agent (0..9) with the same seed -> 5 x 10 attacks per method.
#
# Usage (inside the patched clone):  STAGE2_JOBS=50 nohup bash run.sh > repro.log 2>&1 &
#   CFG=_paper TOKEN=paper   uses configs/algs/{maddpg,mir3}_robot_paper.yaml (max_grad_norm 0.5 as in
#                            paper Table 6; the released robot configs use 10, the SMAC value of Table 2)
# Only teams whose final clean return beats MIN_CLEAN (-40; untrained teams score about -69) are
# attacked, the same rule for both methods (set before the paper-config runs were seen).
set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mir3gpu
STEPS=${STEPS:-10000000}
LAM=${LAM:-0.00005}
SEEDS=${SEEDS:-"0 1 2 3 4"}
NGPU=${NGPU:-8}
STAGE2_JOBS=${STAGE2_JOBS:-50}
CFG=${CFG:-}
TOKEN=${TOKEN:-repro}
MIN_CLEAN=${MIN_CLEAN:--40}
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" >&2; }  # stderr: the command-building loops redirect stdout
final_ckpt() { ls -d results/robot/robot_10_new/none/$1/$TOKEN/$2/models/* 2>/dev/null | sort -t/ -k9 -n | tail -1; }
log "start cfg=[$CFG] token=$TOKEN code=$(git rev-parse --short HEAD) steps=$STEPS lam=$LAM seeds=[$SEEDS]"
git diff HEAD > logs/patch_at_launch_$TOKEN.diff

i=0
for m in maddpg mir3; do for s in $SEEDS; do
  f=logs/${TOKEN}_train_${m}_s$s.log
  extra=""; [ $m = mir3 ] && extra="--param_club $LAM"
  grep -q "timestep $STEPS/$STEPS\|Eval for 32 episodes at timestep 1[0-9]\{7\}" $f 2>/dev/null && continue
  echo "CUDA_VISIBLE_DEVICES=$((i % NGPU)) python -u main.py --alg ${m}_robot$CFG --env robot --token $TOKEN --seed $s --gpu cuda:0 --total_timesteps $STEPS $extra > $f 2>&1"
  i=$((i+1))
done; done > logs/${TOKEN}_train.cmds
xargs -P 10 -I{} bash -c "{}" < logs/${TOKEN}_train.cmds
log "training done"

i=0
for m in maddpg mir3; do for s in $SEEDS; do
  V=$(final_ckpt $m $s)
  [ -z "$V" ] && { log "missing checkpoint $m s$s"; continue; }
  C=$(grep "Mean of return" logs/${TOKEN}_train_${m}_s$s.log | tail -1 | awk '{print $4}')
  if ! awk -v c="$C" -v t="$MIN_CLEAN" 'BEGIN{exit !(c > t)}'; then log "skip $m s$s: clean $C <= $MIN_CLEAN"; continue; fi
  for a in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/${TOKEN}_adv_${m}_s${s}_a$a.log
    grep -q "Eval for 32 episodes at timestep 1[0-9]\{7\}" $f 2>/dev/null && continue
    echo "CUDA_VISIBLE_DEVICES=$((i % NGPU)) python -u main.py --alg maddpg_traitor_mi_robot --env robot --token ${TOKEN}_adv_${m}_s${s}_a$a --seed $s --gpu cuda:0 --adv_agent_id $a --total_timesteps $STEPS --victim_checkpoint $V/ > $f 2>&1"
    i=$((i+1))
  done
done; done > logs/${TOKEN}_adv.cmds
xargs -P $STAGE2_JOBS -I{} bash -c "{}" < logs/${TOKEN}_adv.cmds
log "attacks done"
touch logs/${TOKEN}_DONE
