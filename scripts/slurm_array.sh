#!/usr/bin/env bash
# Submit commands from stdin as a SLURM job array (1 CPU, no GPU per task).
# Usage: scripts/cmds.sh train | scripts/slurm_array.sh NAME [MAX_CONCURRENT] [TIME]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1
MAXC=${2:-15}
TIME=${3:-08:00:00}
mkdir -p slurm
CMDS="slurm/${NAME}_$(date +%Y%m%d_%H%M%S).cmds"
cat > "$CMDS"
N=$(wc -l < "$CMDS")
[ "$N" -eq 0 ] && { echo "no commands"; exit 0; }
VENV=${VENV:-$HOME/venvs/marl0}
sbatch --parsable --job-name="$NAME" --partition=defq --cpus-per-task=1 --mem=4G \
  --time="$TIME" --array=1-"$N"%"$MAXC" --output="slurm/%x_%A_%a.out" \
  --wrap="source $VENV/bin/activate && cd $PWD && eval \"\$(sed -n \${SLURM_ARRAY_TASK_ID}p $CMDS)\""
echo "submitted $N tasks from $CMDS (max $MAXC concurrent)"
