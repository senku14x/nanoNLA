#!/usr/bin/env bash
# Phase 4 launcher: one tmux worker per GPU, each does its ar_sft shards then its rl
# shards. GPU->shard map per spec:
#   GPU0: 0,1   GPU1: 2,3   GPU2: 4,5   GPU3: 6,7   (for both ar_sft and rl)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
W="$HERE/regen_worker.sh"
declare -A MAP=( [0]="0,1" [1]="2,3" [2]="4,5" [3]="6,7" )
for g in 0 1 2 3; do
  S="${MAP[$g]}"; LOG="$LOGS/regen-gpu${g}.log"
  tmux kill-session -t "regen-gpu${g}" 2>/dev/null || true
  tmux new-session -d -s "regen-gpu${g}" \
    "bash '$W' $g ar_sft '$S' 2>&1 | tee -a '$LOG'; bash '$W' $g rl '$S' 2>&1 | tee -a '$LOG'; echo '=== DONE gpu${g} ===' | tee -a '$LOG'"
  echo "launched tmux regen-gpu${g}: ar_sft+rl shards $S  -> $LOG"
done
cat <<EOF

[launch] 4 workers running. Monitor:
  tmux ls
  nvidia-smi
  tail -f $LOGS/regen-gpu0.log    # …gpu1 / gpu2 / gpu3
  rclone lsf $DRIVE_REMOTE/shards

Each GPU runs sequentially: its ar_sft shards, then its rl shards. Re-running this
launcher is safe — workers skip any shard that already exists AND validates.
EOF
