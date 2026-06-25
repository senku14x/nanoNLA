#!/usr/bin/env bash
# Phase 5: pull the 8 AV wide-bank shards from the OLD H200 via rsync-over-SSH, then
# validate -> checksum -> manifest -> upload to Drive (same as AR/RL). Read-only on the
# old box: nothing is deleted there. Resumable.
#   required env: H200_HOST, H200_SSH_KEY    optional: H200_USER (default root)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
: "${H200_HOST:?set H200_HOST (old H200 ip/host)}"
: "${H200_SSH_KEY:?set H200_SSH_KEY (path to private key for the old H200)}"
H200_USER="${H200_USER:-root}"
REMOTE_DIR="/data/mlnla/published_L24x_window"
cd "$REPO"; GIT="$(git rev-parse HEAD)"; NN="$(printf '%02d' "$NUM_SHARDS")"
mkdir -p "$REGEN" "$MANIFESTS"

cat <<EOF
[import] If SSH auth fails, on THIS machine run:
  ssh-keygen -t ed25519 -f "$H200_SSH_KEY" -N ''
  cat "${H200_SSH_KEY}.pub"     # paste this line into the OLD H200's ~/.ssh/authorized_keys
EOF

for i in $(seq 0 $((NUM_SHARDS-1))); do
  II="$(printf '%02d' "$i")"; FNAME="av_sft.shard${II}of${NN}.parquet"; FINAL="$REGEN/$FNAME"
  if [ -f "$FINAL" ] && python -m multilayer_nla.ops.lib_finalize --parquet "$FINAL" --layers "$SAVE_LAYERS" --check >/dev/null 2>&1; then
    echo "[import] skip validated $FNAME"; continue
  fi
  echo "[import] rsync $FNAME from old H200..."
  rsync -aP -e "ssh -i '$H200_SSH_KEY' -o StrictHostKeyChecking=accept-new" \
    "${H200_USER}@${H200_HOST}:${REMOTE_DIR}/${FNAME}" "$FINAL.tmp"
  mv -f "$FINAL.tmp" "$FINAL"
  # AV shards were regenerated on the old box WITHOUT --length-bucket -> length_bucket=false
  python -m multilayer_nla.ops.lib_finalize --parquet "$FINAL" --layers "$SAVE_LAYERS" \
    --final-name "$FNAME" --git-commit "$GIT" --model "$MODEL" --center "$CENTER" \
    --max-length "$MAXLEN" --batch-size "$BATCH" --length-bucket false
  cp -f "$FINAL.manifest.json" "$MANIFESTS/$FNAME.manifest.json"
  rclone copyto "$FINAL"               "$DRIVE_REMOTE/shards/$FNAME"
  rclone copyto "$FINAL.sha256"        "$DRIVE_REMOTE/shards/$FNAME.sha256"
  rclone copyto "$FINAL.manifest.json" "$DRIVE_REMOTE/manifests/$FNAME.manifest.json"
  echo "[import] DONE + uploaded $FNAME"
done
cat <<'EOF'

!!! Do NOT destroy the old H200 until all eight AV shards have been copied,
    validated locally, AND verified uploaded to Google Drive (run audit_activation_bank.sh). !!!
EOF
