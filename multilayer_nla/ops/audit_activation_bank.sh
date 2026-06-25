#!/usr/bin/env bash
# Phase 6: verify all 24 wide-bank shards (8 av + 8 ar + 8 rl) — rows, L19-29 columns,
# local SHA256 (vs sidecar), and local-vs-Drive checksum. Prints a PASS/FAIL summary.
# Does NOT merge shards, train, or delete anything.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
cd "$REPO"; NN="$(printf '%02d' "$NUM_SHARDS")"; FAIL=0; TMP="$(mktemp)"

echo "=== local validation (rows, activation_L${SAVE_LAYERS} columns, sha256 sidecar) ==="
for SUBSET in av_sft ar_sft rl; do
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    II="$(printf '%02d' "$i")"; FNAME="${SUBSET}.shard${II}of${NN}.parquet"; FINAL="$REGEN/$FNAME"
    if python -m multilayer_nla.ops.lib_finalize --parquet "$FINAL" --layers "$SAVE_LAYERS" --check >"$TMP" 2>&1; then
      echo "  PASS(local) $FNAME — $(tr -d '\n' <"$TMP")"
    else
      echo "  FAIL(local) $FNAME — $(tr -d '\n' <"$TMP")"; FAIL=1
    fi
  done
done

echo "=== Drive checksum (rclone check; uses the backend hash, e.g. Drive MD5) ==="
# one-way: every local shard must match its Drive copy
if rclone check "$REGEN" "$DRIVE_REMOTE/shards" --one-way --include "*.parquet" 2>"$TMP"; then
  echo "  PASS(drive) all local shards match Google Drive"
else
  echo "  FAIL(drive): differences vs Drive:"; sed 's/^/    /' "$TMP"; FAIL=1
fi

rm -f "$TMP"
if [ "$FAIL" = 0 ]; then
  echo "[audit] ALL 24 SHARDS PASS (local + Drive)."
else
  echo "[audit] FAILURES PRESENT — do NOT destroy any machine."; exit 1
fi
