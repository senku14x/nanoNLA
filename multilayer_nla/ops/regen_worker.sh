#!/usr/bin/env bash
# Phase 4 worker: regenerate the given shards of one subset on ONE GPU, then
# validate -> checksum -> manifest -> atomic-rename -> upload. Resumable.
#   usage: regen_worker.sh <gpu_id> <subset> <shard_csv>     e.g.  regen_worker.sh 0 ar_sft 0,1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
GPU="$1"; SUBSET="$2"; SHARDS="$3"
cd "$REPO"
GIT="$(git rev-parse HEAD)"; NN="$(printf '%02d' "$NUM_SHARDS")"
mkdir -p "$REGEN" "$LOGS" "$MANIFESTS"
[ -f "$PUB/${SUBSET}.parquet" ] || { echo "ERROR: missing $PUB/${SUBSET}.parquet (run download_published)" >&2; exit 1; }

for i in ${SHARDS//,/ }; do
  II="$(printf '%02d' "$i")"
  FINAL="$REGEN/${SUBSET}.shard${II}of${NN}.parquet"
  TMP_OUT="$REGEN/.work_gpu${GPU}_${SUBSET}.parquet"            # regen appends .shardIIofNN
  TMP="$REGEN/.work_gpu${GPU}_${SUBSET}.shard${II}of${NN}.parquet"

  # resume: skip only if FINAL exists AND fully validates (rows, L19-29 cols, sha256)
  if [ -f "$FINAL" ] && python -m multilayer_nla.ops.lib_finalize --parquet "$FINAL" --layers "$SAVE_LAYERS" --check >/dev/null 2>&1; then
    echo "[gpu$GPU] skip validated $FINAL"; continue
  fi
  rm -f "$TMP" "$TMP.sha256" "$TMP.manifest.json"
  echo "[gpu$GPU] regen $SUBSET shard $i/$NUM_SHARDS -> $TMP"
  CUDA_VISIBLE_DEVICES="$GPU" python -m multilayer_nla.regenerate_multilayer_activations \
    --in "$PUB/${SUBSET}.parquet" --out "$TMP_OUT" --base-model "$MODEL" \
    --center-layer "$CENTER" --save-layers "$SAVE_LAYERS" --max-length "$MAXLEN" \
    --batch-size "$BATCH" --length-bucket --max-drop-frac "$MAXDROP" \
    --num-shards "$NUM_SHARDS" --shard-index "$i"

  # validate + sha256 + manifest on the TMP (records the FINAL name); gates the rename
  python -m multilayer_nla.ops.lib_finalize --parquet "$TMP" --layers "$SAVE_LAYERS" \
    --final-name "$(basename "$FINAL")" --git-commit "$GIT" --model "$MODEL" \
    --center "$CENTER" --max-length "$MAXLEN" --batch-size "$BATCH" --length-bucket true

  # atomic rename ONLY after regen+validation succeed
  mv -f "$TMP"               "$FINAL"
  mv -f "$TMP.sha256"        "$FINAL.sha256"
  mv -f "$TMP.manifest.json" "$FINAL.manifest.json"
  cp -f "$FINAL.manifest.json" "$MANIFESTS/$(basename "$FINAL").manifest.json"

  # upload ONLY validated artifacts; copyto (never sync)
  rclone copyto "$FINAL"               "$DRIVE_REMOTE/shards/$(basename "$FINAL")"
  rclone copyto "$FINAL.sha256"        "$DRIVE_REMOTE/shards/$(basename "$FINAL").sha256"
  rclone copyto "$FINAL.manifest.json" "$DRIVE_REMOTE/manifests/$(basename "$FINAL").manifest.json"
  echo "[gpu$GPU] DONE + uploaded $(basename "$FINAL")"
done
echo "[gpu$GPU] worker finished: $SUBSET shards=$SHARDS"
