#!/bin/bash
# 3-copy data-parallel datagen with SEQUENTIAL loads (concurrent loads die on this
# box at the kernel-select->weight-load transition). Load shards one at a time;
# each starts datagen as soon as it's loaded, so extraction still runs 3x concurrent.
set -u
PY=/workspace/venv-v5/bin/python
mkdir -p /workspace/logs /workspace/nla-ckpts
for i in 0 1 2; do
  GPUS="$((2*i)),$((2*i+1))"
  LOG=/workspace/logs/datagen_shard$i.log
  echo "[launcher] loading shard $i on GPUs $GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid $PY /workspace/397b_datagen_shard.py \
      --shard $i --nshards 3 --n-data 250000 --batch 64 > $LOG 2>&1 &
  PID=$!
  # block until THIS shard finishes loading (or dies) before launching the next
  for w in $(seq 1 140); do            # up to ~70 min
    sleep 30
    if grep -q "model loaded" "$LOG" 2>/dev/null; then echo "[launcher] shard $i loaded"; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "[launcher] shard $i DIED during load"; break; fi
  done
done
echo "[launcher] all shards launched; waiting for datagen to finish"
wait
echo "[launcher] === merging ==="
# no --expect-total: the over-length drop filter makes the post-filter total
# data-dependent (< n-data); merge cross-checks completeness via shardmeta_*.json
$PY /workspace/merge_shards.py > /workspace/logs/merge.log 2>&1
cat /workspace/logs/merge.log
echo "[launcher] === DATAGEN-DP COMPLETE ==="
