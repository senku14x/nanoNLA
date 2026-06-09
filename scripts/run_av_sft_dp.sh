#!/bin/bash
# 2-rank data-parallel AV-SFT, 3 GPUs/copy (2-GPU copies are too tight for the
# train backward pass; 3-GPU gives ~110GB free / 20 layers each -> micro 16 fits).
# SEQUENTIAL loads (concurrent gptqmodel loads die); dist.init AFTER load.
set -u
PY=/workspace/venv-v5/bin/python
mkdir -p /workspace/logs
GPUSETS=("0,1,2" "3,4,5")
WORLD=2
for i in 0 1; do
  GPUS="${GPUSETS[$i]}"
  LOG=/workspace/logs/av_sft_rank$i.log
  echo "[launcher] loading rank $i on GPUs $GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" RANK=$i WORLD=$WORLD MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
      setsid $PY /workspace/397b_av_sft_dp.py > "$LOG" 2>&1 &
  PID=$!
  for w in $(seq 1 140); do        # up to ~70 min for this rank to load
    sleep 30
    if grep -q "model loaded" "$LOG" 2>/dev/null; then echo "[launcher] rank $i loaded"; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "[launcher] rank $i DIED during load"; break; fi
  done
done
echo "[launcher] all ranks launched; dist syncs them post-load, then training"
wait
echo "[launcher] === AV-SFT-DP COMPLETE ==="
