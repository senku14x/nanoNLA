#!/usr/bin/env bash
# Phase 3 gate: 2,048-row slice of ar_sft -> baseline (no bucket) vs bucketed on GPU 0,
# identical settings otherwise. Compares row identity + metadata order + numerics, and
# reports wall time. Decision rule is enforced by bench_compare.py (it exits non-zero to
# STOP). Does NOT launch the full AR/RL run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
cd "$REPO"
B="$WORK_ROOT/bench"; mkdir -p "$B"
[ -f "$PUB/ar_sft.parquet" ] || { echo "ERROR: missing $PUB/ar_sft.parquet (run download_published)" >&2; exit 1; }

python - <<PY
import pyarrow.parquet as pq
t = pq.read_table("$PUB/ar_sft.parquet").slice(0, 2048)
pq.write_table(t, "$B/ar_slice.parquet"); print("slice rows:", t.num_rows)
PY

run () {  # $1 = tag ; $2.. = extra flags
  local tag="$1"; shift
  echo "=== $tag ==="
  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python -m multilayer_nla.regenerate_multilayer_activations \
    --in "$B/ar_slice.parquet" --out "$B/${tag}.parquet" --base-model "$MODEL" \
    --center-layer "$CENTER" --save-layers "$SAVE_LAYERS" --max-length "$MAXLEN" \
    --batch-size "$BATCH" --max-drop-frac "$MAXDROP" "$@" 2>&1 | tee "$B/${tag}.log"
}
# --num-shards defaults to 1 -> writes directly to --out (no shard suffix)
run baseline
run bucketed --length-bucket

echo; echo "=== wall-clock (incl. ~constant model load; the DELTA reflects per-row speedup) ==="
for tag in baseline bucketed; do
  W=$(grep -m1 'wall clock' "$B/${tag}.log" | sed 's/.*: //')
  echo "  $tag: wall=$W"
done
echo; echo "=== numerical + identity comparator (enforces the STOP rule) ==="
python -m multilayer_nla.ops.bench_compare --baseline "$B/baseline.parquet" \
  --bucketed "$B/bucketed.parquet" --layers "$SAVE_LAYERS"
echo "[bench] If the comparator printed PASS and bucketed is materially faster, proceed to launch_4xh200.sh."
