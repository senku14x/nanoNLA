#!/usr/bin/env bash
# Rebuild-fix for the ONE truncated condition: retrain av_s2_19_21_23 on the now-full
# parquet, then re-run its 3 eval cuts — in the correct order, with hard preconditions so
# it can't eval a missing adapter or a stale-data parquet. Run ON THE H200.
#
#   BASE=Qwen/Qwen3-8B AR=<3tap AR iter dir> AR_L24=<L24-only AR iter dir> \
#     bash multilayer_nla/scripts/fix_s2_19_21_23.sh
#
# Everything else (the other 5 conditions, the shared AR) is untouched — their eval outputs
# stay valid byte-for-byte. After this, re-run analyze_sweep + make_datacard (cheap, CPU).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

C=s2_19_21_23
BASE="${BASE:?set BASE (e.g. Qwen/Qwen3-8B)}"
DATA="${DATA:-/data/mlnla}"
SWEEP="${SWEEP:-$DATA/sweep}"
CKPT="${CKPT:-$DATA/sweep_ckpt}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"
AR="${AR:?set AR = the 3-tap shared AR iter dir (the one used in the converged eval)}"
AR_L24="${AR_L24:?set AR_L24 = the L24-only AR iter dir (the one used in test_arL24)}"
EVAL_BATCH="${EVAL_BATCH:-1024}"
AVDIR="$CKPT/av_$C"
AV="$AVDIR/iter_0001000"

# ── precondition 1: the parquet must be the FULL size (== av_local), not the truncated one ──
exp=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$SWEEP/av_local.parquet').metadata.num_rows)")
got=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$SWEEP/av_$C.parquet').metadata.num_rows)")
[ "$got" = "$exp" ] || { echo "ABORT: av_$C.parquet has $got rows, expected $exp (== av_local). Rebuild it first."; exit 1; }
echo "[fix] av_$C.parquet OK: $got rows (matches av_local)"

# ── precondition 2: the AR ckpts must exist ──
[ -f "$AR/adapter_config.json" ]     || { echo "ABORT: AR ckpt not found at $AR";      exit 1; }
[ -f "$AR_L24/adapter_config.json" ] || { echo "ABORT: AR_L24 ckpt not found at $AR_L24"; exit 1; }

# ── clean slate: drop the (absent) ckpt + any stale s2_19_21_23 eval outputs so nothing is skipped ──
rm -rf "$AVDIR"
rm -f "$EVALC"/dev/dev_${C}_ar3000_av1000.{json,jsonl} \
      "$EVALC"/test/test_${C}.{json,jsonl} \
      "$EVALC"/test_arL24/test_${C}.{json,jsonl}

# ── 1. retrain the AV (defaults == the matched recipe: batch 64, accum 1, 1000 steps, seed 0) ──
python -m multilayer_nla.train_av_multi \
  --base-ckpt "$BASE" --use-lora --quant none \
  --parquet "$SWEEP/av_$C.parquet" --save-dir "$AVDIR" \
  --num-steps 1000 --save-every 500 --batch-size 64 --gradient-accumulation-steps 1 \
  --seed 0 --wandb-project "multi layer nla" --wandb-name "av-$C"

# ── HARD GATE: do not eval unless the adapter actually got written ──
[ -f "$AV/adapter_config.json" ] || { echo "ABORT: retrain did not produce $AV/adapter_config.json"; exit 1; }
echo "[fix] adapter present at $AV — proceeding to eval"

# ── 2. re-eval the 3 cuts (dev 3-tap, test 3-tap, test L24-only) ──
python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR" \
  --eval-parquet "$SWEEP/rl_dev_$C.parquet"  --condition "$C" --batch-size "$EVAL_BATCH" --seed 0 \
  --out "$EVALC/dev/dev_${C}_ar3000_av1000.jsonl" --summary "$EVALC/dev/dev_${C}_ar3000_av1000.json"
python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR" \
  --eval-parquet "$SWEEP/rl_test_$C.parquet" --condition "$C" --batch-size "$EVAL_BATCH" --seed 0 \
  --out "$EVALC/test/test_${C}.jsonl" --summary "$EVALC/test/test_${C}.json"
python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR_L24" \
  --eval-parquet "$SWEEP/rl_test_$C.parquet" --condition "$C" --batch-size "$EVAL_BATCH" --seed 0 \
  --out "$EVALC/test_arL24/test_${C}.jsonl" --summary "$EVALC/test_arL24/test_${C}.json"

echo "[fix] DONE — $C retrained + re-evaled. Next: re-run analyze_sweep + make_datacard (CPU)."
