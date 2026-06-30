#!/usr/bin/env bash
# Progressive Reader v0 — THE HEADLINE MATRIX. Run AFTER bootstrap's audit gate passes
# (coverage@96 >= 50%) and the smoke run is green. Sequential on one GPU; resumable at
# RUN granularity (skips a stage whose checkpoint/eval already exists).
#
#   REGEN=/data/mlnla/bank bash multilayer_nla/scripts/run_progressive_matrix.sh
#   GC=--gradient-checkpointing REGEN=... bash ... run_progressive_matrix.sh   # if micro-batch 32 OOMs
#
# THREE trainings (Flat × {layer_balanced} is degenerate with Flat × stage_mean — uniform
# supervision counts — so it is intentionally NOT run):
#   flat      = configs/..._flat.yaml          (primary baseline; loss mode is irrelevant for flat)
#   prog_sm   = configs/..._progressive.yaml   --loss-mode progressive_stage_mean
#   prog_lb   = configs/..._progressive.yaml   --loss-mode progressive_layer_balanced
# THREE evals on held-out TEST (real / no_text / shuffled controls, doc-level bootstrap CIs):
#   eval flat FIRST (its test_per_example.jsonl is the paired-compare baseline), then both prog
#   arms with --compare-to flat -> bootstrap_comparisons.json (ΔG_local / ΔG_outer vs Flat).
#
# Headline reads (per docs/progressive_reader_v0.md pre-registration):
#   * hierarchy real  <=>  prog_lb G_outer steeper than G_local AND ΔG_outer(prog_lb−flat) CI>0
#                          AND real−shuffled>0 at the outer cells.
#   * null            <=>  budget-monotone but depth-flat AND Progressive≈Flat.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export REGEN="${REGEN:-/data/mlnla/bank}"
ROOT="${ROOT:-runs/progressive_reader_v0}"
PROG_CFG="${PROG_CFG:-configs/progressive_reader_v0_progressive.yaml}"
FLAT_CFG="${FLAT_CFG:-configs/progressive_reader_v0_flat.yaml}"
GC="${GC:-}"                       # set GC=--gradient-checkpointing if micro-batch 32 OOMs
SPLIT="${SPLIT:-test}"
# wandb: OFF by default (metrics are still written to disk: train_log.jsonl + dev_matrix_step*.json).
# WANDB=1 -> live dashboard (run `wandb login` first, or export WANDB_API_KEY / WANDB_MODE=offline).
WANDB="${WANDB:-0}"
NW="--no-wandb"; [ "$WANDB" = 1 ] && NW=""
mkdir -p "$ROOT/logs"

train_one () {  # $1 config  $2 loss-mode  $3 run-dir
  if [ -f "$3/best/reader.safetensors" ]; then echo "[matrix] SKIP train $3 (checkpoint exists)"; return; fi
  echo "[matrix] TRAIN $3  ($2)"
  python -m multilayer_nla.progressive_reader.train --config "$1" \
    --loss-mode "$2" --run-dir "$3" $NW $GC 2>&1 | tee "$ROOT/logs/train_$(basename "$3").log"
}

eval_one () {  # $1 config  $2 run-dir  $3 out  [$4 compare-to]
  if [ -f "$3/${SPLIT}_matrix.json" ]; then echo "[matrix] SKIP eval $3 (matrix exists)"; return; fi
  local cmp=(); [ -n "${4:-}" ] && cmp=(--compare-to "$4")
  echo "[matrix] EVAL  $3  (split=$SPLIT${4:+, compare-to=$(basename "$(dirname "$4")")})"
  python -m multilayer_nla.progressive_reader.evaluate --checkpoint "$2/best" \
    --config "$1" --split "$SPLIT" --out "$3" "${cmp[@]}" 2>&1 | tee "$ROOT/logs/eval_$(basename "$3").log"
}

# 1) TRAIN (flat first so its per_example.jsonl exists before the prog evals compare against it)
train_one "$FLAT_CFG" progressive_stage_mean      "$ROOT/flat"
train_one "$PROG_CFG" progressive_stage_mean      "$ROOT/prog_sm"
train_one "$PROG_CFG" progressive_layer_balanced  "$ROOT/prog_lb"

# 2) EVAL on held-out TEST. Flat first; both prog arms paired-compare vs flat's REAL records.
FLAT_REAL="$ROOT/eval/flat/${SPLIT}_per_example.jsonl"
eval_one "$FLAT_CFG" "$ROOT/flat"    "$ROOT/eval/flat"
eval_one "$PROG_CFG" "$ROOT/prog_sm" "$ROOT/eval/prog_sm" "$FLAT_REAL"
eval_one "$PROG_CFG" "$ROOT/prog_lb" "$ROOT/eval/prog_lb" "$FLAT_REAL"

echo "========================================================================"
echo "[matrix] DONE. Read:"
echo "  $ROOT/eval/flat/summary.md"
echo "  $ROOT/eval/prog_sm/summary.md   + $ROOT/eval/prog_sm/bootstrap_comparisons.json (ΔG vs Flat)"
echo "  $ROOT/eval/prog_lb/summary.md   + $ROOT/eval/prog_lb/bootstrap_comparisons.json (ΔG vs Flat)  <- HEADLINE"
echo "  per-cell matrices: $ROOT/eval/*/${SPLIT}_matrix.json   heatmaps: $ROOT/eval/*/plots"
echo "========================================================================"
