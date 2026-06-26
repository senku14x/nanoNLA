#!/usr/bin/env bash
# Regenerate the §7 held-out EVAL outputs (test_<cond>.jsonl incl. generated_text, the
# qualitative explanations) from the HF bank + checkpoints. DETERMINISTIC; NO retraining.
#
# Mirrors run_sweep.sh steps 1-2 (splits + build) and 10-11 (one-shot test eval + report)
# using the already-SELECTED checkpoints (AR@1000 + AV@1000 per result_table.md), so it
# skips the dev grid + selection. The split is rebuilt with the SAME seed/subsets, and the
# eval is greedy, so the regenerated text matches the original run bit-for-bit (modulo rare
# GPU float drift). Idempotent: every step skips if its output exists.
#
#   BASE=Qwen/Qwen3-8B DATA=/data/mlnla EVAL_BATCH=256 bash multilayer_nla/scripts/regen_eval.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── config (override via env) ───────────────────────────────────────────
BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
REGEN="${REGEN:-$DATA/bank}"          # the L19-29 activation bank (HF dataset, downloaded here)
SWEEP="${SWEEP:-$DATA/sweep}"         # rebuilt split manifests + datasets
CKPT="${CKPT:-$DATA/sweep_ckpt}"      # AR + per-condition AV checkpoints (HF model repo)
EVAL="${EVAL:-$DATA/sweep_eval}"      # regenerated summaries + per-example jsonl + table
EVAL_BATCH="${EVAL_BATCH:-256}"       # generation batch (pure speed; greedy => identical results)
SPLIT_SEED="${SPLIT_SEED:-42}"; DEV_SUBSET="${DEV_SUBSET:-256}"; TEST_SUBSET="${TEST_SUBSET:-1000}"
SEED="${SEED:-0}"                     # eval seed (matched to the original run)
AR_STEP="${AR_STEP:-1000}"; AV_STEP="${AV_STEP:-1000}"   # the selected checkpoints
HF_DATASET="${HF_DATASET:-senku21x/qwen3-8b-nla-multilayer-L19-29}"
HF_CKPTS="${HF_CKPTS:-senku21x/qwen3-8b-nla-multilayer-sweep}"
CONDS=(local duplicate wide single)
iterdir() { printf '%s/iter_%07d' "$1" "$2"; }   # trainers save to iter_{step+1:07d}
mkdir -p "$SWEEP" "$EVAL/test"

# ── 0. fetch bank + checkpoints (SKIP_DOWNLOAD=1 if already present) ─────
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  [ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$REGEN"
  huggingface-cli download "$HF_CKPTS"                      --local-dir "$CKPT"
fi
echo "[check] checkpoint tree under $CKPT:"; find "$CKPT" -maxdepth 2 -type d | sort
[ -f "$(iterdir "$CKPT/ar" "$AR_STEP")/ar_meta.json" ] || {
  echo "!! AR ckpt not at $(iterdir "$CKPT/ar" "$AR_STEP") — adjust CKPT/AR_STEP to the layout above"; exit 1; }
for c in "${CONDS[@]}"; do
  [ -f "$(iterdir "$CKPT/av_$c" "$AV_STEP")/adapter_config.json" ] || {
    echo "!! AV[$c] ckpt missing at $(iterdir "$CKPT/av_$c" "$AV_STEP")"; exit 1; }
done

# ── 1. deterministic splits + datasets (SAME seed/subsets as the original run) ──
[ -f "$SWEEP/rl_split_manifest.json" ] || python -m multilayer_nla.splits \
    --source "$REGEN/rl.shard*of*.parquet" --name rl --out-dir "$SWEEP" --seed "$SPLIT_SEED" \
    --fracs 0.8,0.1,0.1 --dev-subset "$DEV_SUBSET" --test-subset "$TEST_SUBSET"
[ -f "$SWEEP/ar_split_manifest.json" ] || python -m multilayer_nla.splits \
    --source "$REGEN/ar_sft.shard*of*.parquet" --name ar --out-dir "$SWEEP" --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1
[ -f "$SWEEP/sweep_build_manifest.json" ] || python -m multilayer_nla.build_sweep --mode all \
    --in-dir "$REGEN" --out-dir "$SWEEP" --rl-split-manifest "$SWEEP/rl_split_manifest.json" \
    --ar-split-manifest "$SWEEP/ar_split_manifest.json" --ar-target-layers 23,24,25 \
    --base-ckpt "$BASE" --allow-existing

# ── 2. ONE-SHOT test eval per condition -> test_<cond>.jsonl (generated_text) ──
for c in "${CONDS[@]}"; do
  [ -f "$EVAL/test/test_$c.json" ] || python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" \
      --av-ckpt "$(iterdir "$CKPT/av_$c" "$AV_STEP")" --ar-ckpt "$(iterdir "$CKPT/ar" "$AR_STEP")" \
      --eval-parquet "$SWEEP/rl_test_$c.parquet" --condition "$c" --batch-size "$EVAL_BATCH" \
      --out "$EVAL/test/test_$c.jsonl" --summary "$EVAL/test/test_$c.json" --seed "$SEED"
done

# ── 3. AR-only gold ceiling + result table ──────────────────────────────
[ -f "$EVAL/test/ar_gold_dev.json" ]  || python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" \
    --ar-ckpt "$(iterdir "$CKPT/ar" "$AR_STEP")" --eval-parquet "$SWEEP/ar_dev.parquet" \
    --summary "$EVAL/test/ar_gold_dev.json"  --batch-size "$EVAL_BATCH"
[ -f "$EVAL/test/ar_gold_test.json" ] || python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" \
    --ar-ckpt "$(iterdir "$CKPT/ar" "$AR_STEP")" --eval-parquet "$SWEEP/ar_test.parquet" \
    --summary "$EVAL/test/ar_gold_test.json" --batch-size "$EVAL_BATCH"
cat > "$EVAL/selection.json" <<JSON
{"metric":"pen_fve_overall","chosen_ar_step":$AR_STEP,"chosen_av_step":{"local":$AV_STEP,"duplicate":$AV_STEP,"wide":$AV_STEP,"single":$AV_STEP}}
JSON
python -m multilayer_nla.select_and_report --mode report --test-dir "$EVAL/test" \
    --selection "$EVAL/selection.json" --ar-gold-dev "$EVAL/test/ar_gold_dev.json" \
    --ar-gold-test "$EVAL/test/ar_gold_test.json" --out "$EVAL/result_table.md"

# ── 4. push eval outputs back to HF so they survive the box (SKIP_UPLOAD=1 to skip) ──
[ -n "${SKIP_UPLOAD:-}" ] || huggingface-cli upload "$HF_DATASET" "$EVAL" sweep_eval --repo-type dataset
echo "[regen] DONE -> $EVAL/test/test_<cond>.jsonl (generated_text preserved); table -> $EVAL/result_table.md"
echo "[regen] sanity: regenerated test_<cond>.json FVE should match result_table.md to the decimal."
