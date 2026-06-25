#!/usr/bin/env bash
# §7 SFT control sweep — ONE H200, sequential. NO regen. NO RL. Does not touch the
# smoke / coherent dirs (everything lands under $SWEEP, $CKPT, $EVAL).
#
# Question: does multi-layer AV *input* improve end-to-end reconstruction of the SAME
# fixed target state [L23,L24,L25]? The AR target is identical for every condition;
# only the AV input slots vary (local 23/24/25, duplicate 24/24/24, wide 20/24/28,
# single 24). Primary test: local vs duplicate. Do NOT claim a winner before the
# held-out TEST table exists.
#
# Prereq: the regenerated wide bank (L19-L29 shards with doc_id) already exists at
# $REGEN — this script does NOT regenerate. Run order: split -> build -> shared AR ->
# 4x AV -> dev e2e grid -> select (dev only) -> one-shot test -> report -> STOP.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── config (override via env) ───────────────────────────────────────
BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
REGEN="${REGEN:-$DATA/published_L24x_window}"     # EXISTING wide bank (L19-29 shards). Not regenerated.
SWEEP="${SWEEP:-$DATA/sweep}"                     # built datasets + split manifests (NOT the smoke dir)
CKPT="${CKPT:-$DATA/sweep_ckpt}"                  # AR + per-condition AV checkpoints
EVAL="${EVAL:-$DATA/sweep_eval}"                  # dev/test summaries + result table
WANDB_PROJECT="${WANDB_PROJECT:-multi layer nla}"   # same project as the coherent run
SEED="${SEED:-0}"                                 # SFT + eval seed (matched across conditions)
SPLIT_SEED="${SPLIT_SEED:-42}"
DEV_SUBSET="${DEV_SUBSET:-256}"                   # locked dev docs for ckpt selection (cheap)
TEST_SUBSET="${TEST_SUBSET:-1000}"               # locked test docs for the one-shot final eval
STEPS="${STEPS:-1000}"; SAVE_EVERY="${SAVE_EVERY:-500}"   # ckpts at 500 and 1000
AR_TAPS="23,24,25"                                # AR reconstruction target — FIXED for every condition
CONDS=(local duplicate wide single)
mkdir -p "$SWEEP" "$CKPT" "$EVAL"

iterdir() { printf '%s/iter_%07d' "$1" "$2"; }    # trainers save to iter_{step+1:07d}
# Matched SFT hyperparameters — IDENTICAL for the shared AR and all four AV runs.
SFT_COMMON=(--base-ckpt "$BASE" --use-lora --quant none --num-steps "$STEPS" \
            --save-every "$SAVE_EVERY" --seed "$SEED" --wandb-project "$WANDB_PROJECT")

# Each step skips if its output already exists, so a re-run RESUMES after a crash/OOM.

# ── 1. document-level splits (rl for end-to-end, ar for AR-only gold) ──
[ -f "$SWEEP/rl_split_manifest.json" ] || \
  python -m multilayer_nla.splits --source "$REGEN/rl.shard*of*.parquet" --name rl \
      --out-dir "$SWEEP" --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1 \
      --dev-subset "$DEV_SUBSET" --test-subset "$TEST_SUBSET"
[ -f "$SWEEP/ar_split_manifest.json" ] || \
  python -m multilayer_nla.splits --source "$REGEN/ar_sft.shard*of*.parquet" --name ar \
      --out-dir "$SWEEP" --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1

# ── 2. build datasets (ar_common/dev/test, av_<cond>, rl_dev/test_<cond>) + preflight ──
#     The preflight HALTS the run on any data defect (layer placement, marker count,
#     fixed-target drift, split leak) — a green build means the datasets are correct.
[ -f "$SWEEP/sweep_build_manifest.json" ] || \
  python -m multilayer_nla.build_sweep --mode all --in-dir "$REGEN" --out-dir "$SWEEP" \
      --rl-split-manifest "$SWEEP/rl_split_manifest.json" \
      --ar-split-manifest "$SWEEP/ar_split_manifest.json" --ar-target-layers "$AR_TAPS" \
      --base-ckpt "$BASE" --allow-existing

# ── 3. shared AR (train once on ar_train; gold dev FVE logged during training) ──
[ -d "$(iterdir "$CKPT/ar" "$STEPS")" ] || \
  python -m multilayer_nla.train_ar_multi "${SFT_COMMON[@]}" --parquet "$SWEEP/ar_common.parquet" \
      --save-dir "$CKPT/ar" --tap-layers "$AR_TAPS" --eval-parquet "$SWEEP/ar_dev.parquet" \
      --wandb-name ar-shared

# ── 4-7. four AV conditions (matched settings; the condition lives in the data) ──
for c in "${CONDS[@]}"; do
  [ -d "$(iterdir "$CKPT/av_$c" "$STEPS")" ] || \
    python -m multilayer_nla.train_av_multi "${SFT_COMMON[@]}" --parquet "$SWEEP/av_$c.parquet" \
        --save-dir "$CKPT/av_$c" --wandb-name "av-$c"
done

# ── 8. dev end-to-end eval grid: AR {500,1000} x AV {500,1000} x condition ──
mkdir -p "$EVAL/dev"
for c in "${CONDS[@]}"; do
  for ars in "$SAVE_EVERY" "$STEPS"; do
    for avs in "$SAVE_EVERY" "$STEPS"; do
      S="$EVAL/dev/dev_${c}_ar${ars}_av${avs}.json"
      [ -f "$S" ] || python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" \
          --av-ckpt "$(iterdir "$CKPT/av_$c" "$avs")" --ar-ckpt "$(iterdir "$CKPT/ar" "$ars")" \
          --eval-parquet "$SWEEP/rl_dev_$c.parquet" --condition "$c" \
          --out "$EVAL/dev/dev_${c}_ar${ars}_av${avs}.jsonl" --summary "$S" --seed "$SEED"
    done
  done
done

# ── 9. select checkpoints — DEV ONLY (shared AR by mean dev FVE; per-cond AV) ──
python -m multilayer_nla.select_and_report --mode select --dev-dir "$EVAL/dev" \
    --out "$EVAL/selection.json" --env-out "$EVAL/selection.env"
source "$EVAL/selection.env"   # -> CHOSEN_AR_STEP, CHOSEN_AV_<COND>_STEP

# ── 10. ONE-SHOT test eval for the selected ckpts + AR-only gold dev/test ──
mkdir -p "$EVAL/test"
ARSEL="$(iterdir "$CKPT/ar" "$CHOSEN_AR_STEP")"
for c in "${CONDS[@]}"; do
  vn="CHOSEN_AV_${c^^}_STEP"; avstep="${!vn}"
  [ -f "$EVAL/test/test_$c.json" ] || python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" \
      --av-ckpt "$(iterdir "$CKPT/av_$c" "$avstep")" --ar-ckpt "$ARSEL" \
      --eval-parquet "$SWEEP/rl_test_$c.parquet" --condition "$c" \
      --out "$EVAL/test/test_$c.jsonl" --summary "$EVAL/test/test_$c.json" --seed "$SEED"
done
[ -f "$EVAL/test/ar_gold_dev.json" ]  || python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" \
    --ar-ckpt "$ARSEL" --eval-parquet "$SWEEP/ar_dev.parquet"  --summary "$EVAL/test/ar_gold_dev.json"
[ -f "$EVAL/test/ar_gold_test.json" ] || python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" \
    --ar-ckpt "$ARSEL" --eval-parquet "$SWEEP/ar_test.parquet" --summary "$EVAL/test/ar_gold_test.json"

# ── 11. result table — STOP (no RL) ──
python -m multilayer_nla.select_and_report --mode report --test-dir "$EVAL/test" \
    --selection "$EVAL/selection.json" --ar-gold-dev "$EVAL/test/ar_gold_dev.json" \
    --ar-gold-test "$EVAL/test/ar_gold_test.json" --out "$EVAL/result_table.md"
echo "[sweep] DONE -> $EVAL/result_table.md   (NO RL launched; do not change anything post-test.)"
