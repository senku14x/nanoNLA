#!/usr/bin/env bash
# Rebuild the §7 sweep TRAINING/EVAL datasets on a fresh instance, deterministically.
# Given the L19-L29 activation bank ($REGEN), this reproduces the EXACT ar/av/rl parquets
# (document-level splits, seed 42) for all 6 conditions, runs the build preflight, and the
# report-all integrity gate. CPU only (the only model touch is the tokenizer for the
# marker-count check; drop --base-ckpt below to skip even that, offline).
#
# Prereq: the bank must already exist at $REGEN as {av_sft,ar_sft,rl}.shard*of*.parquet
# (or {av_sft,ar_sft,rl}.parquet), with columns activation_L19..activation_L29 + doc_id +
# labels. To GET the bank (download from HF, or regenerate from published text+labels on a
# GPU), see REBUILD_RUNBOOK.md §2.
#
#   DATA=/data/mlnla bash multilayer_nla/scripts/rebuild_datasets.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
REGEN="${REGEN:-$DATA/published_L24x_window}"   # L19-29 bank (INPUT)
SWEEP="${SWEEP:-$DATA/sweep}"                   # rebuilt datasets (OUTPUT)
SPLIT_SEED="${SPLIT_SEED:-42}"
DEV_SUBSET="${DEV_SUBSET:-256}"; TEST_SUBSET="${TEST_SUBSET:-1000}"
AR_TAPS="${AR_TAPS:-23,24,25}"                  # FIXED target — do not change for the sweep
mkdir -p "$SWEEP"

# ── 0. bank present? (shards or single file per subset) ──
have() { ls "$REGEN/$1".shard*of*.parquet >/dev/null 2>&1 || [ -f "$REGEN/$1.parquet" ]; }
miss=0
for s in av_sft ar_sft rl; do
  have "$s" || { echo "MISSING bank subset: $REGEN/$s.shard*of*.parquet (or $s.parquet)"; miss=1; }
done
if [ "$miss" = 1 ]; then
  echo "---- the L19-29 bank is not at \$REGEN=$REGEN ----"
  echo "Acquire it first (REBUILD_RUNBOOK.md §2): download from HF, or regenerate on a GPU."
  exit 1
fi
echo "[rebuild] bank OK at $REGEN"

# ── 1. document-level splits (seed 42, 80/10/10): rl (end-to-end) + ar (gold) ──
[ -f "$SWEEP/rl_split_manifest.json" ] || python -m multilayer_nla.splits \
  --source "$REGEN/rl.shard*of*.parquet" --name rl --out-dir "$SWEEP" \
  --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1 --dev-subset "$DEV_SUBSET" --test-subset "$TEST_SUBSET"
[ -f "$SWEEP/ar_split_manifest.json" ] || python -m multilayer_nla.splits \
  --source "$REGEN/ar_sft.shard*of*.parquet" --name ar --out-dir "$SWEEP" \
  --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1

# ── 2. build ALL 6 conditions + preflight (the canonical, gated build) ──
#     --mode all builds every condition in build_sweep.CONDITIONS in one pass AND runs
#     assert_conditions (this is what the per-mode `--mode av` build skipped, which is how
#     the av_s2_19_21_23 truncation slipped through last time — use --mode all here).
[ -f "$SWEEP/sweep_build_manifest.json" ] || python -m multilayer_nla.build_sweep --mode all \
  --in-dir "$REGEN" --out-dir "$SWEEP" \
  --rl-split-manifest "$SWEEP/rl_split_manifest.json" \
  --ar-split-manifest "$SWEEP/ar_split_manifest.json" \
  --ar-target-layers "$AR_TAPS" --base-ckpt "$BASE" --allow-existing

# ── 3. report-all integrity gate (belt-and-suspenders on top of step-2 preflight) ──
python -m multilayer_nla.verify_sweep_integrity --sweep-dir "$SWEEP" \
  --rl-split-manifest "$SWEEP/rl_split_manifest.json"

echo "[rebuild] DONE -> $SWEEP"
echo "[rebuild] built: ar_common/ar_dev/ar_test, av_<cond>, rl_dev/test_<cond> for all 6 conditions."
echo "[rebuild] next: train (REBUILD_RUNBOOK.md §4) then evaluate (§5)."
