#!/usr/bin/env bash
# Rebuild the §7 train/eval datasets from the L19-29 bank — deterministic, CPU only.
# This is REBUILD_RUNBOOK.md step 2. Asserts the bank is present, then:
#   splits.py (rl: locked dev=256/test=1000; ar: full 80/10/10; seed 42)
#   build_sweep --mode all (all 6 conditions, fixed AR target [23,24,25], + marker preflight)
#   verify_sweep_integrity (report-all gate; expect 83/83)
# Resumable: each step skips if its output exists (delete an output to force a redo).
#
#   REGEN=/data/mlnla/bank SWEEP=/data/mlnla/sweep bash multilayer_nla/scripts/rebuild_datasets.sh
#
# Use --mode all (NOT per-mode --mode av/rl-eval) so the preflight runs — that's the gate the
# av_s2_19_21_23 truncation slipped past before.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
REGEN="${REGEN:-$DATA/bank}"          # L19-29 bank: {av_sft,ar_sft,rl}.shard*of*.parquet
SWEEP="${SWEEP:-$DATA/sweep}"         # rebuilt datasets + split manifests
SPLIT_SEED="${SPLIT_SEED:-42}"; DEV_SUBSET="${DEV_SUBSET:-256}"; TEST_SUBSET="${TEST_SUBSET:-1000}"
mkdir -p "$SWEEP"

# ── precondition: all three bank subsets present ─────────────────────────────
shopt -s nullglob
for sub in av_sft ar_sft rl; do
  s=("$REGEN"/$sub.shard*of*.parquet)
  [ ${#s[@]} -gt 0 ] || { echo "ABORT: no $sub.shard*of*.parquet in REGEN=$REGEN — get the bank first (REBUILD_RUNBOOK.md §1, or scripts/recreate_sweep.sh)"; exit 1; }
done
shopt -u nullglob

# ── splits (rl gets the locked subsets; ar gets the full dev/test for gold) ───
[ -f "$SWEEP/rl_split_manifest.json" ] || python -m multilayer_nla.splits \
  --source "$REGEN/rl.shard*of*.parquet" --name rl --out-dir "$SWEEP" \
  --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1 --dev-subset "$DEV_SUBSET" --test-subset "$TEST_SUBSET"
[ -f "$SWEEP/ar_split_manifest.json" ] || python -m multilayer_nla.splits \
  --source "$REGEN/ar_sft.shard*of*.parquet" --name ar --out-dir "$SWEEP" \
  --seed "$SPLIT_SEED" --fracs 0.8,0.1,0.1

# ── build all 6 conditions + AR datasets (preflight + marker check inside) ────
[ -f "$SWEEP/sweep_build_manifest.json" ] || python -m multilayer_nla.build_sweep --mode all \
  --in-dir "$REGEN" --out-dir "$SWEEP" \
  --rl-split-manifest "$SWEEP/rl_split_manifest.json" \
  --ar-split-manifest "$SWEEP/ar_split_manifest.json" \
  --ar-target-layers 23,24,25 --base-ckpt "$BASE" --allow-existing

# ── report-all integrity gate (expect 83/83) ────────────────────────────────
python -m multilayer_nla.verify_sweep_integrity --sweep-dir "$SWEEP" \
  --rl-split-manifest "$SWEEP/rl_split_manifest.json" \
  --ar-split-manifest "$SWEEP/ar_split_manifest.json"

echo "[rebuild] datasets ready in $SWEEP: ar_common/dev/test + av_<cond> + rl_{dev,test}_<cond>"
echo "[rebuild] 6 conditions: local duplicate wide single s2_19_21_23 s2_20_22_24"
echo "[rebuild] next: train/eval (REBUILD_RUNBOOK.md §3-4) or the averaged-input run (scripts/run_mean_input.sh)"
