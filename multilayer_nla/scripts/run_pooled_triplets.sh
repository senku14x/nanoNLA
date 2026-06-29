#!/usr/bin/env bash
# Train + eval the three remaining mean-pooled triplets, generalizing the mean ablation
# beyond the adjacent `local` triplet (which gave: pooled content +0.46pp, slot-distinctness
# +1.98pp). Each condition = ONE AV train (wandb av-<cond>) + the 3 eval cuts against the
# FROZEN ARs, then a refreshed analysis. Resumable (every step skips if its output exists).
#
#   mean_20_24_28  (pool of wide)        — far layers L20/L28 drag the mean off the target
#   mean_19_21_23  (pool of s2_19_21_23) — all below target
#   mean_20_22_24  (pool of s2_20_22_24) — spans up to target (incl. L24)
#
# AR / AR_L24 default to the downloaded model-repo layout; override if yours differ.
# Run INSIDE tmux so a disconnect can't kill it.
#   bash multilayer_nla/scripts/run_pooled_triplets.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="${DATA:-/data/mlnla}"
export BASE="${BASE:-Qwen/Qwen3-8B}"
export REGEN="${REGEN:-$DATA/bank}"
export SWEEP="${SWEEP:-$DATA/sweep}"
export CKPT="${CKPT:-$DATA/sweep_ckpt}"
export EVALC="${EVALC:-$DATA/sweep_eval_converged}"
export AR="${AR:-$DATA/weights/ar/iter_0003000}"          # frozen shared 3-tap AR (downloaded)
export AR_L24="${AR_L24:-$DATA/weights/ar_L24}"           # frozen L24-only AR (downloaded)

run() { echo "===== $1  (pool $2) ====="; COND="$1" POOL_LAYERS="$2" bash "$HERE/run_mean_input.sh"; }
run mean_20_24_28 20,24,28
run mean_19_21_23 19,21,23
run mean_20_22_24 20,22,24

# Refresh the analysis. The published 6-condition jsonls must be in $EVALC for the paired
# contrasts (you pulled them earlier via results/sft_control_sweep/*); if missing, the
# original-condition contrasts will say "missing" but the pooled ones still compute.
python -m multilayer_nla.analyze_sweep --eval-dir "$EVALC" --split-seed 42 --bank "$REGEN" \
  --out "$EVALC/analysis_mean.md"

echo "[pooled] DONE — 3 pooled triplets trained + evaled."
echo "[pooled] analysis -> $EVALC/analysis_mean.md  (push to HF to share:"
echo "         hf upload senku21x/qwen3-8b-nla-multilayer-L19-29 \"$EVALC/analysis_mean.md\" \\"
echo "           results/mean_ablation/analysis_mean.md --repo-type dataset )"
