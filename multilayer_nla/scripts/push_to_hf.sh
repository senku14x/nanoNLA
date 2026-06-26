#!/usr/bin/env bash
# Publish the §7 SFT control-sweep RESULTS + DATACARD + selected WEIGHTS to one HF repo.
#   - results + datacard  -> a SEPARATE folder  ($RESULTS_PREFIX, default results/sft_control_sweep)
#   - selected weights    -> the SAME repo      ($WEIGHTS_PREFIX, default weights/)
#
# Run ON THE H200 (HF is reachable there; it is blocked from the dev box). Token via the
# HF_TOKEN env var ONLY — never an interactive prompt. Invoke with `bash`, not `source`:
#   HF_REPO=org/name HF_TOKEN=hf_xxx bash multilayer_nla/scripts/push_to_hf.sh
#
# Idempotent-ish: re-running re-uploads (HF dedups by content). The repo is auto-created by
# `hf upload` on recent CLIs; if yours doesn't, create it once first:
#   hf repo create "$HF_REPO" --repo-type model
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

: "${HF_REPO:?set HF_REPO=org/name (the repo; results + weights both land here)}"
: "${HF_TOKEN:?export HF_TOKEN (no interactive prompt)}"
DATA="${DATA:-/data/mlnla}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"     # 3-tap converged eval (test/, dev/, test_arL24/)
SWEEP="${SWEEP:-$DATA/sweep}"                     # built datasets (for row counts in the card)
CKPT="${CKPT:-$DATA/sweep_ckpt}"                  # AR + per-condition AV checkpoints
REGEN="${REGEN:-$DATA/published_L24x_window}"     # rl bank (for source tokens in the qualitative compare)
ARL24="${ARL24:-$EVALC/test_arL24}"              # L24-only-AR cut (optional)
REPO_TYPE="${REPO_TYPE:-model}"
RESULTS_PREFIX="${RESULTS_PREFIX:-results/sft_control_sweep}"
WEIGHTS_PREFIX="${WEIGHTS_PREFIX:-weights}"
AR_STEP="${AR_STEP:-3000}"; AV_STEP="${AV_STEP:-1000}"   # the SELECTED checkpoints in the table
CONDS=(local duplicate wide single s2_19_21_23 s2_20_22_24)
iterdir() { printf '%s/iter_%07d' "$1" "$2"; }

# ── 1. (re)generate analysis WITH source tokens, then the datacard (all numbers from disk) ──
python -m multilayer_nla.analyze_sweep --eval-dir "$EVALC" --split-seed 42 \
    --bank "$REGEN" --out "$EVALC/analysis.md"
ARL24_ARGS=()
if [ -d "$ARL24" ]; then
  python -m multilayer_nla.analyze_sweep --test-dir "$ARL24" --eval-dir "$EVALC" \
      --split-seed 42 --bank "$REGEN" --out "$EVALC/analysis_arL24.md"
  ARL24_ARGS=(--arl24-dir "$ARL24")
fi
python -m multilayer_nla.make_datacard --eval-dir "$EVALC" "${ARL24_ARGS[@]}" \
    --sweep-dir "$SWEEP" --weights-repo "$HF_REPO" --out "$EVALC/DATACARD.md"

# ── 2. push RESULTS + DATACARD to the separate folder (whole converged-eval dir) ──
hf upload "$HF_REPO" "$EVALC" "$RESULTS_PREFIX" --repo-type "$REPO_TYPE" \
    --commit-message "§7 SFT control sweep — results + datacard"

# ── 3. push the SELECTED weights (shared AR + per-condition AV) into the same repo ──
hf upload "$HF_REPO" "$(iterdir "$CKPT/ar" "$AR_STEP")" \
    "$WEIGHTS_PREFIX/ar/iter_$(printf '%07d' "$AR_STEP")" --repo-type "$REPO_TYPE" \
    --commit-message "shared AR (step $AR_STEP)"
for c in "${CONDS[@]}"; do
  hf upload "$HF_REPO" "$(iterdir "$CKPT/av_$c" "$AV_STEP")" \
      "$WEIGHTS_PREFIX/av_$c/iter_$(printf '%07d' "$AV_STEP")" --repo-type "$REPO_TYPE" \
      --commit-message "AV $c (step $AV_STEP)"
done

echo "[push] DONE -> https://huggingface.co/$HF_REPO"
echo "[push]   results+datacard: $RESULTS_PREFIX   weights: $WEIGHTS_PREFIX"
