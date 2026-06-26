#!/usr/bin/env bash
# Publish the §7 SFT control-sweep to Hugging Face — TWO destinations:
#   - results + datacard -> DATASET repo, into a SEPARATE folder, additive (sits next to the
#       L19-29 activation bank; does NOT touch the existing shards).
#       default: senku21x/qwen3-8b-nla-multilayer-L19-29  @ results/sft_control_sweep
#   - selected weights   -> MODEL repo (the dedicated sweep weights repo).
#       default: senku21x/qwen3-8b-nla-multilayer-sweep
#
# Run ON THE H200 (HF is reachable there; it is blocked from the dev box). Token via the
# HF_TOKEN env var ONLY (needs WRITE scope) — never an interactive prompt. Use `bash`:
#   HF_TOKEN=hf_xxx bash multilayer_nla/scripts/push_to_hf.sh
#
# `hf upload` auto-creates the target repo on recent CLIs; if yours doesn't, create once:
#   hf repo create senku21x/qwen3-8b-nla-multilayer-sweep --repo-type model
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

: "${HF_TOKEN:?export HF_TOKEN (write scope; no interactive prompt)}"
DATA="${DATA:-/data/mlnla}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"     # 3-tap converged eval (test/, dev/, test_arL24/)
SWEEP="${SWEEP:-$DATA/sweep}"                     # built datasets (row counts in the card)
CKPT="${CKPT:-$DATA/sweep_ckpt}"                  # AR + per-condition AV checkpoints
REGEN="${REGEN:-$DATA/published_L24x_window}"     # rl bank (source tokens in the qualitative compare)
ARL24="${ARL24:-$EVALC/test_arL24}"              # L24-only-AR cut (optional)

# ── destinations (override via env) ─────────────────────────────────
RESULTS_REPO="${RESULTS_REPO:-senku21x/qwen3-8b-nla-multilayer-L19-29}"   # DATASET repo (the bank)
RESULTS_REPO_TYPE="${RESULTS_REPO_TYPE:-dataset}"
RESULTS_PREFIX="${RESULTS_PREFIX:-results/sft_control_sweep}"             # separate folder in it
WEIGHTS_REPO="${WEIGHTS_REPO:-senku21x/qwen3-8b-nla-multilayer-sweep}"    # MODEL repo (weights)
WEIGHTS_REPO_TYPE="${WEIGHTS_REPO_TYPE:-model}"
WEIGHTS_PREFIX="${WEIGHTS_PREFIX:-}"             # "" = repo root; e.g. "weights" to nest

AR_STEP="${AR_STEP:-3000}"; AV_STEP="${AV_STEP:-1000}"   # the SELECTED checkpoints in the table
SKIP_RESULTS="${SKIP_RESULTS:-0}"               # 1 -> don't re-upload results (already on HF)
SKIP_WEIGHTS="${SKIP_WEIGHTS:-0}"               # 1 -> only push results
CONDS=(local duplicate wide single s2_19_21_23 s2_20_22_24)
iterdir() { printf '%s/iter_%07d' "$1" "$2"; }
rjoin()   { if [ -n "$1" ]; then printf '%s/%s' "$1" "$2"; else printf '%s' "$2"; fi; }

# Checkpoint locations — OVERRIDE these if your dirs are named differently. The shared AR
# trained to a custom dir (e.g. $CKPT/ar_3tap_bs256e_3k), NOT $CKPT/ar — so set AR_CKPT.
AR_CKPT="${AR_CKPT:-$(iterdir "$CKPT/ar" "$AR_STEP")}"   # full path to the AR iter dir
AV_BASE="${AV_BASE:-$CKPT}"                              # dir holding av_<cond>/iter_*

# ── 1. (re)generate analysis WITH source tokens, then the datacard (numbers from disk) ──
python -m multilayer_nla.analyze_sweep --eval-dir "$EVALC" --split-seed 42 \
    --bank "$REGEN" --out "$EVALC/analysis.md"
ARL24_ARGS=()
if [ -d "$ARL24" ]; then
  python -m multilayer_nla.analyze_sweep --test-dir "$ARL24" --eval-dir "$EVALC" \
      --split-seed 42 --bank "$REGEN" --out "$EVALC/analysis_arL24.md"
  ARL24_ARGS=(--arl24-dir "$ARL24")
fi
python -m multilayer_nla.make_datacard --eval-dir "$EVALC" "${ARL24_ARGS[@]}" \
    --sweep-dir "$SWEEP" --weights-repo "$WEIGHTS_REPO" --results-repo "$RESULTS_REPO" \
    --out "$EVALC/DATACARD.md"

# ── 2. results + datacard -> DATASET repo (separate folder; additive) ──
if [ "$SKIP_RESULTS" != 1 ]; then
  hf upload "$RESULTS_REPO" "$EVALC" "$RESULTS_PREFIX" --repo-type "$RESULTS_REPO_TYPE" \
      --commit-message "§7 SFT control sweep — results + datacard"
else
  echo "[push] SKIP_RESULTS=1 — not re-uploading results"
fi

# ── 3. selected weights (shared AR + per-condition AV) -> MODEL repo ──
if [ "$SKIP_WEIGHTS" != 1 ]; then
  # fail fast (before any upload) with a discovery listing if a ckpt dir is missing
  miss=0
  [ -d "$AR_CKPT" ] || { echo "MISSING AR ckpt: $AR_CKPT"; miss=1; }
  for c in "${CONDS[@]}"; do
    d="$(iterdir "$AV_BASE/av_$c" "$AV_STEP")"
    [ -d "$d" ] || { echo "MISSING AV ckpt: $d"; miss=1; }
  done
  if [ "$miss" = 1 ]; then
    echo "---- candidate checkpoint dirs under $CKPT: ----"
    ls -d "$CKPT"/*/iter_* 2>/dev/null || ls -d "$CKPT"/* 2>/dev/null || true
    echo "Set AR_CKPT=/abs/path/to/ar/iter_XXXXXXX (and AV_BASE=... if AVs live elsewhere), then re-run."
    exit 1
  fi
  hf upload "$WEIGHTS_REPO" "$AR_CKPT" \
      "$(rjoin "$WEIGHTS_PREFIX" "ar/iter_$(printf '%07d' "$AR_STEP")")" \
      --repo-type "$WEIGHTS_REPO_TYPE" --commit-message "shared AR (step $AR_STEP)"
  # optional: the L24-only reconstructor used for the test_arL24 cut (set AR_L24_CKPT to push)
  if [ -n "${AR_L24_CKPT:-}" ]; then
    [ -d "$AR_L24_CKPT" ] || { echo "MISSING AR_L24 ckpt: $AR_L24_CKPT"; exit 1; }
    hf upload "$WEIGHTS_REPO" "$AR_L24_CKPT" "$(rjoin "$WEIGHTS_PREFIX" "ar_L24")" \
        --repo-type "$WEIGHTS_REPO_TYPE" --commit-message "L24-only AR"
  fi
  for c in "${CONDS[@]}"; do
    hf upload "$WEIGHTS_REPO" "$(iterdir "$AV_BASE/av_$c" "$AV_STEP")" \
        "$(rjoin "$WEIGHTS_PREFIX" "av_$c/iter_$(printf '%07d' "$AV_STEP")")" \
        --repo-type "$WEIGHTS_REPO_TYPE" --commit-message "AV $c (step $AV_STEP)"
  done
fi

echo "[push] results -> https://huggingface.co/datasets/$RESULTS_REPO/tree/main/$RESULTS_PREFIX"
echo "[push] weights -> https://huggingface.co/$WEIGHTS_REPO/tree/main"
