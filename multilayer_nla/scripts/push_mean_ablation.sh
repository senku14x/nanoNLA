#!/usr/bin/env bash
# Publish the mean-pool ablation:
#   - the 4 new AV adapters  -> MODEL repo  (av_<cond>/iter_0001000, matching the existing 8)
#   - analysis + eval summaries/jsonls -> DATASET repo (results/mean_ablation/, ADDITIVE —
#     does NOT touch the published results/sft_control_sweep/).
#
# Needs WRITE-scope HF auth: either `hf auth login` with a write token, or HF_TOKEN=hf_xxx.
# Run from the repo root, ON THE H200:
#   bash multilayer_nla/scripts/push_mean_ablation.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

DATA="${DATA:-/data/mlnla}"
CKPT="${CKPT:-$DATA/sweep_ckpt}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"
WEIGHTS_REPO="${WEIGHTS_REPO:-senku21x/qwen3-8b-nla-multilayer-sweep}"
RESULTS_REPO="${RESULTS_REPO:-senku21x/qwen3-8b-nla-multilayer-L19-29}"
RESULTS_PREFIX="${RESULTS_PREFIX:-results/mean_ablation}"
AV_STEP="${AV_STEP:-1000}"
MEAN_CONDS=(mean mean_20_24_28 mean_19_21_23 mean_20_22_24)
iterdir() { printf '%s/av_%s/iter_%07d' "$CKPT" "$1" "$AV_STEP"; }

# ── fail fast if any adapter is missing (discovery listing) ──────────────────
miss=0
for c in "${MEAN_CONDS[@]}"; do
  [ -f "$(iterdir "$c")/adapter_config.json" ] || { echo "MISSING AV ckpt: $(iterdir "$c")"; miss=1; }
done
if [ "$miss" = 1 ]; then
  echo "---- candidate AV dirs under $CKPT: ----"; ls -d "$CKPT"/av_*/iter_* 2>/dev/null || true
  echo "Set CKPT=... if the adapters live elsewhere, then re-run."; exit 1
fi

# ── 1. AV adapters -> MODEL repo ─────────────────────────────────────────────
for c in "${MEAN_CONDS[@]}"; do
  echo "[push] av_$c -> $WEIGHTS_REPO ..."
  hf upload "$WEIGHTS_REPO" "$(iterdir "$c")" "av_$c/iter_$(printf '%07d' "$AV_STEP")" \
    --repo-type model --commit-message "AV $c (mean-pool ablation, step $AV_STEP)"
done

# ── 2. analysis + eval outputs -> DATASET repo (mirror dev/test/test_arL24) ───
stage="$(mktemp -d)"; mkdir -p "$stage"/{dev,test,test_arL24}
[ -f "$EVALC/analysis_mean.md" ] && cp "$EVALC/analysis_mean.md" "$stage/"
for c in "${MEAN_CONDS[@]}"; do
  cp "$EVALC"/dev/dev_${c}_*.{json,jsonl}      "$stage/dev/"        2>/dev/null || true
  cp "$EVALC"/test/test_${c}.{json,jsonl}      "$stage/test/"       2>/dev/null || true
  cp "$EVALC"/test_arL24/test_${c}.{json,jsonl} "$stage/test_arL24/" 2>/dev/null || true
done
echo "[push] analysis + summaries -> $RESULTS_REPO/$RESULTS_PREFIX ..."
hf upload "$RESULTS_REPO" "$stage" "$RESULTS_PREFIX" --repo-type dataset \
  --commit-message "mean-pool ablation — analysis + eval summaries/jsonls"
rm -rf "$stage"

echo "[push] weights  -> https://huggingface.co/$WEIGHTS_REPO/tree/main  (av_{${MEAN_CONDS[*]// /,}})"
echo "[push] analysis -> https://huggingface.co/datasets/$RESULTS_REPO/tree/main/$RESULTS_PREFIX"
echo "[push] NOTE: the model-repo README still lists the original 8 adapters — regen with make_datacard if you want it to include these 4."
