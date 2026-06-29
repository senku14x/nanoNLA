#!/usr/bin/env bash
# Fresh-box one-shot: FETCH the published artifacts from HF, then REBUILD the §7
# train/eval datasets. CPU + network only (no GPU). Resumable.
#
# This automates REBUILD_RUNBOOK.md step 1 (get the bank) + step 2 (rebuild_datasets.sh):
#   1. download the L19-29 bank ({av_sft,ar_sft,rl} shards) from the dataset repo
#      and the 8 trained adapters from the model repo (so NOTHING but av_mean needs training);
#   2. delegate splits + build_sweep --mode all + verify_sweep_integrity to rebuild_datasets.sh.
#
# Requires `hf auth login` (read scope) first. Run from the repo root.
#   DATA=/data/mlnla bash multilayer_nla/scripts/recreate_sweep.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
DS_REPO="${DS_REPO:-senku21x/qwen3-8b-nla-multilayer-L19-29}"      # dataset repo (the bank)
MODEL_REPO="${MODEL_REPO:-senku21x/qwen3-8b-nla-multilayer-sweep}"  # model repo (8 adapters)
REGEN="${REGEN:-$DATA/bank}"        # bank shards land here
WEIGHTS="${WEIGHTS:-$DATA/weights}" # downloaded adapters land here
SWEEP="${SWEEP:-$DATA/sweep}"       # rebuilt datasets
mkdir -p "$REGEN" "$WEIGHTS" "$SWEEP"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # per-file multi-threaded transfer
HF_WORKERS="${HF_WORKERS:-8}"   # concurrent files (24 bank shards -> downloads 8 at a time)

command -v hf >/dev/null || { echo "ABORT: 'hf' CLI not found. pip install -U 'huggingface_hub[cli]' && hf auth login"; exit 1; }

# ── 1. FETCH — bank (dataset repo) + 8 adapters (model repo) ──────────────────
# Need ALL THREE subsets (a prior mean-only pull may have grabbed only av_sft+rl). hf download
# with --include is idempotent, so this fills any missing subset without re-pulling what's there.
shopt -s nullglob; need_dl=0
for sub in av_sft ar_sft rl; do s=("$REGEN"/$sub.shard*of*.parquet); [ ${#s[@]} -gt 0 ] || need_dl=1; done
shopt -u nullglob
if [ "$need_dl" -eq 1 ]; then
  echo "[recreate] downloading bank shards (av_sft+ar_sft+rl) from $DS_REPO with $HF_WORKERS workers ..."
  hf download "$DS_REPO" --repo-type dataset --local-dir "$REGEN" --max-workers "$HF_WORKERS" \
    --include "av_sft.shard*of*.parquet" "ar_sft.shard*of*.parquet" "rl.shard*of*.parquet"
else
  echo "[recreate] all three bank subsets present in $REGEN — skip"
fi
if [ ! -f "$WEIGHTS/ar/iter_0003000/ar_meta.json" ]; then
  echo "[recreate] downloading 8 adapters from $MODEL_REPO ..."
  hf download "$MODEL_REPO" --local-dir "$WEIGHTS" --max-workers "$HF_WORKERS"   # idempotent
else
  echo "[recreate] adapters already present in $WEIGHTS — skip"
fi

# ── 2. REBUILD datasets (splits + build_sweep --mode all + integrity) ─────────
REGEN="$REGEN" SWEEP="$SWEEP" BASE="$BASE" bash "$HERE/rebuild_datasets.sh"

# ── done: the env downstream steps need ──────────────────────────────────────
cat <<EOF

[recreate] DONE. Bank in $REGEN, 8 adapters in $WEIGHTS, datasets in $SWEEP.

Export for the downstream steps:
  export BASE=$BASE  REGEN=$REGEN  SWEEP=$SWEEP
  export CKPT=$DATA/sweep_ckpt              # the NEW av_mean trains here; downloaded AVs stay in \$WEIGHTS
  export EVALC=$DATA/sweep_eval_converged
  export AR=$WEIGHTS/ar/iter_0003000        # shared 3-tap AR (downloaded)
  export AR_L24=$WEIGHTS/ar_L24             # L24-only AR (downloaded)
  # the 6 published AVs: $WEIGHTS/av_<cond>/iter_0001000

NEXT:
  bash multilayer_nla/scripts/run_mean_input.sh    # the averaged-input experiment (trains av-mean, evals)
  # to reproduce the 6 published conditions' eval (frozen ARs, downloaded AVs): REBUILD_RUNBOOK.md §4.

Retraining is NOT needed (adapters downloaded). If you DO retrain the 3-tap AR, use the converged
recipe: --batch-size 64 --gradient-accumulation-steps 4 (effective 256; the AR has no
gradient-checkpointing so batch 256 won't fit), --num-steps 3000. See REBUILD_RUNBOOK.md §3.
EOF
