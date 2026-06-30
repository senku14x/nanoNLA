#!/usr/bin/env bash
# Fresh-box bootstrap for the Progressive Reader experiment. Run from the repo root after
# cloning. Idempotent: download av_sft (if missing) -> install deps -> audit gate -> smoke run.
#
#   HF_TOKEN=hf_xxx REGEN=/data/mlnla/bank bash multilayer_nla/scripts/bootstrap_progressive_reader.sh
#
# Env: REGEN (where av_sft lands; default /data/mlnla/bank), RUN_SMOKE=0 to skip the smoke run.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export REGEN="${REGEN:-/data/mlnla/bank}"
DS_REPO="${DS_REPO:-senku21x/qwen3-8b-nla-multilayer-L19-29}"
mkdir -p "$REGEN"

# 1) data — download av_sft if missing (BEFORE any hub downgrade, so the fast Xet path is used)
if ! ls "$REGEN"/av_sft.shard*of*.parquet >/dev/null 2>&1; then
  command -v hf >/dev/null || pip install -q "huggingface_hub[cli]"
  echo "[bootstrap] downloading av_sft -> $REGEN ..."
  hf download "$DS_REPO" --repo-type dataset --local-dir "$REGEN" \
    --include "av_sft.shard*of*.parquet" --max-workers "${HF_WORKERS:-12}"
else
  echo "[bootstrap] av_sft already present in $REGEN"
fi

# 2) deps — project (pins transformers==4.57.1, keeps the preinstalled CUDA torch) + plots
pip install -e . -q
pip install -q matplotlib
python -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) audit gate (CPU + tokenizer)
python -m multilayer_nla.progressive_reader.audit \
  --data "$REGEN/av_sft.shard*of*.parquet" --base-ckpt Qwen/Qwen3-8B \
  --out runs/progressive_reader_v0/data_audit.json

# 4) smoke run (validates the GPU path end-to-end; RUN_SMOKE=0 to skip)
if [ "${RUN_SMOKE:-1}" = 1 ]; then
  echo "[bootstrap] smoke run ..."
  python -m multilayer_nla.progressive_reader.train \
    --config configs/progressive_reader_v0_smoke.yaml \
    --run-dir runs/pr_smoke/prog --no-wandb
fi
echo "[bootstrap] DONE. audit -> runs/progressive_reader_v0/data_audit.json"
