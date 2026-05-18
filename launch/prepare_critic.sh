#!/bin/bash
# Truncate Qwen3-8B to layers 0..24 + d×d head, save as critic-init checkpoint.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
export PYTHONUNBUFFERED=1

AR_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/ar_sft_shuf.parquet
CRITIC_INIT=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_critic_init

mkdir -p "$(dirname "$CRITIC_INIT")"

echo "=== prepare_critic_checkpoint ==="
date

python -m nla.scripts.prepare_critic_checkpoint \
    --base-model Qwen/Qwen3-8B \
    --num-layers 24 \
    --dataset-sidecar "$AR_PARQUET" \
    --output "$CRITIC_INIT"

echo "=== DONE ==="
date
ls -la "$CRITIC_INIT"
