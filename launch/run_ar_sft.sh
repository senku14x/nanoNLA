#!/bin/bash
# AR (critic) SFT for Qwen3-8B at layer 24.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export WANDB_PROJECT=nla-qwen3-8b
export WANDB_RUN_GROUP=qwen3_8b_L24_v1
export WANDB_NAME=ar_sft
export PYTHONUNBUFFERED=1

export AR_SFT_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/ar_sft_shuf.parquet
export CRITIC_INIT_CKPT=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_critic_init
export SAVE_DIR=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft

mkdir -p "$SAVE_DIR"

echo "=== AR SFT START ==="
date
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Use miles' train.py. Override actor-num-gpus-per-node from default 8 → 2 for our setup.
# Use FSDP backend. Keep sdpa attention (no flash-attn for now to avoid build flap).
cd $(python -c "import miles, os; print(os.path.dirname(miles.__file__))")/..

bash /workspace-vast/celeste/nla-experiments/configs/critic_sft.sh \
    --actor-num-gpus-per-node 2 \
    --attn-implementation sdpa \
    --rollout-batch-size 64 \
    --global-batch-size 64 \
    --micro-batch-size 16 \
    --lr 2e-5 --min-lr 2e-6 \
    --lr-warmup-iters 50 \
    --lr-decay-style cosine \
    --num-rollout 500 \
    --save-interval 250 \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_RUN_GROUP"

echo "=== AR SFT END ==="
date
