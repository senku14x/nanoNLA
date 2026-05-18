#!/bin/bash
# AR (critic) SFT retry with conservative hparams to prevent the step-29 divergence.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export WANDB_PROJECT=nla-qwen3-8b
export WANDB_RUN_GROUP=qwen3_8b_L24_v1
export WANDB_NAME=ar_sft_safe
export PYTHONUNBUFFERED=1

# Use the EXISTING critic init (it's verified clean — backbone non-NaN, value_head identity).
export AR_SFT_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/ar_sft_shuf_clean.parquet
export NLA_FREEZE_VALUE_HEAD=1
export CRITIC_INIT_CKPT=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_critic_init
export SAVE_DIR=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe

mkdir -p "$SAVE_DIR"

cd $(python -c "import miles, os; print(os.path.dirname(miles.__file__))")/..

# PAPER'S EXACT CRITIC CONFIG (TRAINING_NOTES.md Qwen 2.5-7B):
# global_batch 256, micro 64. Larger batch averages 4× more samples per grad
# step, drastically reducing stochastic gradient noise that was destabilizing
# our 64/16 setup on Qwen3-8B.
bash /workspace-vast/celeste/nla-experiments/configs/critic_sft.sh \
    --actor-num-gpus-per-node 2 \
    --attn-implementation sdpa \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 64 \
    --lr 2e-5 --min-lr 2e-6 \
    --lr-warmup-iters 50 \
    --lr-decay-style cosine \
    --num-rollout 1000 \
    --save-interval 200 \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_RUN_GROUP"
