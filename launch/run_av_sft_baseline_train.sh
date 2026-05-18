#!/bin/bash
# Baseline AV SFT retrained on av_train (90% slice) — for fair head-to-head vs Karvonen.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export WANDB_PROJECT=nla-qwen3-8b
export WANDB_RUN_GROUP=qwen3_8b_L24_v1
export WANDB_NAME=av_sft_baseline_fair
export PYTHONUNBUFFERED=1

# NO Karvonen — standard embedding-replace path.
unset NLA_KARVONEN_INJECTION

export AV_SFT_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/av_train.parquet
export INSTRUCT_MODEL=Qwen/Qwen3-8B
export INJ_SCALE=sqrt_d_model
export SAVE_DIR=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_baseline_fair

mkdir -p "$SAVE_DIR"

cd $(python -c "import miles, os; print(os.path.dirname(miles.__file__))")/..

bash /workspace-vast/celeste/nla-experiments/configs/actor_sft.sh \
    --actor-num-gpus-per-node 2 \
    --attn-implementation sdpa \
    --rollout-batch-size 32 \
    --global-batch-size 32 \
    --micro-batch-size 4 \
    --lr 2e-5 --min-lr 2e-6 \
    --lr-warmup-iters 50 \
    --lr-decay-style cosine \
    --num-rollout 1000 \
    --save-interval 500 \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_RUN_GROUP"
