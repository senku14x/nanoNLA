#!/bin/bash
# RL training — joint AV (GRPO) + AR (MSE) on Qwen3-8B at layer 24.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export WANDB_PROJECT=nla-qwen3-8b
export WANDB_RUN_GROUP=qwen3_8b_L24_v1
export WANDB_NAME=rl
export PYTHONUNBUFFERED=1

export NLA_EMBED_DUMP_DIR=/dev/shm/nla
mkdir -p "$NLA_EMBED_DUMP_DIR"

# Latest SFT iterations — pick the highest-numbered iter dir.
AV_SFT_BASE=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft
AR_SFT_BASE=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft
ACTOR_SFT_CKPT=$(ls -1d $AV_SFT_BASE/iter_* 2>/dev/null | sort | tail -1)
CRITIC_SL_CKPT=$(ls -1d $AR_SFT_BASE/iter_*/hf 2>/dev/null | sort | tail -1)
test -n "$ACTOR_SFT_CKPT" || { echo "no AV SFT checkpoint found in $AV_SFT_BASE"; exit 1; }
test -n "$CRITIC_SL_CKPT" || { echo "no AR SFT/hf checkpoint found in $AR_SFT_BASE"; exit 1; }

export RL_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/rl_shuf.parquet
export INSTRUCT_MODEL=Qwen/Qwen3-8B
export ACTOR_SFT_CKPT
export CRITIC_SL_CKPT
export RUN_DIR=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl
mkdir -p "$RUN_DIR"

# Single-node 8x H200 layout: 4 actor + 2 critic + 2 rollout
export ACTOR_NODES=1
export ACTOR_GPUS=2
export CRITIC_NODES=1
export CRITIC_GPUS=2
export ROLLOUT_GPUS=1

echo "=== RL START ==="
date
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "  ACTOR_SFT_CKPT=$ACTOR_SFT_CKPT"
echo "  CRITIC_SL_CKPT=$CRITIC_SL_CKPT"

cd $(python -c "import miles, os; print(os.path.dirname(miles.__file__))")/..

bash /workspace-vast/celeste/nla-experiments/configs/rl.sh \
    --attn-implementation sdpa \
    --num-rollout 500 \
    --rollout-batch-size 256 \
    --n-samples-per-prompt 8 \
    --rollout-temperature 0.7 \
    --ref-load "$ACTOR_SFT_CKPT/hf" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_RUN_GROUP"

echo "=== RL END ==="
date
