#!/bin/bash
# Variant F: AV SFT with Karvonen ADD norm-matched injection at layer 1.
# Differs from baseline run_av_sft.sh: NLA_KARVONEN_INJECTION=1, trained on
# av_train (90% slice) so val is genuinely held out, separate SAVE_DIR.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export WANDB_PROJECT=nla-qwen3-8b
export WANDB_RUN_GROUP=qwen3_8b_L24_v1
export WANDB_NAME=av_sft_karvonen
export PYTHONUNBUFFERED=1

# THE KEY KNOB
export NLA_KARVONEN_INJECTION=1

export AV_SFT_PARQUET=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/av_train.parquet
export INSTRUCT_MODEL=Qwen/Qwen3-8B
export INJ_SCALE=raw   # ignored under Karvonen mode but actor_sft.sh requires it set
export SAVE_DIR=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen

mkdir -p "$SAVE_DIR"

echo "=== AV SFT (Karvonen) START ==="
date
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

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

echo "=== AV SFT (Karvonen) END ==="
date
