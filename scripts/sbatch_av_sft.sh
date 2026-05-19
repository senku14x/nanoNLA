#!/bin/bash
#SBATCH --job-name=qwen3_av_sft
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# AV (Activation Verbalizer) SFT — Karvonen ADD norm-matched injection on
# layer-1 residual. Self-contained, no Miles dependency. batch=64,
# gradient_checkpointing ON, bitsandbytes AdamW8bit, FA2 attention.
# ~1.5h on a single H200 for 1000 steps.

set -euo pipefail
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

python -m nla.train_sft \
  --mode av \
  --base-ckpt Qwen/Qwen3-8B \
  --parquet $DATA/av_train.parquet \
  --sidecar $DATA/av_train.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft \
  --num-steps 1000 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --lr 2e-5 \
  --min-lr 2e-6 \
  --lr-warmup-steps 50 \
  --max-grad-norm 1.0 \
  --attn-implementation flash_attention_2 \
  --gradient-checkpointing \
  --save-every 500 \
  --wandb-project nla-qwen3-8b \
  --wandb-name av_sft_v1 \
  --seed 0
