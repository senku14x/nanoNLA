#!/bin/bash
#SBATCH --job-name=qwen3_ar_sft
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# AR (Activation Reconstructor) SFT — truncated K+1=25-layer Qwen3-8B
# backbone + Linear(d, d) value_head (identity-init in-script, no
# prepare_critic step needed). Self-contained, no Miles dependency.
# batch=64, gradient_checkpointing OFF (smaller model + shorter seq fits),
# bitsandbytes AdamW8bit, SDPA attention. ~50min on a single H200.

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
  --mode ar \
  --base-ckpt Qwen/Qwen3-8B \
  --parquet $DATA/ar_sft_shuf_clean.parquet \
  --sidecar $DATA/ar_sft_shuf_clean.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft \
  --num-steps 1000 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --ar-num-layers 25 \
  --lr 2e-5 \
  --min-lr 2e-6 \
  --lr-warmup-steps 50 \
  --max-grad-norm 1.0 \
  --attn-implementation sdpa \
  --no-gradient-checkpointing \
  --save-every 500 \
  --wandb-project nla-qwen3-8b \
  --wandb-name ar_sft_v1 \
  --seed 0
