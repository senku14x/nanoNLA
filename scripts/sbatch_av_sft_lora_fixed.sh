#!/bin/bash
#SBATCH --job-name=qwen3_av_sft_lora_fixed
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out
set -euo pipefail
source /workspace-vast/celeste/.env
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments
DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k
cd /workspace-vast/celeste/nla-experiments
python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3-8B \
  --parquet $DATA/av_sft_shuf.parquet --sidecar $DATA/av_sft_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_lora_fixed \
  --num-steps 1000 --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --quant 4bit --lora-r 128 --lora-alpha 16 \
  --lr 3e-5 --min-lr 3e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --save-every 500 --wandb-project nla-qwen3-8b --wandb-name av_sft_lora_fixed --seed 0
