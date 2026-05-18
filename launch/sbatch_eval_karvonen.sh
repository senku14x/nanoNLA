#!/bin/bash
#SBATCH --job-name=qwen3_val_karvonen
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

set -euo pipefail
cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
export PYTHONUNBUFFERED=1

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

python launch/eval_av_val_loss.py \
  --ckpt-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --val-parquet $DATA/av_val.parquet \
  --sidecar $DATA/av_val.parquet \
  --injection karvonen \
  --max-rows 500 \
  --output-json /workspace-vast/celeste/nla-experiments/logs/val_karvonen.json

echo "---SAMPLES---"
python launch/sample_av_explanations.py \
  --ckpt-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --val-parquet $DATA/av_val.parquet \
  --sidecar $DATA/av_val.parquet \
  --injection karvonen \
  --n-samples 5
