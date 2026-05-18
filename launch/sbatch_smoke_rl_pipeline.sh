#!/bin/bash
#SBATCH --job-name=qwen3_rl_smoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

set -euo pipefail
source /workspace-vast/celeste/envs/vllm-lens/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

python /workspace-vast/celeste/nla-experiments/launch/smoke_rl_pipeline.py \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --n-rows 20
