#!/bin/bash
#SBATCH --job-name=qwen3_hf_vs_vllm
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

set -euo pipefail
source /workspace-vast/celeste/envs/vllm-lens/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
export PYTHONUNBUFFERED=1
# vllm-lens env doesn't have the nla package — add it via PYTHONPATH.
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

python /workspace-vast/celeste/nla-experiments/launch/compare_hf_vs_vllm_lens.py \
  --ckpt-hf /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --val-parquet /workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/av_val.parquet \
  --sidecar /workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/av_val.parquet \
  --n-samples 3 \
  --max-new-tokens 60
