#!/bin/bash
#SBATCH --job-name=qwen3_vllm_lens_smoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

set -euo pipefail
source /workspace-vast/celeste/envs/vllm-lens/bin/activate
# vllm-lens 1.0 (pinned for cu128 compat)
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
export PYTHONUNBUFFERED=1

python /workspace-vast/celeste/nla-experiments/launch/smoke_vllm_lens.py
