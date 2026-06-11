#!/bin/bash
#SBATCH --job-name=karvonen_v2_onpolicy_build
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# v2 Karvonen dataset builder.
#   Stage 1: Opus 4.7 filter — does Qwen3-8B even exhibit each quirk? (the
#            original investigation was on 32B, so many won't transfer)
#   Stage 2: per-rollout Sonnet 4.6 audit on kept prompts
#   Output: parquet of quirk-positive rollouts + layer-24 activations at
#           4 strategic positions per rollout

set -euo pipefail
source /workspace-vast/celeste/.env
source /workspace-vast/celeste/envs/nla/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace-vast/pretrained_ckpts
export KARVONEN_CORPUS_DIR=/workspace-vast/celeste/karvonen_corpus
: "${HF_TOKEN:?set HF_TOKEN}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY for judge calls}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

cd /workspace-vast/celeste/nla-experiments

python scripts/karvonen_v2_build_onpolicy.py \
  --n-rollouts 5 \
  --temperature 0.7 \
  --max-new-tokens 2000 \
  --min-interest 3 \
  --min-verif 7 \
  --layer 24 \
  --model Qwen/Qwen3-8B \
  --filter-model claude-opus-4-7 \
  --judge-model claude-sonnet-4-6 \
  --judge-key-env ANTHROPIC_API_KEY \
  --judge-concurrency 16 \
  --corpus-dir /workspace-vast/celeste/karvonen_corpus \
  --output /workspace-vast/celeste/karvonen_corpus/v2_onpolicy.parquet \
  --seed 0
