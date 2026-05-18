#!/bin/bash
#SBATCH --job-name=qwen3_eval_clean
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# Re-run post-RL eval on rl_shuf.parquet rows (DOC-disjoint from av_train,
# so the actor is genuinely held-out). The original eval used av_val which
# was row-sampled from av_train (99.99% doc overlap).
#
# We sample rows past the RL training cursor (max_rows=20000 was used during
# training) to also avoid any rows the in-training actor was rewarded on.

set -euo pipefail
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

# Use the 250-step ckpt (the one already evaluated). After the 1500-step run
# completes, re-run this against iter_001500 for a fairer comparison.
python launch/eval_post_rl.py \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-lora /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_overnight/iter_000250 \
  --val-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --skip-rows 25000 \
  --n-rows 128 \
  --max-new 150
