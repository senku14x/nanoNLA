#!/bin/bash
#SBATCH --job-name=qwen3_rl_v5_smoother
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# v5: smoother trajectory — half the LR of v4 (1e-5 → 5e-6), checkpoint every
# 10 steps for a fine-grained trajectory plot, run hallucination +
# karvonen_confusion evals inline every 10 steps so wandb has the full
# capability-vs-faithfulness signal as it trains.

set -euo pipefail
source /workspace-vast/celeste/.env
source /workspace-vast/celeste/envs/nla/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY for judge calls}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

# Prefer high-prio Anthropic key if available; fall back to the standard key.
JUDGE_KEY_ENV=ANTHROPIC_API_KEY
if [ -n "${ANTHROPIC_API_KEY_FALLBACK:-}" ]; then
  JUDGE_KEY_ENV=ANTHROPIC_API_KEY_FALLBACK
fi
echo "[launch] using judge key env: \$$JUDGE_KEY_ENV"

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

python -m nla.train_rl_self_contained \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_v5_smoother \
  --num-steps 800 \
  --batch-prompts 16 \
  --group-size 16 \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --lr 5e-6 \
  --kl-beta 0.01 \
  --clip-eps 0.2 \
  --lora-r 128 \
  --lora-alpha 16 \
  --use-rslora \
  --train-critic \
  --critic-lr 5e-5 \
  --logp-micro-batch 2 \
  --max-rows 30000 \
  --save-every 10 \
  --eval-every 10 \
  --eval-n-prompts 20 \
  --eval-skip-rows 35000 \
  --external-evals hallucination,karvonen_confusion \
  --eval-n-hallucination 100 \
  --eval-n-karvonen 97 \
  --judge-key-env $JUDGE_KEY_ENV \
  --judge-concurrency 32 \
  --max-grad-norm 1.0 \
  --wandb-project nla-qwen3-8b \
  --wandb-name qwen3_8b_L24_rl_grpo_v5_smoother \
  --seed 0
