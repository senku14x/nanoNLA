#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_overnight
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# Self-contained GRPO RL for Qwen3-8B NLA.
#
# Paper-faithful settings (from configs/rl.sh, the Qwen2.5-7B reference):
#   - --n-samples-per-prompt 4 → --group-size 4
#   - --advantage-estimator grpo → group-relative advantage (per-prompt mean/std)
#   - --kl-loss-coef 0.01 → --kl-beta 0.01
#   - --lr 1e-6 constant
#   - --rollout-max-response-len 150 → --max-new-tokens 150
#
# Documented deviations (memory / single-GPU constraint):
#   1. LoRA actor (r=16, ~15M trainable) instead of full 8B fine-tune.
#      Paper does full FT on 2x H100; we have 1x H200 budgeted.
#   2. Frozen AR critic — paper co-trains it. Frozen is safer; risk is actor
#      finds adversarial explanations the static critic decodes well. We
#      mitigate by short run + close KL leash.
#   3. Effective batch 8 prompts × 4 samples = 32 per step, vs paper's
#      128 × 4 = 512 — fewer prompts/step but more steps for similar wall.

set -euo pipefail
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

python -m nla.train_rl_self_contained \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_overnight \
  --num-steps 250 \
  --batch-prompts 8 \
  --group-size 4 \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --lr 1e-6 \
  --kl-beta 0.01 \
  --clip-eps 0.2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --logp-micro-batch 2 \
  --max-rows 10000 \
  --save-every 50 \
  --max-grad-norm 1.0 \
  --wandb-project nla-qwen3-8b \
  --wandb-name qwen3_8b_L24_rl_grpo_overnight \
  --seed 0
