#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_long
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# Long GRPO run: 1500 steps, ~13h at 33s/step, paper-faithful hparams.
# Follow-up to the overnight 250-step run which showed real reward improvement
# (+12.4%) — see docs/qwen3_8b_run.md.

set -euo pipefail
source /workspace-vast/celeste/envs/nla/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_v4_lrbump \
  --resume-from-lora /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_v4_lrbump/iter_000100 \
  --start-step 100 \
  --num-steps 1000 \
  --batch-prompts 16 \
  --group-size 16 \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --lr 1e-5 \
  --kl-beta 0.01 \
  --clip-eps 0.2 \
  --lora-r 128 \
  --lora-alpha 16 \
  --use-rslora \
  --train-critic \
  --critic-lr 5e-5 \
  --logp-micro-batch 2 \
  --max-rows 30000 \
  --save-every 100 \
  --eval-every 10 \
  --eval-n-prompts 20 \
  --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-qwen3-8b \
  --wandb-name qwen3_8b_L24_rl_grpo_v4_lrbump_normvh \
  --seed 0
