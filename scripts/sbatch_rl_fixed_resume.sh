#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_fixed_leg
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:50:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out
# Self-chaining RL leg: resumes qwen3_8b_L24_rl_grpo_fixed from its latest
# checkpoint and trains toward TARGET steps. Submits its own successor with
# afterany:self, so the chain survives the per-job time limit. A leg that
# starts with the target already reached exits without submitting.
set -euo pipefail
TARGET=2000
SAVE=/workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_fixed
LAST=$(ls -d $SAVE/iter_* 2>/dev/null | sort | tail -1)
[ -z "$LAST" ] && { echo "no checkpoint to resume"; exit 1; }
START=$((10#$(basename $LAST | sed "s/iter_0*//")))
echo "leg starting from $LAST (step $START, target $TARGET)"
[ "$START" -ge "$TARGET" ] && { echo "already complete"; exit 0; }
# Submit successor BEFORE training (this script dies unceremoniously at the
# time limit, so the tail of the script never runs).
sbatch --dependency=afterany:$SLURM_JOB_ID "$0"
source /workspace-vast/celeste/.env
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments
DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k
cd /workspace-vast/celeste/nla-experiments
python -m nla.train_rl_self_contained \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_lora_fixed/iter_0001000 \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_lora_fixed/iter_0001000 \
  --base-ckpt Qwen/Qwen3-8B --quant 4bit \
  --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
  --save-dir $SAVE \
  --resume-from-lora $LAST --start-step $START \
  --num-steps $TARGET --batch-prompts 16 --group-size 16 \
  --max-new-tokens 150 --temperature 1.0 \
  --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 \
  --logp-micro-batch 2 --max-rows 30000 \
  --save-every 50 --eval-every 10 --eval-n-prompts 20 --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-qwen3-8b --wandb-name rl_grpo_fixed_leg$START --seed 0
