#!/bin/bash
# Smoke test of the fixed pipeline: AV-SFT -> AR-SFT -> RL -> RL-resume, tiny steps.
set -euo pipefail
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments
cd /workspace-vast/celeste/nla-experiments
DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k
SMOKE=/workspace-vast/celeste/nla-ckpts/smoke_fixed_20260610
rm -rf $SMOKE && mkdir -p $SMOKE

echo "=========== [1/4] AV-SFT smoke (5 steps, LoRA+4bit) ==========="
python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3-8B \
  --parquet $DATA/av_sft_shuf.parquet --sidecar $DATA/av_sft_shuf.parquet \
  --save-dir $SMOKE/av --num-steps 5 --batch-size 8 --max-rows 400 \
  --use-lora --quant 4bit --lr 3e-5 --lr-warmup-steps 2 \
  --save-every 5 --no-wandb --seed 0

echo "=========== [2/4] AR-SFT smoke (5 steps, LoRA+4bit, norm-stripped) ==========="
python -m nla.train_sft --mode ar --base-ckpt Qwen/Qwen3-8B \
  --parquet $DATA/ar_sft_shuf_clean.parquet --sidecar $DATA/ar_sft_shuf_clean.parquet \
  --save-dir $SMOKE/ar --num-steps 5 --batch-size 8 --max-rows 400 \
  --use-lora --quant 4bit --ar-num-layers 25 --lr 3e-5 --lr-warmup-steps 2 \
  --save-every 5 --no-wandb --seed 0
echo "--- ar_meta.json:" && cat $SMOKE/ar/iter_0000005/ar_meta.json

echo "=========== [3/4] RL smoke (2 steps) ==========="
python -m nla.train_rl_self_contained \
  --av-ckpt $SMOKE/av/iter_0000005 --ar-ckpt $SMOKE/ar/iter_0000005 \
  --base-ckpt Qwen/Qwen3-8B --quant 4bit \
  --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
  --save-dir $SMOKE/rl --num-steps 2 --batch-prompts 2 --group-size 4 \
  --max-new-tokens 60 --temperature 1.0 --max-rows 100 \
  --train-critic --critic-lr 5e-5 --lr 1e-5 \
  --eval-every 0 --save-every 1 --no-wandb --seed 0

echo "=========== [4/4] RL resume smoke (1 step from iter_000002) ==========="
ls $SMOKE/rl/iter_000002/ $SMOKE/rl/iter_000002/critic/
python -m nla.train_rl_self_contained \
  --av-ckpt $SMOKE/av/iter_0000005 --ar-ckpt $SMOKE/ar/iter_0000005 \
  --base-ckpt Qwen/Qwen3-8B --quant 4bit \
  --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
  --save-dir $SMOKE/rl --resume-from-lora $SMOKE/rl/iter_000002 --start-step 2 \
  --num-steps 3 --batch-prompts 2 --group-size 4 \
  --max-new-tokens 60 --temperature 1.0 --max-rows 100 \
  --train-critic --critic-lr 5e-5 --lr 1e-5 \
  --eval-every 0 --save-every 1 --no-wandb --seed 0
echo "=========== SMOKE ALL PASS ==========="
