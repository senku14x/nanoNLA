#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_vllm
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# vLLM-rollout GRPO with TRL-style colocate weight sync.
#
# Same algorithmic config as the v4 HF-generate run (B=16 G=16 paper-faithful,
# co-trained AR, rsLoRA r=128, LRs 1e-5 / 5e-5), but rollouts go through vLLM
# with vllm-lens SteeringVector for Karvonen injection. Expected ~3-5×
# speedup vs HF-generate (rollouts dominate step time).
#
# Weight sync every 20 steps: merge LoRA into base, push state_dict into
# vLLM via collective_rpc("load_weights"), unmerge LoRA. TIS-clip (cap=2.0)
# in the GRPO loss handles residual vLLM/HF engine mismatch.

set -euo pipefail
# Use a dedicated venv so vLLM/vllm-lens doesn't clash with the SGLang/Miles
# stack that the SFT runs need. First-time setup auto-bootstraps.
VENV=/workspace-vast/celeste/envs/nla_vllm
if [ ! -d "$VENV" ]; then
  echo "[setup] bootstrapping $VENV"
  uv venv "$VENV" --python 3.12
  source "$VENV/bin/activate"
  uv pip install --python "$VENV/bin/python" \
    vllm-lens "peft==0.13.0" bitsandbytes wandb pyarrow \
    "transformers>=4.45" safetensors pyyaml numpy huggingface_hub
fi
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

python -m nla.train_rl_vllm \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_vllm \
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
  --critic-micro-batch 4 \
  --max-rows 30000 \
  --save-every 100 \
  --eval-every 10 \
  --eval-n-prompts 20 \
  --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --vllm-gpu-mem 0.35 \
  --vllm-max-len 1024 \
  --vllm-sync-every 20 \
  --tis-cap 2.0 \
  --wandb-project nla-qwen3-8b \
  --wandb-name qwen3_8b_L24_rl_grpo_vllm_v1 \
  --seed 0
