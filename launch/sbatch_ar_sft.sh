#!/bin/bash
#SBATCH --job-name=qwen3_ar_sft
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

bash /workspace-vast/celeste/nla-experiments/launch/run_ar_sft.sh
