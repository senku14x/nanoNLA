#!/bin/bash
#SBATCH --job-name=qwen3_prepcritic
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

bash /workspace-vast/celeste/nla-experiments/launch/prepare_critic.sh
