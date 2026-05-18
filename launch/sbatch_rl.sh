#!/bin/bash
#SBATCH --job-name=qwen3_rl
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

bash /workspace-vast/celeste/nla-experiments/launch/run_rl.sh
